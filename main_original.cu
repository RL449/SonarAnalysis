#include <iostream> // Standard input/output stream operations
#include <string> // Standard string class
#include <filesystem> // File path / directory operations
#include <fstream> // File stream operations
#include <cmath> // Standard math functions
#include <algorithm> // Implements common algorithms
#include <sndfile.h> // Read / write audio files
#include <stdexcept> // Handles exceptions
#include <complex> // Used for FFT / Hilbert transforms
#include <fftw3.h> // FFT computations
#include <cstring> // String operations
#include <thread> // Support multithreading
#include <mutex> // Synchronize threads
#include <cassert> // Debugging
#include <regex> // Parse date / time
#include <ctime> // Output time formatting
#include <memory>
#include <unordered_map>

// Limit thread count to # of cores
#include <queue>
#include <condition_variable>
#include <atomic>

// CUDA
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <cublas_v2.h>

using namespace std; // Standard namespace
namespace fs = filesystem; // Rename filesystem

const int MAX_FILES = 1000;

// Declare structs

struct SampleRange {
    int startSample; // First sample index
    int endSample; // Last sample index

    // Constructor with default range
    SampleRange(int start = 1, int end = -1) {
        startSample = start;
        endSample = end;
    }
};

struct BandpassFilter {
    double* filteredTimeSeries; // Time domain signal after filtering
    double* amplitudeSpectrum; // Frequency domain amplitude spectrum
    int length; // # of samples

    // Constructor
    BandpassFilter(double* ts, double* spec, int len)
        : filteredTimeSeries(ts), amplitudeSpectrum(spec), length(len) {}

    // Destructor
    ~BandpassFilter() {
        delete[] filteredTimeSeries;
        delete[] amplitudeSpectrum;
    }
};

struct Correlation {
    double* correlationValues; // Cross-correlation values between two signals
    double* lags; // Corresponding lag values
    int length; // Length of the arrays

    // Constructor
    Correlation(double* corr, double* lag, int len)
        : correlationValues(corr), lags(lag), length(len) {}

    // Destructor
    ~Correlation() {
        delete[] correlationValues;
        delete[] lags;
    }
};

// Extracted audio features
struct AudioFeatures {
    int* segmentDuration = nullptr; // Duration per segment (seconds)
    double* SPLrms = nullptr; // SPLrms
    double* SPLpk = nullptr; // Peak SPL
    double* impulsivity = nullptr; // Kurtosis
    double* dissim = nullptr; // Dissimilarity between segments
    int* peakcount = nullptr; // # of peaks
    double** autocorr = nullptr; // Autocorrelation matrix

    // # of segments
    int segmentDurationLen = 0;
    int SPLrmsLen = 0;
    int SPLpkLen = 0;
    int impulsivityLen = 0;
    int dissimLen = 0;
    int peakcountLen = 0;
    int autocorrRows = 0; // Time segments
    int autocorrCols = 0; // Lags
};

struct AudioData {
    double** samples; // 2D array of audio samples [channel][frame]
    int numChannels; // # of audio channels
    int numFrames; // # of frames per channel
    int sampleRate; // Sample rate (Hz)
};

struct AudioInfo {
    int sampleRate; // Sampling rate (Hz)
    double duration; // Duration of audio (seconds)
};

// Periodicity / impulsivity
struct SoloPerGM2 {
    int* peakcount; // # of peaks per time window
    double** autocorr; // Autocorrelation per segment
    int peakcount_length; // Length of peakcount array
    int autocorr_rows; // # of time windows - rows
    int autocorr_cols; // # of lags - columns
};

struct ArrayShiftFFT {
    double* data; // Array of samples after shift
    int length; // Length of array

    // Destructor
    ~ArrayShiftFFT() {
        delete[] data;
    }
};

// RAII wrapper for FFTW complex buffer + plan
struct FFTWHandler {
    fftw_complex* buf = nullptr;
    fftw_plan fwd_plan = nullptr;
    fftw_plan inv_plan = nullptr;
    int size = 0;

    FFTWHandler(int N) : size(N) {
        buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        if (!buf) { throw bad_alloc(); }

        fwd_plan = fftw_plan_dft_1d(size, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
        if (!fwd_plan) {
            fftw_free(buf);
            throw runtime_error("FFTW forward plan creation failed");
        }

        inv_plan = fftw_plan_dft_1d(size, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!inv_plan) {
            fftw_destroy_plan(fwd_plan);
            fftw_free(buf);
            throw runtime_error("FFTW inverse plan creation failed");
        }
    }

    ~FFTWHandler() {
        if (fwd_plan) { fftw_destroy_plan(fwd_plan); }
        if (inv_plan) { fftw_destroy_plan(inv_plan); }
        if (buf) { fftw_free(buf); }
    }
};

struct ThreadArgs {
    atomic<int>* nextIndex;
    int totalFiles;
    char (*filePaths)[512];
    AudioFeatures* allFeatures;
    char (*filenames)[512];
    int num_bits, peak_volts, timewin, fft_win, arti, flow, fhigh, downsample;
    double RS, avtime;
    bool omit_partial_minute;
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#define CUFFT_CHECK(call) \
    do { \
        cufftResult error = call; \
        if (error != CUFFT_SUCCESS) { \
            fprintf(stderr, "CUFFT error at %s:%d - %d\n", __FILE__, __LINE__, error); \
            exit(1); \
        } \
    } while(0)

// Replaces backslashes with forward slashes to work with Windows file paths
string fixFilePath(const string& path) {
    string fixedPath = path;
    replace(fixedPath.begin(), fixedPath.end(), '\\', '/');
    return fixedPath;
}

// Read audio samples / extract recording metadata
AudioData audioread(const string& filename, SampleRange range = {1, -1}) {
    SNDFILE* file; // Sound file
    SF_INFO sfinfo = {}; // Sound metadata

    file = sf_open(filename.c_str(), SFM_READ, &sfinfo); // Open file in read mode
    if (!file) { throw runtime_error("Error opening audio file: " + string(sf_strerror(file))); }

    // Sample range to read
    int totalFrames = sfinfo.frames; // Frames per channel
    int endSample;
    if (range.endSample == -1) { endSample = totalFrames; } // No range specified
    else { endSample = min(range.endSample, totalFrames); }
    
    int startSample = max(0, range.startSample - 1);
    int numFramesToRead = endSample - startSample;

    if (numFramesToRead <= 0) { // Invalid arguments provided
        sf_close(file);
        throw runtime_error("Invalid sample range");
    }

    sf_seek(file, startSample, SEEK_SET); // Starting frame in sound file

    int numChannels = sfinfo.channels; // # of audio channels
    double* interleavedSamples = new double[numFramesToRead * numChannels]; // Samples from all channels

    int format = sfinfo.format & SF_FORMAT_SUBMASK; // Determine bit depth

    switch (format) {
        case SF_FORMAT_PCM_16: { // 16 bit
            short* temp = new short[numFramesToRead * numChannels];
            sf_readf_short(file, temp, numFramesToRead);
            for (int i = 0; i < numFramesToRead * numChannels; ++i) { interleavedSamples[i] = static_cast<double>(temp[i]); }
            delete[] temp;
            break;
        }
        case SF_FORMAT_PCM_24:
        case SF_FORMAT_PCM_32: { // 24 or 32 bit
            int* temp = new int[numFramesToRead * numChannels];
            sf_readf_int(file, temp, numFramesToRead);
            for (int i = 0; i < numFramesToRead * numChannels; ++i) { interleavedSamples[i] = static_cast<double>(temp[i]); }
            delete[] temp;
            break;
        }
        case SF_FORMAT_FLOAT: { // 32 bit float
            float* temp = new float[numFramesToRead * numChannels];
            sf_readf_float(file, temp, numFramesToRead);
            for (int i = 0; i < numFramesToRead * numChannels; ++i) { interleavedSamples[i] = static_cast<double>(temp[i]); }
            delete[] temp;
            break;
        }
        case SF_FORMAT_DOUBLE: { // 64 bit double
            sf_readf_double(file, interleavedSamples, numFramesToRead);
            break;
        }
        default:
            // Unsupported bit depth
            sf_close(file);
            delete[] interleavedSamples;
            throw runtime_error("Unsupported bit format");
    }

    sf_close(file);

    // Channel data matrix
    double** samples = new double*[numChannels];
    for (int ch = 0; ch < numChannels; ++ch) { samples[ch] = new double[numFramesToRead]; }

    // Separate indices per channel
    for (int i = 0; i < numFramesToRead; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) { samples[ch][i] = interleavedSamples[i * numChannels + ch]; }
    }

    delete[] interleavedSamples; // Deallocate memory

    return AudioData{samples, numChannels, numFramesToRead, sfinfo.samplerate}; // Metadata
}

AudioInfo audioread_info(const string& file_path) {
    SF_INFO sfInfo = {0}; // Struct containing sound metadata (frames, samplerate, channels, format)
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sfInfo); // Open audio file in read mode

    if (!file) { throw runtime_error("Error opening audio file: " + file_path); } // Error opening file

    int sampleRate = sfInfo.samplerate; // Get sample rate
    int numFrames = sfInfo.frames; // Get # of frames

    float duration = static_cast<float>(numFrames) / sampleRate; // Calculate duration (seconds)

    sf_close(file); // Close file after reading info

    return {sampleRate, duration};
}

__global__ void downsample_kernel(const double* x, double* result, int length, int factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * factor < length) { result[idx] = x[idx * factor]; }
}

// Reduce sampling rate by analyzing (1 / factor) samples
double* downsample(const double* x, int length, int factor, int& newLength) {
    if (factor <= 0) { throw invalid_argument("Factor must be positive"); }

    newLength = (length + factor - 1) / factor;

    // Allocate GPU memory
    double* d_input;
    double* d_output;
    cudaMalloc(&d_input, sizeof(double) * length);
    cudaMalloc(&d_output, sizeof(double) * newLength);

    // Copy input to device
    cudaMemcpy(d_input, x, sizeof(double) * length, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256;
    int blocks = (newLength + threads - 1) / threads;
    downsample_kernel << <blocks, threads >> > (d_input, d_output, length, factor);

    // Copy result back to host
    double* result = new double[newLength];
    cudaMemcpy(result, d_output, sizeof(double) * newLength, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return result;
}

__global__ void fftshift_kernel(const double* input, double* shifted, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) { shifted[i] = input[(i + (length / 2)) % length]; }
}

// Manually shift zero-frequency to center of array
ArrayShiftFFT fftshift(double* input, int length) {
    double* d_input;
    double* d_output;

    cudaMalloc(&d_input, sizeof(double) * length);
    cudaMalloc(&d_output, sizeof(double) * length);

    cudaMemcpy(d_input, input, sizeof(double) * length, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (length + threads - 1) / threads;
    fftshift_kernel << <blocks, threads >> > (d_input, d_output, length);

    double* shifted = new double[length];
    cudaMemcpy(shifted, d_output, sizeof(double) * length, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    return { shifted, length };
}

// CUDA kernel to compute shifted frequency array and apply bandpass filter
__global__ void apply_bandpass_filter(cufftDoubleComplex* freq_data, int N, double freq_step, double flow, double fhigh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) { return; }

    int shifted_index = (i + N / 2) % N;
    double freq = (shifted_index - N / 2) * freq_step;
    double abs_freq = fabs(freq);

    if (abs_freq < flow || abs_freq > fhigh) {
        freq_data[i].x = 0.0;
        freq_data[i].y = 0.0;
    }
}

// CUDA kernel to normalize inverse FFT and compute amplitude spectrum
__global__ void normalize_and_compute_amplitude(const cufftDoubleComplex* time_data, double* output_time, double* output_amp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) { return; }

    double norm = 1.0 / N;
    output_time[i] = time_data[i].x * norm;
    output_amp[i] = sqrt(time_data[i].x * time_data[i].x + time_data[i].y * time_data[i].y);
}

// Apply bandpass filter in frequency domain
BandpassFilter bandpass_filter(const double* time_series, int num_pts, double frequency, double flow, double fhigh) {
    // Allocate memory on device
    cufftDoubleComplex* d_freq_data;
    cudaMalloc(&d_freq_data, sizeof(cufftDoubleComplex) * num_pts);

    // Copy time_series to device, real part only
    cufftDoubleComplex* h_input = new cufftDoubleComplex[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        h_input[i].x = time_series[i];
        h_input[i].y = 0.0;
    }
    cudaMemcpy(d_freq_data, h_input, sizeof(cufftDoubleComplex) * num_pts, cudaMemcpyHostToDevice);
    delete[] h_input;

    // Execute forward FFT
    cufftHandle plan_fwd;
    cufftPlan1d(&plan_fwd, num_pts, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan_fwd, d_freq_data, d_freq_data, CUFFT_FORWARD);

    // Apply bandpass filter
    double reclen = num_pts * frequency;
    double freq_step = 1.0 / reclen;
    if (fhigh == 0.0) { fhigh = 0.5 / frequency; }

    int threads = 256;
    int blocks = (num_pts + threads - 1) / threads;
    apply_bandpass_filter << <blocks, threads >> > (d_freq_data, num_pts, freq_step, flow, fhigh);
    cudaDeviceSynchronize();

    // Execute inverse FFT
    cufftExecZ2Z(plan_fwd, d_freq_data, d_freq_data, CUFFT_INVERSE); // Reuse plan

    // Allocate output arrays
    double* d_time_out;
    double* d_amp_out;
    cudaMalloc(&d_time_out, sizeof(double) * num_pts);
    cudaMalloc(&d_amp_out, sizeof(double) * num_pts);

    normalize_and_compute_amplitude << <blocks, threads >> > (d_freq_data, d_time_out, d_amp_out, num_pts);
    cudaDeviceSynchronize();

    // Copy results to host
    double* time_series_filt = new double[num_pts];
    double* amp_spectrum = new double[num_pts];
    cudaMemcpy(time_series_filt, d_time_out, sizeof(double) * num_pts, cudaMemcpyDeviceToHost);
    cudaMemcpy(amp_spectrum, d_amp_out, sizeof(double) * num_pts, cudaMemcpyDeviceToHost);

    // Cleanup
    cufftDestroy(plan_fwd);
    cudaFree(d_freq_data);
    cudaFree(d_time_out);
    cudaFree(d_amp_out);

    return BandpassFilter(time_series_filt, amp_spectrum, num_pts);
}

__global__ void partial_sums_kernel(const double* data, double* sum_out, double* sum_sq_out, int N) {
    __shared__ double local_sum[256];
    __shared__ double local_sum_sq[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = (i < N) ? data[i] : 0.0;
    local_sum[tid] = val;
    local_sum_sq[tid] = val * val;
    __syncthreads();

    // Reduce within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            local_sum[tid] += local_sum[tid + s];
            local_sum_sq[tid] += local_sum_sq[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        sum_out[blockIdx.x] = local_sum[0];
        sum_sq_out[blockIdx.x] = local_sum_sq[0];
    }
}

__global__ void fourth_moment_kernel(const double* data, double* fourth_out, double mean, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) { return; }

    double centered = data[i] - mean;
    fourth_out[i] = centered * centered * centered * centered;
}

// Calculate kurtosis used for impulsivity of a signal
double calculate_kurtosis(const double* h_data, int N) {
    if (N <= 0 || h_data == nullptr) { throw invalid_argument("Input array is empty or null"); }

    // Allocate device memory
    double* d_data, * d_sum_partial, * d_sum_sq_partial, * d_fourth;
    cudaMalloc(&d_data, N * sizeof(double));
    cudaMemcpy(d_data, h_data, N * sizeof(double), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaMalloc(&d_sum_partial, blocks * sizeof(double));
    cudaMalloc(&d_sum_sq_partial, blocks * sizeof(double));

    // First pass: Compute mean / variance components
    partial_sums_kernel << <blocks, threads >> > (d_data, d_sum_partial, d_sum_sq_partial, N);
    cudaDeviceSynchronize();

    thrust::device_ptr<double> sum_ptr(d_sum_partial);
    thrust::device_ptr<double> sum_sq_ptr(d_sum_sq_partial);

    double total_sum = thrust::reduce(sum_ptr, sum_ptr + blocks, 0.0, thrust::plus<double>());
    double total_sum_sq = thrust::reduce(sum_sq_ptr, sum_sq_ptr + blocks, 0.0, thrust::plus<double>());

    double mean = total_sum / N;
    double mean_sq = mean * mean;
    double variance = (total_sum_sq / N) - mean_sq;

    if (variance < 1e-12) {
        cudaFree(d_data); cudaFree(d_sum_partial); cudaFree(d_sum_sq_partial);
        return 0.0;
    }

    // Second pass: Compute fourth moment
    cudaMalloc(&d_fourth, N * sizeof(double));
    fourth_moment_kernel << <blocks, threads >> > (d_data, d_fourth, mean, N);
    cudaDeviceSynchronize();

    thrust::device_ptr<double> fourth_ptr(d_fourth);
    double fourth_moment = thrust::reduce(fourth_ptr, fourth_ptr + N, 0.0, thrust::plus<double>()) / N;

    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sum_partial);
    cudaFree(d_sum_sq_partial);
    cudaFree(d_fourth);

    return fourth_moment / (variance * variance);
}

// CUDA kernel for envelope calculation
__global__ void envelope_kernel(const cufftDoubleComplex* hilbert, double* envelope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        envelope[idx] = sqrt(hilbert[idx].x * hilbert[idx].x + hilbert[idx].y * hilbert[idx].y);
    }
}

// CUDA kernel for correlation computation
__global__ void correlation_kernel(const double* x, const double* y, int series_length, 
                                 double* corr_vals, int max_lag, int offset) {
    int lag = blockIdx.x * blockDim.x + threadIdx.x;
    if (lag > max_lag) { return; }
    
    double sumX = 0.0, sumY = 0.0, sumXSquare = 0.0, sumYSquare = 0.0, sumXYProd = 0.0;
    int sampleCount = 0;
    
    for (int k = 0; k < series_length - (lag + offset); k++) {
        double x_val = x[k];
        double y_val = y[k + lag + offset];
        
        if (!isnan(x_val) && !isnan(y_val)) {
            sumX += x_val;
            sumY += y_val;
            sumXSquare += x_val * x_val;
            sumYSquare += y_val * y_val;
            sumXYProd += x_val * y_val;
            sampleCount++;
        }
    }
    
    if (sampleCount == 0) {
        corr_vals[lag] = NAN;
        return;
    }
    
    double meanX = sumX / sampleCount;
    double meanY = sumY / sampleCount;
    double meanXSquare = sumXSquare / sampleCount;
    double meanYSquare = sumYSquare / sampleCount;
    
    double covar = (sumXYProd / sampleCount) - (meanX * meanY);
    double denomX = sqrt(meanXSquare - (meanX * meanX));
    double denomY = sqrt(meanYSquare - (meanY * meanY));
    
    if (denomX == 0.0 || denomY == 0.0) { corr_vals[lag] = NAN; }
    else { corr_vals[lag] = covar / (denomX * denomY); }
}

// CUDA kernel for FFT magnitude calculation
__global__ void fft_magnitude_kernel(const cufftDoubleComplex* fft_data, double* magnitude, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {  magnitude[idx] = sqrt(fft_data[idx].x * fft_data[idx].x + fft_data[idx].y * fft_data[idx].y); }
}

// GPU-accelerated correlation function
Correlation correl_5_gpu(const double* time_series1, const double* time_series2, 
            int series_length, int lags, int offset) {
    int len = lags + 1;
    
    // Allocate GPU memory
    double *d_x, *d_y, *d_corr_vals;
    CUDA_CHECK(cudaMalloc(&d_x, sizeof(double) * series_length));
    CUDA_CHECK(cudaMalloc(&d_y, sizeof(double) * series_length));
    CUDA_CHECK(cudaMalloc(&d_corr_vals, sizeof(double) * len));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_x, time_series1, sizeof(double) * series_length, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, time_series2, sizeof(double) * series_length, cudaMemcpyHostToDevice));
    
    // Launch correlation kernel
    dim3 block(256);
    dim3 grid((len + block.x - 1) / block.x);
    correlation_kernel<<<grid, block>>>(d_x, d_y, series_length, d_corr_vals, lags, offset);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Allocate host memory / copy results back
    double* corr_vals = new double[len];
    double* lag_vals = new double[len];
    
    CUDA_CHECK(cudaMemcpy(corr_vals, d_corr_vals, sizeof(double) * len, cudaMemcpyDeviceToHost));
    
    // Fill lag values
    for (int i = 0; i < len; ++i) { lag_vals[i] = static_cast<double>(i); }
    
    // Cleanup GPU memory
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_corr_vals));
    
    return Correlation(corr_vals, lag_vals, len);
}

// Kernel to square / segment the input
__global__ void square_and_segment(const double* input, double* output, int samp_window_size, int num_time_wins) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = samp_window_size * num_time_wins;

    if (idx < total) {
        double val = input[idx];
        output[idx] = val * val;
    }
}

// Kernel to average squared values
__global__ void compute_averages(const double* squared, double* output_avg, int samp_window_size, int avg_win_size, int numavwin, int num_time_wins) {
    int seg = blockIdx.x;   // segment index
    int win = threadIdx.x;  // averaging window index

    if (seg < num_time_wins && win < numavwin) {
        int base_idx = seg * samp_window_size + win * avg_win_size;
        double sum = 0.0;
        for (int i = 0; i < avg_win_size; i++) { sum += squared[base_idx + i]; }
        output_avg[seg * numavwin + win] = sum / avg_win_size;
    }
}

// Calculate autocorrelation / peak counts
SoloPerGM2 f_solo_per_GM2(const double* p_filt_input, int input_length, double fs, double timewin, double avtime) {
    int samp_window_size = static_cast<int>(fs * timewin);
    int num_time_wins = input_length / samp_window_size;
    if (num_time_wins == 0) { throw runtime_error("Empty time window"); }

    int total_samples = samp_window_size * num_time_wins;
    int avg_win_size = static_cast<int>(fs * avtime);
    int numavwin = samp_window_size / avg_win_size;

    // GPU memory allocations
    double* d_input, * d_squared, * d_avg;
    cudaMalloc(&d_input, total_samples * sizeof(double));
    cudaMalloc(&d_squared, total_samples * sizeof(double));
    cudaMalloc(&d_avg, num_time_wins * numavwin * sizeof(double));

    cudaMemcpy(d_input, p_filt_input, total_samples * sizeof(double), cudaMemcpyHostToDevice);

    // Launch square + segment kernel
    int threads = 256;
    int blocks = (total_samples + threads - 1) / threads;
    square_and_segment << <blocks, threads >> > (d_input, d_squared, samp_window_size, num_time_wins);

    // Launch average kernel
    compute_averages << <num_time_wins, numavwin >> > (d_squared, d_avg, samp_window_size, avg_win_size, numavwin, num_time_wins);

    // Copy averages back to host
    double* h_avg = new double[num_time_wins * numavwin];
    cudaMemcpy(h_avg, d_avg, num_time_wins * numavwin * sizeof(double), cudaMemcpyDeviceToHost);

    // Correlation + peak count (CPU, per segment)
    int p_avtot_rows = numavwin;
    int lag_limit = static_cast<int>(p_avtot_rows * 0.7);
    int p_avtot_cols = num_time_wins;

    double** acorr = new double* [p_avtot_cols];
    int* pkcount = new int[p_avtot_cols];

    for (int zz = 0; zz < p_avtot_cols; zz++) {
        Correlation corr_result = correl_5_gpu(&h_avg[zz * numavwin], &h_avg[zz * numavwin], p_avtot_rows, lag_limit, 0);
        acorr[zz] = new double[lag_limit + 1];
        copy(corr_result.correlationValues, corr_result.correlationValues + lag_limit + 1, acorr[zz]);

        // Peak counting
        int peak_count = 0;
        for (int i = 1; i < lag_limit; i++) {
            if (acorr[zz][i] > acorr[zz][i - 1] && acorr[zz][i] > acorr[zz][i + 1]) {
                double left_min = acorr[zz][i];
                for (int j = i - 1; j >= 0 && acorr[zz][j] < acorr[zz][i]; j--) { left_min = min(left_min, acorr[zz][j]); }
                double right_min = acorr[zz][i];
                for (int j = i + 1; j <= lag_limit && acorr[zz][j] < acorr[zz][i]; j++) { right_min = min(right_min, acorr[zz][j]); }
                double prominence = acorr[zz][i] - max(left_min, right_min);
                if (prominence > 0.5) { peak_count++; }
            }
        }
        pkcount[zz] = peak_count;
    }

    // Free GPU memory
    cudaFree(d_input);
    cudaFree(d_squared);
    cudaFree(d_avg);
    delete[] h_avg;

    // Return result
    SoloPerGM2 result;
    result.peakcount = pkcount;
    result.autocorr = acorr;
    result.peakcount_length = p_avtot_cols;
    result.autocorr_rows = p_avtot_cols;
    result.autocorr_cols = lag_limit + 1;

    return result;
}

// CUDA kernel for Hilbert transform filter application
__global__ void hilbert_filter_kernel(cufftDoubleComplex* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) { return; }
    
    int half = n / 2;
    int upper = (n % 2 == 0) ? half - 1 : half;
    
    if (idx >= 1 && idx <= upper) {
        // Multiply positive frequencies by 2
        data[idx].x *= 2.0;
        data[idx].y *= 2.0;
    } else if (idx > half) {
        // Zero out negative frequencies
        data[idx].x = 0.0;
        data[idx].y = 0.0;
    }
}

// GPU-accelerated Hilbert transform
#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err)  \
                      << "\"" << endl;                                   \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

#define CUFFT_CHECK(call)                                                     \
    {                                                                         \
        cufftResult err = call;                                               \
        if (err != CUFFT_SUCCESS) {                                           \
            cerr << "CUFFT error at " << __FILE__ << ":" << __LINE__    \
                      << " code=" << err << endl;                        \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

// CUDA kernel to normalize FFT result in-place
__global__ void normalize_kernel(cufftDoubleComplex* data, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < length) {
        data[idx].x /= length;
        data[idx].y /= length;
    }
}

__global__ void initialize_complex(double* input, cufftDoubleComplex* output, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        output[i].x = input[i];
        output[i].y = 0.0;
    }
}

__global__ void normalize_result(cufftDoubleComplex* data, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        data[i].x /= len;
        data[i].y /= len;
    }
}

fftw_complex* hilbert_raw_gpu(const double* input, int input_len) {
    if (!input || input_len <= 0) {
        cerr << "Invalid input\n";
        return nullptr;
    }

    cufftDoubleComplex* d_data = nullptr;
    CUDA_CHECK(cudaMalloc(&d_data, sizeof(cufftDoubleComplex) * input_len));

    double* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, sizeof(double) * input_len));
    CUDA_CHECK(cudaMemcpy(d_input, input, sizeof(double) * input_len, cudaMemcpyHostToDevice));

    // Initialize d_data = input + 0j
    dim3 block(256);
    dim3 grid((input_len + block.x - 1) / block.x);
    initialize_complex << <grid, block >> > (d_input, d_data, input_len);
    CUDA_CHECK(cudaFree(d_input)); // Free d_input early

    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, input_len, CUFFT_Z2Z, 1));
    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD));

    hilbert_filter_kernel << <grid, block >> > (d_data, input_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUFFT_CHECK(cufftExecZ2Z(plan, d_data, d_data, CUFFT_INVERSE));

    // Normalize in-place
    normalize_result << <grid, block >> > (d_data, input_len);

    // Copy result to host
    fftw_complex* result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * input_len);
    if (!result) {
        cerr << "Host allocation failed\n";
        cudaFree(d_data);
        cufftDestroy(plan);
        return nullptr;
    }

    CUDA_CHECK(cudaMemcpy(result, d_data, sizeof(cufftDoubleComplex) * input_len, cudaMemcpyDeviceToHost));

    // Cleanup
    cudaFree(d_data);
    cufftDestroy(plan);

    return result;
}

// GPU-accelerated dissimilarity function
double* f_solo_dissim_GM1_gpu(double** timechunk_matrix, int pts_per_timewin, int num_timewin,
    double fft_win, double fs, int& out_len) {
    
    int pts_per_fft = static_cast<int>(fft_win * fs);
    if (pts_per_fft <= 0 || pts_per_timewin <= 0 || num_timewin <= 1) {
        out_len = 0;
        return nullptr;
    }

    int numfftwin = (pts_per_timewin - pts_per_fft) / pts_per_fft + 1;
    if (numfftwin <= 0) {
        out_len = 0;
        return nullptr;
    }

    out_len = num_timewin - 1;

    // Create CUFFT plan
    cufftHandle fft_plan;
    if (cufftPlan1d(&fft_plan, pts_per_fft, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        out_len = 0;
        return nullptr;
    }

    // Allocate GPU buffers
    cufftDoubleComplex* d_fft_input = nullptr;
    cufftDoubleComplex* d_fft_output = nullptr;
    double* d_magnitude = nullptr;
    double* d_envelope1 = nullptr;
    double* d_envelope2 = nullptr;
    double* diss = nullptr;

    // Allocate GPU memory with error checking
    if (cudaMalloc(&d_fft_input, sizeof(cufftDoubleComplex) * pts_per_fft) != cudaSuccess ||
        cudaMalloc(&d_fft_output, sizeof(cufftDoubleComplex) * pts_per_fft) != cudaSuccess ||
        cudaMalloc(&d_magnitude, sizeof(double) * pts_per_fft) != cudaSuccess ||
        cudaMalloc(&d_envelope1, sizeof(double) * pts_per_timewin) != cudaSuccess ||
        cudaMalloc(&d_envelope2, sizeof(double) * pts_per_timewin) != cudaSuccess) {

        // Clean up allocated memory
        if (d_fft_input) { cudaFree(d_fft_input); }
        if (d_fft_output) { cudaFree(d_fft_output); }
        if (d_magnitude) { cudaFree(d_magnitude); }
        if (d_envelope1) { cudaFree(d_envelope1); }
        if (d_envelope2) { cudaFree(d_envelope2); }
        cufftDestroy(fft_plan);
        out_len = 0;
        return nullptr;
    }

    // Allocate host result array
    diss = new(nothrow) double[out_len];
    if (!diss) {
        cudaFree(d_fft_input);
        cudaFree(d_fft_output);
        cudaFree(d_magnitude);
        cudaFree(d_envelope1);
        cudaFree(d_envelope2);
        cufftDestroy(fft_plan);
        out_len = 0;
        return nullptr;
    }

    // Pre-allocate host vectors to avoid repeated allocation/deallocation
    vector<cufftDoubleComplex> hil1_host(pts_per_timewin);
    vector<cufftDoubleComplex> hil2_host(pts_per_timewin);
    vector<cufftDoubleComplex> fft_input_host(pts_per_fft);

    dim3 block(256);
    dim3 grid_fft((pts_per_fft + block.x - 1) / block.x);
    dim3 grid_env((pts_per_timewin + block.x - 1) / block.x);

    for (int i = 0; i < out_len; ++i) {
        fftw_complex* hil1 = hilbert_raw_gpu(timechunk_matrix[i], pts_per_timewin);
        fftw_complex* hil2 = hilbert_raw_gpu(timechunk_matrix[i + 1], pts_per_timewin);

        if (!hil1 || !hil2) {
            diss[i] = NAN;
            if (hil1) { fftw_free(hil1); }
            if (hil2) { fftw_free(hil2); }
            continue;
        }

        // Create device vectors for hilbert transforms
        thrust::device_vector<cufftDoubleComplex> gpu_hil1(pts_per_timewin);
        thrust::device_vector<cufftDoubleComplex> gpu_hil2(pts_per_timewin);

        // Convert / copy hilbert data to device
        for (int k = 0; k < pts_per_timewin; ++k) {
            hil1_host[k].x = hil1[k][0];
            hil1_host[k].y = hil1[k][1];
            hil2_host[k].x = hil2[k][0];
            hil2_host[k].y = hil2[k][1];
        }

        gpu_hil1 = hil1_host;
        gpu_hil2 = hil2_host;

        // Free hilbert transforms
        fftw_free(hil1);
        fftw_free(hil2);

        // Calculate envelopes
        envelope_kernel << <grid_env, block >> > (thrust::raw_pointer_cast(gpu_hil1.data()), d_envelope1, pts_per_timewin);
        envelope_kernel << <grid_env, block >> > (thrust::raw_pointer_cast(gpu_hil2.data()), d_envelope2, pts_per_timewin);

        if (cudaDeviceSynchronize() != cudaSuccess) {
            diss[i] = NAN;
            continue;
        }

        // Create device vectors for envelopes
        thrust::device_vector<double> env1(pts_per_timewin);
        thrust::device_vector<double> env2(pts_per_timewin);

        // Copy envelope data
        thrust::copy(thrust::device_pointer_cast(d_envelope1),
            thrust::device_pointer_cast(d_envelope1) + pts_per_timewin,
            env1.begin());
        thrust::copy(thrust::device_pointer_cast(d_envelope2),
            thrust::device_pointer_cast(d_envelope2) + pts_per_timewin,
            env2.begin());

        // Normalize envelopes
        double sum1 = thrust::reduce(env1.begin(), env1.end(), 0.0);
        double sum2 = thrust::reduce(env2.begin(), env2.end(), 0.0);

        if (sum1 > 1e-12) {
            thrust::transform(env1.begin(), env1.end(), env1.begin(),
                [sum1] __device__(double x) { return x / sum1; });
        }
        if (sum2 > 1e-12) {
            thrust::transform(env2.begin(), env2.end(), env2.begin(),
                [sum2] __device__(double x) { return x / sum2; });
        }

        // Calculate time dissimilarity
        double timeDiss = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(env1.begin(), env2.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(env1.end(), env2.end())),
            [] __device__(const thrust::tuple<double, double>&t) -> double {
            return fabs(thrust::get<0>(t) - thrust::get<1>(t));
        },
            0.0,
            thrust::plus<double>()) * 0.5;

        // Calculate frequency dissimilarity
        thrust::device_vector<double> fftA(pts_per_fft, 0.0);
        thrust::device_vector<double> fftB(pts_per_fft, 0.0);

        for (int w = 0; w < numfftwin; ++w) {
            int base = w * pts_per_fft;

            // Process first time chunk
            for (int j = 0; j < pts_per_fft; ++j) {
                fft_input_host[j].x = timechunk_matrix[i][base + j];
                fft_input_host[j].y = 0.0;
            }

            if (cudaMemcpy(d_fft_input, fft_input_host.data(),
                sizeof(cufftDoubleComplex) * pts_per_fft, cudaMemcpyHostToDevice) != cudaSuccess) {
                continue;
            }

            if (cufftExecZ2Z(fft_plan, d_fft_input, d_fft_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
                continue;
            }

            fft_magnitude_kernel << <grid_fft, block >> > (d_fft_output, d_magnitude, pts_per_fft);

            if (cudaDeviceSynchronize() != cudaSuccess) {  continue; }

            // Add to fftA
            thrust::device_vector<double> temp_mag(pts_per_fft);
            thrust::copy(thrust::device_pointer_cast(d_magnitude),
                thrust::device_pointer_cast(d_magnitude) + pts_per_fft,
                temp_mag.begin());
            thrust::transform(fftA.begin(), fftA.end(), temp_mag.begin(), fftA.begin(), thrust::plus<double>());

            // Process second time chunk
            for (int j = 0; j < pts_per_fft; ++j) {
                fft_input_host[j].x = timechunk_matrix[i + 1][base + j];
                fft_input_host[j].y = 0.0;
            }

            if (cudaMemcpy(d_fft_input, fft_input_host.data(),
                sizeof(cufftDoubleComplex) * pts_per_fft, cudaMemcpyHostToDevice) != cudaSuccess) {
                continue;
            }

            if (cufftExecZ2Z(fft_plan, d_fft_input, d_fft_output, CUFFT_FORWARD) != CUFFT_SUCCESS) {
                continue;
            }

            fft_magnitude_kernel << <grid_fft, block >> > (d_fft_output, d_magnitude, pts_per_fft);

            if (cudaDeviceSynchronize() != cudaSuccess) { continue; }

            // Add to fftB
            thrust::copy(thrust::device_pointer_cast(d_magnitude),
                thrust::device_pointer_cast(d_magnitude) + pts_per_fft,
                temp_mag.begin());
            thrust::transform(fftB.begin(), fftB.end(), temp_mag.begin(), fftB.begin(), thrust::plus<double>());
        }

        // Normalize frequency spectra
        double totalA = thrust::reduce(fftA.begin(), fftA.end(), 0.0);
        double totalB = thrust::reduce(fftB.begin(), fftB.end(), 0.0);

        if (totalA > 1e-12) {
            thrust::transform(fftA.begin(), fftA.end(), fftA.begin(),
                [totalA] __device__(double x) { return x / totalA; });
        }
        if (totalB > 1e-12) {
            thrust::transform(fftB.begin(), fftB.end(), fftB.begin(),
                [totalB] __device__(double x) { return x / totalB; });
        }

        // Calculate frequency dissimilarity
        double freqDiss = thrust::transform_reduce(
            thrust::make_zip_iterator(thrust::make_tuple(fftA.begin(), fftB.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(fftA.end(), fftB.end())),
            [] __device__(const thrust::tuple<double, double>&t) -> double {
            return fabs(thrust::get<0>(t) - thrust::get<1>(t));
        },
            0.0,
            thrust::plus<double>()) * 0.5;

        diss[i] = timeDiss * freqDiss;
    }

    // Final cleanup
    cudaFree(d_fft_input);
    cudaFree(d_fft_output);
    cudaFree(d_magnitude);
    cudaFree(d_envelope1);
    cudaFree(d_envelope2);
    cufftDestroy(fft_plan);

    return diss;
}

void freeAudioData(AudioData& audio) {
    for (int ch = 0; ch < audio.numChannels; ++ch) { delete[] audio.samples[ch]; }
    delete[] audio.samples;
}

void freeAudioFeatures(AudioFeatures& features) {
    delete[] features.segmentDuration;
    delete[] features.SPLrms;
    delete[] features.SPLpk;
    delete[] features.impulsivity;
    delete[] features.dissim;
    delete[] features.peakcount;

    if (features.autocorr) {
        for (int i = 0; i < features.autocorrRows; ++i) { delete[] features.autocorr[i]; }
        delete[] features.autocorr;
    }

    // Reset pointers
    features.segmentDuration = nullptr;
    features.SPLrms = nullptr;
    features.SPLpk = nullptr;
    features.impulsivity = nullptr;
    features.dissim = nullptr;
    features.peakcount = nullptr;
    features.autocorr = nullptr;
}

__global__ void convertToPressureKernel(
    const double* samples, double* pressure, int numSamples,
    int num_bits, double peak_volts, double refSens) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSamples) { return; }

    double s = samples[idx];

    if (num_bits == 24) { s = static_cast<double>(static_cast<int>(s) >> 8); }
    else if (num_bits == 32) { s = static_cast<double>(static_cast<int>(s) >> 16); }

    pressure[idx] = s * (peak_volts / static_cast<double>(1 << num_bits)) * (1.0 / pow(10.0, refSens / 20.0));
}

void gpu_convert_to_pressure(
    const double* h_samples, double* h_pressure, int length,
    int num_bits, double peak_volts, double refSens) {

    double* d_samples;
    double* d_pressure;
    cudaMalloc(&d_samples, sizeof(double) * length);
    cudaMalloc(&d_pressure, sizeof(double) * length);

    cudaMemcpy(d_samples, h_samples, sizeof(double) * length, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    convertToPressureKernel << <blocks, threadsPerBlock >> > (
        d_samples, d_pressure, length, num_bits, peak_volts, refSens);

    cudaMemcpy(h_pressure, d_pressure, sizeof(double) * length, cudaMemcpyDeviceToHost);
    cudaFree(d_samples);
    cudaFree(d_pressure);
}

// Main feature extraction
AudioFeatures feature_extraction(int num_bits, int peak_volts, const fs::path& file_path,
    double refSens, int timewin, double avtime, int fft_win,
    int calTone, int flow, int fhigh, int downsample_factor, bool omit_partial_minute) {

    string fixed_file_path = fixFilePath(file_path.string());
    AudioInfo info = audioread_info(fixed_file_path);

    if (omit_partial_minute) { info.duration = floor(info.duration / 60.0) * 60.0; }

    int total_samples = static_cast<int>(info.sampleRate * info.duration);
    AudioData audio = audioread(file_path.string(), SampleRange{ 1, total_samples });

    int fs = audio.sampleRate;
    int audioSamplesLen = audio.numFrames;

    double* pressure = nullptr;
    try { pressure = new double[audioSamplesLen]; }
    catch (bad_alloc& e) {
        AudioFeatures empty_features = {};
        return empty_features;
    }

    double* flat_samples = audio.samples[0];
    gpu_convert_to_pressure(flat_samples, pressure, audioSamplesLen, num_bits, peak_volts, refSens);
    freeAudioData(audio);

    if (downsample_factor != -1) {
        int newLen = 0;
        double* downsampled = downsample(pressure, audioSamplesLen, downsample_factor, newLen);
        delete[] pressure;
        pressure = downsampled;
        audioSamplesLen = newLen;
        fs /= downsample_factor;
    }

    if (calTone == 1 && audioSamplesLen > 6 * fs) {
        int newLen = audioSamplesLen - 6 * fs;
        double* shifted = new double[newLen];
        memcpy(shifted, pressure + 6 * fs, sizeof(double) * newLen);
        delete[] pressure;
        pressure = shifted;
        audioSamplesLen = newLen;
    }

    BandpassFilter filt = bandpass_filter(pressure, audioSamplesLen, 1.0 / fs, flow, fhigh);
    delete[] pressure;

    int pts_per_timewin = timewin * fs;
    int num_timewin = filt.length / pts_per_timewin;
    int remainder = filt.length % pts_per_timewin;
    if (remainder > 0) { ++num_timewin; }

    int padded_len = num_timewin * pts_per_timewin;
    double* padded_signal = new double[padded_len]();
    memcpy(padded_signal, filt.filteredTimeSeries, sizeof(double) * filt.length);

    AudioFeatures features = {};
    features.segmentDurationLen = num_timewin;
    features.segmentDuration = new int[num_timewin];

    double** timechunk_matrix = new double* [num_timewin];
    for (int i = 0; i < num_timewin; ++i) {
        timechunk_matrix[i] = &padded_signal[i * pts_per_timewin];
        features.segmentDuration[i] = (i == num_timewin - 1 && remainder > 0)
            ? static_cast<int>(round(static_cast<double>(remainder) / fs)) : timewin;
    }

    features.SPLrmsLen = features.SPLpkLen = features.impulsivityLen = num_timewin;
    features.SPLrms = new double[num_timewin];
    features.SPLpk = new double[num_timewin];
    features.impulsivity = new double[num_timewin];

    for (int i = 0; i < num_timewin; ++i) {
        const double* chunk = timechunk_matrix[i];
        double sumsq = 0.0, peak = 0.0;
        for (int j = 0; j < pts_per_timewin; ++j) {
            double v = chunk[j];
            sumsq += v * v;
            peak = fabs(v) > peak ? fabs(v) : peak;
        }
        double rms = sqrt(sumsq / pts_per_timewin);
        features.SPLrms[i] = 20.0 * log10(max(rms, 1e-12));
        features.SPLpk[i] = 20.0 * log10(max(peak, 1e-12));
        features.impulsivity[i] = calculate_kurtosis(chunk, pts_per_timewin);
    }

    SoloPerGM2 gm2 = f_solo_per_GM2(padded_signal, padded_len, fs, timewin, avtime);

    features.peakcountLen = num_timewin;
    features.peakcount = new int[num_timewin];
    for (int i = 0; i < num_timewin; ++i) {
        features.peakcount[i] = gm2.peakcount[i];
    }
    delete[] gm2.peakcount;

    features.autocorrRows = gm2.autocorr_rows;
    features.autocorrCols = gm2.autocorr_cols;
    features.autocorr = new double* [gm2.autocorr_rows];
    for (int i = 0; i < gm2.autocorr_rows; ++i) { features.autocorr[i] = gm2.autocorr[i]; }
    delete[] gm2.autocorr;

    int dissim_len = 0;
    features.dissim = f_solo_dissim_GM1_gpu(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs, dissim_len);
    features.dissimLen = dissim_len;

    delete[] timechunk_matrix;
    delete[] padded_signal;

    return features;
}

tm extractBaseTime(const string& filename) {
    tm baseTime = {}; // Fields initialized to zero
    smatch match;
    regex pattern(R"((\d{8})_(\d{6}))"); // Matches YYYYMMDD_HHMMSS

    // Find date / time from file name
    if (regex_search(filename, match, pattern) && match.size() == 3) {
        string date = match[1]; // Date
        string time = match[2]; // Time

        baseTime.tm_year = stoi(date.substr(0, 4)) - 1900; // Years since 1900
        baseTime.tm_mon = stoi(date.substr(4, 2)) - 1; // Zero based month
        baseTime.tm_mday = stoi(date.substr(6, 2)); // Day of month
        baseTime.tm_hour = stoi(time.substr(0, 2)) - 1; // Hour
        baseTime.tm_min = stoi(time.substr(2, 2)); // Minute
        baseTime.tm_sec = stoi(time.substr(4, 2)); // Second
    }

    return baseTime;
}

// Export saved features to CSV file
void saveFeaturesToCSV(const char* filename, const char** filenames, int numFiles, const AudioFeatures* allFeatures) {
    ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open output file: " << filename << endl;
        return;
    }

    // Determine max autocorr matrix size
    int maxAutocorrRows = 0;
    int maxAutocorrCols = 0;

    for (int i = 0; i < numFiles; ++i) {
        const AudioFeatures& feature = allFeatures[i];
        if (feature.autocorr != nullptr && feature.autocorrRows > 0 && feature.autocorrCols > 0) {
            if (feature.autocorrRows > maxAutocorrRows) { maxAutocorrRows = feature.autocorrRows; }
            if (feature.autocorrCols > maxAutocorrCols) { maxAutocorrCols = feature.autocorrCols; }
        }
    }

    // Allocate array for valid autocorr columns
    bool* validAutocorrCols = new bool[maxAutocorrCols];
    for (int j = 0; j < maxAutocorrCols; ++j) { validAutocorrCols[j] = false; }

    // Remove extra autocorr columns
    for (int i = 0; i < numFiles; ++i) {
        const AudioFeatures& feature = allFeatures[i];
        if (feature.autocorr != nullptr) {
            for (int r = 0; r < feature.autocorrRows; ++r) {
                for (int c = 0; c < feature.autocorrCols; ++c) {
                    if (!isnan(feature.autocorr[r][c])) { validAutocorrCols[c] = true; }
                }
            }
        }
    }

    // CSV Header
    outputFile << "Filename,Year,Month,Day,Hour,Minute,SegmentDuration,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount";
    for (int j = 0; j < maxAutocorrCols; ++j) {
        if (validAutocorrCols[j]) { outputFile << ",Autocorr_" << j; }
    }
    outputFile << "\n";

    // Write data
    for (int fileIdx = 0; fileIdx < numFiles; ++fileIdx) {
        const AudioFeatures& features = allFeatures[fileIdx];

        int maxLength = max({ features.SPLrmsLen, features.SPLpkLen, features.impulsivityLen,
                features.dissimLen, features.peakcountLen });

        tm baseTime = extractBaseTime(filenames[fileIdx]);
        time_t baseEpoch = mktime(&baseTime);
        tm* firstTime = localtime(&baseEpoch);

        bool useNanTimestamp = false;
        if (!firstTime || (firstTime->tm_year + 1900) < 1900) { useNanTimestamp = true; }

        for (int i = 0; i < maxLength; ++i) {
            time_t currentEpoch = baseEpoch + i * 60;
            tm* currentTime = localtime(&currentEpoch);

            outputFile << filenames[fileIdx] << ",";

            if (useNanTimestamp || !currentTime) {  outputFile << "NaN,NaN,NaN,NaN,NaN,"; }
            else {
                outputFile << (currentTime->tm_year + 1900) << ","
                           << (currentTime->tm_mon + 1) << ","
                           << currentTime->tm_mday << ","
                           << currentTime->tm_hour << ","
                           << currentTime->tm_min << ",";
            }

            if (i < features.segmentDurationLen) { outputFile << features.segmentDuration[i]; }
            else { outputFile << "NaN"; }
            outputFile << ",";

            if (i < features.SPLrmsLen) outputFile << features.SPLrms[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            if (i < features.SPLpkLen) outputFile << features.SPLpk[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            if (i < features.impulsivityLen) outputFile << features.impulsivity[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            if (i < features.dissimLen) outputFile << features.dissim[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            if (i < features.peakcountLen) outputFile << features.peakcount[i];
            else { outputFile << "NaN"; }
            
            for (int j = 0; j < maxAutocorrCols; ++j) {
                if (validAutocorrCols[j]) {
                    outputFile << ",";
                    if (features.autocorr && i < features.autocorrRows && j < features.autocorrCols) { outputFile << features.autocorr[i][j]; }
                    else { outputFile << "NaN"; }
                }
            }

            outputFile << "\n";
        }
    }

    // Clean up
    delete[] validAutocorrCols;
    outputFile.close();
}

// Worker thread function
void threadWork(atomic<int>& nextIndex, int totalFiles, char filePaths[][512], AudioFeatures* allFeatures,
                char filenames[][512], int num_bits, int peak_volts, double RS, int timewin, double avtime,
                int fft_win, int arti, int flow, int fhigh, int downsample, bool omit_partial_minute) {
    while (true) {
        int index = nextIndex++;
        if (index >= totalFiles) { break; }

        // Convert the raw C-string to filesystem::path
        fs::path filepath(filePaths[index]);

        // Extract filename as string / copy to filenames[]
        string fname = filepath.filename().string();
        strncpy(filenames[index], fname.c_str(), 511);
        filenames[index][511] = '\0'; // Ensure null-termination

        // Extract features from recordings
        allFeatures[index] = feature_extraction(
            num_bits, peak_volts, filepath, RS,
            timewin, avtime, fft_win, arti,
            flow, fhigh, downsample, omit_partial_minute);
    }
}

void bubbleSort(char arr[][512], int n) {
    char temp[512];
    for (int i = 0; i < n-1; ++i) {
        for (int j = 0; j < n-i-1; ++j) {
            if (strcmp(arr[j], arr[j+1]) > 0) {
                strcpy(temp, arr[j]);
                strcpy(arr[j], arr[j+1]);
                strcpy(arr[j+1], temp);
            }
        }
    }
}

void threadWrapper(ThreadArgs& args) {
    try {
        while (true) {
            int idx = args.nextIndex->fetch_add(1);
            if (idx >= args.totalFiles) { break; }

            cout << "Processing file index " << idx << ": " << args.filePaths[idx] << "\n";
            cerr.flush();

            fs::path file_path(args.filePaths[idx]);

            AudioFeatures features = feature_extraction(
                args.num_bits,
                args.peak_volts,
                file_path,
                args.RS,
                args.timewin,
                args.avtime,
                args.fft_win,
                args.arti,
                args.flow,
                args.fhigh,
                args.downsample,
                args.omit_partial_minute
            );

            args.allFeatures[idx] = features;

            string filename_str = file_path.filename().string();
            strncpy(args.filenames[idx], filename_str.c_str(), 511);
            args.filenames[idx][511] = '\0';
        }
    }
    catch (const exception& e) {
        cerr << "Exception in thread: " << e.what() << "\n";
    }
    catch (...) {
        cerr << "Unknown exception in thread\n";
    }
}

// Process directory of sound files with user-given parameters
int main(int argc, char* argv[]) {
    using namespace std;
    using namespace chrono;

    auto start = high_resolution_clock::now();

    // Defaults
    char input_dir[512] = {}, output_file[512] = {};
    int num_bits = 16, peak_volts = 2, arti = 1;
    int timewin = 60, fft_win = 1, flow = 1, fhigh = 192000;
    double RS = -178.3, avtime = 0.1;
    int max_threads = 4, downsample = -1;
    bool omit_partial_minute = false;

    // CLI Parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--omit_partial_minute") == 0) { omit_partial_minute = true; }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) { strncpy(input_dir, argv[++i], sizeof(input_dir) - 1); }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { strncpy(output_file, argv[++i], sizeof(output_file) - 1); }
        else if (strcmp(argv[i], "--num_bits") == 0) { num_bits = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--RS") == 0) { RS = atof(argv[++i]); }
        else if (strcmp(argv[i], "--peak_volts") == 0) { peak_volts = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--arti") == 0) { arti = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--timewin") == 0) { timewin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fft_win") == 0) { fft_win = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--avtime") == 0) { avtime = atof(argv[++i]); }
        else if (strcmp(argv[i], "--flow") == 0) { flow = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fhigh") == 0) { fhigh = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max_threads") == 0) { max_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--downsample") == 0) { downsample = atoi(argv[++i]); }
    }

    // Allocate heap memory for large arrays
    auto filePaths = new char[MAX_FILES][512];
    auto filenames = new char[MAX_FILES][512];
    auto allFeatures = new AudioFeatures[MAX_FILES];

    int totalFiles = 0;

    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".wav") {
            if (totalFiles >= MAX_FILES) {
                cerr << "Too many .wav files. Increase MAX_FILES.\n";
                delete[] filePaths;
                delete[] filenames;
                delete[] allFeatures;
                return 1;
            }
            strncpy(filePaths[totalFiles], entry.path().string().c_str(), 511);
            filePaths[totalFiles][511] = '\0'; // Null terminate
            totalFiles++;
        }
    }

    if (totalFiles == 0) {
        cerr << "No valid .wav files found in " << input_dir << "\n";
        delete[] filePaths;
        delete[] filenames;
        delete[] allFeatures;
        return 1;
    }

    bubbleSort(filePaths, totalFiles);

    atomic<int> nextIndex(0);
    int availableThreads = max(1u, thread::hardware_concurrency());
    int numThreads = min(max_threads, availableThreads);

    if (fhigh <= 16000 && numThreads > 2) { numThreads = 2; }
    else if (fhigh <= 48000 && numThreads > 4) { numThreads = 4; }

    // Prepare thread args
    ThreadArgs args;
    args.nextIndex = &nextIndex;
    args.totalFiles = totalFiles;
    args.filePaths = filePaths;
    args.filenames = filenames;
    args.allFeatures = allFeatures;
    args.num_bits = num_bits;
    args.peak_volts = peak_volts;
    args.RS = RS;
    args.timewin = timewin;
    args.avtime = avtime;
    args.fft_win = fft_win;
    args.arti = arti;
    args.flow = flow;
    args.fhigh = fhigh;
    args.downsample = downsample;
    args.omit_partial_minute = omit_partial_minute;

    // Launch threads
    vector<thread> threads;
    for (int i = 0; i < numThreads; ++i) { threads.emplace_back(threadWrapper, ref(args)); }
    for (auto& t : threads) { t.join(); }

    // Save to CSV
    vector<const char*> file_names(totalFiles);
    for (int i = 0; i < totalFiles; ++i) { file_names[i] = filenames[i]; }

    saveFeaturesToCSV(output_file, file_names.data(), totalFiles, allFeatures);
    cout << "Saved features for " << totalFiles << " files to " << output_file << "\n";

    // Cleanup
    for (int i = 0; i < totalFiles; ++i) { freeAudioFeatures(allFeatures[i]); }

    delete[] filePaths;
    delete[] filenames;
    delete[] allFeatures;

    fftw_cleanup();

    auto stop = high_resolution_clock::now();

    using chrono::duration;
    using chrono::duration_cast;

    duration<double> elapsed = duration_cast<duration<double>>(stop - start);
    cout << "Runtime: " << elapsed.count() << " seconds\n";

    return 0;
}