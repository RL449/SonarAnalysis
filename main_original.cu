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
#include <iomanip> // Setfill, setw

// Limit thread count to # of cores
#include <queue> // Sequential reading of files for threads
#include <condition_variable> // Thread synchronization
#include <atomic> // Atomic operations for multithreading

// CUDA
#include <cufft.h> // CUDA FFT
#include <thrust/reduce.h> // Reduction operations
#include <thrust/device_ptr.h> // Smart pointers

using namespace std; // Standard namespace
namespace fs = filesystem; // Rename filesystem

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
    double* time_series_filt; // Signal
    double* amp_spectrum; // Frequency amplitude spectrum
    int num_pts; // # of samples

    BandpassFilter(double* ts, double* amp, int n) : time_series_filt(ts), amp_spectrum(amp), num_pts(n) {}
};

struct Correlation {
    double* correlationValues; // Cross-correlation values between two signals
    double* lags; // Corresponding lag values
    int length; // Length of the arrays

    // Constructor
    Correlation(double* corr, double* lag, int len) : correlationValues(corr), lags(lag), length(len) {}

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
    int* peakCount = nullptr; // # of peaks
    double** autocorr = nullptr; // Autocorrelation matrix

    // # of segments
    int segmentDurationLen = 0;
    int SPLrmsLen = 0;
    int SPLpkLen = 0;
    int impulsivityLen = 0;
    int dissimLen = 0;
    int peakCountLen = 0;
    int autocorrRows = 0; // Time segments
    int autocorrCols = 0; // Lags

    // Cross-file dissimilarity
    double* first60s = nullptr;
    int first60sLen = 0;
    double* last60s = nullptr;
    int last60sLen = 0;
    int sampleRate = 0;
};

struct AudioData {
    double** samples; // 2D array of audio samples [channel][frame]
    int numChannels; // # of audio channels
    int numFrames; // # of frames per channel
    int sampleRate; // Sampling rate (Hz)
    double duration; // Duration of audio (seconds)
};

// Periodicity / impulsivity
struct SoloPer {
    int* peakCount; // # of peaks per time window
    double** autocorr; // Autocorrelation per segment
    int peakcountLength; // Length of peakcount array
    int autocorrRows; // # of time windows - rows
    int autocorrCols; // # of lags - columns
};

struct ArrayShiftFFT {
    double* data; // Samples after shift
    int size; // Length of array
    ArrayShiftFFT(double* d, int s) : data(d), size(s) {} // Constructor
    ~ArrayShiftFFT() { delete[] data; } // Destructor
};

// FFTW complex buffer + plan
struct FFTWHandler {
    fftw_complex* buf = nullptr; // Buffer for FFT computation
    fftw_plan forwardPlan = nullptr; // Forward FFT plan: Frequency to time
    fftw_plan inversePlan = nullptr; // Inverse FFT plan: Time to frequency
    int size = 0; // # of points
    
    // Constructor
    FFTWHandler(int N) : size(N) {
        buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N); // Allocate FFTW buffer
        if (!buf) { throw bad_alloc(); } // Allocation unsuccessful
        
        forwardPlan = fftw_plan_dft_1d(size, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE); // Create forward plan
        if (!forwardPlan) { // Error creating forward plan
            fftw_free(buf);
            throw runtime_error("FFTW forward plan creation failed");
        }
        
        inversePlan = fftw_plan_dft_1d(size, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE); // Create inverse plan
        if (!inversePlan) { // Error creating inverse plan
            fftw_destroy_plan(forwardPlan);
            fftw_free(buf);
            throw runtime_error("FFTW inverse plan creation failed");
        }
    }
    
    // Destructor
    ~FFTWHandler() {
        if (forwardPlan) { fftw_destroy_plan(forwardPlan); }
        if (inversePlan) { fftw_destroy_plan(inversePlan); }
        if (buf) { fftw_free(buf); }
    }
    
    // Delete copy constructor / assignment operator to prevent copying between threads
    FFTWHandler(const FFTWHandler&) = delete;
    FFTWHandler& operator=(const FFTWHandler&) = delete;
};

// Hold extracted time information per file
struct FileTimeInfo {
    tm baseTime; // Struct containing date/time components
    bool timeExtracted; // Time successfully extracted
    string filename;
};

struct ThreadArgs { // Worker threads for parallel processing
    atomic<int>* nextIndex; // Counter for thread-safe file indexing
    int totalFiles; // # of audio files to process
    char (*filePaths)[128]; // Input file paths
    AudioFeatures* allFeatures; // Feature extraction results
    char (*filenames)[128]; // Names of files
    FileTimeInfo* fileTimeInfo;

    // User-given arguments
    int numBits, peakVolts, timeWin, fftWin, fLow, fHigh, downSample;
    double RS, avTime, artiLen;
    bool omitPartialMinute, debugOutput;
};

// Thread-safe FFTW handler cache
class FFTWHandlerCache {
private:
    static unordered_map<int, shared_ptr<FFTWHandler>> cache; // Fast mapping to FFTWHandler instance
    static mutex cache_mutex; // Guard concurrent access to cache

public:
    static shared_ptr<FFTWHandler> getHandler(int size) { // Reusable pointer for FFTWHandler
        // Check if handler exists
        {
            lock_guard<mutex> lock(cache_mutex);
            auto it = cache.find(size);
            if (it != cache.end()) { return it->second; } // If requested size exits
        }

        // Create new handler outside of lock to avoid holding lock during allocation
        auto new_handler = make_shared<FFTWHandler>(size);

        // Ensure no other thread created another handler
        {
            lock_guard<mutex> lock(cache_mutex);
            auto it = cache.find(size);
            if (it != cache.end()) { return it->second; }
            cache[size] = new_handler;
            return new_handler;
        }
    }

    // Thread safe clears cached FFTWHandlers
    static void clearCache() {
        lock_guard<mutex> lock(cache_mutex);
        cache.clear(); // Release shared pointers
    }

    // Thread safe read lock to return # of cached FFT sizes
    static size_t getCacheSize() {
        lock_guard<mutex> lock(cache_mutex);
        return cache.size();
    }
};

// Replaces backslashes with forward slashes to work with Windows file paths
string fixFilePath(const string& path) {
    string fixedPath = path;
    replace(fixedPath.begin(), fixedPath.end(), '\\', '/'); // Replace '\\' with '/'
    return fixedPath;
}

// Read audio samples from files
AudioData audioRead(const string& filename, SampleRange range = { 1, -1 }) {
    SF_INFO sfinfo = {}; // Audio metadata (# of channels, sample rate, etc.)
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo); // Open file for reading
    // File open unsuccessful
    if (!file) { throw runtime_error("Error opening audio file: " + string(sf_strerror(file))); }
    // Sample range calculation to fit within file bounds
    int totalFrames = sfinfo.frames;
    int endSample;
    if (range.endSample == -1) { endSample = totalFrames; }
    else { endSample = min(range.endSample, totalFrames); }
    int startSample = max(0, range.startSample - 1);
    int numFramesToRead = endSample - startSample;

    if (numFramesToRead <= 0) { // Range invalid
        sf_close(file);
        throw runtime_error("Invalid sample range");
    }

    sf_seek(file, startSample, SEEK_SET); // Adjust file position to startSample

    int numChannels = sfinfo.channels;
    double* interleavedSamples = new double[numFramesToRead * numChannels]; // Raw interleaved samples
    int format = sfinfo.format & SF_FORMAT_SUBMASK; // Extract audio subtype from full format

    // Read / convert audio samples
    switch (format) { // Convert samples according to bit format
    case SF_FORMAT_PCM_16: {
        short* tempBuffer = new short[numFramesToRead * numChannels];
        sf_readf_short(file, tempBuffer, numFramesToRead);
        for (int i = 0; i < numFramesToRead * numChannels; ++i)
        {
            interleavedSamples[i] = static_cast<double>(tempBuffer[i]);
        }
        delete[] tempBuffer;
        break;
    }
                         // 24 or 32 bits
    case SF_FORMAT_PCM_24:
    case SF_FORMAT_PCM_32: {
        int* tempBuffer = new int[numFramesToRead * numChannels];
        sf_readf_int(file, tempBuffer, numFramesToRead);
        for (int i = 0; i < numFramesToRead * numChannels; ++i)
        {
            interleavedSamples[i] = static_cast<double>(tempBuffer[i]);
        }
        delete[] tempBuffer;
        break;
    }
    default: // Bit format invalid
        sf_close(file);
        delete[] interleavedSamples;
        throw runtime_error("Unsupported bit format");
    }

    sf_close(file); // Close file

    double** samples = new double* [numChannels]; // Deinterleave samples into separate channels
    for (int ch = 0; ch < numChannels; ++ch) { samples[ch] = new double[numFramesToRead]; } // Allocate memory per channel
    // Populate records for each channel
    for (int i = 0; i < numFramesToRead; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) { samples[ch][i] = interleavedSamples[i * numChannels + ch]; }
    }

    delete[] interleavedSamples; // Deallocate memory
    double duration = static_cast<double>(sfinfo.frames) / sfinfo.samplerate; // Recording length (seconds)

    return AudioData{ samples, numChannels, numFramesToRead, sfinfo.samplerate, duration };
}

__global__ void downsampleKernel(const double* x, double* result, int length, int factor) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Compute global index of thread
    // Determine if downsampled data is in input bounds
    if (index * factor < length) { result[index] = x[index * factor]; } // Compute output sample if within bounds
}

// Reduce sampling rate to lower frequency
double* downSample(const double* x, int length, int factor, int& newLength) {
    if (factor <= 0) { throw invalid_argument("Factor must be positive"); } // Validate input

    newLength = (length + factor - 1) / factor; // # of samples in downsampled signal

    // Allocate GPU memory
    double* deviceInput;
    double* deviceOutput;
    cudaMalloc(&deviceInput, sizeof(double) * length);
    cudaMalloc(&deviceOutput, sizeof(double) * newLength);

    // Copy input signal from host to device
    cudaMemcpy(deviceInput, x, sizeof(double) * length, cudaMemcpyHostToDevice);

    // Launch kernel
    int threads = 256; // # of threads per block
    int blocks = (newLength + threads - 1) / threads; // # of blocks
    downsampleKernel <<<blocks, threads >>> (deviceInput, deviceOutput, length, factor);

    double* result = new double[newLength]; // Allocate memory on host
    // Copy downsampled result to host
    cudaMemcpy(result, deviceOutput, sizeof(double) * newLength, cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return result; // Downsampled signal
}

// CUDA kernel to compute shifted frequency array / apply bandpass filter
__global__ void applyBandpassFilter(cufftDoubleComplex* freqData, int numPoints,
            double freqStep, double fLow, double fHigh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Thread indexing and bounds checking
    if (i >= numPoints) { return; }

    int shiftedIndex = (i + numPoints / 2) % numPoints; // Calculate index

    double freq = (shiftedIndex - numPoints / 2) * freqStep; // Convert index to frequency with index centering
    double absFreq = fabs(freq);

    // Zero out components outside specified range
    if (absFreq < fLow || absFreq > fHigh) {
        freqData[i].x = 0.0;
        freqData[i].y = 0.0;
    }
}

// CUDA kernel to normalize inverse FFT / compute amplitude spectrum
__global__ void normalizeAndComputeAmplitude(const cufftDoubleComplex* timeData, double* outputTime,
            double* outputAmp, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Compute thread global index
    if (i >= numPoints) { return; } // Ensure valid thread range

    double norm = 1.0 / numPoints;
    outputTime[i] = timeData[i].x * norm; // Normalize inverse FFT real output
    // Compute magnitude at each point
    outputAmp[i] = sqrt(timeData[i].x * timeData[i].x + timeData[i].y * timeData[i].y);
}

// Apply bandpass filter in frequency domain
BandpassFilter bandpassFilter(const double* timeSeries, int numPts, double frequency, double flow, double freqHigh) {
    // Allocate memory on device
    cufftDoubleComplex* deviceFreqData;
    cudaMalloc(&deviceFreqData, sizeof(cufftDoubleComplex) * numPts);

    // Copy real timeSeries to device
    cufftDoubleComplex* hostInput = new cufftDoubleComplex[numPts]; // Convert to complex
    for (int i = 0; i < numPts; ++i) {
        hostInput[i].x = timeSeries[i];
        hostInput[i].y = 0.0;
    }

    // Copy frequency data to device
    cudaMemcpy(deviceFreqData, hostInput, sizeof(cufftDoubleComplex) * numPts, cudaMemcpyHostToDevice);
    delete[] hostInput; // Free host memory

    // Execute forward FFT
    cufftHandle planForward;
    cufftPlan1d(&planForward, numPts, CUFFT_Z2Z, 1);
    cufftExecZ2Z(planForward, deviceFreqData, deviceFreqData, CUFFT_FORWARD);

    // Apply bandpass filter
    double recordingLen = numPts * frequency; // Time span (seconds)
    double freqStep = 1.0 / recordingLen;
    if (freqHigh == 0.0) { freqHigh = 0.5 / frequency; }

    // Zero out frequencies outside given range
    int threads = 256;
    int blocks = (numPts + threads - 1) / threads;
    applyBandpassFilter <<<blocks, threads >>> (deviceFreqData, numPts, freqStep, flow, freqHigh);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    // Execute inverse FFT
    cufftExecZ2Z(planForward, deviceFreqData, deviceFreqData, CUFFT_INVERSE); // Reuse plan

    // Allocate output arrays
    double* deviceTimeOut;
    double* deviceAmplitudeOut;
    cudaMalloc(&deviceTimeOut, sizeof(double) * numPts);
    cudaMalloc(&deviceAmplitudeOut, sizeof(double) * numPts);

    normalizeAndComputeAmplitude <<<blocks, threads >>> (deviceFreqData, deviceTimeOut, deviceAmplitudeOut, numPts);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    // Copy results to host
    double* timeSeriesFilt = new double[numPts];
    double* amplitudeSpectrum = new double[numPts];
    // Copy magnitude spectrum / time series data to host
    cudaMemcpy(timeSeriesFilt, deviceTimeOut, sizeof(double) * numPts, cudaMemcpyDeviceToHost);
    cudaMemcpy(amplitudeSpectrum, deviceAmplitudeOut, sizeof(double) * numPts, cudaMemcpyDeviceToHost);

    // Cleanup
    cufftDestroy(planForward);
    cudaFree(deviceFreqData);
    cudaFree(deviceTimeOut);
    cudaFree(deviceAmplitudeOut);

    return BandpassFilter(timeSeriesFilt, amplitudeSpectrum, numPts);
}

__global__ void partialSumsKernel(const double* data, double* sumOut, double* sumSqOut, int pointsPerTimeWin) {
    // Shared memory per thread block
    __shared__ double localSum[256];
    __shared__ double localSumSq[256];

    // Thread global index
    int threadIndex = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double val; // Load current val
    if (index < pointsPerTimeWin) { val = data[index]; }
    else { val = 0.0; }
    // Write data / squared value into shared memory
    localSum[threadIndex] = val;
    localSumSq[threadIndex] = val * val;
    __syncthreads(); // Ensure all threads have written

    // Reduce within block
    for (int stepSize = blockDim.x / 2; stepSize > 0; stepSize >>= 1) { // Halves active threads on each pass
        if (threadIndex < stepSize) {
            // Calculate sums
            localSum[threadIndex] += localSum[threadIndex + stepSize];
            localSumSq[threadIndex] += localSumSq[threadIndex + stepSize];
        }
        __syncthreads(); // Ensure all threads completed before next step
    }

    // First thread per block writes result to global memory
    if (threadIndex == 0) {
        sumOut[blockIdx.x] = localSum[0]; // Block sum
        sumSqOut[blockIdx.x] = localSumSq[0]; // Block sum of squares
    }
}

__global__ void fourthMomentKernel(const double* data, double* fourthOut, double mean, int pointsPerTimeWin) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    if (i >= pointsPerTimeWin) { return; } // Protect against out of bounds threads

    // Fourth central moment
    double centered = data[i] - mean;
    fourthOut[i] = centered * centered * centered * centered;
}

__global__ void reduceSumKernel(const double* input, double* output, int n) {
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load shared memory with either data or 0
    if (i < n) { sdata[tid] = input[i]; }
    else { sdata[tid] = 0.0; }
    __syncthreads();

    // Reduce in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sdata[tid] += sdata[tid + s]; }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {  output[blockIdx.x] = sdata[0]; }
}

// Calculate kurtosis used for impulsivity of a signal
double calculateKurtosis(const double* hostData, int length) {
    if (length <= 0 || hostData == nullptr) { throw invalid_argument("Input array is empty or null"); }

    // Allocate device memory
    double* deviceData, * deviceSumPartial, * deviceSumSqPartial, * deviceFourth;
    cudaMalloc(&deviceData, length * sizeof(double));
    cudaMemcpy(deviceData, hostData, length * sizeof(double), cudaMemcpyHostToDevice); // Copy data to device

    // Thread / block setup
    int threads = 256;
    int blocks = (length + threads - 1) / threads;

    cudaMalloc(&deviceSumPartial, blocks * sizeof(double)); // Allocate device memory for partial sum
    cudaMalloc(&deviceSumSqPartial, blocks * sizeof(double)); // Allocate device memory for partial sum square

    // Compute mean / variance components
    partialSumsKernel <<<blocks, threads >>> (deviceData, deviceSumPartial, deviceSumSqPartial, length);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    // Reduce partial results to final results
    double* sumReduceBuf = deviceSumPartial;
    int sumN = blocks;
    while (sumN > 1) {
        int sumBlocks = (sumN + threads - 1) / threads;
        reduceSumKernel << <sumBlocks, threads, threads * sizeof(double) >> > (sumReduceBuf, sumReduceBuf, sumN);
        cudaDeviceSynchronize();
        sumN = sumBlocks;
    }

    // Sum values in parallel
    double totalSum;
    cudaMemcpy(&totalSum, sumReduceBuf, sizeof(double), cudaMemcpyDeviceToHost);

    double* sumSqReduceBuf = deviceSumSqPartial;
    int sumSqN = blocks;
    while (sumSqN > 1) {
        int sumBlocks = (sumSqN + threads - 1) / threads;
        reduceSumKernel << <sumBlocks, threads, threads * sizeof(double) >> > (sumSqReduceBuf, sumSqReduceBuf, sumSqN);
        cudaDeviceSynchronize();
        sumSqN = sumBlocks;
    }

    double totalSumSq;
    cudaMemcpy(&totalSumSq, sumSqReduceBuf, sizeof(double), cudaMemcpyDeviceToHost);

    double mean = totalSum / length; // Calculate mean
    double variance = (totalSumSq / length) - (mean * mean); // Calculate variance

    if (variance < 1e-12) { // Avoid divide by zero
        cudaFree(deviceData);
        cudaFree(deviceSumPartial);
        cudaFree(deviceSumSqPartial);
        return 0.0;
    }

    // Compute fourth moment
    cudaMalloc(&deviceFourth, length * sizeof(double));
    fourthMomentKernel <<<blocks, threads >>> (deviceData, deviceFourth, mean, length);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    // Reduce fourth moment array to scalar value / sum values in parallel
    double* fourthReduceBuf = deviceFourth;
    int fourthN = length;
    while (fourthN > 1) {
        int sumBlocks = (fourthN + threads - 1) / threads;
        reduceSumKernel << <sumBlocks, threads, threads * sizeof(double) >> > (fourthReduceBuf, fourthReduceBuf, fourthN);
        cudaDeviceSynchronize();
        fourthN = sumBlocks;
    }

    double fourthMoment;
    cudaMemcpy(&fourthMoment, fourthReduceBuf, sizeof(double), cudaMemcpyDeviceToHost);
    fourthMoment /= length;

    // Cleanup
    cudaFree(deviceData);
    cudaFree(deviceSumPartial);
    cudaFree(deviceSumSqPartial);
    cudaFree(deviceFourth);

    return fourthMoment / (variance * variance); // Kurtosis value
}

// CUDA kernel for envelope calculation
__global__ void envelopeKernel(const cufftDoubleComplex* hilbert, double* envelope, int pointsPerTimeWin) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute magnitude for idx
    if (index < pointsPerTimeWin)
        { envelope[index] = sqrt(hilbert[index].x * hilbert[index].x + hilbert[index].y * hilbert[index].y); }
}

// CUDA kernel for correlation computation
__global__ void correlationKernel(const double* real, const double* imaginary, int seriesLength,
            double* corrVals, int maxLag, int offset) {
    int lag = blockIdx.x * blockDim.x + threadIdx.x;
    if (lag > maxLag) { return; } // Skip out of range lags
    
    // Accumulators for statistical calculations
    double sumReal = 0.0, sumImagninary = 0.0, sumRealSquare = 0.0, sumImaginarySquare = 0.0, sumRealImaginaryProd = 0.0;
    int sampleCount = 0;
    
    // Loop through overlapping samples of current lag
    for (int i = 0; i < seriesLength - (lag + offset); i++) {
        double realVal = real[i];
        double imaginaryVal = imaginary[i + lag + offset];
        
        if (!isnan(realVal) && !isnan(imaginaryVal)) { // Skip NaNs
            sumReal += realVal;
            sumImagninary += imaginaryVal;
            sumRealSquare += realVal * realVal;
            sumImaginarySquare += imaginaryVal * imaginaryVal;
            sumRealImaginaryProd += realVal * imaginaryVal;
            sampleCount++;
        }
    }
    
    if (sampleCount == 0) { // No valid samples
        corrVals[lag] = NAN;
        return;
    }
    
    // Means / variances
    double meanReal = sumReal / sampleCount;
    double meanImaginary = sumImagninary / sampleCount;
    double meanXSquare = sumRealSquare / sampleCount;
    double meanYSquare = sumImaginarySquare / sampleCount;
    
    double covar = (sumRealImaginaryProd / sampleCount) - (meanReal * meanImaginary); // Covariance
    double denomReal = sqrt(meanXSquare - (meanReal * meanReal));
    double denomImaginary = sqrt(meanYSquare - (meanImaginary * meanImaginary)); // Calculate standard deviation
    
    // Normalize correlation
    if (denomReal == 0.0 || denomImaginary == 0.0) { corrVals[lag] = NAN; }
    else { corrVals[lag] = covar / (denomReal * denomImaginary); }
}

// CUDA kernel for FFT magnitude calculation
__global__ void fftMagnitudeKernel(const cufftDoubleComplex* fftData, double* magnitude, int pointsPerFFT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate magnitude
    if (idx < pointsPerFFT) { magnitude[idx] = sqrt(fftData[idx].x * fftData[idx].x + fftData[idx].y * fftData[idx].y); }
}

// GPU-accelerated correlation function
Correlation correl5GPU(const double* timeSeries1, const double* timeSeries2, 
            int seriesLength, int lags, int offset) {
    int len = lags + 1; // # of correlation values to calculate
    
    // Allocate GPU memory
    double * deviceInputSignal1, * deviceInputSignal2, * deviceCorrVals;
    cudaMalloc(&deviceInputSignal1, sizeof(double) * seriesLength);
    cudaMalloc(&deviceInputSignal2, sizeof(double) * seriesLength);
    cudaMalloc(&deviceCorrVals, sizeof(double) * len);
    
    // Copy deviceInputSignal1 / deviceInputSignal2 to device
    cudaMemcpy(deviceInputSignal1, timeSeries1, sizeof(double) * seriesLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInputSignal2, timeSeries2, sizeof(double) * seriesLength, cudaMemcpyHostToDevice);
    
    // Launch correlation kernel
    // Specify dimensions
    dim3 block(256);
    dim3 grid((len + block.x - 1) / block.x); // All data points covered by threads
    correlationKernel<<<grid, block>>>(deviceInputSignal1, deviceInputSignal2, seriesLength, deviceCorrVals, lags, offset);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution
    
    // Allocate host memory
    double* corrVals = new double[len];
    double* lagVals = new double[len];
    
    cudaMemcpy(corrVals, deviceCorrVals, sizeof(double) * len, cudaMemcpyDeviceToHost); // Copy corrVals to host
    
    // Fill lag values
    for (int i = 0; i < len; ++i) { lagVals[i] = static_cast<double>(i); }
    
    // Cleanup GPU memory
    cudaFree(deviceInputSignal1);
    cudaFree(deviceInputSignal2);
    cudaFree(deviceCorrVals);
    
    return Correlation(corrVals, lagVals, len);
}

// Kernel to square / segment the input
__global__ void squareAndSegment(const double* input, double* output, int sampWindowSize, int numTimeWins) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int total = sampWindowSize * numTimeWins; // # of samples

    if (index < total) { // Ensure index in bounds
        double val = input[index];
        output[index] = val * val; // Signal energy
    }
}

// Kernel to average squared values
__global__ void computeAverages(const double* squared, double* outputAvg, int sampWindowSize,
            int avgWinSize, int numavwin, int numTimeWins) {
    int segIndex = blockIdx.x; // Segment index
    int win = threadIdx.x; // Averaging window index

    if (segIndex < numTimeWins && win < numavwin) { // Check bounds
        int baseIndex = segIndex * sampWindowSize + win * avgWinSize;
        double sum = 0.0;
        for (int i = 0; i < avgWinSize; i++) { sum += squared[baseIndex + i]; } // Sum of samples in window
        outputAvg[segIndex * numavwin + win] = sum / avgWinSize; // Mean in window
    }
}

// Calculate autocorrelation / peak counts
SoloPer calculatePeriodicity(const double* pFiltInput, int inputLength, double fs, double timewin, double avtime) {
    // Calculate window sizes
    int sampWindowSize = static_cast<int>(fs * timewin); // # of samples in time window
    int numTimeWins = inputLength / sampWindowSize; // # of time windows
    if (numTimeWins == 0) { throw runtime_error("Empty time window"); }

    int totalSamples = sampWindowSize * numTimeWins;
    int avgWinSize = static_cast<int>(fs * avtime); // Samples in averaging window
    int numAvWin = sampWindowSize / avgWinSize; // Averaging windows per time window

    // GPU memory allocations
    double* deviceInput, * deviceSquared, * deviceAvg;
    cudaMalloc(&deviceInput, totalSamples * sizeof(double));
    cudaMalloc(&deviceSquared, totalSamples * sizeof(double));
    cudaMalloc(&deviceAvg, numTimeWins * numAvWin * sizeof(double));

    cudaMemcpy(deviceInput, pFiltInput, totalSamples * sizeof(double), cudaMemcpyHostToDevice); // Copy input to device

    // Launch square / segment kernel
    int threads = 256;
    int blocks = (totalSamples + threads - 1) / threads;
    squareAndSegment <<<blocks, threads >>> (deviceInput, deviceSquared, sampWindowSize, numTimeWins);

    // Calculate average of squared values
    computeAverages <<<numTimeWins, numAvWin >>> (deviceSquared, deviceAvg, sampWindowSize,
            avgWinSize, numAvWin, numTimeWins);

    // Copy averages back to host
    double* hostAvg = new double[numTimeWins * numAvWin];
    cudaMemcpy(hostAvg, deviceAvg, numTimeWins * numAvWin * sizeof(double), cudaMemcpyDeviceToHost);

    // Outputs for correlation / peak count
    int pAvTotRows = numAvWin;
    int lagLimit = static_cast<int>(pAvTotRows * 0.7); // 70% of lags
    int pAvTotCols = numTimeWins;

    double** acorr = new double* [pAvTotCols]; // Autocorrelation per window
    int* pkcount = new int[pAvTotCols]; // Peak count per window

    // Iterate through time windows - Calculate autocorr / peak count
    for (int i = 0; i < pAvTotCols; i++) {
        // Calculate correlation
        cudaSetDevice(0);
        Correlation corrResult = correl5GPU(&hostAvg[i * numAvWin], &hostAvg[i * numAvWin], pAvTotRows, lagLimit, 0);
        acorr[i] = new double[lagLimit + 1];
        for (int j = 0; j <= lagLimit; ++j) { acorr[i][j] = corrResult.correlationValues[j]; }

        // Calculate peak count
        int peakCount = 0;
        for (int j = 1; j < lagLimit; j++) {
            if (acorr[i][j] > acorr[i][j - 1] && acorr[i][j] > acorr[i][j + 1]) {
                // Find min to left / right to find prominence
                double leftMin = acorr[i][j];
                for (int k = j - 1; k >= 0 && acorr[i][k] < acorr[i][j]; k--) { leftMin = min(leftMin, acorr[i][k]); }
                double rightMin = acorr[i][j];
                for (int k = j + 1; k <= lagLimit && acorr[i][k] < acorr[i][j]; k++)
                    { rightMin = min(rightMin, acorr[i][k]); }
                double prominence = acorr[i][j] - max(leftMin, rightMin);
                if (prominence > 0.5) { peakCount++; } // Threshold reached
            }
        }
        pkcount[i] = peakCount;
    }

    // Free GPU memory
    cudaFree(deviceInput);
    cudaFree(deviceSquared);
    cudaFree(deviceAvg);
    delete[] hostAvg;

    // Return result
    SoloPer result;
    result.peakCount = pkcount;
    result.autocorr = acorr;
    result.peakcountLength = pAvTotCols;
    result.autocorrRows = pAvTotCols;
    result.autocorrCols = lagLimit + 1;

    return result;
}

// Zeroes out negative frequencies / doubles positive frequencies to create frequency signal
__global__ void hilbertFilterKernel(cufftDoubleComplex* data, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) { return; } // Ensure index in valid range
    
    int half = len / 2; // Half way point of FFT
    int upper;
    if (len % 2 == 0) { upper = half - 1; }
    else { upper = half; }
    
    if (index >= 1 && index <= upper) {
        // Multiply positive frequencies by 2
        data[index].x *= 2.0;
        data[index].y *= 2.0;
    } else if (index > half) {
        // Zero out negative frequencies
        data[index].x = 0.0;
        data[index].y = 0.0;
    }
}

// Convert real input to complex array
__global__ void initializeComplex(double* input, cufftDoubleComplex* output, int len) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (threadIndex < len) {
        output[threadIndex].x = input[threadIndex]; // Real
        output[threadIndex].y = 0.0; // Imaginary
    }
}

// Normalize inverse FFT result to preserve signal amplitude
__global__ void normalizeResult(cufftDoubleComplex* data, int len) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (threadIndex < len) {
        data[threadIndex].x /= len; // Normalize real part
        data[threadIndex].y /= len; // Normalize imaginary part
    }
}

// Converts real values to complex analytic signal using FFT
fftw_complex* hilbertRawGPU(const double* input, int inputLen) {
    if (!input || inputLen <= 0) { // Validate input
        cerr << "Invalid input\n";
        return nullptr;
    }

    cufftDoubleComplex* deviceData = nullptr; // Final complex array
    cudaMalloc(&deviceData, sizeof(cufftDoubleComplex) * inputLen);

    double* deviceIinput = nullptr; // Device buffer for input
    cudaMalloc(&deviceIinput, sizeof(double) * inputLen);
    cudaMemcpy(deviceIinput, input, sizeof(double) * inputLen, cudaMemcpyHostToDevice); // Copy input to device

    // Convert to complex
    dim3 block(256);
    dim3 grid((inputLen + block.x - 1) / block.x); // All data points covered by threads
    initializeComplex <<<grid, block >>> (deviceIinput, deviceData, inputLen);
    cudaFree(deviceIinput); // Free deviceIinput early

    cufftHandle plan; // FFT plan
    cufftPlan1d(&plan, inputLen, CUFFT_Z2Z, 1);
    cufftExecZ2Z(plan, deviceData, deviceData, CUFFT_FORWARD); // Apply FFT: Time to frequency

    hilbertFilterKernel <<<grid, block >>> (deviceData, inputLen); // Apply filter
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    cufftExecZ2Z(plan, deviceData, deviceData, CUFFT_INVERSE); // Apply inverse FFT: Frequency to time

    // Normalize to preserve amplitude
    normalizeResult <<<grid, block >>> (deviceData, inputLen);

    // Copy result to host
    fftw_complex* result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * inputLen);
    if (!result) { // Unsuccessful memory allocation
        cerr << "Host allocation failed\n";
        cudaFree(deviceData);
        cufftDestroy(plan);
        return nullptr;
    }

    // Copy results to host
    cudaMemcpy(result, deviceData, sizeof(cufftDoubleComplex) * inputLen, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(deviceData);
    cufftDestroy(plan);

    return result;
}

// Perform per-window envelope comparisons with hilbert transofrms / FFTs to calculate dissimilarity
double* calculateDissimGPU(double** timechunkMatrix, int ptsPerTimewin, int numTimeWin,
            double fftWin, double fs, int& outLen) {

    // # of FFT points
    int ptsPerFFT = static_cast<int>(fftWin * fs);
    if (ptsPerFFT <= 0 || ptsPerTimewin <= 0 || numTimeWin <= 1) {
        outLen = 0;
        return nullptr;
    }

    // # of overlapping FFT windows per time window
    int numfftwin = (ptsPerTimewin - ptsPerFFT) / ptsPerFFT + 1;
    if (numfftwin <= 0) {
        outLen = 0;
        return nullptr;
    }

    outLen = numTimeWin; // # of outputs

    // CUFFT plan
    cufftHandle fftPlan;
    if (cufftPlan1d(&fftPlan, ptsPerFFT, CUFFT_Z2Z, 1) != CUFFT_SUCCESS) {
        outLen = 0;
        return nullptr;
    }

    // Allocate GPU buffers
    cufftDoubleComplex* deviceFFTInput = nullptr;
    cufftDoubleComplex* deviceFFTOutput = nullptr;
    double* deviceMagnitude = nullptr;
    double* deviceEnvelope1 = nullptr;
    double* deviceEnvelope2 = nullptr;

    // Free partial allocations on failure
    if (cudaMalloc(&deviceFFTInput, sizeof(cufftDoubleComplex) * ptsPerFFT) != cudaSuccess ||
        cudaMalloc(&deviceFFTOutput, sizeof(cufftDoubleComplex) * ptsPerFFT) != cudaSuccess ||
        cudaMalloc(&deviceMagnitude, sizeof(double) * ptsPerFFT) != cudaSuccess ||
        cudaMalloc(&deviceEnvelope1, sizeof(double) * ptsPerTimewin) != cudaSuccess ||
        cudaMalloc(&deviceEnvelope2, sizeof(double) * ptsPerTimewin) != cudaSuccess) {

        cudaFree(deviceFFTInput);
        cudaFree(deviceFFTOutput);
        cudaFree(deviceMagnitude);
        cudaFree(deviceEnvelope1);
        cudaFree(deviceEnvelope2);
        cufftDestroy(fftPlan);
        outLen = 0;
        return nullptr;
    }

    // Allocate host result array
    double* diss = new double[outLen];

    diss[0] = NAN; // No previous record for comparison

    // Temporary buffers for hilbert / fft inputs
    cufftDoubleComplex* hil1Host = new cufftDoubleComplex[ptsPerTimewin];
    cufftDoubleComplex* hil2Host = new cufftDoubleComplex[ptsPerTimewin];
    cufftDoubleComplex* fftInputHost = new cufftDoubleComplex[ptsPerFFT];
    double* envelope1Host = new double[ptsPerTimewin];
    double* envelope2Host = new double[ptsPerTimewin];
    double* magnitudeHost = new double[ptsPerFFT];
    double* fftAHost = new double[ptsPerFFT];
    double* fftBHost = new double[ptsPerFFT];

    // Specify dimensions
    dim3 block(256);
    dim3 gridFFT((ptsPerFFT + block.x - 1) / block.x); // All data points covered by threads
    dim3 gridEnv((ptsPerTimewin + block.x - 1) / block.x);

    // Iterate over adjacent pairs of time chunks
    for (int i = 1; i < outLen; ++i) { // diss[0] already NAN
        // Calculate analytic signals
        cudaSetDevice(0); // Use first available GPU
        fftw_complex* hil1 = hilbertRawGPU(timechunkMatrix[i - 1], ptsPerTimewin);
        cudaSetDevice(0); // Use first available GPU
        fftw_complex* hil2 = hilbertRawGPU(timechunkMatrix[i], ptsPerTimewin);

        if (!hil1 || !hil2) { // Failed hilbert computations
            diss[i] = NAN;
            if (hil1) fftw_free(hil1);
            if (hil2) fftw_free(hil2);
            continue;
        }

        // Copy hilbert result to host cufftDoubleComplex arrays
        for (int k = 0; k < ptsPerTimewin; ++k) {
            hil1Host[k].x = hil1[k][0];
            hil1Host[k].y = hil1[k][1];
            hil2Host[k].x = hil2[k][0];
            hil2Host[k].y = hil2[k][1];
        }

        fftw_free(hil1);
        fftw_free(hil2);

        // Copy hilbert data to device
        cufftDoubleComplex* deviceHil1 = nullptr;
        cufftDoubleComplex* deviceHil2 = nullptr;
        cudaMalloc(&deviceHil1, sizeof(cufftDoubleComplex) * ptsPerTimewin);
        cudaMalloc(&deviceHil2, sizeof(cufftDoubleComplex) * ptsPerTimewin);
        cudaMemcpy(deviceHil1, hil1Host, sizeof(cufftDoubleComplex) * ptsPerTimewin, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceHil2, hil2Host, sizeof(cufftDoubleComplex) * ptsPerTimewin, cudaMemcpyHostToDevice);

        // Compute amplitude envelopes on device
        envelopeKernel <<<gridEnv, block >>> (deviceHil1, deviceEnvelope1, ptsPerTimewin);
        envelopeKernel <<<gridEnv, block >>> (deviceHil2, deviceEnvelope2, ptsPerTimewin);
        cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

        // Copy envelopes to host
        cudaMemcpy(envelope1Host, deviceEnvelope1, sizeof(double) * ptsPerTimewin, cudaMemcpyDeviceToHost);
        cudaMemcpy(envelope2Host, deviceEnvelope2, sizeof(double) * ptsPerTimewin, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(deviceHil1);
        cudaFree(deviceHil2);

        // Normalize envelopes
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (int j = 0; j < ptsPerTimewin; ++j) {
            sum1 += envelope1Host[j];
            sum2 += envelope2Host[j];
        }
        // Amplitude between records
        if (sum1 > 1e-12) {
            for (int j = 0; j < ptsPerTimewin; ++j) { envelope1Host[j] /= sum1; }
        }
        if (sum2 > 1e-12) {
            for (int j = 0; j < ptsPerTimewin; ++j) { envelope2Host[j] /= sum2; }
        }

        // Calculate time dissimilarity
        double timeDiss = 0.0;
        for (int j = 0; j < ptsPerTimewin; ++j) { timeDiss += fabs(envelope1Host[j] - envelope2Host[j]); }
        timeDiss *= 0.5;

        // Initialize fftAHost / fftBHost arrays to zero
        for (int j = 0; j < ptsPerFFT; ++j) {
            fftAHost[j] = 0.0;
            fftBHost[j] = 0.0;
        }

        // Frequency domain dissimilarity
        for (int w = 0; w < numfftwin; ++w) {
            int base = w * ptsPerFFT;

            // Process first time chunk FFT
            for (int j = 0; j < ptsPerFFT; ++j) {
                fftInputHost[j].x = timechunkMatrix[i][base + j];
                fftInputHost[j].y = 0.0;
            }
            cudaMemcpy(deviceFFTInput, fftInputHost, sizeof(cufftDoubleComplex) * ptsPerFFT, cudaMemcpyHostToDevice);
            cufftExecZ2Z(fftPlan, deviceFFTInput, deviceFFTOutput, CUFFT_FORWARD);

            fftMagnitudeKernel <<<gridFFT, block >>> (deviceFFTOutput, deviceMagnitude, ptsPerFFT);
            cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

            cudaMemcpy(magnitudeHost, deviceMagnitude, sizeof(double) * ptsPerFFT, cudaMemcpyDeviceToHost);

            for (int j = 0; j < ptsPerFFT; ++j) { fftAHost[j] += magnitudeHost[j]; }

            // Process second time chunk FFT
            for (int j = 0; j < ptsPerFFT; ++j) {
                fftInputHost[j].x = timechunkMatrix[i - 1][base + j];
                fftInputHost[j].y = 0.0;
            }
            cudaMemcpy(deviceFFTInput, fftInputHost, sizeof(cufftDoubleComplex) * ptsPerFFT, cudaMemcpyHostToDevice);
            cufftExecZ2Z(fftPlan, deviceFFTInput, deviceFFTOutput, CUFFT_FORWARD);

            fftMagnitudeKernel <<<gridFFT, block >>> (deviceFFTOutput, deviceMagnitude, ptsPerFFT);
            cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

            cudaMemcpy(magnitudeHost, deviceMagnitude, sizeof(double) * ptsPerFFT, cudaMemcpyDeviceToHost);

            for (int j = 0; j < ptsPerFFT; ++j) { fftBHost[j] += magnitudeHost[j]; }
        }

        // Normalize frequency spectra
        double totalA = 0.0;
        double totalB = 0.0;
        for (int j = 0; j < ptsPerFFT; ++j) {
            totalA += fftAHost[j];
            totalB += fftBHost[j];
        }

        // Populate fft hosts
        if (totalA > 1e-12) {
            for (int j = 0; j < ptsPerFFT; ++j) { fftAHost[j] /= totalA; }
        }
        if (totalB > 1e-12) {
            for (int j = 0; j < ptsPerFFT; ++j) { fftBHost[j] /= totalB; }
        }

        // Calculate frequency dissimilarity
        double freqDiss = 0.0;
        for (int j = 0; j < ptsPerFFT; ++j) { freqDiss += fabs(fftAHost[j] - fftBHost[j]); }
        freqDiss *= 0.5;

        diss[i] = timeDiss * freqDiss; // Combine time / frequency dissimilarity
    }

    // Cleanup
    delete[] hil1Host;
    delete[] hil2Host;
    delete[] fftInputHost;
    delete[] envelope1Host;
    delete[] envelope2Host;
    delete[] magnitudeHost;
    delete[] fftAHost;
    delete[] fftBHost;

    // Free device memory
    cudaFree(deviceFFTInput);
    cudaFree(deviceFFTOutput);
    cudaFree(deviceMagnitude);
    cudaFree(deviceEnvelope1);
    cudaFree(deviceEnvelope2);

    cufftDestroy(fftPlan);

    return diss;
}

// Calculate dissimilarity between last segment of previous file / first segment of next file
double calculateCrossFileDissim(const double* lastSeg, int lastSegLen, const double* firstSeg,
    int firstSegLen, double fftWin, double fs) {
    int segLen = min(lastSegLen, firstSegLen); // Use shorter segment
    if (segLen <= 0) { return NAN; } // Invalid segment length

    int pts_per_fft = static_cast<int>(fftWin * fs); // Calculate FFT size
    if (pts_per_fft <= 0) { return NAN; } // Invalid FFT size

    // For short segments, use entire segment length for FFT
    if (segLen <= pts_per_fft) { pts_per_fft = segLen; }

    // Amplitude envelopes
    double* env1 = new double[segLen];
    double* env2 = new double[segLen];

    // Analytic signals with hilbert transform
    fftw_complex* hil1 = hilbertRawGPU(lastSeg, segLen);
    fftw_complex* hil2 = hilbertRawGPU(firstSeg, segLen);
    if (!hil1 || !hil2) {
        if (hil1) fftw_free(hil1);
        if (hil2) fftw_free(hil2);
        delete[] env1;
        delete[] env2;
        return NAN;
    }

    int i, j; // Iterators
    double sum1 = 0.0, sum2 = 0.0;

    // Amplitude envelopes / sums
    for (i = 0; i < segLen; ++i) {
        env1[i] = hypot(hil1[i][0], hil1[i][1]);
        env2[i] = hypot(hil2[i][0], hil2[i][1]);
        sum1 += env1[i];
        sum2 += env2[i];
    }
    fftw_free(hil1);
    fftw_free(hil2);

    // Envelope normalization
    if (sum1 > 1e-12) {
        for (i = 0; i < segLen; ++i) { env1[i] /= sum1; }
    }
    if (sum2 > 1e-12) {
        for (i = 0; i < segLen; ++i) { env2[i] /= sum2; }
    }

    // Time domain dissimilarity: Distance between envelopes
    double timeDiss = 0.0;
    for (i = 0; i < segLen; ++i) { timeDiss += fabs(env1[i] - env2[i]); }
    timeDiss *= 0.5;

    // Frequency-domain dissimilarity: Calculate magnitude spectrum with FFT
    int half_bins = pts_per_fft / 2 + 1;
    double* mag1 = new double[pts_per_fft]();
    double* mag2 = new double[pts_per_fft]();

    // Create buffers for FFT input
    double* fft_input1 = new double[pts_per_fft]();
    double* fft_input2 = new double[pts_per_fft]();

    // Copy envelope data into FFT buffers (zero-padded if segLen < pts_per_fft)
    int copy_len = min(segLen, pts_per_fft);
    memcpy(fft_input1, env1, copy_len * sizeof(double));
    memcpy(fft_input2, env2, copy_len * sizeof(double));

    // Output buffers
    fftw_complex* fftbuf1 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * half_bins);
    fftw_complex* fftbuf2 = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * half_bins);

    // Create / execute plans
    fftw_plan p1 = fftw_plan_dft_r2c_1d(pts_per_fft, fft_input1, fftbuf1, FFTW_ESTIMATE);
    fftw_plan p2 = fftw_plan_dft_r2c_1d(pts_per_fft, fft_input2, fftbuf2, FFTW_ESTIMATE);
    fftw_execute(p1);
    fftw_execute(p2);

    // Calculate magnitude spectrum
    for (j = 0; j < half_bins; ++j) {
        mag1[j] = sqrt(fftbuf1[j][0] * fftbuf1[j][0] + fftbuf1[j][1] * fftbuf1[j][1]);
        mag2[j] = sqrt(fftbuf2[j][0] * fftbuf2[j][0] + fftbuf2[j][1] * fftbuf2[j][1]);

        // Mirror for negative frequencies
        if (j > 0 && j < pts_per_fft - j) {
            mag1[pts_per_fft - j] = mag1[j];
            mag2[pts_per_fft - j] = mag2[j];
        }
    }

    // Normalize magnitude spectra
    double total1 = 0.0, total2 = 0.0;
    for (j = 0; j < pts_per_fft; ++j) {
        total1 += mag1[j];
        total2 += mag2[j];
    }
    if (total1 > 1e-12) {
        for (j = 0; j < pts_per_fft; ++j) { mag1[j] /= total1; }
    }
    if (total2 > 1e-12) {
        for (j = 0; j < pts_per_fft; ++j) { mag2[j] /= total2; }
    }

    // Frequency domain dissimilarity
    double freqDiss = 0.0;
    for (j = 0; j < pts_per_fft; ++j) { freqDiss += fabs(mag1[j] - mag2[j]); }
    freqDiss *= 0.5;

    // Deallocate resources
    fftw_destroy_plan(p1);
    fftw_destroy_plan(p2);
    fftw_free(fftbuf1);
    fftw_free(fftbuf2);
    delete[] env1;
    delete[] env2;
    delete[] mag1;
    delete[] mag2;
    delete[] fft_input1;
    delete[] fft_input2;

    return timeDiss * freqDiss; // Combined dissimilarity
}

// Deallocate audio data
void freeAudioData(AudioData& audio) {
    for (int ch = 0; ch < audio.numChannels; ++ch) { delete[] audio.samples[ch]; } // Delete channel rows
    delete[] audio.samples; // Delete top level array
}

// Deallocate resources after output
void freeAudioFeatures(AudioFeatures& features) {
    delete[] features.segmentDuration;
    delete[] features.SPLrms;
    delete[] features.SPLpk;
    delete[] features.impulsivity;
    delete[] features.dissim;
    delete[] features.peakCount;

    if (features.autocorr) { // If autocorrelation matrix exists
        for (int i = 0; i < features.autocorrRows; ++i) { delete[] features.autocorr[i]; } // Delete rows
        delete[] features.autocorr; // Delete top level array
    }

    // Reset pointers
    features.segmentDuration = nullptr;
    features.SPLrms = nullptr;
    features.SPLpk = nullptr;
    features.impulsivity = nullptr;
    features.dissim = nullptr;
    features.peakCount = nullptr;
    features.autocorr = nullptr;

    // Delete cross-file dissimilarity
    delete[] features.first60s;
    delete[] features.last60s;
    features.first60s = nullptr;
    features.last60s = nullptr;
}

__global__ void convertToPressureKernel(const double* samples, double* pressure, int numSamples,
            int numBits, double peakVolts, double refSens) {

    int index = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (index >= numSamples) { return; } // Exit if out of bounds

    double samp = samples[index]; // Read sample

    // Right shift align
    if (numBits == 24) { samp = static_cast<double>(static_cast<int>(samp) >> 8); }
    else if (numBits == 32) { samp = static_cast<double>(static_cast<int>(samp) >> 16); }

    // Convert to pressure
    pressure[index] = samp * (peakVolts / static_cast<double>(1 << numBits)) * (1.0 / pow(10.0, refSens / 20.0));
}

void gpuConvertToPressure(const double* hostSamples, double* hostPressure, int length,
            int numBits, double peakVolts, double refSens) {

    // Allocate device memory
    double* deviceSamples;
    double* devicePressure;
    cudaMalloc(&deviceSamples, sizeof(double) * length);
    cudaMalloc(&devicePressure, sizeof(double) * length);

    cudaMemcpy(deviceSamples, hostSamples, sizeof(double) * length, cudaMemcpyHostToDevice); // Copy samples to device

    // CUDA launch configuration
    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    convertToPressureKernel <<<blocks, threadsPerBlock >>> (deviceSamples, devicePressure,
            length, numBits, peakVolts, refSens);

    cudaMemcpy(hostPressure, devicePressure, sizeof(double) * length, cudaMemcpyDeviceToHost); // Copy pressure to host
    
    // Cleanup
    cudaFree(deviceSamples);
    cudaFree(devicePressure);
}

// Main feature extraction
AudioFeatures featureExtraction(int numBits, int peakVolts, const fs::path& filePath,
            double refSens, int timewin, double avtime, int fftWin, int calToneLen, int flow,
            int fhigh, int downsampleFactor, bool omitPartialMinute) {

    string fixedFilePath = fixFilePath(filePath.string()); // Make file path Windows compatible
    AudioData audio = audioRead(filePath.string()); // Read all samples / metadata

    int sampFreq = audio.sampleRate; // Sampling frequency
    int audioSamplesLen = audio.numFrames; // # of audio frames

    double* pressure = new double[audioSamplesLen]; // Convert audio samples to pressure

    double* flatSamples = audio.samples[0]; // First channel
    // Convert to pressure (Pascals)
    gpuConvertToPressure(flatSamples, pressure, audioSamplesLen, numBits, peakVolts, refSens);
    freeAudioData(audio); // Deallocate original audio data

    // Optionally downsample
    if (downsampleFactor != -1) {
        int newLen = 0;
        double* downsampled = downSample(pressure, audioSamplesLen, downsampleFactor, newLen);
        delete[] pressure;
        pressure = downsampled;
        audioSamplesLen = newLen;
        sampFreq /= downsampleFactor;
    }

    // Optionally remove excess noise at start of recording
    if (calToneLen > 0 && audioSamplesLen > calToneLen * sampFreq) {
        int newLen = audioSamplesLen - calToneLen * sampFreq;
        double* shifted = new double[newLen];
        memcpy(shifted, pressure + calToneLen * sampFreq, sizeof(double) * newLen);
        delete[] pressure;
        pressure = shifted;
        audioSamplesLen = newLen;
    }

    // Optionally only include full minutes of recordings
    if ((omitPartialMinute)) {
        double currentDuration = static_cast<double>(audioSamplesLen) / sampFreq;
        double fullMinutesDuration = floor(currentDuration / 60.0) * 60.0; // Find full minutes
        int croppedFrames = static_cast<int>(sampFreq * fullMinutesDuration); // # of frames for full minutes
        if (croppedFrames < audioSamplesLen) { // Trim if full minutes < total time
            double* trimmed = new double[croppedFrames]; // Allocate space for full minute
            memcpy(trimmed, pressure, sizeof(double) * croppedFrames); // Only include full minutes
            delete[] pressure; // Delete original array
            pressure = trimmed; // Set samples to only include full minutes
            audioSamplesLen = croppedFrames; // Update metadata
        }
    }

    // Apply bandpass filter
    BandpassFilter filt = bandpassFilter(pressure, audioSamplesLen, 1.0 / sampFreq, flow, fhigh);
    delete[] pressure;

    // Segment length (samples) / # of time windows
    int ptsPerTimeWin = timewin * sampFreq;
    int numTimeWin = filt.num_pts / ptsPerTimeWin;
    int remainder = filt.num_pts % ptsPerTimeWin;
    if (remainder > 0) { ++numTimeWin; } // Partial minute

    // Pad filtered signal to match time window
    int paddedLen = numTimeWin * ptsPerTimeWin;
    double* paddedSignal = new double[paddedLen]();
    // Include padding for consistent row / column lengths
    memcpy(paddedSignal, filt.time_series_filt, sizeof(double) * filt.num_pts);
    delete[] filt.time_series_filt;
    filt.time_series_filt = nullptr;

    // Initialize audio features struct
    AudioFeatures features = {};
    features.segmentDurationLen = numTimeWin;
    features.segmentDuration = new int[numTimeWin];

    // Signal to time windows
    double** timechunkMatrix = new double* [numTimeWin];
    for (int i = 0; i < numTimeWin; ++i) {
        timechunkMatrix[i] = &paddedSignal[i * ptsPerTimeWin];
        // Duration per segment
        if (i == numTimeWin - 1 && remainder > 0)
        { features.segmentDuration[i] = static_cast<int>(round(static_cast<double>(remainder) / sampFreq)); }
        else { features.segmentDuration[i] = timewin; }
    }

    // Allocate space for features
    features.SPLrmsLen = features.SPLpkLen = features.impulsivityLen = numTimeWin;
    features.SPLrms = new double[numTimeWin];
    features.SPLpk = new double[numTimeWin];
    features.impulsivity = new double[numTimeWin];

    // Calculate SPLrms, SPLpk, / impulsivity for each time segment
    for (int i = 0; i < numTimeWin; ++i) {
        const double* chunk = timechunkMatrix[i];
        double sumSq = 0.0, peak = 0.0;
        for (int j = 0; j < ptsPerTimeWin; ++j) {
            double temp = chunk[j];
            sumSq += temp * temp;
            if (fabs(temp) > peak) { peak = fabs(temp); }
        }
        double rms = sqrt(sumSq / ptsPerTimeWin);
        features.SPLrms[i] = 20.0 * log10(max(rms, 1e-12));
        features.SPLpk[i] = 20.0 * log10(max(peak, 1e-12));
        features.impulsivity[i] = calculateKurtosis(chunk, ptsPerTimeWin);
    }

    // Calculate autocorr / peakcount
    SoloPer per = calculatePeriodicity(paddedSignal, paddedLen, sampFreq, timewin, avtime);

    features.peakCountLen = numTimeWin;
    features.peakCount = new int[numTimeWin];
    for (int i = 0; i < numTimeWin; ++i) { features.peakCount[i] = per.peakCount[i]; } // Write peakCount to features
    delete[] per.peakCount; // Free original array

    features.autocorrRows = per.autocorrRows;
    features.autocorrCols = per.autocorrCols;
    features.autocorr = new double* [per.autocorrRows];
    for (int i = 0; i < per.autocorrRows; ++i) { features.autocorr[i] = per.autocorr[i]; } // Write autocorr to features matrix
    delete[] per.autocorr; // Free original array

    // Calculate dissim
    int dissimLen = 0;
    features.dissim = calculateDissimGPU(timechunkMatrix, ptsPerTimeWin, numTimeWin, fftWin, sampFreq, dissimLen);
    features.dissimLen = dissimLen;

    int pts_per_timewin = timewin * sampFreq;
    int num_timewin = filt.num_pts / pts_per_timewin;

    features.sampleRate = sampFreq;

    // Store first / last segments for cross-file dissimilarity
    if (num_timewin > 0) {
        int actualLength = audioSamplesLen; // Use full audio length if under 60 seconds

        // First segment: Use min of 60 seconds or actual length
        int firstSegLength = min(actualLength, sampFreq * 60);
        features.first60s = new double[firstSegLength];
        memcpy(features.first60s, paddedSignal, sizeof(double) * firstSegLength);
        features.first60sLen = firstSegLength;

        // Last segment: use min of 60 seconds or actual length
        int lastSegLength = min(actualLength, sampFreq * 60);
        features.last60s = new double[lastSegLength];
        int startOffset = max(0, actualLength - lastSegLength);
        memcpy(features.last60s, paddedSignal + startOffset, sizeof(double) * lastSegLength);
        features.last60sLen = lastSegLength;
    }

    // Deallocate temporary arrays
    delete[] timechunkMatrix;
    delete[] paddedSignal;

    return features;
}

// Copy input file name to output file record
tm extractBaseTime(const string& filename) {
    tm baseTime = {}; // Fields initialized to zero
    smatch match; // Will store matched part
    regex pattern1(R"((\d{8})_(\d{6}))"); // Matches YYYYMMDD_HHMMSS
    regex pattern2(R"(.*\.(\d{6})(\d{6}))"); // Matches XXXX.YYMMDDHHMMSS

    // Find date / time from file name
    if (regex_search(filename, match, pattern1) && match.size() == 3) {
        string date = match[1]; // Date
        string time = match[2]; // Time

        baseTime.tm_year = stoi(date.substr(0, 4)) - 1900; // Years since 1900: 4 digits
        baseTime.tm_mon = stoi(date.substr(4, 2)) - 1; // Zero based month
        baseTime.tm_mday = stoi(date.substr(6, 2)); // Day of month
        baseTime.tm_hour = stoi(time.substr(0, 2)) - 1; // Hour
        baseTime.tm_min = stoi(time.substr(2, 2)); // Minute
        baseTime.tm_sec = stoi(time.substr(4, 2)); // Second
    }
    else if (regex_search(filename, match, pattern2) && match.size() == 3) {
        string date = match[1]; // YYMMDD
        string time = match[2]; // HHMMSS

        int year = stoi(date.substr(0, 2)); // Year: 2 digits
        // Assume years 00-40 are 2000s, 41-99 are 1900s
        if (year <= 40) { baseTime.tm_year = year + 100; } // 2000-2040
        else { baseTime.tm_year = year; } // 1941-1999
        baseTime.tm_mon = stoi(date.substr(2, 2)) - 1; // Month
        baseTime.tm_mday = stoi(date.substr(4, 2)); // Day
        baseTime.tm_hour = stoi(time.substr(0, 2)) - 1; // Hour
        baseTime.tm_min = stoi(time.substr(2, 2)); // Minute
        baseTime.tm_sec = stoi(time.substr(4, 2)); // Second
    }

    return baseTime;
}

// Export saved features to CSV file
void saveFeaturesToCSV(const char* filename, const char** filenames, int numFiles,
    const AudioFeatures* allFeatures, const vector<double>& crossFileDissim) {
    ofstream outputFile(filename); // Open file for writing
    if (!outputFile.is_open()) { // Error opening file
        cerr << "Error: Unable to open output file: " << filename << endl;
        return;
    }

    // Write CSV header
    outputFile << "Filename,Year,Month,Day,Hour,Minute,SegmentDuration,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount\n";

    string* fileBuffers = new string[numFiles]; // Allocate file buffers

    int fileIdx, i, maxLength; // Single declaration outside of loops

    // Write CSV rows per file in parallel
#pragma omp parallel for
    for (fileIdx = 0; fileIdx < numFiles; ++fileIdx) {
        const AudioFeatures& features = allFeatures[fileIdx]; // Current file features
        ostringstream oss; // Per thread string builder

        maxLength = features.SPLrmsLen; // Set SPLrmsLen as max length for current file

        // Extract timestamp from filename
        tm baseTime = extractBaseTime(filenames[fileIdx]);
        time_t baseEpoch = mktime(&baseTime); // Convert to seconds
        tm* firstTime = localtime(&baseEpoch); // Convert to local time

        // Timestamp extraction unsuccessful
        bool useNanTimestamp = false;
        if (!firstTime || (firstTime->tm_year + 1900) < 1900) { useNanTimestamp = true; } // Date / time set to NaN

        for (i = 0; i < maxLength; ++i) { // Iterate through time segments
            time_t currentEpoch = baseEpoch + i * 60;
            tm* currentTime = localtime(&currentEpoch);

            oss << filenames[fileIdx] << ","; // Original file name

            if (useNanTimestamp || !currentTime) { oss << "NaN,NaN,NaN,NaN,NaN,"; }
            else {
                oss << (currentTime->tm_year + 1900) << "," // Year
                    << (currentTime->tm_mon + 1) << "," // Month
                    << currentTime->tm_mday << "," // Day
                    << currentTime->tm_hour << "," // Hour
                    << currentTime->tm_min << ","; // Minute
            }

            // Segment duration
            if (i < features.segmentDurationLen) { oss << to_string(features.segmentDuration[i]) << ","; }
            else { oss << "NaN,"; }

            // SPLrms
            if (i < features.SPLrmsLen) { oss << to_string(features.SPLrms[i]) << ","; }
            else { oss << "NaN,"; }

            // SPLpk
            if (i < features.SPLpkLen) { oss << to_string(features.SPLpk[i]) << ","; }
            else { oss << "NaN,"; }

            // Impulsivity
            if (i < features.impulsivityLen) { oss << to_string(features.impulsivity[i]) << ","; }
            else { oss << "NaN,"; }

            // Dissim
            if (i < features.dissimLen) { oss << to_string(features.dissim[i]) << ","; }
            else { oss << "NaN,"; }

            // PeakCount
            if (i < features.peakCountLen) { oss << to_string(features.peakCount[i]); }
            else { oss << "NaN"; }

            oss << "\n"; // End of row
        }

        fileBuffers[fileIdx] = oss.str(); // Per file CSV content in buffer
    }

    // Write all buffers sequentially to CSV file
    for (i = 0; i < numFiles; ++i) { outputFile << fileBuffers[i]; }

    // Deallocate resources
    delete[] fileBuffers;
    outputFile.close();
}

// Sort input files
void bubbleSort(char arr[][128], int n) {
    char temp[128]; // Buffer for swapping strings
    for (int i = 0; i < n - 1; ++i) { // Iterate through array
        for (int j = 0; j < n - i - 1; ++j) { // Compare adjacent elements up to unsorted portion
            if (strcmp(arr[j], arr[j + 1]) > 0) { // Lexicographically compare two strings
                // Swap strings using temp buffer
                strcpy(temp, arr[j]);
                strcpy(arr[j], arr[j + 1]);
                strcpy(arr[j + 1], temp);
            }
        }
    }
}

// Process input files in parallel
void threadWrapper(ThreadArgs& args) {
    while (true) {
        int index = args.nextIndex->fetch_add(1, memory_order_relaxed); // Get next file index
        if (index >= args.totalFiles) { break; } // No more files

        try {
            if (args.debugOutput == 1) {
                // Inform user of file processing progress
                cout << "Processing file index " << index << ": " << args.filePaths[index] << "\n";
                cerr.flush();
            }

            fs::path filePath(args.filePaths[index]); // Convert C-style path to a filesystem::path for safe handling
            string filenameStr = filePath.filename().string(); // Extract filename without path

            if (args.fileTimeInfo) { // Extract date / time data if present
                tm extractedTime = extractBaseTime(filenameStr);
                args.fileTimeInfo[index].baseTime = extractedTime;
                args.fileTimeInfo[index].filename = filenameStr;
                args.fileTimeInfo[index].timeExtracted =
                    (extractedTime.tm_year > 0 || extractedTime.tm_mon >= 0 ||
                        extractedTime.tm_mday > 0 || extractedTime.tm_hour >= 0);
            }

            // Run feature extraction
            args.allFeatures[index] = featureExtraction(
                args.numBits, args.peakVolts, filePath, args.RS,
                args.timeWin, args.avTime, args.fftWin, args.artiLen,
                args.fLow, args.fHigh, args.downSample, args.omitPartialMinute
            );

            // Store filename for CSV output
            strcpy(args.filenames[index], filenameStr.c_str());
            args.filenames[index][sizeof(args.filenames[index]) - 1] = '\0';
        }
        catch (const exception& e)
            { cerr << "Exception in thread while processing index " << index << ": " << e.what() << "\n"; }
        catch (...) { cerr << "Unknown exception in thread while processing index " << index << "\n"; }
    }
}

// Process directory of sound files with user-given parameters
int main(int argc, char* argv[]) {
    using namespace std; // Standard namespace
    using namespace chrono; // Time tracking

    auto start = high_resolution_clock::now(); // Starting time to show runtime performance

    // Use first available GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA devices available.\n");
        exit(EXIT_FAILURE);
    }
    int deviceID = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceID);
    cudaSetDevice(deviceID);

    // Default values if unspecified by user
    char inputDir[128] = {};
    char outputFile[128] = {};
    int numBits = 16, peakVolts = 2;
    int timeWin = 60, fftWin = 1, fLow = 1, fHigh = 16000;
    double RS = -178.3, avTime = 0.1, artiLen = 0.0;
    int maxThreads = 1, downSample = -1, debugOutput = 0;
    bool omitPartialMinute = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--omit_partial_minute") == 0) { omitPartialMinute = true; }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) { strcpy(inputDir, argv[++i]); }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { strcpy(outputFile, argv[++i]); }
        else if (strcmp(argv[i], "--num_bits") == 0 && i + 1 < argc) { numBits = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--RS") == 0 && i + 1 < argc) { RS = atof(argv[++i]); }
        else if (strcmp(argv[i], "--peak_volts") == 0 && i + 1 < argc) { peakVolts = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--arti_len") == 0 && i + 1 < argc) { artiLen = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--timewin") == 0 && i + 1 < argc) { timeWin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fft_win") == 0 && i + 1 < argc) { fftWin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--avtime") == 0 && i + 1 < argc) { avTime = atof(argv[++i]); }
        else if (strcmp(argv[i], "--flow") == 0 && i + 1 < argc) { fLow = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fhigh") == 0 && i + 1 < argc) { fHigh = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max_threads") == 0 && i + 1 < argc) { maxThreads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--downsample") == 0 && i + 1 < argc) { downSample = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--debug_output") == 0 && i + 1 < argc) { debugOutput = atoi(argv[++i]); }
    }

    // Count .wav files
    int totalFiles = 0;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".wav") { ++totalFiles; } // Only read .wav files
    }

    if (totalFiles == 0) { // No .wav files in directory
        cerr << "No valid .wav files found in " << inputDir << "\n";
        return 1;
    }

    // Allocate arrays large enough for all files
    char (*filePaths)[128] = new char[totalFiles][128]; // Path to files
    char (*filenames)[128] = new char[totalFiles][128]; // Base filenames
    AudioFeatures* allFeatures = new AudioFeatures[totalFiles]; // Initialize AudioFeatures struct
    FileTimeInfo* fileTimeInfo = new FileTimeInfo[totalFiles]; // Time info for each file

    // Fill filePaths with file names
    int index = 0;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".wav") {
            strcpy(filePaths[index], entry.path().string().c_str());
            filePaths[index][511] = '\0'; // Reserve final index for terminal character
            ++index;
        }
    }

    bubbleSort(filePaths, totalFiles); // Sort file paths alphabetically for in-order data processing

    // Thread setup
    atomic<int> nextIndex(0);
    int availableThreads = max(1, thread::hardware_concurrency());
    int numThreads = min(maxThreads, availableThreads);

    // Thread args
    ThreadArgs args;
    args.nextIndex = &nextIndex;
    args.totalFiles = totalFiles;
    args.filePaths = filePaths;
    args.allFeatures = allFeatures;
    args.filenames = filenames;
    args.fileTimeInfo = nullptr;
    args.numBits = numBits;
    args.peakVolts = peakVolts;
    args.RS = RS;
    args.timeWin = timeWin;
    args.avTime = avTime;
    args.fftWin = fftWin;
    args.artiLen = artiLen;
    args.fLow = fLow;
    args.fHigh = fHigh;
    args.downSample = downSample;
    args.omitPartialMinute = omitPartialMinute;
    args.debugOutput = debugOutput;

    // Launch threads
    thread* threads = new thread[numThreads];
    for (int i = 0; i < numThreads; ++i) { threads[i] = thread(threadWrapper, ref(args)); } // Reference to args
    for (int i = 0; i < numThreads; ++i) { threads[i].join(); } // Join after all threads are launched
    delete[] threads; // Deallocate threads

    // Compute cross file dissimilarities
    for (int i = 0; i < totalFiles - 1; ++i) {
        // Calculate cross-file dissimilarity between end of current file and start of next file
        double crossFileDissim = calculateCrossFileDissim(
            allFeatures[i].last60s, allFeatures[i].last60sLen,
            allFeatures[i + 1].first60s, allFeatures[i + 1].first60sLen,
            fftWin, allFeatures[i].sampleRate
        );

        // Assign cross-file dissimilarity to the first segment of the next file
        if (allFeatures[i + 1].dissimLen > 0 && allFeatures[i + 1].dissim) {
            allFeatures[i + 1].dissim[0] = crossFileDissim;
        }
    }

    const char** fileNames = new const char* [totalFiles];
    for (int i = 0; i < totalFiles; ++i) { fileNames[i] = filenames[i]; } // Convert to const char* array

    // Save calculated features to output
    vector<double> crossFileDissimVector; // Empty vector since we're handling it differently now
    saveFeaturesToCSV(outputFile, fileNames, totalFiles, allFeatures, crossFileDissimVector);

    if (debugOutput == 1) {
        // Inform user of file processing progress
        cout << "Saved features for " << totalFiles << " files to " << outputFile << endl;
        cerr.flush();
    }

    for (int i = 0; i < totalFiles; ++i) { freeAudioFeatures(allFeatures[i]); } // Free features for each file

    // Cleanup
    delete[] filePaths;
    delete[] filenames;
    delete[] allFeatures;
    delete[] fileTimeInfo;
    delete[] fileNames;
    fftw_cleanup(); // Clean up internal memory

    auto stop = high_resolution_clock::now(); // Ending time to show runtime performance
    duration<double> elapsed = duration_cast<duration<double>>(stop - start);
    if (debugOutput == 1) { cout << "Runtime: " << elapsed.count() << " seconds\n"; } // Total execution time
    
    return 0;
}
