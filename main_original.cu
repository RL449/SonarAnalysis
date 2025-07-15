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
    double* filteredTimeSeries; // Time domain signal after filtering
    double* amplitudeSpectrum; // Frequency domain amplitude spectrum
    int length; // # of samples

    // Constructor
    BandpassFilter(double* ts, double* spec, int len) : filteredTimeSeries(ts), amplitudeSpectrum(spec), length(len) {}

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
    int* peakCount; // # of peaks per time window
    double** autocorr; // Autocorrelation per segment
    int peakcountLength; // Length of peakcount array
    int autocorrRows; // # of time windows - rows
    int autocorrCols; // # of lags - columns
};

struct ArrayShiftFFT {
    double* data; // Array of samples after shift
    int length; // Length of array

    // Destructor
    ~ArrayShiftFFT() { delete[] data; }
};

// RAII wrapper for FFTW complex buffer + plan
struct FFTWHandler {
    fftw_complex* buf = nullptr; // Buffer for FFT computation
    fftw_plan forwardPlan = nullptr; // Forward FFT plan
    fftw_plan inversePlan = nullptr; // Inverse FFT plan
    int size = 0; // # of points in FFT

    // Constructor
    FFTWHandler(int N) : size(N) {
        buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N);
        if (!buf) { throw bad_alloc(); }

        forwardPlan = fftw_plan_dft_1d(size, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
        if (!forwardPlan) {
            fftw_free(buf);
            throw runtime_error("FFTW forward plan creation failed");
        }

        inversePlan = fftw_plan_dft_1d(size, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!inversePlan) {
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
};

struct ThreadArgs {
    atomic<int>* nextIndex; // Counter for thread-safe file indexing
    int totalFiles; // # of audio files to process
    char (*filePaths)[512]; // Input file paths
    AudioFeatures* allFeatures; // Feature extraction results
    char (*filenames)[512]; // Names of files

    // User-given arguments
    int numBits, peakVolts, timeWin, fftWin, arti, fLow, fHigh, downSample;
    double RS, avTime;
    bool omitPartialMinute;
};

// Replaces backslashes with forward slashes to work with Windows file paths
string fixFilePath(const string& path) {
    string fixedPath = path;
    replace(fixedPath.begin(), fixedPath.end(), '\\', '/');
    return fixedPath;
}

// Read audio samples / extract recording metadata
AudioData audioRead(const string& filename, SampleRange range = {1, -1}) {
    SNDFILE* file; // Sound file
    SF_INFO sfinfo = {}; // Sound metadata

    file = sf_open(filename.c_str(), SFM_READ, &sfinfo); // Open file in read mode
    if (!file) { throw runtime_error("Error opening audio file: " + string(sf_strerror(file))); }

    // Sample range to read
    int totalFrames = sfinfo.frames; // Frames per channel
    int endSample;
    if (range.endSample == -1) { endSample = totalFrames; } // No range specified
    else { endSample = min(range.endSample, totalFrames); }
    
    int startSample = max(0, range.startSample - 1); // Zero based indexing
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
            short* tempBuffer = new short[numFramesToRead * numChannels];
            sf_readf_short(file, tempBuffer, numFramesToRead); // Read into temporary short buffer
            for (int i = 0; i < numFramesToRead * numChannels; ++i) { interleavedSamples[i] = static_cast<double>(tempBuffer[i]); }
            delete[] tempBuffer; // Clean up temp buffer
            break;
        }
        case SF_FORMAT_PCM_24:
        case SF_FORMAT_PCM_32: { // 24 or 32 bit
            int* tempBuffer = new int[numFramesToRead * numChannels];
            sf_readf_int(file, tempBuffer, numFramesToRead); // Read into temporary int buffer
            for (int i = 0; i < numFramesToRead * numChannels; ++i) { interleavedSamples[i] = static_cast<double>(tempBuffer[i]); }
            delete[] tempBuffer; // Clean up temp buffer
            break;
        }
        default:
            // Unsupported bit depth
            sf_close(file);
            delete[] interleavedSamples;
            throw runtime_error("Unsupported bit format");
    }

    sf_close(file); // Reading complete

    // Channel data matrix: one array per channel
    double** samples = new double*[numChannels];
    for (int ch = 0; ch < numChannels; ++ch) { samples[ch] = new double[numFramesToRead]; }

    // De-interleave sample buffer into one array per channel
    // Separate indices per channel
    for (int i = 0; i < numFramesToRead; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) { samples[ch][i] = interleavedSamples[i * numChannels + ch]; }
    }

    delete[] interleavedSamples; // Deallocate interleaved buffer

    return AudioData{samples, numChannels, numFramesToRead, sfinfo.samplerate}; // Metadata
}

AudioInfo audioReadInfo(const string& filePath) {
    SF_INFO sfInfo = {0}; // Struct containing sound metadata (frames, samplerate, channels, format)
    SNDFILE* file = sf_open(filePath.c_str(), SFM_READ, &sfInfo); // Open audio file in read mode

    if (!file) { throw runtime_error("Error opening audio file: " + filePath); } // Error opening file

    int sampleRate = sfInfo.samplerate; // Get sample rate
    int numFrames = sfInfo.frames; // Get # of frames
    float duration = static_cast<float>(numFrames) / sampleRate; // Calculate duration (seconds)

    sf_close(file); // Close file after reading info

    return {sampleRate, duration};
}

__global__ void downsampleKernel(const double* x, double* result, int length, int factor) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Compute global index of thread
    if (index * factor < length) { result[index] = x[index * factor]; } // Determine if downsampled data is in input bounds
}

// Reduce sampling rate by analyzing (1 / factor) samples
double* downSample(const double* x, int length, int factor, int& newLength) {
    if (factor <= 0) { throw invalid_argument("Factor must be positive"); } // Validate input

    newLength = (length + factor - 1) / factor; // # of samples in downsampled signal

    // Allocate GPU memory
    double* deviceInput;
    double* deviceOutput;
    cudaMalloc(&deviceInput, sizeof(double) * length);
    cudaMalloc(&deviceOutput, sizeof(double) * newLength);

    cudaMemcpy(deviceInput, x, sizeof(double) * length, cudaMemcpyHostToDevice); // Copy input signal from host to device

    // Launch kernel
    int threads = 256; // # of threads per block
    int blocks = (newLength + threads - 1) / threads; // # of blocks
    downsampleKernel <<<blocks, threads >>> (deviceInput, deviceOutput, length, factor);

    double* result = new double[newLength]; // Allocate memory on host
    cudaMemcpy(result, deviceOutput, sizeof(double) * newLength, cudaMemcpyDeviceToHost); // Copy downsampled result to host

    // Deallocate GPU memory
    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    return result; // Downsampled signal
}

__global__ void fftShiftKernel(const double* input, double* shifted, int length) {
    // Shift array to center around zero index component
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Global index
    if (i < length) { shifted[i] = input[(i + (length / 2)) % length]; } // Process valid indices
}

// CUDA kernel to compute shifted frequency array / apply bandpass filter
__global__ void applyBandpassFilter(cufftDoubleComplex* freqData, int numPoints, double freqStep, double fLow, double fHigh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) { return; }

    int shiftedIndex = (i + numPoints / 2) % numPoints;

    // Convert index to frequency (Hz): Accounts for zero index centering
    double freq = (shiftedIndex - numPoints / 2) * freqStep;
    double absFreq = fabs(freq);

    // Zero out components outside specified range
    if (absFreq < fLow || absFreq > fHigh) {
        freqData[i].x = 0.0;
        freqData[i].y = 0.0;
    }
}

// CUDA kernel to normalize inverse FFT / compute amplitude spectrum
__global__ void normalizeAndComputeAmplitude(const cufftDoubleComplex* timeData, double* outputTime, double* outputAmp, int numPoints) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numPoints) { return; }

    double norm = 1.0 / numPoints;
    outputTime[i] = timeData[i].x * norm; // Normalize inverse FFT real output
    outputAmp[i] = sqrt(timeData[i].x * timeData[i].x + timeData[i].y * timeData[i].y); // Compute magnitude at each point
}

// Apply bandpass filter in frequency domain
BandpassFilter bandpassFilter(const double* timeSeries, int numPts, double frequency, double flow, double freqHigh) {
    // Allocate memory on device
    cufftDoubleComplex* deviceFreqData;
    cudaMalloc(&deviceFreqData, sizeof(cufftDoubleComplex) * numPts);

    // Copy timeSeries to device, real part only
    cufftDoubleComplex* hostInput = new cufftDoubleComplex[numPts];
    for (int i = 0; i < numPts; ++i) {
        hostInput[i].x = timeSeries[i];
        hostInput[i].y = 0.0;
    }
    cudaMemcpy(deviceFreqData, hostInput, sizeof(cufftDoubleComplex) * numPts, cudaMemcpyHostToDevice); // Copy frequency data to device
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
    cudaMemcpy(timeSeriesFilt, deviceTimeOut, sizeof(double) * numPts, cudaMemcpyDeviceToHost); // Copy time series data to host
    cudaMemcpy(amplitudeSpectrum, deviceAmplitudeOut, sizeof(double) * numPts, cudaMemcpyDeviceToHost); // Copy magnitude spectrum to host

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

    // Thread / global index
    int threadIndex = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    double val; // Load current val
    if (index < pointsPerTimeWin) { val = data[index]; }
    else { val = 0.0; }
    localSum[threadIndex] = val;
    localSumSq[threadIndex] = val * val;
    __syncthreads(); // Ensure all threads have written

    // Reduce within block
    for (int stepSize = blockDim.x / 2; stepSize > 0; stepSize >>= 1) {
        if (threadIndex < stepSize) {
            localSum[threadIndex] += localSum[threadIndex + stepSize];
            localSumSq[threadIndex] += localSumSq[threadIndex + stepSize];
        }
        __syncthreads(); // Ensure all threads have written
    }

    // First thread in each block writes result to global memory
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

// Calculate kurtosis used for impulsivity of a signal
double calculateKurtosis(const double* hostData, int pointsPerTimeWin) {
    if (pointsPerTimeWin <= 0 || hostData == nullptr) { throw invalid_argument("Input array is empty or null"); }

    // Allocate device memory
    double* deviceData, * deviceSumPartial, * deviceSumSqPartial, * deviceFourth;
    cudaMalloc(&deviceData, pointsPerTimeWin * sizeof(double));
    cudaMemcpy(deviceData, hostData, pointsPerTimeWin * sizeof(double), cudaMemcpyHostToDevice); // Copy data to device

    // Thread / block setup
    int threads = 256;
    int blocks = (pointsPerTimeWin + threads - 1) / threads;

    cudaMalloc(&deviceSumPartial, blocks * sizeof(double));
    cudaMalloc(&deviceSumSqPartial, blocks * sizeof(double));

    // Compute mean / variance components
    partialSumsKernel <<<blocks, threads >>> (deviceData, deviceSumPartial, deviceSumSqPartial, pointsPerTimeWin);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    // Reduce partial results to final results
    // thrust::device_ptr = Can be used with thrust algorithms
    thrust::device_ptr<double> sumPtr(deviceSumPartial);
    thrust::device_ptr<double> sumSqPtr(deviceSumSqPartial);

    // Sum values in parallel
    double totalSum = thrust::reduce(sumPtr, sumPtr + blocks, 0.0, thrust::plus<double>());
    double totalSumSq = thrust::reduce(sumSqPtr, sumSqPtr + blocks, 0.0, thrust::plus<double>());

    double mean = totalSum / pointsPerTimeWin;
    double variance = (totalSumSq / pointsPerTimeWin) - (mean * mean);

    if (variance < 1e-12) { // Avoid divide by zero
        cudaFree(deviceData);
        cudaFree(deviceSumPartial);
        cudaFree(deviceSumSqPartial);
        return 0.0;
    }

    // Compute fourth moment
    cudaMalloc(&deviceFourth, pointsPerTimeWin * sizeof(double));
    fourthMomentKernel <<<blocks, threads >>> (deviceData, deviceFourth, mean, pointsPerTimeWin);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution

    // Reduce fourth moment array to scalar value / sum values in parallel
    thrust::device_ptr<double> fourthPtr(deviceFourth);
    double fourthMoment = thrust::reduce(fourthPtr, fourthPtr + pointsPerTimeWin, 0.0, thrust::plus<double>()) / pointsPerTimeWin;

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
    if (index < pointsPerTimeWin) { envelope[index] = sqrt(hilbert[index].x * hilbert[index].x + hilbert[index].y * hilbert[index].y); }
}

// CUDA kernel for correlation computation
__global__ void correlationKernel(const double* real, const double* imaginary, int seriesLength,
            double* corrVals, int maxLag, int offset) {
    int lag = blockIdx.x * blockDim.x + threadIdx.x;
    if (lag > maxLag) { return; } // Skip out of range lags
    
    // Accumulators for statistical calculations
    double sumReal = 0.0, sumImagninary = 0.0, sumRealSquare = 0.0, sumImaginarySquare = 0.0, sumRealImaginaryProd = 0.0;
    int sampleCount = 0;
    
    // Loop through overlaping samples of current lag
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
    
    double covar = (sumRealImaginaryProd / sampleCount) - (meanReal * meanImaginary);
    double denomReal = sqrt(meanXSquare - (meanReal * meanReal));
    double denomImaginary = sqrt(meanYSquare - (meanImaginary * meanImaginary));
    
    // Normalize correlation
    if (denomReal == 0.0 || denomImaginary == 0.0) { corrVals[lag] = NAN; }
    else { corrVals[lag] = covar / (denomReal * denomImaginary); }
}

// CUDA kernel for FFT magnitude calculation
__global__ void fftMagnitudeKernel(const cufftDoubleComplex* fftData, double* magnitude, int pointsPerFFT) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pointsPerFFT) { magnitude[idx] = sqrt(fftData[idx].x * fftData[idx].x + fftData[idx].y * fftData[idx].y); } // Calculate magnitude
}

// GPU-accelerated correlation function
Correlation correl5GPU(const double* timeSeries1, const double* timeSeries2, 
            int seriesLength, int lags, int offset) {
    int len = lags + 1;
    
    // Allocate GPU memory
    double * deviceReal, * deviceImaginary, * deviceCorrVals;
    cudaMalloc(&deviceReal, sizeof(double) * seriesLength);
    cudaMalloc(&deviceImaginary, sizeof(double) * seriesLength);
    cudaMalloc(&deviceCorrVals, sizeof(double) * len);
    
    // Copy deviceReal / deviceImaginary to device
    cudaMemcpy(deviceReal, timeSeries1, sizeof(double) * seriesLength, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceImaginary, timeSeries2, sizeof(double) * seriesLength, cudaMemcpyHostToDevice);
    
    // Launch correlation kernel
    // Specify dimensions
    dim3 block(256);
    dim3 grid((len + block.x - 1) / block.x);
    correlationKernel<<<grid, block>>>(deviceReal, deviceImaginary, seriesLength, deviceCorrVals, lags, offset);
    cudaDeviceSynchronize(); // Ensure previous operations are completed before execution
    
    // Allocate host memory
    double* corrVals = new double[len];
    double* lagVals = new double[len];
    
    cudaMemcpy(corrVals, deviceCorrVals, sizeof(double) * len, cudaMemcpyDeviceToHost); // Copy corrVals to host
    
    // Fill lag values
    for (int i = 0; i < len; ++i) { lagVals[i] = static_cast<double>(i); }
    
    // Cleanup GPU memory
    cudaFree(deviceReal);
    cudaFree(deviceImaginary);
    cudaFree(deviceCorrVals);
    
    return Correlation(corrVals, lagVals, len);
}

// Kernel to square / segment the input
__global__ void squareAndSegment(const double* input, double* output, int sampWindowSize, int numTimeWins) {
    int index = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int total = sampWindowSize * numTimeWins; // # of samples

    if (index < total) {
        double val = input[index];
        output[index] = val * val; // Signal energy
    }
}

// Kernel to average squared values
__global__ void computeAverages(const double* squared, double* outputAvg, int sampWindowSize, int avgWinSize, int numavwin, int numTimeWins) {
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
SoloPerGM2 fSoloPerGM2(const double* pFiltInput, int inputLength, double fs, double timewin, double avtime) {
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
    computeAverages <<<numTimeWins, numAvWin >>> (deviceSquared, deviceAvg, sampWindowSize, avgWinSize, numAvWin, numTimeWins);

    // Copy averages back to host
    double* hostAvg = new double[numTimeWins * numAvWin];
    cudaMemcpy(hostAvg, deviceAvg, numTimeWins * numAvWin * sizeof(double), cudaMemcpyDeviceToHost); // Copy averages to host

    // Outputs for correlation / peak count
    int pAvTotRows = numAvWin;
    int lagLimit = static_cast<int>(pAvTotRows * 0.7); // 70% of lags
    int pAvTotCols = numTimeWins;

    double** acorr = new double* [pAvTotCols]; // Autocorrelation per window
    int* pkcount = new int[pAvTotCols]; // Peak count per window

    // Iterate through time windows - Calculate autocorr / peak count
    for (int i = 0; i < pAvTotCols; i++) {
        // Calculate correlation
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
                for (int k = j + 1; k <= lagLimit && acorr[i][k] < acorr[i][j]; k++) { rightMin = min(rightMin, acorr[i][k]); }
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
    SoloPerGM2 result;
    result.peakCount = pkcount;
    result.autocorr = acorr;
    result.peakcountLength = pAvTotCols;
    result.autocorrRows = pAvTotCols;
    result.autocorrCols = lagLimit + 1;

    return result;
}

// CUDA kernel for Hilbert transform filter application
__global__ void hilbertFilterKernel(cufftDoubleComplex* data, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= len) { return; }
    
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

// Convert real input to complex
__global__ void initializeComplex(double* input, cufftDoubleComplex* output, int len) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (threadIndex < len) {
        output[threadIndex].x = input[threadIndex]; // Real
        output[threadIndex].y = 0.0; // Imaginary
    }
}

// Normalize inverse FFT result
__global__ void normalizeResult(cufftDoubleComplex* data, int len) {
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (threadIndex < len) {
        data[threadIndex].x /= len; // Normalize real part
        data[threadIndex].y /= len; // Normalize imaginary part
    }
}

// Calculate analytic signal using hilbert transform with FFT
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
    // Specify dimensions
    dim3 block(256);
    dim3 grid((inputLen + block.x - 1) / block.x);
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
    if (!result) {
        cerr << "Host allocation failed\n";
        cudaFree(deviceData);
        cufftDestroy(plan);
        return nullptr;
    }

    cudaMemcpy(result, deviceData, sizeof(cufftDoubleComplex) * inputLen, cudaMemcpyDeviceToHost); // Copy results to host

    // Cleanup
    cudaFree(deviceData);
    cufftDestroy(plan);

    return result;
}

// Calculate dissimilarity with GPU
double* fSoloDissimGM1GPU(double** timechunkMatrix, int ptsPerTimewin, int numTimeWin,
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

    outLen = numTimeWin - 1; // # of outputs

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
    dim3 gridFFT((ptsPerFFT + block.x - 1) / block.x);
    dim3 gridEnv((ptsPerTimewin + block.x - 1) / block.x);

    // Iterate over adjacent pairs of time chunks
    for (int i = 0; i < outLen; ++i) {
        // Calculate analytic signals
        fftw_complex* hil1 = hilbertRawGPU(timechunkMatrix[i], ptsPerTimewin);
        fftw_complex* hil2 = hilbertRawGPU(timechunkMatrix[i + 1], ptsPerTimewin);

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

        cudaFree(deviceHil1);
        cudaFree(deviceHil2);

        // Normalize envelopes
        double sum1 = 0.0;
        double sum2 = 0.0;
        for (int j = 0; j < ptsPerTimewin; ++j) {
            sum1 += envelope1Host[j];
            sum2 += envelope2Host[j];
        }
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
                fftInputHost[j].x = timechunkMatrix[i + 1][base + j];
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

    cudaFree(deviceFFTInput);
    cudaFree(deviceFFTOutput);
    cudaFree(deviceMagnitude);
    cudaFree(deviceEnvelope1);
    cudaFree(deviceEnvelope2);

    cufftDestroy(fftPlan);

    return diss;
}

// Free allocated memory for audio samples
void freeAudioData(AudioData& audio) {
    for (int ch = 0; ch < audio.numChannels; ++ch) { delete[] audio.samples[ch]; } // Deallocate each channel
    delete[] audio.samples; // Deallocate top level array
}

// Free allocated memory for extracted features
void freeAudioFeatures(AudioFeatures& features) {
    // Free 1D feature arrays
    delete[] features.segmentDuration;
    delete[] features.SPLrms;
    delete[] features.SPLpk;
    delete[] features.impulsivity;
    delete[] features.dissim;
    delete[] features.peakCount;
    
    for (int i = 0; i < features.autocorrRows; ++i) { delete[] features.autocorr[i]; } // Deallocate each row of autocorr
    delete[] features.autocorr; // Deallocate top layer array

    // Reset pointers
    features.segmentDuration = nullptr;
    features.SPLrms = nullptr;
    features.SPLpk = nullptr;
    features.impulsivity = nullptr;
    features.dissim = nullptr;
    features.peakCount = nullptr;
    features.autocorr = nullptr;
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

    double* deviceSamples;
    double* devicePressure;
    cudaMalloc(&deviceSamples, sizeof(double) * length);
    cudaMalloc(&devicePressure, sizeof(double) * length);

    cudaMemcpy(deviceSamples, hostSamples, sizeof(double) * length, cudaMemcpyHostToDevice); // Copy samples to device

    // CUDA launch configuration
    int threadsPerBlock = 256;
    int blocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    convertToPressureKernel <<<blocks, threadsPerBlock >>> (deviceSamples, devicePressure, length, numBits, peakVolts, refSens);

    cudaMemcpy(hostPressure, devicePressure, sizeof(double) * length, cudaMemcpyDeviceToHost); // Copy pressure to host
    
    // Cleanup
    cudaFree(deviceSamples);
    cudaFree(devicePressure);
}

// Main feature extraction
AudioFeatures featureExtraction(int numBits, int peakVolts, const fs::path& filePath,
            double refSens, int timewin, double avtime, int fftWin, int calTone, int flow,
            int fhigh, int downsampleFactor, bool omitPartialMinute) {

    string fixedFilePath = fixFilePath(filePath.string()); // Make file path Windows compatible
    AudioInfo info = audioReadInfo(fixedFilePath); // Read audio metadata

    if (omitPartialMinute) { info.duration = floor(info.duration / 60.0) * 60.0; } // Only include full minutes of recordings

    int totalSamples = static_cast<int>(info.sampleRate * info.duration); // # of samples
    AudioData audio = audioRead(filePath.string(), SampleRange{ 1, totalSamples }); // Load audio data

    int sampFreq = audio.sampleRate; // Sampling frequency
    int audioSamplesLen = audio.numFrames; // # of audio frames

    double* pressure = new double[audioSamplesLen]; // Buffer for pressure waveform

    double* flatSamples = audio.samples[0]; // First channel
    gpuConvertToPressure(flatSamples, pressure, audioSamplesLen, numBits, peakVolts, refSens); // Convert to pressure (Pascals)
    freeAudioData(audio); // Deallocate original audio data

    if (downsampleFactor != -1) { // Downsample
        int newLen = 0;
        double* downsampled = downSample(pressure, audioSamplesLen, downsampleFactor, newLen);
        delete[] pressure;
        pressure = downsampled;
        audioSamplesLen = newLen;
        sampFreq /= downsampleFactor;
    }

    if (calTone == 1 && audioSamplesLen > 6 * sampFreq) { // Remove first six seconds if calibration tone is present
        int newLen = audioSamplesLen - 6 * sampFreq;
        double* shifted = new double[newLen];
        memcpy(shifted, pressure + 6 * sampFreq, sizeof(double) * newLen);
        delete[] pressure;
        pressure = shifted;
        audioSamplesLen = newLen;
    }

    BandpassFilter filt = bandpassFilter(pressure, audioSamplesLen, 1.0 / sampFreq, flow, fhigh); // Apply bandpass filter
    delete[] pressure;

    // Segment length (samples) / # of time windows
    int ptsPerTimeWin = timewin * sampFreq;
    int numTimeWin = filt.length / ptsPerTimeWin;
    int remainder = filt.length % ptsPerTimeWin;
    if (remainder > 0) { ++numTimeWin; } // Padded segment for leftover samples

    // Pad filtered signal to match time window
    int paddedLen = numTimeWin * ptsPerTimeWin;
    double* paddedSignal = new double[paddedLen]();
    // Include padding for consistent row / column lengths
    memcpy(paddedSignal, filt.filteredTimeSeries, sizeof(double) * filt.length);

    // Populate audio features struct
    AudioFeatures features = {};
    features.segmentDurationLen = numTimeWin;
    features.segmentDuration = new int[numTimeWin];

    // Signal to time windows
    double** timechunkMatrix = new double* [numTimeWin];
    for (int i = 0; i < numTimeWin; ++i) {
        timechunkMatrix[i] = &paddedSignal[i * ptsPerTimeWin];
        // Duration per segment
        if (i == numTimeWin - 1 && remainder > 0)
        { features.segmentDuration[i] = static_cast<int>(round(static_cast<double>(remainder) / sampFreq));}
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
    SoloPerGM2 gm2 = fSoloPerGM2(paddedSignal, paddedLen, sampFreq, timewin, avtime);

    features.peakCountLen = numTimeWin;
    features.peakCount = new int[numTimeWin];
    for (int i = 0; i < numTimeWin; ++i) { features.peakCount[i] = gm2.peakCount[i]; }
    delete[] gm2.peakCount;

    features.autocorrRows = gm2.autocorrRows;
    features.autocorrCols = gm2.autocorrCols;
    features.autocorr = new double* [gm2.autocorrRows];
    for (int i = 0; i < gm2.autocorrRows; ++i) { features.autocorr[i] = gm2.autocorr[i]; }
    delete[] gm2.autocorr;

    // Calculate dissim
    int dissimLen = 0;
    features.dissim = fSoloDissimGM1GPU(timechunkMatrix, ptsPerTimeWin, numTimeWin, fftWin, sampFreq, dissimLen);
    features.dissimLen = dissimLen;

    // Deallocate temporary arrays
    delete[] timechunkMatrix;
    delete[] paddedSignal;

    return features;
}

// Copy input file name to output file record
tm extractBaseTime(const string& filename) {
    tm baseTime = {}; // Fields initialized to zero
    smatch match; // Will store matched part
    regex pattern(R"((\d{8})_(\d{6}))"); // Matches YYYYMMDD_HHMMSS

    // Find date / time from file name
    if (regex_search(filename, match, pattern) && match.size() == 3) { // Date / time / full match
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
    if (!outputFile.is_open()) { // Error opening file
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

    // Allocate array for valid autocorr columns - Avoids printing empty columns
    bool* validAutocorrCols = new bool[maxAutocorrCols];
    for (int i = 0; i < maxAutocorrCols; ++i) { validAutocorrCols[i] = false; }

    // Remove extra autocorr columns
    for (int i = 0; i < numFiles; ++i) {
        const AudioFeatures& feature = allFeatures[i];
        if (feature.autocorr != nullptr) {
            for (int row = 0; row < feature.autocorrRows; ++row) {
                for (int col = 0; col < feature.autocorrCols; ++col) {
                    if (!isnan(feature.autocorr[row][col])) { validAutocorrCols[col] = true; }
                }
            }
        }
    }

    // CSV Header
    outputFile << "Filename,Year,Month,Day,Hour,Minute,SegmentDuration,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount";
    for (int i = 0; i < maxAutocorrCols; ++i) {
        if (validAutocorrCols[i]) { outputFile << ",Autocorr_" << i; }
    }
    outputFile << "\n";

    // Write one row per time segment for each file
    for (int fileIdx = 0; fileIdx < numFiles; ++fileIdx) {
        const AudioFeatures& features = allFeatures[fileIdx];

        // Find max length of features (Most features are length n, dissim is n - 1)
        int maxLength = features.SPLrmsLen;

        // Convert timestamp to time_t
        tm baseTime = extractBaseTime(filenames[fileIdx]);
        time_t baseEpoch = mktime(&baseTime);
        tm* firstTime = localtime(&baseEpoch);

        // Use NaN for empty indices - Only applies to dissim
        bool useNanTimestamp = false;
        if (!firstTime || (firstTime->tm_year + 1900) < 1900) { useNanTimestamp = true; }

        // Iterate through segments
        for (int i = 0; i < maxLength; ++i) {
            // Calculate timestamp per minute
            time_t currentEpoch = baseEpoch + i * 60;
            tm* currentTime = localtime(&currentEpoch);

            outputFile << filenames[fileIdx] << ",";

            // Write timestamp or NaN
            if (useNanTimestamp || !currentTime) {  outputFile << "NaN,NaN,NaN,NaN,NaN,"; }
            else {
                outputFile << (currentTime->tm_year + 1900) << ","
                           << (currentTime->tm_mon + 1) << ","
                           << currentTime->tm_mday << ","
                           << currentTime->tm_hour << ","
                           << currentTime->tm_min << ",";
            }

            // Segment duration
            if (i < features.segmentDurationLen) { outputFile << features.segmentDuration[i]; }
            else { outputFile << "NaN"; }
            outputFile << ",";

            // SPLrms
            if (i < features.SPLrmsLen) outputFile << features.SPLrms[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            // SPLpk
            if (i < features.SPLpkLen) outputFile << features.SPLpk[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            // Impulsivity
            if (i < features.impulsivityLen) outputFile << features.impulsivity[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            // Dissim
            if (i < features.dissimLen) outputFile << features.dissim[i];
            else { outputFile << "NaN"; }
            outputFile << ",";

            // Peakcount
            if (i < features.peakCountLen) outputFile << features.peakCount[i];
            else { outputFile << "NaN"; }
            
            // Autocorr
            for (int j = 0; j < maxAutocorrCols; ++j) {
                if (validAutocorrCols[j]) {
                    outputFile << ",";
                    if (features.autocorr && i < features.autocorrRows && j < features.autocorrCols) { outputFile << features.autocorr[i][j]; }
                    else { outputFile << "NaN"; }
                }
            }

            outputFile << "\n"; // End of row
        }
    }

    // Clean up
    delete[] validAutocorrCols;
    outputFile.close();
}

// Sort input files
int compareStrings(const void* ptrA, const void* ptrB) { // Sort strings alphabetically
    // Dereference to get real string names
    const char* str1 = *(const char**)ptrA;
    const char* str2 = *(const char**)ptrB;
    return strcmp(str1, str2); // Return string in alphabetical order
}

void quickSortStrings(char arr[][512], int numFiles) {
    // Create an array of pointers for use with qsort
    char* ptrs[512];
    for (int i = 0; i < numFiles; ++i) {
        ptrs[i] = arr[i]; // Pointers to strings
    }

    // Sort pointers
    qsort(ptrs, numFiles, sizeof(char*), compareStrings);

    // Rearrange original array based on sorted pointers
    char temp[512]; // Buffer for swapping string contents
    for (int i = 0; i < numFiles; ++i) {
        if (ptrs[i] != arr[i]) { // Pointer not in correct location
            strcpy(temp, ptrs[i]); // Move sorted string to temp buffer
            strcpy(ptrs[i], arr[i]); // Move unsorted string to new location
            strcpy(arr[i], temp); // Move string from temp to current index

            // Update array for future swaps
            for (int j = i + 1; j < numFiles; ++j) {
                if (ptrs[j] == arr[i]) {
                    ptrs[j] = ptrs[i]; // Fix pointer inconsistency
                    break;
                }
            }
        }
    }
}

void threadWrapper(ThreadArgs& args) {
    try {
        while (true) { // Find next index
            int index = args.nextIndex->fetch_add(1);
            if (index >= args.totalFiles) { break; } // End of input files

            // Display current file being processed
            cout << "Processing file index " << index << ": " << args.filePaths[index] << "\n";
            cerr.flush();

            fs::path filePath(args.filePaths[index]);

            // Perform feature extraction for current input file
            AudioFeatures features = featureExtraction(args.numBits, args.peakVolts,
                    filePath, args.RS, args.timeWin, args.avTime, args.fftWin,
                    args.arti, args.fLow, args.fHigh, args.downSample, args.omitPartialMinute);

            args.allFeatures[index] = features;

            // Display base filename in output results
            string filename_str = filePath.filename().string();
            strncpy(args.filenames[index], filename_str.c_str(), 511);
            args.filenames[index][511] = '\0'; // Null terminate for safe handling
        }
    }

    // Display any errors
    catch (const exception& e) { cerr << "Exception in thread: " << e.what() << "\n"; }
    catch (...) { cerr << "Unknown exception in thread\n"; }
}

// Process directory of sound files with user-given parameters
int main(int argc, char* argv[]) {
    using namespace std; // Standard namespace
    using namespace chrono; // Time tracking

    auto start = high_resolution_clock::now(); // Starting time to show runtime performance

    // Default arguments if unspecified
    char inputDir[512] = {}, outputFile[512] = {};
    int numBits = 16, peakVolts = 2, arti = 1, timeWin = 60, fftWin = 1,
            fLow = 1, fHigh = 192000, maxThreads = 4, downSample = -1;
    double RS = -178.3, avTime = 0.1;
    bool omitPartialMinute = false;

    // Command line index parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--omit_partial_minute") == 0) { omitPartialMinute = true; }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) { strncpy(inputDir, argv[++i], sizeof(inputDir) - 1); }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { strncpy(outputFile, argv[++i], sizeof(outputFile) - 1); }
        else if (strcmp(argv[i], "--num_bits") == 0) { numBits = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--RS") == 0) { RS = atof(argv[++i]); }
        else if (strcmp(argv[i], "--peak_volts") == 0) { peakVolts = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--arti") == 0) { arti = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--timewin") == 0) { timeWin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fft_win") == 0) { fftWin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--avtime") == 0) { avTime = atof(argv[++i]); }
        else if (strcmp(argv[i], "--flow") == 0) { fLow = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fhigh") == 0) { fHigh = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max_threads") == 0) { maxThreads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--downsample") == 0) { downSample = atoi(argv[++i]); }
    }

    // Count .wav files
    int totalFiles = 0;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".wav") { ++totalFiles; }
    }

    if (totalFiles == 0) { // Empty directory
        cerr << "No valid .wav files found in " << inputDir << "\n";
        return 1;
    }

    // Allocate arrays with dynamic size now that totalFiles is known
    char (*filePaths)[512] = new char[totalFiles][512]; // Path to files
    char (*filenames)[512] = new char[totalFiles][512]; // Base filenames
    AudioFeatures* allFeatures = new AudioFeatures[totalFiles];

    // Fill filePaths with file names
    int index = 0;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.path().extension() == ".wav") {
            strncpy(filePaths[index], entry.path().string().c_str(), 511);
            filePaths[index][511] = '\0';
            ++index;
        }
    }

    quickSortStrings(filePaths, totalFiles); // Sort file paths alphabetically for in-order data processing

    // Thread setup
    atomic<int> nextIndex(0);
    int availableThreads = max(1, thread::hardware_concurrency());
    int numThreads = min(maxThreads, availableThreads);

    // Limit thread count depending on high frequency cutoff
    if (fHigh <= 16000 && numThreads > 2) { numThreads = 2; }
    else if (fHigh <= 48000 && numThreads > 4) { numThreads = 4; }

    // Thread arguments
    ThreadArgs args;
    args.nextIndex = &nextIndex;
    args.totalFiles = totalFiles;
    args.filePaths = filePaths;
    args.filenames = filenames;
    args.allFeatures = allFeatures;
    args.numBits = numBits;
    args.peakVolts = peakVolts;
    args.RS = RS;
    args.timeWin = timeWin;
    args.avTime = avTime;
    args.fftWin = fftWin;
    args.arti = arti;
    args.fLow = fLow;
    args.fHigh = fHigh;
    args.downSample = downSample;
    args.omitPartialMinute = omitPartialMinute;

    // Launch threads
    thread* threads = new thread[numThreads];
    for (int i = 0; i < numThreads; ++i) { threads[i] = thread(threadWrapper, ref(args)); }
    for (int i = 0; i < numThreads; ++i) { threads[i].join(); }
    delete[] threads;

    const char** fileNames = new const char* [totalFiles];
    for (int i = 0; i < totalFiles; ++i) { fileNames[i] = filenames[i]; } // Convert to const char* array

    saveFeaturesToCSV(outputFile, fileNames, totalFiles, allFeatures);
    cout << "Saved features for " << totalFiles << " files to " << outputFile << "\n";

    for (int i = 0; i < totalFiles; ++i) { freeAudioFeatures(allFeatures[i]); } // Free featurees for each file

    // Cleanup
    delete[] filePaths;
    delete[] filenames;
    delete[] allFeatures;
    delete[] fileNames;
    fftw_cleanup(); // Clean up internal memory

    auto stop = high_resolution_clock::now(); // Ending time to show runtime performance
    duration<double> elapsed = duration_cast<duration<double>>(stop - start);
    cout << "Runtime: " << elapsed.count() << " seconds\n"; // Total execution time

    return 0;
}
