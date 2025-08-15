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
#include <memory>
#include <unordered_map>

// Limit thread count to # of cores
#include <queue>
#include <condition_variable>
#include <atomic>

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
    double* time_series_filt;
    double* amp_spectrum;
    int num_pts;

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
    double* data;
    int size;
    ArrayShiftFFT(double* d, int s) : data(d), size(s) {}
    ~ArrayShiftFFT() { delete[] data; }
};

// FFTW complex buffer + plan
struct FFTWHandler {
    fftw_complex* buf = nullptr; // Data buffer
    fftw_plan forwardPlan = nullptr; // Forward FFT plan
    fftw_plan inversePlan = nullptr; // Inverse FFT plan
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
    bool omitPartialMinute;
};

static mutex fftw_plan_mutex; // Thread safe global mutex for FFTW plan creation

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
            { interleavedSamples[i] = static_cast<double>(tempBuffer[i]);}
        delete[] tempBuffer;
        break;
    }
    // 24 or 32 bits
    case SF_FORMAT_PCM_24:
    case SF_FORMAT_PCM_32: {
        int* tempBuffer = new int[numFramesToRead * numChannels];
        sf_readf_int(file, tempBuffer, numFramesToRead);
        for (int i = 0; i < numFramesToRead * numChannels; ++i)
            { interleavedSamples[i] = static_cast<double>(tempBuffer[i]); }
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

// Reduce sampling rate by analyzing (1 / factor) samples
double* downsample(const double* x, int length, int factor, int& newLength) {
    if (factor <= 0) { throw invalid_argument("Factor must be positive"); } // Invalid scaling factor
    
    newLength = (length + factor - 1) / factor; // # of output samples
    double* result = new double[newLength]; // Downsampled output

    int idx = 0;
    for (int i = 0; i < length; i += factor) { result[idx++] = x[i]; } // Copy (1 / factor) samples
    
    return result;
}

// Manually shift zero-frequency to center of array
ArrayShiftFFT fftshift(double* input, int length) {
    double* shifted = new double[length]; // Shifted array
    // Shift indices from center / center zero frequency sample
    for (int i = 0; i < length; ++i) { shifted[i] = input[(i + (length / 2)) % length]; }
    return {shifted, length};
}

// Apply bandpass filter in frequency domain
BandpassFilter bandpass_filter(const double* time_series, int num_pts, double frequency, double flow, double fhigh) {
    // Create local FFTW resources for thread safety
    fftw_complex* buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * num_pts); // Allocate buffer
    if (!buf) { throw bad_alloc(); } // Buffer allocation unsuccessful

    // Load input signal into buffer
    for (int i = 0; i < num_pts; ++i) {
        buf[i][0] = time_series[i]; // Real
        buf[i][1] = 0.0; // Imaginary
    }

    // Use global mutex for thread safety in FFTW creation
    fftw_plan forward_plan, inverse_plan;
    {
        lock_guard<mutex> lock(fftw_plan_mutex);
        // Forward plan to convert time to frequency
        forward_plan = fftw_plan_dft_1d(num_pts, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE);
        if (!forward_plan) { // Plan creation unsuccessful
            fftw_free(buf);
            throw runtime_error("FFTW forward plan creation failed");
        }
        // Inverse plan to convert frequency to time
        inverse_plan = fftw_plan_dft_1d(num_pts, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE);
        if (!inverse_plan) { // Plan creation unsuccessful
            fftw_destroy_plan(forward_plan);
            fftw_free(buf);
            throw runtime_error("FFTW inverse plan creation failed");
        }
    }

    fftw_execute(forward_plan); // Execute forward plan to convert time to frequency

    // Pre-compute frequency array
    double reclen = num_pts * frequency; // Recording length (seconds)
    double freq_step = 1.0 / reclen; // Hz per bin
    double freq_offset = -num_pts * 0.5 * freq_step; // Start frequency for centered axis
    
    // Frequency array before shift
    double* freq = new double[num_pts];
    for (int i = 0; i < num_pts; ++i) { freq[i] = freq_offset + i * freq_step; }
    ArrayShiftFFT shifted_freq = fftshift(freq, num_pts); // FFT shift
    delete[] freq; // Delete unshifted array

    // Set high-frequency cutoff
    if (fhigh == 0.0) { fhigh = 0.5 / frequency; }
    
    // Apply bandpass filter directly to buffer
    for (int i = 0; i < num_pts; ++i) {
        double abs_freq = fabs(shifted_freq.data[i]);
        if (abs_freq < flow || abs_freq > fhigh) { // Zero out values outside flow to fhigh range
            buf[i][0] = 0.0;
            buf[i][1] = 0.0;
        }
    }

    fftw_execute(inverse_plan); // Execute inverse plan

    // Normalize / fill outputs
    double* time_series_filt = new double[num_pts]; // Real valued filtered time tomain signal
    double* amp_spectrum = new double[num_pts]; // Magnitude of each FFT bin after filtering
    for (int i = 0; i < num_pts; ++i) {
        time_series_filt[i] = buf[i][0] * (1.0 / num_pts); // Multiply by normalization factor

        // Calculate amplitude spectrum magnitude
        double real = buf[i][0];
        double imag = buf[i][1];
        amp_spectrum[i] = sqrt(real * real + imag * imag);
    }

    // Clean up local resources
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(buf);

    return BandpassFilter(time_series_filt, amp_spectrum, num_pts);
}

// Calculate kurtosis used for impulsivity of a signal
double calculate_kurtosis(const double* data, int length) {
    if (length <= 0 || data == nullptr) { throw invalid_argument("Input array is empty or null"); }

    double sum = 0.0, sumSquared = 0.0;
    
    // Calculate sums
    for (int i = 0; i < length; i++) {
        double val = data[i];
        sum += val;
        sumSquared += val * val;
    }
    
    double mean = sum / length; // Calculate mean
    
    // Calculate fourth moment
    double sumFourth = 0.0;
    for (int i = 0; i < length; i++) {
        double centeredSquare = (data[i] - mean) * (data[i] - mean);
        sumFourth += centeredSquare * centeredSquare;
    }
    
    double variance = (sumSquared / length) - (mean * mean); // Calculate variance
    if (variance < 1e-12) { return 0.0; } // Avoid division by zero
    
    return (sumFourth / length) / (variance * variance); // Fourth moment / squared variance
}

// Calculate autocorrelation between two signals
Correlation correl_5(const double* time_series1, const double* time_series2, int series_length, int lags, int offset) {
    // Declare reusable variables outside loops to avoid reallocation
    double sumX, sumY, sumXSquare, sumYSquare, sumXYProd;
    int sampleCount;
    double x, y;
    double sum_x_sample_count, sum_y_sample_count, sum_x_square_sample_count, sum_y_square_sample_count;
    double covar1, denom1, denom2;
    int len = lags + 1;

    double* corr_vals = new double[len]; // Correlation values
    double* lag_vals = new double[len]; // Corresponding lag values

    // Calculate correlation for each lag
    for (int i = 0; i <= lags; i++) {
        // Initialize accumulators for each lag iteration
        sumX = 0.0;
        sumY = 0.0;
        sumXSquare = 0.0;
        sumYSquare = 0.0;
        sumXYProd = 0.0;
        sampleCount = 0;

        // Loop through samples for current lag
        for (int k = 0; k < series_length - (i + offset); k++) {
            x = time_series1[k];
            y = time_series2[k + (i + offset)];

            // Ignore nan values
            if (!isnan(x) && !isnan(y)) {
                sumX += x;
                sumY += y;
                sumXSquare += x * x;
                sumYSquare += y * y;
                sumXYProd += x * y;
                sampleCount += 1;
            }
        }

        // Avoid division by zero
        if (sampleCount == 0) {
            corr_vals[i] = NAN;
            lag_vals[i] = static_cast<double>(i);
            continue;
        }

        // Claculate means
        sum_x_sample_count = sumX / sampleCount;
        sum_y_sample_count = sumY / sampleCount;
        sum_x_square_sample_count = sumXSquare / sampleCount;
        sum_y_square_sample_count = sumYSquare / sampleCount;

        // Claculate covariance / standard deviations
        covar1 = (sumXYProd / sampleCount) - (sum_x_sample_count * sum_y_sample_count);
        denom1 = sqrt(sum_x_square_sample_count - (sum_x_sample_count * sum_x_sample_count));
        denom2 = sqrt(sum_y_square_sample_count - (sum_y_sample_count * sum_y_sample_count));

        // Prevent division by zero in correlation calculation
        if (denom1 == 0.0 || denom2 == 0.0) { corr_vals[i] = NAN; }
        else { corr_vals[i] = covar1 / (denom1 * denom2); }
        lag_vals[i] = static_cast<double>(i);
    }

    return Correlation(corr_vals, lag_vals, len);
}

// Calculate autocorrelation / peak counts
SoloPer calculatePeriodicity(double* p_filt_input, int input_length, double fs, double timewin, double avtime) {
    int samp_window_size = static_cast<int>(fs * timewin); // Samples per window
    int num_time_wins = input_length / samp_window_size; // # of full windows
    if (num_time_wins == 0) { throw runtime_error("Empty time window"); } // No full window exists

    // Single declaration of variables outside loop
    int peak_count;
    double left_min, right_min, prominence;
    int j, i, jj, zz; // Iterators
    double* input_start;
    double val;
    
    // Squared input signal segmented into windows
    double** p_filt_reshaped = new double* [num_time_wins];
    for (i = 0; i < num_time_wins; i++) { p_filt_reshaped[i] = new double[samp_window_size]; }

    // Square to emphasize peaks / segment
    for (j = 0; j < num_time_wins; j++) {
        input_start = p_filt_input + j * samp_window_size;
        for (i = 0; i < samp_window_size; i++) {
            val = input_start[i];
            p_filt_reshaped[j][i] = val * val;
        }
    }

    int avg_win_size = static_cast<int>(fs * avtime); // Samples per averaging block
    int numavwin = samp_window_size / avg_win_size; // # of averages per window

    // Pressure averages per window
    double** pressure_avg = new double* [num_time_wins];
    for (i = 0; i < num_time_wins; i++) { pressure_avg[i] = new double[numavwin]; }

    double* row;
    double* segment_start;
    double avg;

    // Calculate block averages per time window
    for (jj = 0; jj < num_time_wins; ++jj) {
        row = p_filt_reshaped[jj];
        for (i = 0; i < numavwin; ++i) {
            segment_start = row + i * avg_win_size;
            avg = 0.0;
            for (j = 0; j < avg_win_size; ++j) { avg += segment_start[j]; }
            pressure_avg[jj][i] = avg / avg_win_size; // Block mean
        }
    }

    // Free squared signal buffer
    for (i = 0; i < num_time_wins; ++i) { delete[] p_filt_reshaped[i]; }
    delete[] p_filt_reshaped;

    // Autocorrelation calculations
    int p_avtot_rows = numavwin; // Length of averaged signal per window
    int lag_limit = static_cast<int>(p_avtot_rows * 0.7); // Maximum lag
    int p_avtot_cols = num_time_wins; // One autocorrelation row per window

    // Autocorrelation result
    double** acorr = new double* [p_avtot_cols];
    for (i = 0; i < p_avtot_cols; i++) { acorr[i] = new double[lag_limit + 1]; }

    int* pkcount = new int[p_avtot_cols]; // Peak count result

    // Compute autocorrelation using Pearson's correlation / peak count per window
    for (zz = 0; zz < p_avtot_cols; zz++) {
        Correlation corr_result = correl_5(pressure_avg[zz], pressure_avg[zz], p_avtot_rows, lag_limit, 0);

        // Copy correlationValues
        for (i = 0; i <= lag_limit; i++) { acorr[zz][i] = corr_result.correlationValues[i]; }

        // Peak counting with prominence
        peak_count = 0;
        for (i = 1; i < lag_limit; i++) {
            if (acorr[zz][i] > acorr[zz][i - 1] && acorr[zz][i] > acorr[zz][i + 1]) {
                left_min = acorr[zz][i]; // Find local minimum to left of peak
                for (j = i - 1; j >= 0; j--) {
                    if (acorr[zz][j] >= acorr[zz][i]) { break; } // Stop if rising again
                    if (acorr[zz][j] < left_min) { left_min = acorr[zz][j]; }
                }
                right_min = acorr[zz][i]; // Find local minimum to right of peak
                for (j = i + 1; j <= lag_limit; j++) {
                    if (acorr[zz][j] >= acorr[zz][i]) { break; } // Stop if rising again
                    if (acorr[zz][j] < right_min) right_min = acorr[zz][j];
                }
                prominence = acorr[zz][i] - max(left_min, right_min); // Height above higher of two minimums
                if (prominence > 0.5) { peak_count++; } // Only include peaks with prominence > 0.5
            }
        }
        pkcount[zz] = peak_count;
    }

    // Free pressure_avg after use
    for (int i = 0; i < num_time_wins; ++i) { delete[] pressure_avg[i]; }
    delete[] pressure_avg;

    // Save results
    SoloPer result;
    result.peakCount = pkcount;
    result.autocorr = acorr;
    result.peakcountLength = p_avtot_cols;
    result.autocorrRows = p_avtot_cols;
    result.autocorrCols = lag_limit + 1;
    return result;
}

// Hilbert transform using cached handlers
fftw_complex* hilbert_raw(const double* input, int input_len) {
    if (input_len <= 0 || input == nullptr) { return nullptr; } // Validate input

    // Create local FFTW resources instead of using shared cache for complex operations
    fftw_complex* buf = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * input_len); // Allocate buffer
    if (!buf) { throw bad_alloc(); } // Buffer allocation unsuccessful

    // Load real input signal into FFTW buffer
    for (int i = 0; i < input_len; ++i) {
        buf[i][0] = input[i]; // Real input
        buf[i][1] = 0.0; // Imaginary input
    }

    // Create thread safe plans with locking
    fftw_plan forward_plan, inverse_plan;
    {
        lock_guard<mutex> lock(fftw_plan_mutex); // Serialize plan creation
        forward_plan = fftw_plan_dft_1d(input_len, buf, buf, FFTW_FORWARD, FFTW_ESTIMATE); // Create forward plan
        if (!forward_plan) { // Error creating forward plan
            fftw_free(buf);
            throw runtime_error("FFTW forward plan creation failed");
        }
        inverse_plan = fftw_plan_dft_1d(input_len, buf, buf, FFTW_BACKWARD, FFTW_ESTIMATE); // Create inverse plan
        if (!inverse_plan) { // Error creating inverse plan
            fftw_destroy_plan(forward_plan);
            fftw_free(buf);
            throw runtime_error("FFTW inverse plan creation failed");
        }
    }

    fftw_execute(forward_plan); // Execute forward plan to convert time to frequency

    // Apply Hilbert filter in frequency domain
    const int half = input_len >> 1;
    int upper_limit;
    if (input_len & 1) { upper_limit = half; }
    else { upper_limit = half - 1; }

    // Double positive frequencies for analytical signal
    for (int i = 1; i <= upper_limit; ++i) {
        buf[i][0] *= 2.0;
        buf[i][1] *= 2.0;
    }

    // Zero out negative frequencies
    if (half + 1 < input_len) { memset(buf + half + 1, 0, sizeof(fftw_complex) * (input_len - half - 1)); }

    fftw_execute(inverse_plan); // Execute inverse plan to convert frequency to time

    fftw_complex* result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * input_len); // Allocate output buffer
    if (!result) { // Error creating output buffer
        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(inverse_plan);
        fftw_free(buf);
        throw bad_alloc();
    }

    // Normalize inverse FFT / copy
    const double norm_factor = 1.0 / input_len;
    for (int i = 0; i < input_len; ++i) {
        result[i][0] = buf[i][0] * norm_factor;
        result[i][1] = buf[i][1] * norm_factor;
    }

    // Deallocate resources
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(inverse_plan);
    fftw_free(buf);

    return result; // caller must fftw_free(result)
}

// Apply Hilbert transform filter to FFT data in place
void hilbertFilter(fftw_complex* data, int len) {
    int half = len / 2; // Center of FFT - Nyquist frequency
    int upper;

    // Find last positive frequency index
    if (len % 2 == 0) { upper = half - 1; } // Even length FFT
    else { upper = half; } // Odd length FFT

    // Double positive frequencies for analytical signal
    for (int i = 0; i < len; ++i) {
        if (i >= 1 && i <= upper) {
            data[i][0] *= 2.0; // Real part
            data[i][1] *= 2.0; // Imaginary part
        }
        // Zero out negative frequencies
        else if (i > half) {
            data[i][0] = 0.0; // Real part
            data[i][1] = 0.0; // Imaginary part
        }
    }
}

// Convert real input to complex array
void initializeComplex(const double* input, fftw_complex* output, int len) {
    for (int i = 0; i < len; ++i) {
        output[i][0] = input[i]; // Real part
        output[i][1] = 0.0; // Imaginary part
    }
}

// Normalize inverse FFT result
void normalizeResult(fftw_complex* data, int len) {
    for (int i = 0; i < len; ++i) {
        data[i][0] /= len; // Normalize real part
        data[i][1] /= len; // Normalize imaginary part
    }
}

// Compute Hilbert transform of real input (analytic signal)
fftw_complex* hilbertRaw(const double* input, int inputLen) {
    if (!input || inputLen <= 0) { // Validate input
        cerr << "Invalid input to hilbertRaw\n";
        return nullptr;
    }

    fftw_complex* data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * inputLen); // Create FFT buffer
    if (!data) { // Error creating buffer
        cerr << "Failed to allocate fftw_complex array\n";
        return nullptr;
    }

    initializeComplex(input, data, inputLen); // Write real input to complex buffer

    fftw_plan forwardPlan = fftw_plan_dft_1d(inputLen, data, data, FFTW_FORWARD, FFTW_ESTIMATE); // Convert time to freq
    fftw_execute(forwardPlan); // Execute forward plan

    hilbertFilter(data, inputLen); // In place hilber filter

    fftw_plan inversePlan = fftw_plan_dft_1d(inputLen, data, data, FFTW_BACKWARD, FFTW_ESTIMATE); // Convert freq to time
    fftw_execute(inversePlan);

    normalizeResult(data, inputLen); // Normalize inverse FFT result

    // Deallocate resources
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(inversePlan);

    return data;
}

// Find time frequency dissimilarity between a time window / its previous window
double* calculateDissim(double** timechunkMatrix, int ptsPerTimewin, int numTimeWin,
            double fftWin, double fs, int& outLen) {
    int pts_per_fft = static_cast<int>(fftWin * fs); // # of samples per FFT
    if (pts_per_fft <= 0 || ptsPerTimewin <= 0 || numTimeWin <= 1) { return nullptr; } // Validate input
    
    int numfftwin = (ptsPerTimewin - pts_per_fft) / pts_per_fft + 1; // # of FFT windows per segment
    if (numfftwin <= 0) { return nullptr; }

    outLen = numTimeWin; // # of dissimilarity records for output

    // Use local FFTW resources instead of cached ones for thread safety
    double* envelope = new double[ptsPerTimewin]; // Normalized amplitude of previous window
    double* env2 = new double[ptsPerTimewin]; // Normalized amplitude of current window
    double* fftA = new double[pts_per_fft]; // FFT magnitude of previous window
    double* fftB = new double[pts_per_fft]; // FFT magnitude of current window

    double* diss = new double[outLen]; // Dissimilarity results

    // Create batch FFT resources with thread safe plan creation
    double* batch_in = (double*)fftw_malloc(sizeof(double) * pts_per_fft * numfftwin);
    fftw_complex* batch_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * pts_per_fft * numfftwin);

    fftw_plan batch_plan; // Thread safe FFTW batch plan for all FFT windows
    {
        lock_guard<mutex> lock(fftw_plan_mutex); // Ensure plan creation is serialized
        batch_plan = fftw_plan_many_dft_r2c(
            1, // 1D FFT
            &pts_per_fft, // array of fft sizes
            numfftwin, // # of FFT windows per batch
            batch_in, // input array
            NULL, // input strides (default)
            1, // input distance between batches
            pts_per_fft, // input stride (elements between input points)
            batch_out, // output array
            NULL, // output strides (default)
            1, // output distance between batches
            (pts_per_fft / 2) + 1, // output stride (complex output length)
            FFTW_ESTIMATE); // Plan flag
    }

    if (!batch_plan) { // Plan creation unsuccessful
        fftw_free(batch_in);
        fftw_free(batch_out);
        delete[] envelope;
        delete[] env2;
        delete[] fftA;
        delete[] fftB;
        delete[] diss;
        return nullptr;
    }

    // Calculate normalized magnitude spectrum with batched FFT
    auto calc_fft_mag_batched = [&](const double* input, double* output) {
        
        // Single declarations outisde loops
        int i, j, base, half_bins;
        double mag_sum, total;

        fill(output, output + pts_per_fft, 0.0);// Clear output

        // Write all FFT windows into batch input
        for (i = 0; i < numfftwin; ++i) {
            base = i * pts_per_fft;
            memcpy(batch_in + i * pts_per_fft, input + base, sizeof(double) * pts_per_fft);
        }

        fftw_execute(batch_plan); // Execute batch FFT

        half_bins = pts_per_fft / 2 + 1; // Positive frequencies

        // Sum magnitudes across all FFT windows
        for (j = 0; j < half_bins; ++j) {
            mag_sum = 0.0;
            for (i = 0; i < numfftwin; ++i) {
                fftw_complex& c = batch_out[i * half_bins + j];
                mag_sum += sqrt(c[0] * c[0] + c[1] * c[1]); // Calculate magnitude using hypotenuse
            }
            output[j] = mag_sum;
            if (j > 0 && j < pts_per_fft / 2) { output[pts_per_fft - j] = mag_sum; } // Mirror negative frequencies
        }

        // Normalize magnitude spectrum
        total = 0.0;
        for (j = 0; j < pts_per_fft; ++j) { total += output[j]; }
        if (total > 1e-12) {
            for (j = 0; j < pts_per_fft; ++j) { output[j] /= total; }
        }
        else { fill(output, output + pts_per_fft, 0.0); } // Avoid division by zero
        };

    diss[0] = NAN;  // Set first element to NaN

    // Start at i=1: No previous index for i=0
    int i, k;
    double sum1, sum2, timeDiss, env2_norm, freqDiss;

    // Iterate through consecutive time windows to calculate dissimilarity between them
    for (i = 1; i < outLen; ++i) {
        fftw_complex* hil1 = hilbert_raw(timechunkMatrix[i - 1], ptsPerTimewin);
        fftw_complex* hil2 = hilbert_raw(timechunkMatrix[i], ptsPerTimewin);

        if (!hil1 || !hil2) { // Error in allocation
            diss[i] = NAN;
            if (hil1) fftw_free(hil1);
            if (hil2) fftw_free(hil2);
            continue;
        }

        // Calculate amplitude envelopes
        sum1 = 0.0, sum2 = 0.0;
        for (k = 0; k < ptsPerTimewin; ++k) {
            envelope[k] = hypot(hil1[k][0], hil1[k][1]);
            env2[k] = hypot(hil2[k][0], hil2[k][1]);
            sum1 += envelope[k];
            sum2 += env2[k];
        }

        // Deallocate resources
        fftw_free(hil1);
        fftw_free(hil2);

        // Normalize envelopes
        if (sum1 > 1e-12) {
            for (k = 0; k < ptsPerTimewin; ++k) { envelope[k] /= sum1; }
        }
        else { fill(envelope, envelope + ptsPerTimewin, 0.0); }

        // Time domain dissimilarity
        timeDiss = 0.0;
        if (sum2 > 0.0) {
            for (k = 0; k < ptsPerTimewin; ++k) {
                env2_norm = env2[k] / sum2;
                timeDiss += fabs(envelope[k] - env2_norm);
            }
        }
        else {
            for (k = 0; k < ptsPerTimewin; ++k) { timeDiss += envelope[k]; }
        }
        timeDiss *= 0.5; // Normalize scaling

        // Frequency domain dissimilarity using batched FFT
        calc_fft_mag_batched(timechunkMatrix[i - 1], fftA);
        calc_fft_mag_batched(timechunkMatrix[i], fftB);

        freqDiss = 0.0;
        for (int j = 0; j < pts_per_fft; ++j) { freqDiss += fabs(fftA[j] - fftB[j]); }
        freqDiss *= 0.5;

        diss[i] = timeDiss * freqDiss; // Combined dissimilarity: time * frequency
    }

    // Clean up local resources
    fftw_destroy_plan(batch_plan);
    fftw_free(batch_in);
    fftw_free(batch_out);

    delete[] envelope;
    delete[] env2;
    delete[] fftA;
    delete[] fftB;

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
    fftw_complex* hil1 = hilbert_raw(lastSeg, segLen);
    fftw_complex* hil2 = hilbert_raw(firstSeg, segLen);
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
    for (int j = 0; j < half_bins; ++j) {
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
    for (int j = 0; j < pts_per_fft; ++j) {
        total1 += mag1[j];
        total2 += mag2[j];
    }
    if (total1 > 1e-12) {
        for (int j = 0; j < pts_per_fft; ++j) { mag1[j] /= total1; }
    }
    if (total2 > 1e-12) {
        for (int j = 0; j < pts_per_fft; ++j) { mag2[j] /= total2; }
    }

    // Frequency domain dissimilarity
    double freqDiss = 0.0;
    for (int j = 0; j < pts_per_fft; ++j) { freqDiss += fabs(mag1[j] - mag2[j]); }
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

// Main feature extraction
AudioFeatures featureExtraction(int numBits, int peakVolts, const fs::path& filePath,
    double refSens, int timewin, double avtime, int fftWin, int calToneLen, int flow,
    int fhigh, int downsampleFactor, bool omitPartialMinute) {

    string fixedFilePath = fixFilePath(filePath.string()); // Make file path Windows compatible
    AudioData audio = audioRead(filePath.string()); // Read all samples / metadata

    int total_samples = static_cast<int>(audio.sampleRate * audio.duration);
    int sampFreq = audio.sampleRate;
    int audioSamplesLen = audio.numFrames;

    // Allocate / convert audio samples to pressure
    double* pressure = new double[audioSamplesLen];

    // Optionally downsample
    if (downsampleFactor != -1) {
        int newLen = 0;
        double* downsampled = downsample(pressure, audioSamplesLen, downsampleFactor, newLen);
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

    // Convert raw samples to pressure (Pa)
    double scale_factor = peakVolts / static_cast<double>(1 << numBits);
    double ref_factor = 1.0 / pow(10.0, refSens / 20.0);
    double combined_factor = scale_factor * ref_factor;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < audioSamplesLen; ++i) {
        double s = static_cast<double>(audio.samples[0][i]);
        if (numBits == 24) { s = static_cast<double>(static_cast<int>(s) >> 8); } // Shift for 24 bits
        else if (numBits == 32) { s = static_cast<double>(static_cast<int>(s) >> 16); } // Shift for 32 bits
        pressure[i] = s * combined_factor; // Convert to pressure
    }

    freeAudioData(audio); // Deallocate audio data

    // Apply bandpass filter
    BandpassFilter filt = bandpass_filter(pressure, audioSamplesLen, 1.0 / sampFreq, flow, fhigh);
    delete[] pressure;

    // Segment signal into time windows
    int pts_per_timewin = timewin * sampFreq;
    int num_timewin = filt.num_pts / pts_per_timewin;
    int remainder = filt.num_pts % pts_per_timewin;
    if (remainder > 0) { ++num_timewin; }

    int padded_len = num_timewin * pts_per_timewin;
    double* padded_signal = new double[padded_len](); // Padding for consistent row / column lengths
    memcpy(padded_signal, filt.time_series_filt, sizeof(double) * filt.num_pts);
    delete[] filt.time_series_filt;
    filt.time_series_filt = nullptr;

    // Initialize audio features struct
    AudioFeatures features = {};
    features.segmentDurationLen = num_timewin;
    features.segmentDuration = new int[num_timewin];

    double** timechunk_matrix = new double* [num_timewin]; // Time window pointers

    // Parallel initialization of time chunks / segment durations
#pragma omp parallel for
    for (int i = 0; i < num_timewin; ++i) {
        timechunk_matrix[i] = &padded_signal[i * pts_per_timewin];
        if (i == num_timewin - 1 && remainder > 0)
            { features.segmentDuration[i] = static_cast<int>(round(static_cast<double>(remainder) / sampFreq)); }
        else { features.segmentDuration[i] = timewin; }
    }

    // Allocate feature arrays
    features.SPLrmsLen = features.SPLpkLen = features.impulsivityLen = num_timewin;
    features.SPLrms = new double[num_timewin];
    features.SPLpk = new double[num_timewin];
    features.impulsivity = new double[num_timewin];

    // Calculate SPLpeak, SPLrms, impulsivity for each window
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_timewin; ++i) {
        const double* chunk = timechunk_matrix[i];
        double sumsq = 0.0, peak = 0.0;

        // Use current window length for last segment if partial
        int current_pts;
        if (i == num_timewin - 1 && remainder > 0) { current_pts = remainder; }
        else { current_pts = pts_per_timewin; }

        // Vectorizable inner loop
#pragma omp simd reduction(+:sumsq) reduction(max:peak)
        for (int j = 0; j < current_pts; ++j) {
            double v = chunk[j];
            double abs_v = fabs(v);
            sumsq += v * v;
            if (abs_v > peak) { peak = abs_v; }
        }

        double rms = sqrt(sumsq / current_pts);
        features.SPLrms[i] = 20.0 * log10(fmax(rms, 1e-12));
        features.SPLpk[i] = 20.0 * log10(fmax(peak, 1e-12));
        features.impulsivity[i] = calculate_kurtosis(chunk, current_pts);
    }

    // Autocorrelation / Peak count
    SoloPer per = calculatePeriodicity(padded_signal, padded_len, sampFreq, timewin, avtime);
    features.peakCountLen = num_timewin;
    features.peakCount = new int[num_timewin];

    // Parallel copy of peak counts
#pragma omp parallel for
    for (int i = 0; i < num_timewin; ++i) { features.peakCount[i] = per.peakCount[i]; }
    delete[] per.peakCount;

    features.autocorrRows = per.autocorrRows;
    features.autocorrCols = per.autocorrCols;
    features.autocorr = new double* [per.autocorrRows];

    // Parallel pointer assignment
#pragma omp parallel for
    for (int i = 0; i < per.autocorrRows; ++i) { features.autocorr[i] = per.autocorr[i]; } // Reuse raw pointer
    delete[] per.autocorr; // Free array of pointers

    // Dissimilarity calculation
    features.dissimLen = num_timewin;
    features.dissim = new double[num_timewin];

    // Initialize all dissimilarity values to NaN first
    for (int i = 0; i < num_timewin; ++i) { features.dissim[i] = NAN; }

    // Calculate dissimilarity only if we have more than one time window
    if (num_timewin > 1) {
        int dissim_len = 0;
        double* calculated_dissim = calculateDissim(timechunk_matrix, pts_per_timewin, num_timewin, fftWin, sampFreq, dissim_len);

        if (calculated_dissim && dissim_len > 0) {
            // Copy calculated dissimilarity values
            for (int i = 0; i < min(dissim_len, num_timewin); ++i) {
                features.dissim[i] = calculated_dissim[i];
            }
            delete[] calculated_dissim;
        }
    }

    features.sampleRate = sampFreq;

    // Store first / last segments for cross-file dissimilarity
    if (num_timewin > 0) {
        int actualLength = audioSamplesLen; // Use full audio length if under 60 seconds

        // First segment: use min of 60 seconds or actual length
        int firstSegLength = min(actualLength, sampFreq * 60);
        features.first60s = new double[firstSegLength];
        memcpy(features.first60s, padded_signal, sizeof(double) * firstSegLength);
        features.first60sLen = firstSegLength;

        // Last segment: use min of 60 seconds or actual length
        int lastSegLength = min(actualLength, sampFreq * 60);
        features.last60s = new double[lastSegLength];
        int startOffset = max(0, actualLength - lastSegLength);
        memcpy(features.last60s, padded_signal + startOffset, sizeof(double) * lastSegLength);
        features.last60sLen = lastSegLength;
    }

    delete[] timechunk_matrix;
    delete[] padded_signal;

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

// Worker thread function
void threadWork(atomic<int>& nextIndex, int totalFiles, char filePaths[][128], AudioFeatures* allFeatures,
                char filenames[][128], int num_bits, int peak_volts, double RS, int timewin, double avtime,
                int fft_win, double artiLen, int flow, int fhigh, int downsample, bool omit_partial_minute) {
    while (true) {
        int index = nextIndex++; // Get next thread index
        if (index >= totalFiles) { break; } // No more files to be processed

        fs::path filepath(filePaths[index]); // Convert file path string to filesystem::path object
        string fname = filepath.filename().string(); // Extract filename as string
        strcpy(filenames[index], fname.c_str()); // Write filename to filenames array
        filenames[index][511] = '\0'; // Ensure null-termination

        // Extract features
        allFeatures[index] = featureExtraction(num_bits, peak_volts, filepath, RS,
            timewin, avtime, fft_win, artiLen, flow, fhigh, downsample, omit_partial_minute);
    }
}

// Sort file names for output
void bubbleSort(char arr[][128], int n) {
    int i, j;
    char temp[128]; // Temporary storage for swapping
    for (i = 0; i < n-1; ++i) { // # of passes through array
        for (j = 0; j < n-i-1; ++j) { // Compare adjacent elements / swap if needed
            // Swap if current string is lexicographically greater than previous one
            if (strcmp(arr[j], arr[j+1]) > 0) {
                strcpy(temp, arr[j]);
                strcpy(arr[j], arr[j+1]);
                strcpy(arr[j+1], temp);
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
            { cerr << "Exception in thread while processing index " << index  << ": " << e.what() << "\n"; }
        catch (...) { cerr << "Unknown exception in thread while processing index " << index << "\n"; }
    }
}

const int MAX_FILES = 1000;

// Process directory of sound files with user-given parameters
int main(int argc, char* argv[]) {
    // Default values if unspecified by user
    char input_dir[128] = {};
    char output_file[128] = {};
    int num_bits = 16, peak_volts = 2;
    int timewin = 60, fft_win = 1, flow = 1, fhigh = 16000;
    double RS = -178.3, avtime = 0.1, artiLen = 0.0;
    int max_threads = 1, downsample = -1;
    bool omit_partial_minute = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--omit_partial_minute") == 0) { omit_partial_minute = true; }
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) { strcpy(input_dir, argv[++i]); }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) { strcpy(output_file, argv[++i]); }
        else if (strcmp(argv[i], "--num_bits") == 0 && i + 1 < argc) { num_bits = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--RS") == 0 && i + 1 < argc) { RS = atof(argv[++i]); }
        else if (strcmp(argv[i], "--peak_volts") == 0 && i + 1 < argc) { peak_volts = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--arti_len") == 0 && i + 1 < argc) { artiLen = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--timewin") == 0 && i + 1 < argc) { timewin = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fft_win") == 0 && i + 1 < argc) { fft_win = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--avtime") == 0 && i + 1 < argc) { avtime = atof(argv[++i]); }
        else if (strcmp(argv[i], "--flow") == 0 && i + 1 < argc) { flow = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--fhigh") == 0 && i + 1 < argc) { fhigh = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--max_threads") == 0 && i + 1 < argc) { max_threads = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--downsample") == 0 && i + 1 < argc) { downsample = atoi(argv[++i]); }
    }

    // Collect file paths
    char filePaths[MAX_FILES][128]; // Full paths to .wav files
    char filenames[MAX_FILES][128]; // File names without paths
    AudioFeatures allFeatures[MAX_FILES]; // Extracted features per file
    int totalFiles = 0; // # of .wav files in input directory

    // Iterate through input directory to collect .wav files
    fs::directory_iterator endIter;
    for (fs::directory_iterator iter(input_dir); iter != endIter; ++iter) {
        if (iter->path().extension() == ".wav") {
            if (totalFiles >= MAX_FILES) {
                cout << "Too many .wav files, increase MAX_FILES\n";
                return 1;
            }
            // Write file path into filePaths
            strcpy(filePaths[totalFiles], iter->path().string().c_str());
            filePaths[totalFiles][127] = '\0'; // Ensure null termination
            totalFiles++;
        }
    }

    if (totalFiles == 0) { // No .wav files exist in input directory
        cout << "No valid .wav files were found in " << input_dir << endl;
        return 1;
    }

    bubbleSort(filePaths, totalFiles); // Sort to process files in order

    // Thread setup
    atomic<int> nextIndex(0); // Shared atomic counter for file indexing
    int availableThreads = thread::hardware_concurrency();
    if (availableThreads <= 0) { availableThreads = 1; } // Default threads to use
    int numThreads = min(max_threads, availableThreads);

    // Thread args
    ThreadArgs args;
    args.nextIndex = &nextIndex;
    args.totalFiles = totalFiles;
    args.filePaths = filePaths;
    args.allFeatures = allFeatures;
    args.filenames = filenames;
    args.fileTimeInfo = nullptr;
    args.numBits = num_bits;
    args.peakVolts = peak_volts;
    args.RS = RS;
    args.timeWin = timewin;
    args.avTime = avtime;
    args.fftWin = fft_win;
    args.artiLen = artiLen;
    args.fLow = flow;
    args.fHigh = fhigh;
    args.downSample = downsample;
    args.omitPartialMinute = omit_partial_minute;

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
            fft_win, allFeatures[i].sampleRate
        );

        // Assign cross-file dissimilarity to the first segment of the next file
        if (allFeatures[i + 1].dissimLen > 0 && allFeatures[i + 1].dissim) {
            allFeatures[i + 1].dissim[0] = crossFileDissim;
        }
    }

    // Prepare filenames for CSV
    const char* file_names[MAX_FILES];
    for (int i = 0; i < totalFiles; ++i) { file_names[i] = filenames[i]; }

    // Write output
    vector<double> crossFileDissimVector; // Empty vector since we're handling it differently now
    saveFeaturesToCSV(output_file, file_names, totalFiles, allFeatures, crossFileDissimVector);
    
    for (int i = 0; i < totalFiles; ++i) { freeAudioFeatures(allFeatures[i]); } // Deallocate memory
    fftw_cleanup(); // Clean up FFTW resources

    return 0;
}