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
        if (!buf) {
            throw bad_alloc();
        }

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
        if (fwd_plan) {
            fftw_destroy_plan(fwd_plan);
        }
        if (inv_plan) {
            fftw_destroy_plan(inv_plan);
        }
        if (buf) {
            fftw_free(buf);
        }
    }
};

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
    if (!file) {
        throw runtime_error("Error opening audio file: " + string(sf_strerror(file)));
    }

    // Sample range to read
    int totalFrames = sfinfo.frames; // Frames per channel
    int endSample;
    if (range.endSample == -1) { // No range specified
        endSample = totalFrames;
    }
    else {
        endSample = min(range.endSample, totalFrames);
    }
    
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
            for (int i = 0; i < numFramesToRead * numChannels; ++i) {
                interleavedSamples[i] = static_cast<double>(temp[i]);
            }
            delete[] temp;
            break;
        }
        case SF_FORMAT_PCM_24:
        case SF_FORMAT_PCM_32: { // 24 or 32 bit
            int* temp = new int[numFramesToRead * numChannels];
            sf_readf_int(file, temp, numFramesToRead);
            for (int i = 0; i < numFramesToRead * numChannels; ++i) {
                interleavedSamples[i] = static_cast<double>(temp[i]);
            }
            delete[] temp;
            break;
        }
        case SF_FORMAT_FLOAT: { // 32 bit float
            float* temp = new float[numFramesToRead * numChannels];
            sf_readf_float(file, temp, numFramesToRead);
            for (int i = 0; i < numFramesToRead * numChannels; ++i) {
                interleavedSamples[i] = static_cast<double>(temp[i]);
            }
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
    for (int ch = 0; ch < numChannels; ++ch) {
        samples[ch] = new double[numFramesToRead];
    }

    // Separate indices per channel
    for (int i = 0; i < numFramesToRead; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) {
            samples[ch][i] = interleavedSamples[i * numChannels + ch];
        }
    }

    delete[] interleavedSamples; // Deallocate memory

    return AudioData{samples, numChannels, numFramesToRead, sfinfo.samplerate}; // Metadata
}

AudioInfo audioread_info(const string& file_path) {
    SF_INFO sfInfo = {0}; // Struct containing sound metadata (frames, samplerate, channels, format)
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sfInfo); // Open audio file in read mode

    if (!file) { // Error opening file
        throw runtime_error("Error opening audio file: " + file_path);
    }

    int sampleRate = sfInfo.samplerate; // Get sample rate
    int numFrames = sfInfo.frames; // Get # of frames

    float duration = static_cast<float>(numFrames) / sampleRate; // Calculate duration (seconds)

    sf_close(file); // Close file after reading info

    return {sampleRate, duration};
}

// Reduce sampling rate by analyzing (1 / factor) samples
double* downsample(const double* x, int length, int factor, int& newLength) {
    if (factor <= 0) { // Invalid scaling factor
        throw invalid_argument("Factor must be positive");
    }

    newLength = (length + factor - 1) / factor; // # of output samples
    double* result = new double[newLength]; // Downsampled output

    int idx = 0;
    for (int i = 0; i < length; i += factor) { // Copy (1 / factor) samples
        result[idx++] = x[i];
    }

    return result;
}

// Manually shift zero-frequency to center of array
ArrayShiftFFT fftshift(double* input, int length) {
    double* shifted = new double[length]; // Shifted array
    for (int i = 0; i < length; ++i) { // Shift indices and center zero frequency sample
        shifted[i] = input[(i + (length / 2)) % length]; // Shift from center
    }
    return {shifted, length};
}

// Apply bandpass filter in frequency domain
BandpassFilter bandpass_filter(const double* time_series, int num_pts, double frequency, double flow, double fhigh) {
    // Use RAII wrapper for better memory management
    FFTWHandler fftw(num_pts);

    // Load input signal directly
    for (int i = 0; i < num_pts; ++i) {
        fftw.buf[i][0] = time_series[i];
        fftw.buf[i][1] = 0.0;
    }

    fftw_execute(fftw.fwd_plan);

    // Pre-compute frequency array with optimization
    double reclen = num_pts * frequency;
    double freq_step = 1.0 / reclen;
    double freq_offset = -num_pts * 0.5 * freq_step;
    double* freq = new double[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        freq[i] = freq_offset + i * freq_step;
    }
    ArrayShiftFFT shifted_freq = fftshift(freq, num_pts);
    delete[] freq;

    // Set high-frequency cutoff if zero
    if (fhigh == 0.0) {
        fhigh = 0.5 / frequency; // Nyquist frequency
    }

    // Apply bandpass filter directly to FFTW buffer
    for (int i = 0; i < num_pts; ++i) {
        double abs_freq = fabs(shifted_freq.data[i]);
        if (abs_freq < flow || abs_freq > fhigh) {
            fftw.buf[i][0] = 0.0;
            fftw.buf[i][1] = 0.0;
        }
    }

    fftw_execute(fftw.inv_plan);

    // Normalize and fill outputs
    double norm_factor = 1.0 / num_pts;

    // Allocate output arrays
    double* time_series_filt = new double[num_pts];
    double* amp_spectrum = new double[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        time_series_filt[i] = fftw.buf[i][0] * norm_factor;
        double real = fftw.buf[i][0];
        double imag = fftw.buf[i][1];
        amp_spectrum[i] = sqrt(real*real + imag*imag);
    }

    return BandpassFilter(time_series_filt, amp_spectrum, num_pts);
}

// Calculate kurtosis used for impulsivity of a signal
double calculate_kurtosis(const double* data, int length) {
    if (length <= 0 || data == nullptr) {
        throw invalid_argument("Input array is empty or null");
    }

    // Single-pass algorithm for better cache performance
    double sum = 0.0, sum_sq = 0.0;
    
    // First pass: calculate sums
    for (int i = 0; i < length; i++) {
        double val = data[i];
        sum += val;
        sum_sq += val * val;
    }
    
    double mean = sum / length;
    double mean_sq = mean * mean;
    
    // Second pass: calculate fourth moment
    double sum_fourth = 0.0;
    for (int i = 0; i < length; i++) {
        double centered_sq = (data[i] - mean) * (data[i] - mean);
        sum_fourth += centered_sq * centered_sq;
    }
    
    double variance = (sum_sq / length) - mean_sq;
    double fourth_moment = sum_fourth / length;
    
    if (variance < 1e-12) return 0.0; // Avoid division by zero
    
    return (sum_fourth / length) / (variance * variance); // Fourth moment / squared variance
}

// Calculate autocorrelation between two signals
Correlation correl_5(const double* time_series1, const double* time_series2, int series_length, int lags, int offset) {
    // Declare variables outside loops
    double sumX, sumY, sumXSquare, sumYSquare, sumXYProd;
    double sampleCount;
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
        sampleCount = 0.0;

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
                sampleCount += 1.0;
            }
        }

        // Avoid division by zero
        if (sampleCount == 0) {
            corr_vals[i] = NAN;
            lag_vals[i] = static_cast<double>(i);
            continue;
        }

        sum_x_sample_count = sumX / sampleCount;
        sum_y_sample_count = sumY / sampleCount;
        sum_x_square_sample_count = sumXSquare / sampleCount;
        sum_y_square_sample_count = sumYSquare / sampleCount;

        covar1 = (sumXYProd / sampleCount) - (sum_x_sample_count * sum_y_sample_count);
        denom1 = sqrt(sum_x_square_sample_count - (sum_x_sample_count * sum_x_sample_count));
        denom2 = sqrt(sum_y_square_sample_count - (sum_y_sample_count * sum_y_sample_count));

        // Prevent division by zero in correlation calculation
        if (denom1 == 0.0 || denom2 == 0.0) {
            corr_vals[i] = NAN;
        }
        else {
            corr_vals[i] = covar1 / (denom1 * denom2);
        }
        lag_vals[i] = static_cast<double>(i);
    }

    return Correlation(corr_vals, lag_vals, len);
}

// Calculate autocorrelation / peak counts
SoloPerGM2 f_solo_per_GM2(const double* p_filt_input, int input_length, double fs, double timewin, double avtime) {
    int samp_window_size = static_cast<int>(fs * timewin); // # of samples per window
    int num_time_wins = input_length / samp_window_size; // # of time windows

    if (num_time_wins == 0) { // Invalid # of time windows
        throw runtime_error("Empty time window");
    }

    // Matrix for pressure signal per window
    double** p_filt_reshaped = new double*[num_time_wins];
    for (int i = 0; i < num_time_wins; i++) {
        p_filt_reshaped[i] = new double[samp_window_size];
    }

    // Square input signals / segment into windows
    for (int j = 0; j < num_time_wins; j++) {
        for (int i = 0; i < samp_window_size; i++) {
            double val = p_filt_input[j * samp_window_size + i];
            p_filt_reshaped[j][i] = val * val;
        }
    }

    int avg_win_size = static_cast<int>(fs * avtime); // Average window size
    int numavwin = samp_window_size / avg_win_size; // # of averaging windows per segment

    // Allocate pressure averages
    double** pressure_avg = new double*[num_time_wins];
    for (int i = 0; i < num_time_wins; i++) {
        pressure_avg[i] = new double[numavwin];
    }

    // Calculate averages per window
    for (int jj = 0; jj < num_time_wins; ++jj) {
        const double* row = p_filt_reshaped[jj];
        for (int i = 0; i < numavwin; ++i) {
            double* start = const_cast<double*>(&row[i * avg_win_size]);
            double avg = 0.0;
            for (int j = 0; j < avg_win_size; ++j) {
                avg += start[j];
            }
            pressure_avg[jj][i] = avg / avg_win_size;
        }
    }

    for (int i = 0; i < num_time_wins; ++i) {
        delete[] p_filt_reshaped[i];
    }
    delete[] p_filt_reshaped;

    int p_avtot_rows = numavwin; // # of lags per window
    int lag_limit = static_cast<int>(p_avtot_rows * 0.7); // Max lag

    int p_avtot_cols = num_time_wins; // # of segments
    double** acorr = new double*[p_avtot_cols]; // Autocorrelation matrix
    for (int i = 0; i < p_avtot_cols; i++) {
        acorr[i] = new double[lag_limit + 1];
    }

    // Peak count array
    int* pkcount = new int[p_avtot_cols];

    for (int zz = 0; zz < p_avtot_cols; zz++) {
        Correlation corr_result = correl_5(pressure_avg[zz], pressure_avg[zz], p_avtot_rows, lag_limit, 0);

        // Copy correlationValues to acorr[zz]
        for (int i = 0; i <= lag_limit; i++) {
            acorr[zz][i] = corr_result.correlationValues[i];
        }

        // Count peaks
        int peak_count = 0;
        for (int i = 1; i < lag_limit; i++) {
            if (acorr[zz][i] > acorr[zz][i - 1] && acorr[zz][i] > acorr[zz][i + 1]) {
                // calculate prominence
                double left_min = acorr[zz][i];
                for (int j = i - 1; j >= 0; j--) {
                    if (acorr[zz][j] >= acorr[zz][i]) {
                        break;
                    }
                    if (acorr[zz][j] < left_min) {
                        left_min = acorr[zz][j];
                    }
                }
                double right_min = acorr[zz][i];
                for (int j = i + 1; j <= lag_limit; j++) {
                    if (acorr[zz][j] >= acorr[zz][i]) {
                        break;
                    }
                    if (acorr[zz][j] < right_min) {
                        right_min = acorr[zz][j];
                    }
                }
                double prominence = acorr[zz][i] - max(left_min, right_min);
                if (prominence > 0.5) {
                    peak_count++;
                }
            }
        }
        pkcount[zz] = peak_count;
    }
    
    for (int i = 0; i < num_time_wins; ++i) {
        delete[] pressure_avg[i];
    }
    delete[] pressure_avg;

    // Return result struct with raw arrays
    SoloPerGM2 result;
    result.peakcount = pkcount;
    result.autocorr = acorr;
    result.peakcount_length = p_avtot_cols;
    result.autocorr_rows = p_avtot_cols;
    result.autocorr_cols = lag_limit + 1;

    return result;
}

// Calculate analytic signal (real / imaginary) with hilbert transform
fftw_complex* hilbert_raw(const double* input, int input_len) {
    if (input_len <= 0 || input == nullptr) {
        return nullptr;
    }

    // Use RAII wrapper for buffer and plans
    FFTWHandler fftw(input_len);

    // Load input into FFTW buffer (imaginary = 0)
    for (int i = 0; i < input_len; ++i) {
        fftw.buf[i][0] = input[i];
        fftw.buf[i][1] = 0.0;
    }

    // Forward FFT
    fftw_execute(fftw.fwd_plan);

    // Apply Hilbert filter in frequency domain
    int half = input_len / 2;
    int upper = (input_len % 2 == 0) ? half - 1 : half;

    for (int i = 1; i <= upper; ++i) {
        fftw.buf[i][0] *= 2.0;
        fftw.buf[i][1] *= 2.0;
    }
    for (int i = half + 1; i < input_len; ++i) {
        fftw.buf[i][0] = 0.0;
        fftw.buf[i][1] = 0.0;
    }

    // Inverse FFT
    fftw_execute(fftw.inv_plan);

    // Normalize and allocate output buffer for caller
    fftw_complex* result = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * input_len);
    if (!result) {
        throw bad_alloc();
    }

    for (int i = 0; i < input_len; ++i) {
        result[i][0] = fftw.buf[i][0] / input_len;
        result[i][1] = fftw.buf[i][1] / input_len;
    }

    return result; // caller must fftw_free(result)
}

double* f_solo_dissim_GM1(double** timechunk_matrix, int pts_per_timewin, int num_timewin,
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

    // --- NEW: Thread-local reusable FFTWHandler ---
    thread_local FFTWHandler fft_handler(pts_per_fft);
    fftw_complex* fft_buf = fft_handler.buf;

    // --- Allocate buffers once ---
    double* envelope = new double[pts_per_timewin];
    double* env2 = new double[pts_per_timewin];
    double* fftA = new double[pts_per_fft];
    double* fftB = new double[pts_per_fft];

    auto calc_fft_mag = [&](const double* input, double* output) {
        fill(output, output + pts_per_fft, 0.0);

        for (int w = 0; w < numfftwin; ++w) {
            int base = w * pts_per_fft;
            for (int j = 0; j < pts_per_fft; ++j) {
                fft_buf[j][0] = input[base + j];
                fft_buf[j][1] = 0.0;
            }
            fftw_execute(fft_handler.fwd_plan); // FFT

            for (int j = 0; j < pts_per_fft; ++j) {
                output[j] += hypot(fft_buf[j][0], fft_buf[j][1]);
            }
        }

        double total = 0.0;
        for (int j = 0; j < pts_per_fft; ++j) {
            total += output[j];
        }
        if (total > 1e-12) {
            for (int j = 0; j < pts_per_fft; ++j) {
                output[j] /= total;
            }
        } else {
            fill(output, output + pts_per_fft, 0.0);
        }
    };

    double* diss = new double[out_len];

    for (int i = 0; i < out_len; ++i) {
        fftw_complex* hil1 = hilbert_raw(timechunk_matrix[i], pts_per_timewin);
        fftw_complex* hil2 = hilbert_raw(timechunk_matrix[i + 1], pts_per_timewin);
        if (!hil1 || !hil2) {
            diss[i] = NAN;
            if (hil1) fftw_free(hil1);
            if (hil2) fftw_free(hil2);
            continue;
        }

        double sum1 = 0.0, sum2 = 0.0;
        for (int k = 0; k < pts_per_timewin; ++k) {
            envelope[k] = hypot(hil1[k][0], hil1[k][1]);
            env2[k] = hypot(hil2[k][0], hil2[k][1]);
            sum1 += envelope[k];
            sum2 += env2[k];
        }

        fftw_free(hil1);
        fftw_free(hil2);

        if (sum1 > 1e-12) {
            for (int k = 0; k < pts_per_timewin; ++k) {
                envelope[k] /= sum1;
            }
        } else {
            fill(envelope, envelope + pts_per_timewin, 0.0);
        }

        double timeDiss = 0.0;
        if (sum2 > 0.0) {
            for (int k = 0; k < pts_per_timewin; ++k) {
                double env2_norm = env2[k] / sum2;
                timeDiss += fabs(envelope[k] - env2_norm);
            }
        } else {
            for (int k = 0; k < pts_per_timewin; ++k) {
                timeDiss += envelope[k];
            }
        }
        timeDiss *= 0.5;

        // Frequency-domain dissimilarity
        calc_fft_mag(timechunk_matrix[i], fftA);
        calc_fft_mag(timechunk_matrix[i + 1], fftB);

        double freqDiss = 0.0;
        for (int j = 0; j < pts_per_fft; ++j) {
            freqDiss += fabs(fftA[j] - fftB[j]);
        }
        freqDiss *= 0.5;

        diss[i] = timeDiss * freqDiss;
    }

    delete[] envelope;
    delete[] env2;
    delete[] fftA;
    delete[] fftB;

    return diss;
}

void freeAudioData(AudioData& audio) {
    for (int ch = 0; ch < audio.numChannels; ++ch) {
        delete[] audio.samples[ch];
    }
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
        for (int i = 0; i < features.autocorrRows; ++i) {
            delete[] features.autocorr[i];
        }
        delete[] features.autocorr;
    }

    // Reset pointers (optional safety)
    features.segmentDuration = nullptr;
    features.SPLrms = nullptr;
    features.SPLpk = nullptr;
    features.impulsivity = nullptr;
    features.dissim = nullptr;
    features.peakcount = nullptr;
    features.autocorr = nullptr;
}

// Main feature extraction
AudioFeatures feature_extraction(int num_bits, int peak_volts, const fs::path &file_path,
        double refSens, int timewin, double avtime, int fft_win,
        int calTone, int flow, int fhigh, int downsample_factor, bool omit_partial_minute) {

    string fixed_file_path = fixFilePath(file_path.string());
    AudioInfo info = audioread_info(fixed_file_path);
    if (omit_partial_minute) {
        info.duration = floor(info.duration / 60.0) * 60.0;
    }

    int total_samples = static_cast<int>(info.sampleRate * info.duration);
    AudioData audio = audioread(file_path.string(), SampleRange{1, total_samples});
    int fs = audio.sampleRate;
    int audioSamplesLen = audio.numFrames;

    // Allocate and convert audio samples to pressure
    double* pressure = new double[audioSamplesLen];
    for (int i = 0; i < audioSamplesLen; ++i) {
        double s = static_cast<double>(audio.samples[0][i]);
        if (num_bits == 24) {
            s = static_cast<double>(static_cast<int>(s) >> 8);
        }
        else if (num_bits == 32) {
            s = static_cast<double>(static_cast<int>(s) >> 16);
        }
        pressure[i] = s * (peak_volts / static_cast<double>(1 << num_bits)) * (1.0 / pow(10.0, refSens / 20.0)); // Convert to pressure
    }

    freeAudioData(audio);

    // Downsample in-place if needed
    if (downsample_factor != -1) {
        int newLen = 0;
        double* downsampled = downsample(pressure, audioSamplesLen, downsample_factor, newLen);
        delete[] pressure;
        pressure = downsampled;
        audioSamplesLen = newLen;
        fs /= downsample_factor;
    }

    // Remove calibration tone
    if (calTone == 1 && audioSamplesLen > 6 * fs) {
        int newLen = audioSamplesLen - 6 * fs;
        double* shifted = new double[newLen];
        memcpy(shifted, pressure + 6 * fs, sizeof(double) * newLen);
        delete[] pressure;
        pressure = shifted;
        audioSamplesLen = newLen;
    }

    // Apply bandpass filter
    BandpassFilter filt = bandpass_filter(pressure, audioSamplesLen, 1.0 / fs, flow, fhigh);
    delete[] pressure;

    int pts_per_timewin = timewin * fs;
    int num_timewin = filt.length / pts_per_timewin;
    int remainder = filt.length % pts_per_timewin;
    if (remainder > 0) {
        ++num_timewin;
    }

    int padded_len = num_timewin * pts_per_timewin;
    double* padded_signal = new double[padded_len]();
    memcpy(padded_signal, filt.filteredTimeSeries, sizeof(double) * filt.length);

    AudioFeatures features = {};
    
    // Segment durations
    features.segmentDurationLen = num_timewin;
    features.segmentDuration = new int[num_timewin];

    double** timechunk_matrix = new double*[num_timewin]; // Time window pointers
    
    for (int i = 0; i < num_timewin; ++i) {
        timechunk_matrix[i] = &padded_signal[i * pts_per_timewin];
        features.segmentDuration[i] = (i == num_timewin - 1 && remainder > 0)
            ? static_cast<int>(round(static_cast<double>(remainder) / fs)) : timewin;
    }
    if (remainder > 0) {
        features.segmentDuration[num_timewin - 1] = static_cast<int>(round(static_cast<double>(remainder) / fs));
    }
    else {
        features.segmentDuration[num_timewin - 1] = timewin;
    }

    // Allocate feature arrays
    features.SPLrmsLen = features.SPLpkLen = features.impulsivityLen = num_timewin;
    features.SPLrms = new double[num_timewin];
    features.SPLpk = new double[num_timewin];
    features.impulsivity = new double[num_timewin];

    // Extract features (combined loop)
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

    // Autocorrelation + Peak Count
    SoloPerGM2 gm2 = f_solo_per_GM2(padded_signal, padded_len, fs, timewin, avtime);
    features.peakcountLen = num_timewin;
    features.peakcount = new int[num_timewin];
    for (int i = 0; i < num_timewin; ++i) {
        features.peakcount[i] = gm2.peakcount[i];
    }
    delete[] gm2.peakcount;

    features.autocorrRows = gm2.autocorr_rows;
    features.autocorrCols = gm2.autocorr_cols;
    features.autocorr = new double*[gm2.autocorr_rows];
    for (int i = 0; i < gm2.autocorr_rows; ++i) {
        features.autocorr[i] = gm2.autocorr[i]; // reuse raw ptr
    }
    delete[] gm2.autocorr; // free array of ptrs only

    // Dissimilarity
    int dissim_len = 0;
    features.dissim = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs, dissim_len);
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
            if (feature.autocorrRows > maxAutocorrRows) {
                maxAutocorrRows = feature.autocorrRows;
            }
            if (feature.autocorrCols > maxAutocorrCols) {
                maxAutocorrCols = feature.autocorrCols;
            }
        }
    }

    // Allocate array for valid autocorr columns
    bool* validAutocorrCols = new bool[maxAutocorrCols];
    for (int j = 0; j < maxAutocorrCols; ++j) {
        validAutocorrCols[j] = false;
    }

    // Remove extra autocorr columns
    for (int i = 0; i < numFiles; ++i) {
        const AudioFeatures& feature = allFeatures[i];
        if (feature.autocorr != nullptr) {
            for (int r = 0; r < feature.autocorrRows; ++r) {
                for (int c = 0; c < feature.autocorrCols; ++c) {
                    if (!isnan(feature.autocorr[r][c])) {
                        validAutocorrCols[c] = true;
                    }
                }
            }
        }
    }

    // CSV Header
    outputFile << "Filename,Year,Month,Day,Hour,Minute,SegmentDuration,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount";
    for (int j = 0; j < maxAutocorrCols; ++j) {
        if (validAutocorrCols[j]) {
            outputFile << ",Autocorr_" << j;
        }
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
        if (!firstTime || (firstTime->tm_year + 1900) < 1900) {
            useNanTimestamp = true;
        }

        for (int i = 0; i < maxLength; ++i) {
            time_t currentEpoch = baseEpoch + i * 60;
            tm* currentTime = localtime(&currentEpoch);

            outputFile << filenames[fileIdx] << ",";

            if (useNanTimestamp || !currentTime) {
                outputFile << "NaN,NaN,NaN,NaN,NaN,";
            }
            else {
                outputFile << (currentTime->tm_year + 1900) << ","
                           << (currentTime->tm_mon + 1) << ","
                           << currentTime->tm_mday << ","
                           << currentTime->tm_hour << ","
                           << currentTime->tm_min << ",";
            }

            if (i < features.segmentDurationLen) outputFile << features.segmentDuration[i];
            else outputFile << "NaN";
            outputFile << ",";

            if (i < features.SPLrmsLen) outputFile << features.SPLrms[i];
            else outputFile << "NaN";
            outputFile << ",";

            if (i < features.SPLpkLen) outputFile << features.SPLpk[i];
            else outputFile << "NaN";
            outputFile << ",";

            if (i < features.impulsivityLen) outputFile << features.impulsivity[i];
            else outputFile << "NaN";
            outputFile << ",";

            if (i < features.dissimLen) outputFile << features.dissim[i];
            else outputFile << "NaN";
            outputFile << ",";

            if (i < features.peakcountLen) outputFile << features.peakcount[i];
            else outputFile << "NaN";
            
            for (int j = 0; j < maxAutocorrCols; ++j) {
                if (validAutocorrCols[j]) {
                    outputFile << ",";
                    if (features.autocorr && i < features.autocorrRows && j < features.autocorrCols) outputFile << features.autocorr[i][j];
                    else outputFile << "NaN";
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
        if (index >= totalFiles) {
            break;
        }

        // Convert the raw C-string to a filesystem::path
        fs::path filepath(filePaths[index]);

        // Extract just the filename as string and copy to filenames[]
        string fname = filepath.filename().string();
        strncpy(filenames[index], fname.c_str(), 511);
        filenames[index][511] = '\0'; // ensure null-termination

        // Call the actual feature extraction (assumes it accepts fs::path)
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

void threadWrapper(ThreadArgs args) {
    threadWork(
        *args.nextIndex, args.totalFiles, args.filePaths,
        args.allFeatures, args.filenames,
        args.num_bits, args.peak_volts, args.RS, args.timewin, args.avtime,
        args.fft_win, args.arti, args.flow, args.fhigh,
        args.downsample, args.omit_partial_minute
    );
}

const int MAX_FILES = 1000;

// Process directory of sound files with user-given parameters
int main(int argc, char* argv[]) {
    using namespace chrono;
    auto start = high_resolution_clock::now();

    // Defaults
    char input_dir[512] = {};
    char output_file[512] = {};
    int num_bits = 16, peak_volts = 2, arti = 1;
    int timewin = 60, fft_win = 1, flow = 1, fhigh = 192000;
    double RS = -178.3, avtime = 0.1;
    int max_threads = 4, downsample = -1;
    bool omit_partial_minute = false;

    // Parse CLI args
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--omit_partial_minute") == 0) omit_partial_minute = true;
        else if (strcmp(argv[i], "--input") == 0 && i+1 < argc) strncpy(input_dir, argv[++i], sizeof(input_dir)-1);
        else if (strcmp(argv[i], "--output") == 0 && i+1 < argc) strncpy(output_file, argv[++i], sizeof(output_file)-1);
        else if (strcmp(argv[i], "--num_bits") == 0) num_bits = atoi(argv[++i]);
        else if (strcmp(argv[i], "--RS") == 0) RS = atof(argv[++i]);
        else if (strcmp(argv[i], "--peak_volts") == 0) peak_volts = atoi(argv[++i]);
        else if (strcmp(argv[i], "--arti") == 0) arti = atoi(argv[++i]);
        else if (strcmp(argv[i], "--timewin") == 0) timewin = atoi(argv[++i]);
        else if (strcmp(argv[i], "--fft_win") == 0) fft_win = atoi(argv[++i]);
        else if (strcmp(argv[i], "--avtime") == 0) avtime = atof(argv[++i]);
        else if (strcmp(argv[i], "--flow") == 0) flow = atoi(argv[++i]);
        else if (strcmp(argv[i], "--fhigh") == 0) fhigh = atoi(argv[++i]);
        else if (strcmp(argv[i], "--max_threads") == 0) max_threads = atoi(argv[++i]);
        else if (strcmp(argv[i], "--downsample") == 0) downsample = atoi(argv[++i]);
    }

    // Collect file paths
    char filePaths[MAX_FILES][512];
    char filenames[MAX_FILES][512];
    AudioFeatures allFeatures[MAX_FILES];
    int totalFiles = 0;

    fs::directory_iterator endIter;
    for (fs::directory_iterator iter(input_dir); iter != endIter; ++iter) {
        if (iter->path().extension() == ".wav") {
            if (totalFiles >= MAX_FILES) {
                cout << "Too many .wav files, increase MAX_FILES\n";
                return 1;
            }
            strncpy(filePaths[totalFiles], iter->path().string().c_str(), 511);
            filePaths[totalFiles][511] = '\0'; // ensure null termination
            totalFiles++;
        }
    }

    if (totalFiles == 0) {
        cout << "No valid .wav files were found in " << input_dir << endl;
        return 1;
    }

    // Sort to process files in order
    bubbleSort(filePaths, totalFiles);

    // Decide thread count
    atomic<int> nextIndex(0);
    int availableThreads = thread::hardware_concurrency();
    if (availableThreads <= 0) {
        availableThreads = 1;
    }
    int numThreads = min(max_threads, availableThreads);

    if (fhigh <= 16000 && numThreads > 2) {
        cout << "fhigh = " << fhigh << " Hz is low\nReducing threads to 2\n";
        numThreads = 2;
    }
    else if (fhigh <= 48000 && numThreads > 4) {
        cout << "fhigh = " << fhigh << " Hz is low\nReducing threads to 4\n";
        numThreads = 4;
    }

    // Thread args
    ThreadArgs args;
    args.nextIndex = &nextIndex;
    args.totalFiles = totalFiles;
    args.filePaths = filePaths;
    args.allFeatures = allFeatures;
    args.filenames = filenames;
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
    thread* threads = new thread[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        threads[i] = thread(threadWrapper, args);
    }
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join(); // Join after all threads are launched
    }
    delete[] threads;

    // Prepare filenames for CSV
    const char* file_names[MAX_FILES];
    for (int i = 0; i < totalFiles; ++i) {
        file_names[i] = filenames[i];
    }

    // Write output
    saveFeaturesToCSV(output_file, file_names, totalFiles, allFeatures);
    cout << "Saved features for " << totalFiles << " files to " << output_file << endl;

    // Deallocate memory
    for (int i = 0; i < totalFiles; ++i) {
        freeAudioFeatures(allFeatures[i]);
    }

    fftw_cleanup(); // Clean up FFTW resources
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    cout << duration.count() / 1000000.0 << " seconds" << endl;
    
    return 0;
}