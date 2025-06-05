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
    double* peakcount = nullptr; // # of peaks
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

    int numChannels = sfinfo.channels; // # of audio channels
    int totalFrames = sfinfo.frames; // Frames per channel

    // Sample range to read
    int startSample = max(0, range.startSample - 1);
    int endSample;
    if (range.endSample == -1) { // No range specified
        endSample = totalFrames;
    }
    else {
        endSample = min(range.endSample, totalFrames);
    }
    int numFramesToRead = endSample - startSample;

    if (numFramesToRead <= 0) { // Invalid arguments provided
        sf_close(file);
        throw runtime_error("Invalid sample range");
    }

    sf_seek(file, startSample, SEEK_SET); // Starting frame in sound file

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
ArrayShiftFFT fftshift(const double* input, int length) {
    double* shifted = new double[length]; // Shifted array
    int half = length / 2; // Middle index
    for (int i = 0; i < length; ++i) { // Shift indices and center zero frequency sample
        shifted[i] = input[(i + half) % length];
    }
    return {shifted, length};
}

// Apply bandpass filter in frequency domain
BandpassFilter dylan_bpfilt(const double* time_series, int num_pts, double samint, double flow, double fhigh) {
    double reclen = num_pts * samint;

    // FFTW allocations
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * num_pts);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * num_pts);
    fftw_plan plan = fftw_plan_dft_1d(num_pts, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Load input signal into FFTW array
    for (int i = 0; i < num_pts; ++i) {
        in[i][0] = time_series[i];
        in[i][1] = 0.0;
    }

    fftw_execute(plan); // Perform FFT

    // Construct / fftshift frequency array
    double* freq = new double[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        freq[i] = (-num_pts / 2.0 + i) / reclen;
    }
    ArrayShiftFFT shifted_freq = fftshift(freq, num_pts);
    delete[] freq;

    // Copy FFT output to complex array for filtering
    complex<double>* spectrum = new complex<double>[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        spectrum[i] = complex<double>(out[i][0], out[i][1]);
    }

    // Set high-frequency cutoff if zero
    if (fhigh == 0.0) {
        fhigh = 1.0 / (2.0 * samint);
    }

    // Apply bandpass filter
    for (int i = 0; i < num_pts; ++i) {
        if (abs(shifted_freq.data[i]) < flow || abs(shifted_freq.data[i]) > fhigh) {
            spectrum[i] = 0.0;
        }
    }

    // Prepare IFFT
    fftw_plan ifft_plan = fftw_plan_dft_1d(num_pts, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
    for (int i = 0; i < num_pts; ++i) {
        out[i][0] = spectrum[i].real();
        out[i][1] = spectrum[i].imag();
    }

    fftw_execute(ifft_plan); // Perform IFFT

    // Allocate / fill filtered time series
    double* time_series_filt = new double[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        time_series_filt[i] = in[i][0] / num_pts;
    }

    // Allocate / fill amplitude spectrum
    double* amp_spectrum = new double[num_pts];
    for (int i = 0; i < num_pts; ++i) {
        amp_spectrum[i] = abs(spectrum[i]);
    }

    // Deallocate memory
    delete[] spectrum;
    fftw_destroy_plan(plan);
    fftw_destroy_plan(ifft_plan);
    fftw_free(in);
    fftw_free(out);

    return BandpassFilter(time_series_filt, amp_spectrum, num_pts);
}

// Convert RMS pressure to SPL (dB)
double* computeSPL(const double* rms_values, int length, int& out_len) {
    constexpr double eps = 1e-12; // Small value to prevent log(0)
    if (length <= 0 || rms_values == nullptr) {
        out_len = 0;
        return nullptr;
    }
    out_len = length;

    double* spl_values = new double[out_len]; // Allocate memory
    for (int i = 0; i < out_len; ++i) {
        double rms = rms_values[i];
        spl_values[i] = 20.0 * log10(max(rms, eps)); // Use max() to prevent log(0)
    }

    return spl_values; // Calculated SPL array
}

// Calculate kurtosis used for impulsivity of a signal
double calculate_kurtosis(const double* data, int length) {
    if (length <= 0 || data == nullptr) { // Invalid input
        throw invalid_argument("Input array is empty or null");
    }

    // Calculate mean
    double mean = 0.0;
    for (int i = 0; i < length; i++) {
        mean += data[i];
    }
    mean /= length;

    double variance = 0.0;
    double fourth_moment = 0.0;

    // Calculate variance / fourth moment
    for (int i = 0; i < length; i++) {
        double diff = data[i] - mean;
        double secondMoment = pow(diff, 2);
        variance += secondMoment;
        double fourthMoment = pow(diff, 4);
        fourth_moment += fourthMoment;
    }
    variance /= length;
    fourth_moment /= length;
    double varianceSquare = pow(variance, 2);

    double kurtosis = fourth_moment / varianceSquare;
    return kurtosis; // Return raw kurtosis
}

// Calculate autocorrelation between two signals
Correlation correl_5(const double* time_series1, const double* time_series2, int series_length, int lags, int offset) {
    int len = lags + 1;
    double* corr_vals = new double[len]; // Correlation values
    double* lag_vals = new double[len]; // Corresponding lag values

    // Calculate correlation for each lag
    for (int i = 0; i <= lags; i++) {
        // Calculated statistics
        double sampleCount = 1.0;
        double sumX = 2.0;
        double sumY = 3.0;
        double sumXSquare = 4.0;
        double sumYSquare = 5.0;
        double sumXYProd = 6.0;

        for (int k = 0; k < series_length - (i + offset); k++) {
            double x = time_series1[k];
            double y = time_series2[k + (i + offset)];

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

        // Sums
        double sum_x_sample_count = sumX / sampleCount;
        double sum_y_sample_count = sumY / sampleCount;
        double sum_x_square_sample_count = sumXSquare / sampleCount;
        double sum_y_square_sample_count = sumYSquare / sampleCount;
        
        // Variance / normalization
        double covar1 = (sumXYProd / sampleCount) - (sum_x_sample_count * sum_y_sample_count);
        double denom1 = sqrt(sum_x_square_sample_count - pow(sum_x_sample_count, 2));
        double denom2 = sqrt(sum_y_square_sample_count - pow(sum_y_sample_count, 2));

        // Correlation for lag indices
        corr_vals[i] = covar1 / (denom1 * denom2);
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
    for (int jj = 0; jj < num_time_wins; jj++) {
        for (int i = 0; i < numavwin; i++) {
            double sum = 0.0;
            for (int j = 0; j < avg_win_size; j++) {
                sum += p_filt_reshaped[jj][i * avg_win_size + j];
            }
            pressure_avg[jj][i] = sum / avg_win_size;
        }
    }

    int p_avtot_rows = numavwin; // # of lags per window
    int p_avtot_cols = num_time_wins; // # of segments
    int lag_limit = static_cast<int>(p_avtot_rows * 0.7); // Max lag

    // Autocorrelation matrix
    double** acorr = new double*[p_avtot_cols];
    for (int i = 0; i < p_avtot_cols; i++) {
        acorr[i] = new double[lag_limit + 1];
    }

    // Peak count array
    int* pkcount = new int[p_avtot_cols];

    for (int zz = 0; zz < p_avtot_cols; zz++) {
        auto corr_result = correl_5(pressure_avg[zz], pressure_avg[zz], p_avtot_rows, lag_limit, 0);

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

    // Deallocate memory
    for (int i = 0; i < num_time_wins; i++) {
        delete[] p_filt_reshaped[i];
    }
    delete[] p_filt_reshaped;
    for (int i = 0; i < num_time_wins; i++) {
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
complex<double>* hilbert(const double* input, int input_len, int& output_len) {
    output_len = input_len;
    
    // Allocate input buffer (zero-padded)
    double* in = (double*)fftw_malloc(sizeof(double) * output_len);
    fill(in, in + output_len, 0.0);
    copy(input, input + input_len, in); // Copy input data

    // Allocate output buffer for real-to-complex FFT
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * output_len);
    fftw_plan plan = fftw_plan_dft_r2c_1d(output_len, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Apply Hilbert transform filter in frequency domain
    double* hilbert_filter = new double[output_len];
    fill(hilbert_filter, hilbert_filter + output_len, 0.0);
    hilbert_filter[0] = 1.0;

    if (output_len % 2 == 0) {
        hilbert_filter[output_len / 2] = 1.0;
        for (int i = 1; i < output_len / 2; ++i) {
            hilbert_filter[i] = 2.0;
        }
    }
    else {
        for (int i = 1; i <= output_len / 2; ++i) {
            hilbert_filter[i] = 2.0;
        }
    }

    for (int i = 0; i < output_len; ++i) {
        out[i][0] *= hilbert_filter[i];
        out[i][1] *= hilbert_filter[i];
    }

    // Perform inverse FFT (complex-to-complex)
    fftw_complex* inverse = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * output_len);
    fftw_plan plan_inv = fftw_plan_dft_1d(output_len, out, inverse, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_inv);

    // Normalize / store result as complex output
    complex<double>* analytic = new complex<double>[output_len];
    for (int i = 0; i < output_len; ++i) {
        analytic[i] = complex<double>(inverse[i][0] / output_len, inverse[i][1] / output_len);
    }

    // Deallocate memory
    fftw_destroy_plan(plan);
    fftw_destroy_plan(plan_inv);
    fftw_free(in);
    fftw_free(out);
    fftw_free(inverse);
    delete[] hilbert_filter;

    return analytic;
}

// Dissimilarity between consecutive time chunks
double* f_solo_dissim_GM1(double** timechunk_matrix, int pts_per_timewin, int num_timewin,
                                double fft_win, double fs, int& out_len) {
    int pts_per_fft = static_cast<int>(fft_win * fs); // # of samples in FFT
    int numfftwin = (pts_per_timewin - pts_per_fft) / pts_per_fft + 1; // # of FFT windows per time window
    out_len = num_timewin - 1; // Dissimilarity between two consecutive segments

    double* diss = new double[out_len]; // Calculated dissimilarity

    // Allocate FFT memory
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
    fftw_plan plan = fftw_plan_dft_1d(pts_per_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int kk = 0; kk < out_len; ++kk) {
        // Use Hilbert transform to calculate analytic signals for two consecutive segments
        int len1 = 0, len2 = 0;
        complex<double>* analytic1 = hilbert(timechunk_matrix[kk], pts_per_timewin, len1);
        complex<double>* analytic2 = hilbert(timechunk_matrix[kk + 1], pts_per_timewin, len2);

        // Skip if Hilbert transform failed or incorrect lengths
        if (!analytic1 || !analytic2 || len1 != len2 || len1 != pts_per_timewin) {
            cout << "Error in Hilbert calculation" << endl;
            delete[] analytic1;
            delete[] analytic2;
            continue;
        }

        // Calculate sums for normalization 
        double* at1 = new double[len1];
        double* at2 = new double[len2];
        double sum1 = 0.0, sum2 = 0.0;

        for (int i = 0; i < len1; ++i) {
            at1[i] = abs(analytic1[i]);
            sum1 += at1[i];
        }
        for (int i = 0; i < len1; ++i) {
            if (sum1 != 0.0) {
                at1[i] = at1[i] / sum1;
            }
            else {
                at1[i] = 0.0;
            }
        }

        for (int i = 0; i < len2; ++i) {
            at2[i] = abs(analytic2[i]);
            sum2 += at2[i];
        }
        for (int i = 0; i < len2; ++i) {
            if (sum2 != 0.0) {
                at2[i] = at2[i] / sum2;
            }
            else {
                at2[i] = 0.0;
            }
        }

        // Time domain dissimilarity
        double timeDomain = 0.0;
        for (int i = 0; i < pts_per_timewin; ++i) {
            timeDomain += abs(at1[i] - at2[i]);
        }
        timeDomain /= 2.0;

        // Skip if not enough windows
        if (numfftwin <= 0) {
            cout << "Error in window calculation" << endl;
            delete[] analytic1;
            delete[] analytic2;
            delete[] at1;
            delete[] at2;
            continue;
        }

        // Calculate FFT magnitude
        double* timeDomainSampleA = new double[pts_per_timewin];
        copy(timechunk_matrix[kk], timechunk_matrix[kk] + pts_per_timewin, timeDomainSampleA);
        double* magSpectrumA = new double[pts_per_timewin]();

        // FFT magnitude calculation for timeDomainSampleA
        for (int i = 0; i < numfftwin; ++i) {
            if (i * pts_per_fft + pts_per_fft > pts_per_timewin) {
                break;
            }

            for (int j = 0; j < pts_per_fft; ++j) {
                in[j][0] = timeDomainSampleA[i * pts_per_fft + j];
                in[j][1] = 0.0;
            }
            fftw_execute(plan);
            for (int j = 0; j < pts_per_fft; ++j) {
                magSpectrumA[i * pts_per_fft + j] = sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) / pts_per_fft;
            }
        }

        // Average magnitude across time windows
        double* avgFFTMagA = new double[pts_per_fft];
        for (int i = 0; i < pts_per_fft; ++i) {
            double sum = 0.0;
            for (int j = 0; j < numfftwin; ++j) {
                if ((j * pts_per_fft + i) < pts_per_timewin) {
                    sum += magSpectrumA[i + j * pts_per_fft];
                }
            }
            avgFFTMagA[i] = sum / numfftwin;
        }

        // Normalize magnitude
        double sumFFTMagA = 0.0;
        for (int i = 0; i < pts_per_fft; ++i) {
            sumFFTMagA += avgFFTMagA[i];
        }
        for (int i = 0; i < pts_per_fft; ++i) {
            if (sumFFTMagA != 0.0) {
                avgFFTMagA[i] = avgFFTMagA[i] / sumFFTMagA;
            }
            else {
                avgFFTMagA[i] = 0.0;
            }
        }

        // Calculate FFT magnitude
        double* timeDomainSampleB = new double[pts_per_timewin];
        copy(timechunk_matrix[kk + 1], timechunk_matrix[kk + 1] + pts_per_timewin, timeDomainSampleB);
        double* magSpectrumB = new double[pts_per_timewin]();

        // FFT magnitude calculation for timeDomainSampleA
        for (int i = 0; i < numfftwin; ++i) {
            if (i * pts_per_fft + pts_per_fft > pts_per_timewin) {
                break;
            }

            for (int j = 0; j < pts_per_fft; ++j) {
                in[j][0] = timeDomainSampleB[i * pts_per_fft + j];
                in[j][1] = 0.0;
            }
            fftw_execute(plan);
            for (int j = 0; j < pts_per_fft; ++j) {
                magSpectrumB[i * pts_per_fft + j] = sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) / pts_per_fft;
            }
        }

        // Average magnitude across time windows
        double* avgFFTMagB = new double[pts_per_fft];
        for (int i = 0; i < pts_per_fft; ++i) {
            double sum = 0.0;
            for (int j = 0; j < numfftwin; ++j) {
                if ((j * pts_per_fft + i) < pts_per_timewin) {
                    sum += magSpectrumB[i + j * pts_per_fft];
                }
            }
            avgFFTMagB[i] = sum / numfftwin;
        }

        // Normalize magnitude
        double sumFFTMagB = 0.0;
        for (int i = 0; i < pts_per_fft; ++i) {
            sumFFTMagB += avgFFTMagB[i];
        }
        for (int i = 0; i < pts_per_fft; ++i) {
            if (sumFFTMagB != 0.0) {
                avgFFTMagB[i] = avgFFTMagB[i] / sumFFTMagB;
            }
            else {
                avgFFTMagB[i] = 0.0;
            }
        }

        // Frequency domain dissimilarity
        double domainFrequency = 0.0;
        for (int i = 0; i < pts_per_fft; ++i) {
            domainFrequency += abs(avgFFTMagB[i] - avgFFTMagA[i]);
        }
        domainFrequency /= 2.0;

        // Error calculating frequency domain / time domain
        if (isnan(timeDomain) || isnan(domainFrequency) || isinf(timeDomain) || isinf(domainFrequency)) {
            cout << "Error in dissimilarity calculation" << endl;
        }
        else {
            diss[kk] = timeDomain * domainFrequency;
        }

        // Deallocate memory
        delete[] analytic1;
        delete[] analytic2;
        delete[] at1;
        delete[] at2;
        delete[] timeDomainSampleA;
        delete[] magSpectrumA;
        delete[] avgFFTMagA;
        delete[] timeDomainSampleB;
        delete[] magSpectrumB;
        delete[] avgFFTMagB;
    }

    // Deallocate FFTW memory
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return diss;
}

// Calculate RMS per time window
double rms(const double* row, int length) {
    double sum_squares = 0.0;

    // Calculate sum of squared values
    for (int i = 0; i < length; ++i) {
        double rowSquared = row[i] * row[i];
        sum_squares += rowSquared;
    }
    // Calculate RMS
    double meanSquares = sqrt(sum_squares / length);
    return meanSquares;
}

// Calculate RMS across time windows
double* rms(const double** matrix, int rows, int cols) {
    // Validate input
    if (rows <= 0 || cols <= 0 || matrix == nullptr) {
        return nullptr;
    }

    double* row_rms = new double[rows]; // Output for RMS SPL

    // RMS per row
    for (int i = 0; i < rows; ++i) {
        double sum_squares = 0.0;
        // Sum of squares per row
        for (int j = 0; j < cols; ++j) {
            double val = matrix[i][j];
            sum_squares += val * val;
        }
        row_rms[i] = sqrt(sum_squares / cols); // RMS per row
    }

    return row_rms;
}

// Peak SPL values of each time window
double* calculateSPLpkhold(const double** matrix, int rows, int cols) {
    // Validate input
    if (rows <= 0 || cols <= 0 || matrix == nullptr) {
        return nullptr;
    }

    double* SPLpkhold = new double[rows]; // Output for peak SPL
    const double epsilon = 1e-12; // Near-zero value to prevent log10(0)

    for (int i = 0; i < rows; ++i) { // Peak per time window
        double maxVal = 0.0;
        // Maximum absolute value
        for (int j = 0; j < cols; ++j) {
            double val = matrix[i][j];
            double abs_val = abs(val);
            if (abs_val > maxVal) {
                maxVal = abs_val;
            }
        }

        // Prevent log(0)
        if (maxVal < epsilon) {
            maxVal = epsilon;
        }

        SPLpkhold[i] = 20.0 * log10(maxVal); // Peak pressure - linear to dB
    }

    return SPLpkhold;
}

// Main feature extraction
AudioFeatures f_WAV_frankenfunction_reilly(
    int num_bits, int peak_volts, const fs::path &file_path,
    double refSens, int timewin, double avtime, int fft_win,
    int calTone, int flow, int fhigh, int downsample_factor, bool omit_partial_minute) {

    AudioFeatures features;

    string fixed_file_path = fixFilePath(file_path.string()); // Normalize file path
    // Scaling constants
    double refSensitivity = pow(10, refSens / 20.0);
    double max_count = pow(2, num_bits);
    double conv_factor = peak_volts / max_count;

    AudioInfo info = audioread_info(fixed_file_path); // Audio metadata

    // Optionally ignore incomplete minutes of audio
    if (omit_partial_minute) {
        info.duration = floor(info.duration / 60.0) * 60.0;
    }

    int total_samples = static_cast<int>(info.sampleRate * info.duration);

    // Read audio signal
    auto audio = audioread(file_path.string(), SampleRange{1, total_samples});
    int fs = audio.sampleRate;
    int audioSamplesLen = audio.numFrames;
    double* audioSamples = new double[audioSamplesLen];
    copy(audio.samples[0], audio.samples[0] + audioSamplesLen, audioSamples);

    // Optionally downsample
    if (downsample_factor != -1) {
        int newLength = 0;
        double* downsampled = downsample(audioSamples, audioSamplesLen, downsample_factor, newLength);
        delete[] audioSamples;
        audioSamples = downsampled;
        audioSamplesLen = newLength;
        fs /= downsample_factor;
    }

    // Bitshift if bit depth more than 16
    if (num_bits == 24) {
        for (int i = 0; i < audioSamplesLen; i++) {
            int temp = static_cast<int>(audioSamples[i]);
            temp >>= 8;
            audioSamples[i] = static_cast<double>(temp);
        }
    }
    else if (num_bits == 32) {
        for (int i = 0; i < audioSamplesLen; i++) {
            int temp = static_cast<int>(audioSamples[i]);
            temp >>= 16;
            audioSamples[i] = static_cast<double>(temp);
        }
    }

    // Convert to voltage / pressure
    double* voltage = new double[audioSamplesLen];
    double* pressure = new double[audioSamplesLen];
    for (int i = 0; i < audioSamplesLen; ++i) {
        voltage[i] = audioSamples[i] * conv_factor;
        pressure[i] = voltage[i] / refSensitivity;
    }
    // Deallocate memory
    delete[] audioSamples;
    delete[] voltage;

    // Remove calibration tone
    if (calTone == 1) {
        int start_idx = 6 * fs;
        int newLen = audioSamplesLen - start_idx;
        if (newLen > 0) {
            double* shiftedPressure = new double[newLen + 1];
            shiftedPressure[0] = 0.0;
            copy(pressure + start_idx, pressure + audioSamplesLen, shiftedPressure + 1);
            delete[] pressure;
            pressure = shiftedPressure;
            audioSamplesLen = newLen + 1;
        }
    }

    // Apply bandpass filter
    BandpassFilter filt = dylan_bpfilt(pressure, audioSamplesLen, 1.0 / fs, flow, fhigh);
    delete[] pressure;

    // Create segment lengths
    int pts_per_timewin = timewin * fs;
    int num_timewin = filt.length / pts_per_timewin;
    int remainder = filt.length % pts_per_timewin;
    if (remainder > 0) {
        num_timewin += 1;
    }

    // Segment durations
    features.segmentDurationLen = num_timewin;
    features.segmentDuration = new int[num_timewin];
    for (int i = 0; i < num_timewin; ++i) {
        features.segmentDuration[i] = timewin; // full window duration
    }
    if (remainder > 0) { // Not full minute
        double last_segment_sec = static_cast<double>(remainder) / fs;
        features.segmentDuration[num_timewin - 1] = static_cast<int>(round(last_segment_sec));
    }
    else if (num_timewin > 0) { // Process full minutes
        features.segmentDuration[num_timewin - 1] = timewin;
    }

    // Pad filtered time series to fit full windows
    int padding_length = (num_timewin * pts_per_timewin) - filt.length;
    int padded_len = filt.length + padding_length;
    double* p_filt_padded = new double[padded_len];
    copy(filt.filteredTimeSeries, filt.filteredTimeSeries + filt.length, p_filt_padded);
    for (int i = filt.length; i < padded_len; ++i) {
        p_filt_padded[i] = 0.0; // zero padding
    }

    // Time window matrix
    double** timechunk_matrix = new double*[num_timewin];
    for (int i = 0; i < num_timewin; ++i) {
        timechunk_matrix[i] = &p_filt_padded[i * pts_per_timewin];
    }

    // RMS
    double* rms_array = rms(const_cast<const double**>(timechunk_matrix), num_timewin, pts_per_timewin);
    int spl_len = 0;
    double* spl_array = computeSPL(rms_array, num_timewin, spl_len);

    features.SPLrmsLen = spl_len;
    features.SPLrms = new double[spl_len];
    copy(spl_array, spl_array + spl_len, features.SPLrms);
    
    delete[] rms_array;
    delete[] spl_array;

    // SPL peak
    double* splpk_result = calculateSPLpkhold(const_cast<const double**>(timechunk_matrix), num_timewin, pts_per_timewin);
    features.SPLpkLen = num_timewin;
    features.SPLpk = new double[num_timewin];
    copy(splpk_result, splpk_result + num_timewin, features.SPLpk);
    delete[] splpk_result;

    // Impulsivity (kurtosis)
    features.impulsivityLen = num_timewin;
    features.impulsivity = new double[num_timewin];
    for (int row = 0; row < num_timewin; ++row) {
        features.impulsivity[row] = calculate_kurtosis(timechunk_matrix[row], pts_per_timewin);
    }

    // SoloPerGM2 (peakcount, autocorr)
    SoloPerGM2 result = f_solo_per_GM2(p_filt_padded, padded_len, fs, timewin, avtime);

    features.peakcountLen = result.peakcount_length;
    features.peakcount = new double[features.peakcountLen];
    copy(result.peakcount, result.peakcount + features.peakcountLen, features.peakcount);

    features.autocorrRows = result.autocorr_rows;
    features.autocorrCols = result.autocorr_cols;
    features.autocorr = new double*[features.autocorrRows];
    for (int i = 0; i < features.autocorrRows; ++i) {
        features.autocorr[i] = new double[features.autocorrCols];
        copy(result.autocorr[i], result.autocorr[i] + features.autocorrCols, features.autocorr[i]);
    }

    // Deallocate memory
    delete[] result.peakcount;
    for (int i = 0; i < result.autocorr_rows; ++i) {
        delete[] result.autocorr[i];
    }
    delete[] result.autocorr;

    // Dissimilarity calculation
    int dissim_len = 0;
    double* dissim_array = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs, dissim_len);
    
    features.dissimLen = dissim_len;
    features.dissim = new double[dissim_len];
    copy(dissim_array, dissim_array + dissim_len, features.dissim);
    delete[] dissim_array;

    // Deallocate memory
    delete[] p_filt_padded;
    delete[] timechunk_matrix;

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

void extractTimestamp(const string& filename, int& year, int& month, int& day, int& hour, int& minute, int& second) {
    year = month = day = hour = minute = second = 0;
    smatch match;
    regex pattern(R"((\d{8})_(\d{6}))"); // Extract YYYYMMDD_HHMMSS

    // Check if file name matches pattern
    if (regex_search(filename, match, pattern)) {
        if (match.size() == 3) {
            string date = match[1]; // Date
            string time = match[2]; // Time
            year = stoi(date.substr(0, 4));
            month = stoi(date.substr(4, 2));
            day = stoi(date.substr(6, 2));
            hour = stoi(time.substr(0, 2));
            minute = stoi(time.substr(2, 2));
            second = stoi(time.substr(4, 2));
        }
    }
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

        int maxLength = max({ features.SPLrmsLen, features.SPLpkLen,
                              features.impulsivityLen, features.dissimLen,
                              features.peakcountLen });

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
                outputFile << "nan,nan,nan,nan,nan,";
            }
            else {
                outputFile << (currentTime->tm_year + 1900) << ","
                           << (currentTime->tm_mon + 1) << ","
                           << currentTime->tm_mday << ","
                           << currentTime->tm_hour << ","
                           << currentTime->tm_min << ",";
            }

            if (i < features.segmentDurationLen) {
                outputFile << features.segmentDuration[i];
            }
            else {
                outputFile << NAN;
            }
            outputFile << ",";

            if (i < features.SPLrmsLen) {
                outputFile << features.SPLrms[i];
            }
            else {
                outputFile << NAN;
            }
            outputFile << ",";

            if (i < features.SPLpkLen) {
                outputFile << features.SPLpk[i];
            }
            else {
                outputFile << NAN;
            }
            outputFile << ",";

            if (i < features.impulsivityLen) {
                outputFile << features.impulsivity[i];
            }
            else {
                outputFile << NAN;
            }
            outputFile << ",";

            if (i < features.dissimLen) {
                outputFile << features.dissim[i];
            }
            else {
                outputFile << NAN;
            }
            outputFile << ",";

            if (i < features.peakcountLen) {
                outputFile << features.peakcount[i];
            }
            else {
                outputFile << NAN;
            }

            for (int j = 0; j < maxAutocorrCols; ++j) {
                if (validAutocorrCols[j]) {
                    outputFile << ",";
                    if (features.autocorr && i < features.autocorrRows && j < features.autocorrCols) {
                        outputFile << features.autocorr[i][j];
                    }
                    else {
                        outputFile << "nan";
                    }
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
void threadWork(atomic<int>& nextIndex, int totalFiles, string* filePaths, AudioFeatures* allFeatures,
        string* filenames, int num_bits, int peak_volts, double RS, int timewin, double avtime, int fft_win,
        int arti, int flow, int fhigh, int downsample, bool omit_partial_minute) {
    while (true) {
        // Process next index
        int index = nextIndex++;
        if (index >= totalFiles) { // All processes complete
            return;
        }

        try {
            fs::path filepath(filePaths[index]); // Create file path from string

            // Extract metrics from sound files
            allFeatures[index] = f_WAV_frankenfunction_reilly(num_bits, peak_volts, filepath, RS, timewin,
                avtime, fft_win, arti, flow, fhigh, downsample, omit_partial_minute
            );

            filenames[index] = filepath.filename().string();
        } catch (const exception& e) { // Error extracting metrics
            cerr << "Error processing " << filePaths[index] << ": " << e.what() << endl;
        }
    }
}

// Process directory of sound files with user-given parameters
int main(int argc, char* argv[]) {
    // Default parameters if unspecified
    string input_dir; // Directory containing sound files
    string output_file; // File containing extracted features
    int num_bits = 16; // Bit depth
    double RS = -178.3; // Reference sensitivity (dB)
    int peak_volts = 2; // Peak voltage for scaling
    int arti = 1; // Calibration tone presence
    int timewin = 60; // Time segment length (seconds)
    int fft_win = 1; // FFT window length (seconds)
    double avtime = 0.1; // Averaging window for autocorrelation (seconds)
    int flow = 1; // Low frequency cutoff (Hz)
    int fhigh = 192000; // High frequency cutoff (Hz)
    int max_threads = 4; // # of threads for parallel processing
    int downsample = -1; // Downsampling factor (No downsampling if negative)
    bool omit_partial_minute = false; // Ignore incomplete time segments

    // Command line argument parsing
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) {
                input_dir = argv[++i];
            }
        }
        else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) {
                output_file = argv[++i];
            }
        }
        else if (arg == "--num_bits" || arg == "-nb") {
            if (i + 1 < argc) {
                num_bits = stoi(argv[++i]);
            }
        }
        else if (arg == "--RS" || arg == "-rs") {
            if (i + 1 < argc) {
                RS = stod(argv[++i]);
            }
        }
        else if (arg == "--peak_volts" || arg == "-pv") {
            if (i + 1 < argc) {
                peak_volts = stoi(argv[++i]);
            }
        }
        else if (arg == "--arti" || arg == "-a") {
            if (i + 1 < argc) {
                arti = stoi(argv[++i]);
            }
        }
        else if (arg == "--timewin" || arg == "-tw") {
            if (i + 1 < argc) {
                timewin = stoi(argv[++i]);
            }
        }
        else if (arg == "--fft_win" || arg == "-fw") {
            if (i + 1 < argc) {
                fft_win = stoi(argv[++i]);
            }
        }
        else if (arg == "--avtime" || arg == "-at") {
            if (i + 1 < argc) {
                avtime = stod(argv[++i]);
            }
        }
        else if (arg == "--flow" || arg == "-fl") {
            if (i + 1 < argc) {
                flow = stoi(argv[++i]);
            }
        }
        else if (arg == "--fhigh" || arg == "-fh") {
            if (i + 1 < argc) {
                fhigh = stoi(argv[++i]);
            }
        }
        else if (arg == "--max_threads" || arg == "-mt") {
            if (i + 1 < argc) {
                max_threads = stoi(argv[++i]);
            }
        }
        else if (arg == "--downsample" || arg == "-ds") {
            if (i + 1 < argc) {
                downsample = stoi(argv[++i]);
            }
        }
        else if (arg == "--omit_partial_minute" || arg == "-opm") {
            omit_partial_minute = true;
        }
    }

    // Count .wav files in input directory
    fs::directory_iterator iter(input_dir);
    fs::directory_iterator endIter;
    int totalFiles = 0;
    // Iterate through directory
    while (iter != endIter) {
        if ((*iter).path().extension() == ".wav") {
            totalFiles++;
        }
        ++iter;
    }

    // Collect file path names
    string* filePaths = new string[totalFiles];

    // Reset iterator to beginning
    iter = fs::directory_iterator(input_dir);

    int idx = 0;
    while (iter != endIter) {
        if ((*iter).path().extension() == ".wav") {
            filePaths[idx++] = (*iter).path().string();
        }
        ++iter;
    }
    // Sort file paths alphabetically
    sort(filePaths, filePaths + totalFiles);

    // Allocate memory for extracted features
    AudioFeatures* allFeatures = new AudioFeatures[totalFiles];
    string* filenames = new string[totalFiles];

    atomic<int> nextIndex(0); // Shared atomic index for in-order output

    // Determine maximum thread count
    int availableThreads = thread::hardware_concurrency();
    if (availableThreads <= 0) { // Invalid # of threads
        availableThreads = 1;
    }
    int numThreads = min(max_threads, availableThreads);

    // Create / launch threads using workerFunction
    thread* threads = new thread[numThreads];
    for (int i = 0; i < numThreads; ++i) {
        threads[i] = thread(threadWork, ref(nextIndex), totalFiles, filePaths, allFeatures, filenames, num_bits,
            peak_volts, RS, timewin, avtime, fft_win, arti, flow, fhigh, downsample, omit_partial_minute);
    }

    // Wait for all threads to complete
    for (int i = 0; i < numThreads; ++i) {
        threads[i].join();
    }
    delete[] threads; // Free memory

    // Prepare filenames for .csv export
    const char** file_names = new const char*[totalFiles];
    for (int i = 0; i < totalFiles; ++i) {
        file_names[i] = filenames[i].c_str();
    }

    // Write features to output file
    if (totalFiles > 0) {
        saveFeaturesToCSV(output_file.c_str(), file_names, totalFiles, allFeatures);
        cout << "Successfully saved features for " << totalFiles << " files to " << output_file << endl;
    }
    else {
        cout << "No valid .wav files were processed" << endl;
    }

    // Deallocate memory
    delete[] file_names;
    delete[] allFeatures;
    delete[] filenames;
    delete[] filePaths;

    return 0;
}