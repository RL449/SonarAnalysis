#include <iostream>     // Standard input/output stream operations
#include <string>       // Implements string class
#include <filesystem>   // File path and directory operations
#include <vector>       // Dynamic arrays
#include <fstream>      // File stream operations
#include <cmath>        // Standard math functions
#include <algorithm>    // Implements common algorithms
#include <sndfile.h>    // Read/write audio files
#include <stdexcept>    // Exception classes
#include <complex>      // Implements complex for FFT or Hilbert transforms
#include <fftw3.h>      // FFT computations
#include <cstring>      // C-style string operations
#include <thread>       // Implements multithreading
#include <mutex>        // Implements thread synchronization
// Limit thread count to # of cores
#include <queue>
#include <condition_variable>
#include <atomic>

using namespace std;    // Standard namespace

// Declare structs

struct SampleRange {
    int startSample;    // Starting sample index
    int endSample;      // Ending sample index
    // Constructor with default range
    SampleRange(int start = 1, int end = -1) : startSample(start), endSample(end) {}
};

struct BandpassFilter {
    vector<double> filtered_timeseries; // Time-domain signal after filtering
    vector<double> amplitude_spectrum;  // Frequency-domain amplitude spectrum
    // Constructor initializing both vectors
    BandpassFilter(const vector<double>& ts, const vector<double>& spec)
        : filtered_timeseries(ts), amplitude_spectrum(spec) {}
};

struct Correlation {
    vector<double> correlation_values;  // Cross-correlation values between two signals
    vector<double> lags;                // Corresponding lag values
};

// Extracted audio features
struct AudioFeatures {
    vector<double> SPLrms;              // Root mean square sound pressure levels
    vector<double> SPLpk;               // Peak sound pressure levels
    vector<double> impulsivity;         // Measure signal impulsivity levels
    vector<int> peakcount;              // Number of peaks detected
    vector<vector<double>> autocorr;    // Autocorrelation vectors for each segment
    vector<double> dissim;              // Dissimilarity between segments
};

struct AudioData {
    vector<vector<double>> samples; // 2D array of audio samples
    int sampleRate;                 // Sampling rate in Hz
};

struct AudioInfo {
    int sampleRate;     // Sampling rate in Hz
    double duration;    // Duration of the audio in seconds
};

struct SoloPerGM2 {
    vector<int> peakcount;              // Number of peaks per segment
    vector<vector<double>> autocorr;    // Autocorrelation values for each segment
};

struct ArrayShiftFFT {
    double* data;
    int length;

    ~ArrayShiftFFT() {
        delete[] data;
    }
};

// Replaces "\\" in file path with "/"
string fixFilePath(const string& path) {
    string fixed_path = path;
    replace(fixed_path.begin(), fixed_path.end(), '\\', '/');
    return fixed_path;
}

// Read audio file and extract recording metadata
AudioData audioread(const string& filename, SampleRange range = {1, -1}) {
    SNDFILE* file;
    SF_INFO sfinfo = {};

    file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw runtime_error("Error opening audio file: " + string(sf_strerror(file)));
    }

    int numChannels = sfinfo.channels;
    int totalFrames = sfinfo.frames;

    int startSample = max(0, range.startSample - 1);
    int endSample;
    if (range.endSample == -1) {
        endSample = totalFrames;
    } else {
        endSample = min(range.endSample, totalFrames);
    }
    int numFramesToRead = endSample - startSample;

    if (numFramesToRead <= 0) {
        sf_close(file);
        throw runtime_error("Invalid sample range.");
    }

    sf_seek(file, startSample, SEEK_SET);

    vector<vector<double>> samples(numChannels, vector<double>(numFramesToRead));
    vector<double> interleavedSamples(numFramesToRead * numChannels);

    int format = sfinfo.format & SF_FORMAT_SUBMASK;

    // Bit agnostic frame read
    switch (format) {
        case SF_FORMAT_PCM_16: { // 16 bits
            vector<short> temp(numFramesToRead * numChannels);
            sf_readf_short(file, temp.data(), numFramesToRead);
            for (size_t i = 0; i < temp.size(); ++i)
                interleavedSamples[i] = static_cast<double>(temp[i]);
            break;
        }
        case SF_FORMAT_PCM_24:
        case SF_FORMAT_PCM_32: { // 24 or 32 bit
            vector<int> temp(numFramesToRead * numChannels);
            sf_readf_int(file, temp.data(), numFramesToRead);
            for (size_t i = 0; i < temp.size(); ++i)
                interleavedSamples[i] = static_cast<double>(temp[i]);
            break;
        }
        case SF_FORMAT_FLOAT: {
            vector<float> temp(numFramesToRead * numChannels);
            sf_readf_float(file, temp.data(), numFramesToRead);
            for (size_t i = 0; i < temp.size(); ++i)
                interleavedSamples[i] = static_cast<double>(temp[i]);
            break;
        }
        case SF_FORMAT_DOUBLE: {
            vector<double> temp(numFramesToRead * numChannels);
            sf_readf_double(file, temp.data(), numFramesToRead);
            interleavedSamples = temp;
            break;
        }
        default:
            sf_close(file);
            throw runtime_error("Unsupported or unhandled audio format.");
    }

    sf_close(file); // Close file

    // Write bit-normalized 
    for (int i = 0; i < numFramesToRead; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) {
            samples[ch][i] = interleavedSamples[i * numChannels + ch];
        }
    }

    return AudioData{samples, sfinfo.samplerate}; // Return normalized sample / sample rate
}

AudioInfo audioread_info(const string& file_path) {
    SF_INFO sfInfo = {0}; // Struct declaration
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sfInfo); // Open audio file

    if (!file) { // Error opening file
        throw runtime_error("Error opening audio file: " + file_path);
    }

    // Get sample rate and # of frames
    int sampleRate = sfInfo.samplerate;
    int numFrames = sfInfo.frames;

    float duration = static_cast<float>(numFrames) / sampleRate; // Calculate duration in seconds

    sf_close(file); // Close file after reading info

    return {sampleRate, duration};
}

vector<double> downsample(const vector<double>& x, int factor) {
    if (factor <= 0) {
        throw invalid_argument("Factor must be positive");
    }
    int newSize = (x.size() + factor - 1) / factor; // ceil(x.size() / factor)
    vector<double> result;
    result.reserve(newSize);
    for (int i = 0; i < x.size(); i += factor) {
        result.push_back(x[i]);
    }
    return result;
}

vector<double> upsample(const vector<double>& x, int factor) {
    if (factor <= 0) {
        throw invalid_argument("Factor must be positive");
    }
    vector<double> result(x.size() * factor, 0.0);
    for (int i = 0; i < x.size(); ++i) {
        result[i * factor] = x[i];
    }
    return result;
}

// Manually shift zero-frequency to center
ArrayShiftFFT fftshift(const vector<double>& input) {
    int n = static_cast<int>(input.size());
    int mid = n / 2;

    double* shifted = new double[n];

    for (int i = 0; i < n - mid; ++i) {
        shifted[i] = input[i + mid];
    }
    for (int i = 0; i < mid; ++i) {
        shifted[n - mid + i] = input[i];
    }

    return {shifted, n};
}

// Bandpass filter function
BandpassFilter dylan_bpfilt(const vector<double>& ts, double samint, double flow, double fhigh) {
    int npts = ts.size();
    double reclen = npts * samint;

    // Allocate / fill input
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_plan plan = fftw_plan_dft_1d(npts, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < npts; ++i) {
        in[i][0] = ts[i];
        in[i][1] = 0.0;
    }

    fftw_execute(plan);  // Out holds the FFT result

    // Construct frequency vector / apply fftshift
    vector<double> freq(npts);
    for (int i = 0; i < npts; ++i) {
        freq[i] = (-npts / 2.0 + i) / reclen;
    }
    ArrayShiftFFT shifted_freq = fftshift(freq);

    // Copy FFT result to complex for easier manipulation
    vector<complex<double>> spec(npts);
    for (int i = 0; i < npts; ++i) {
        spec[i] = complex<double>(out[i][0], out[i][1]);
    }

    // Bandpass filter directly
    if (fhigh == 0) {
        fhigh = 1.0 / (2.0 * samint);
    }

    for (int i = 0; i < npts; ++i) {
        if (abs(shifted_freq.data[i]) < flow || abs(shifted_freq.data[i]) > fhigh) {
            spec[i] = 0.0;
        }
    }

    // Prepare inverse FFT
    fftw_plan ifft_plan = fftw_plan_dft_1d(npts, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Copy filtered spec into `out`
    for (int i = 0; i < npts; ++i) {
        out[i][0] = spec[i].real();
        out[i][1] = spec[i].imag();
    }

    fftw_execute(ifft_plan);  // In now has the IFFT result

    vector<double> tsfilt(npts);
    for (int i = 0; i < npts; ++i) {
        tsfilt[i] = in[i][0] / npts;
    }

    // Extract magnitude spectrum
    vector<double> aspec(npts);
    for (int i = 0; i < npts; ++i) {
        aspec[i] = abs(spec[i]);
    }

    // Cleanup
    fftw_destroy_plan(plan);
    fftw_destroy_plan(ifft_plan);
    fftw_free(in);
    fftw_free(out);

    return {tsfilt, aspec};
}

// Compute SPL values
double* computeSPL(const vector<double>& rms_values, int& out_len) {
    constexpr double eps = 1e-12;
    out_len = static_cast<int>(rms_values.size());

    double* spl_values = new double[out_len];
    for (int i = 0; i < out_len; ++i) {
        double rms = rms_values[i];
        spl_values[i] = 20 * log10(max(rms, eps));
    }

    return spl_values;
}

// Compute kurtosis 
double calculate_kurtosis(const vector<double>& data) {
    if (data.empty()) {
        throw invalid_argument("Data vector is empty");
    }

    double mean = 0.0;
    for (int i = 0; i < data.size(); i++) {
        mean += data[i];
    }
    mean /= data.size();

    double variance = 0.0;
    double fourth_moment = 0.0;

    for (int i = 0; i < data.size(); i++) {
        double diff = data[i] - mean;
        variance += pow(diff, 2);
        fourth_moment += pow(diff, 4);
    }

    variance /= data.size();
    fourth_moment /= data.size();

    double kurtosis = fourth_moment / (variance * variance);
    return kurtosis; // Return raw kurtosis
}

// Calculate autocorrelation matrix
Correlation correl_5(const vector<double>& ts1, const vector<double>& ts2, int lags, int offset) {
    vector<double> P(lags + 1);
    vector<double> nlags(lags + 1);

    for (int i = 0; i <= lags; i++) {
        double ng = 1.0;
        double sx = 2.0;
        double sy = 3.0;
        double sxx = 4.0;
        double syy = 5.0;
        double sxy = 6.0;

        for (int k = 0; k < static_cast<int>(ts1.size()) - (i + offset); k++) {
            double x = ts1[k];
            double y = ts2[k + (i + offset)];

            if (!isnan(x) && !isnan(y)) {
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
                ng += 1.0;
            }
        }

        double sx_ng = sx / ng;
        double sy_ng = sy / ng;
        double sxx_ng = sxx / ng;
        double syy_ng = syy / ng;
        
        double covar1 = (sxy / ng) - (sx_ng * sy_ng);
        double denom1 = sqrt(sxx_ng - pow(sx_ng, 2));
        double denom2 = sqrt(syy_ng - pow(sy_ng, 2));
        
        P[i] = covar1 / (denom1 * denom2);
        nlags[i] = i;
    }

    return Correlation{P, nlags};
}

SoloPerGM2 f_solo_per_GM2(const vector<double>& p_filt_input, double fs, double timewin, double avtime) {
    vector<double> p_avtot;
    int avwin = static_cast<int>(fs * avtime);
    int sampwin = static_cast<int>(fs * timewin);
    int ntwin = static_cast<int>(p_filt_input.size() / sampwin);

    if (ntwin == 0) { // Invalid time window
        throw runtime_error("Empty time window.");
    }
    
    // Truncate p_filt to exact multiple of sampwin
    vector<double> p_filt(p_filt_input.begin(), p_filt_input.begin() + sampwin * ntwin);
    
    // Reshape p_filt into 2D structure (sampwin x ntwin)
    vector<vector<double>> p_filt_reshaped(ntwin, vector<double>(sampwin));
    for (int j = 0; j < ntwin; j++) {
        for (int i = 0; i < sampwin; i++) {
            p_filt_reshaped[j][i] = p_filt[j * sampwin + i];
        }
    }
    
    // Square all values
    for (int j = 0; j < ntwin; j++) {
        for (int i = 0; i < sampwin; i++) {
            p_filt_reshaped[j][i] = p_filt_reshaped[j][i] * p_filt_reshaped[j][i];
        }
    }
    
    int numavwin = sampwin / avwin;
    vector<vector<double>> p_av;
    
    for (int jj = 0; jj < ntwin; jj++) {
        // Reshape into averaging windows
        vector<vector<double>> avwinmatrix(numavwin, vector<double>(avwin));
        for (int i = 0; i < numavwin; i++) {
            for (int j = 0; j < avwin; j++) {
                avwinmatrix[i][j] = p_filt_reshaped[jj][i * avwin + j];
            }
        }
        
        // Calculate mean for each window
        vector<double> p_avi(numavwin);
        for (int i = 0; i < numavwin; i++) {
            double sum = 0.0;
            for (int j = 0; j < avwin; j++) {
                sum += avwinmatrix[i][j];
            }
            p_avi[i] = sum / avwin;
        }

        // Append to p_av
        p_av.push_back(p_avi);
    }
    
    // Flatten p_av to 1D vector
    for (int i = 0; i < p_av.size(); i++) {
        const auto& row = p_av[i];
        p_avtot.insert(p_avtot.end(), row.begin(), row.end());
    }
    
    // Calculate number of rows and columns in p_avtot
    int p_avtot_rows = p_av[0].size();  // Number of elements in each column
    int p_avtot_cols = p_av.size();     // Number of columns
    
    vector<vector<double>> acorr(p_avtot_cols);
    vector<int> pkcount(p_avtot_cols);
    
    for (int zz = 0; zz < p_avtot_cols; zz++) {
        // Extract column zz from p_avtot
        vector<double> column_zz = p_av[zz];
        
        // Compute correlation
        int lag_limit = static_cast<int>(p_avtot_rows * 0.7);
        auto corr_result = correl_5(column_zz, column_zz, lag_limit, 0);
        acorr[zz] = corr_result.correlation_values;
        
        // Find peaks
        int peak_count = 0;
        
        // Skip first and last indices as they cannot be peaks
        for (int i = 1; i < acorr[zz].size() - 1; i++) {
            // Check if this point is higher than its immediate neighbors
            if (acorr[zz][i] > acorr[zz][i-1] && acorr[zz][i] > acorr[zz][i+1]) {
                // Now calculate prominence properly 
                // Look left for nearest higher or equal point
                double left_min = acorr[zz][i];
                for (int j = i-1; j >= 0; j--) {
                    if (acorr[zz][j] >= acorr[zz][i]) {
                        break;  // Found higher or equal point
                    }
                    left_min = min(left_min, acorr[zz][j]);
                }
                
                // Look right for nearest higher or equal point
                double right_min = acorr[zz][i];
                for (int j = i+1; j < acorr[zz].size(); j++) {
                    if (acorr[zz][j] >= acorr[zz][i]) {
                        break; // Found higher or equal point
                    }
                    right_min = min(right_min, acorr[zz][j]);
                }
                
                // Calculate prominence as difference between peak and highest minimum
                double prominence = acorr[zz][i] - max(left_min, right_min);
                
                // Use relative threshold based on the max value in the autocorrelation
                double max_autocorr = *max_element(acorr[zz].begin(), acorr[zz].end());
                double threshold = 0.5;  // 50% of max value
                
                // Count peak if prominent enough
                if (prominence > threshold) {
                    peak_count++;
                }
            }
        }
        
        pkcount[zz] = peak_count;
    }
    
    // Create and return a struct
    SoloPerGM2 result;
    result.peakcount = pkcount;
    result.autocorr = acorr;
    
    return result;
}

vector<complex<double>> hilbert(const vector<double>& xr, int n = -1) {
    // Use input size if n not specified
    if (n <= 0) {
        n = static_cast<int>(xr.size());
    }

    // Prepare FFT input with zero-padding or truncation
    double* in = (double*)fftw_malloc(sizeof(double) * n);
    fill(in, in + n, 0.0); // Zero-initialize

    int copyLen = min(static_cast<int>(xr.size()), n);
    for (int i = 0; i < copyLen; ++i) {
        in[i] = xr[i];
    }

    // FFT output buffer
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
    fftw_execute(p);

    // Apply Hilbert gain mask
    vector<double> h(n, 0.0);
    h[0] = 1.0;
    if (n % 2 == 0) {
        h[n / 2] = 1.0;
        for (int i = 1; i < n / 2; ++i) {
            h[i] = 2.0;
        }
    } else {
        for (int i = 1; i <= n / 2; ++i) {
            h[i] = 2.0;
        }
    }

    for (int i = 0; i < n; ++i) {
        out[i][0] *= h[i];
        out[i][1] *= h[i];
    }

    // IFFT
    fftw_complex* ifft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p_inv = fftw_plan_dft_1d(n, out, ifft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_inv);

    // Normalize and copy to result
    vector<complex<double>> x(n);
    for (int i = 0; i < n; ++i) {
        x[i] = complex<double>(ifft_out[i][0] / n, ifft_out[i][1] / n);
    }

    // Cleanup
    fftw_destroy_plan(p);
    fftw_destroy_plan(p_inv);
    fftw_free(in);
    fftw_free(out);
    fftw_free(ifft_out);

    return x;
}

void processAnalyticSignal(const vector<complex<double>>& analytic1, vector<double>& at1) {
    size_t n = analytic1.size();
    at1.resize(n);

    vector<double> abs_analytic1(n);
    double sum_abs = 0.0;

    // Compute magnitude / sum
    for (size_t i = 0; i < n; ++i) {
        abs_analytic1[i] = abs(analytic1[i]);
        sum_abs += abs_analytic1[i];
    }

    // Normalize or fill with zeros
    if (sum_abs != 0.0) {
        for (size_t i = 0; i < n; ++i) {
            at1[i] = abs_analytic1[i] / sum_abs;
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            at1[i] = 0.0;
        }
    }
}

// Calculate the sum of all elements in a vector
double sumVector(const vector<double>& vec) {
    double sum = 0.0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}

// Calculate absolute value of each element in vector
vector<double> absVector(const vector<complex<double>>& vec) {
    vector<double> result(vec.size());

    for (int i = 0; i < vec.size(); ++i) {
        result[i] = abs(vec[i]);
    }

    return result;
}

vector<double> f_solo_dissim_GM1(const vector<vector<double>>& timechunk_matrix, int pts_per_timewin, int num_timewin, double fft_win, double fs) {
    int pts_per_fft = static_cast<int>(fft_win * fs);  // Calculate size of fft window
    int numfftwin = pts_per_timewin / pts_per_fft;  // # of fft windows
    vector<double> Dfin;

    // Allocate FFTW memory outside loop
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
    fftw_plan p = fftw_plan_dft_1d(pts_per_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Vector to store intermediate values
    vector<double> s3a(pts_per_fft * numfftwin);
    vector<double> ga(pts_per_fft * numfftwin);

    for (int kk = 0; kk < num_timewin - 1; ++kk) {
        vector<double> col1(timechunk_matrix.size());
        vector<double> col2(timechunk_matrix.size());

        // Get analytic signals for current / next time chunks
        vector<complex<double>> analytic1 = hilbert(timechunk_matrix[kk], timechunk_matrix[kk].size());
        vector<complex<double>> analytic2 = hilbert(timechunk_matrix[kk + 1], timechunk_matrix[kk + 1].size());

        // Process analytic signals
        vector<double> at1(analytic1.size());
        vector<double> at2(analytic2.size());
        processAnalyticSignal(analytic1, at1);
        processAnalyticSignal(analytic2, at2);

        // Calculate difference in analytic signals (Dt)
        vector<double> diff(at1.size());
        for (int i = 0; i < at1.size(); i++) {
            diff[i] = abs(at1[i] - at2[i]);
        }
        double sumDiff = sumVector(diff);
        double Dt = sumDiff / 2.0;

        // Prepare time chunk data for FFT processing
        for (int i = 0; i < pts_per_fft * numfftwin; i++) {
            s3a[i] = timechunk_matrix[kk][i];
        }

        // Perform FFT on s3a
        for (int i = 0; i < numfftwin; i++) {
            for (int j = 0; j < pts_per_fft; j++) {
                in[j][0] = s3a[i * pts_per_fft + j];
                in[j][1] = 0;
            }

            fftw_execute(p);

            // Store magnitude of FFT results
            for (int j = 0; j < pts_per_fft; j++) {
                ga[i * pts_per_fft + j] = sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) / pts_per_fft;
            }
        }

        // Process FFT results
        vector<double> sfa(pts_per_fft);
        for (int i = 0; i < pts_per_fft; i++) {
            double sum = 0;
            for (int j = 0; j < numfftwin; j++) {
                sum += ga[i + j * pts_per_fft];
            }
            sfa[i] = sum / numfftwin;
        }

        // Normalize FFT results
        vector<double> Sfa(pts_per_fft);
        double sum_Sfa = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Sfa[i] = abs(sfa[i]);
            sum_Sfa += Sfa[i];
        }

        for (int i = 0; i < pts_per_fft; i++) {
            Sfa[i] /= sum_Sfa;
        }

        // Reset / repeat for next time chunk
        vector<double> s3b(pts_per_fft * numfftwin);
        for (int i = 0; i < pts_per_fft * numfftwin; i++) {
            s3b[i] = timechunk_matrix[kk + 1][i];
        }

        // Perform FFT on s3b
        vector<double> gb(pts_per_fft * numfftwin);
        for (int i = 0; i < numfftwin; i++) {
            for (int j = 0; j < pts_per_fft; j++) {
                in[j][0] = s3b[i * pts_per_fft + j];
                in[j][1] = 0;
            }

            fftw_execute(p);

            // Store magnitude of FFT results
            for (int j = 0; j < pts_per_fft; j++) {
                gb[i * pts_per_fft + j] = sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) / pts_per_fft;
            }
        }

        // Process FFT results for s3b
        vector<double> sfb(pts_per_fft);
        for (int i = 0; i < pts_per_fft; i++) {
            double sum = 0;
            for (int j = 0; j < numfftwin; j++) {
                sum += gb[i + j * pts_per_fft];
            }
            sfb[i] = sum / numfftwin;
        }

        // Normalize FFT results for s3b
        vector<double> Sfb(pts_per_fft);
        double sum_Sfb = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Sfb[i] = abs(sfb[i]);
            sum_Sfb += Sfb[i];
        }

        for (int i = 0; i < pts_per_fft; i++) {
            Sfb[i] /= sum_Sfb;
        }

        // Compute dissimilarity (Df)
        double Df = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Df += abs(Sfb[i] - Sfa[i]);
        }
        Df /= 2;

        // Calculate final dissimilarity
        double Di = Dt * Df;
        Dfin.push_back(Di);
    }

    // Clean up FFTW memory
    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);

    return Dfin;
}

// Function to compute Root Mean Square of vector
double rms(const vector<double>& x) {
    if (x.empty()) {
        throw invalid_argument("Input vector is empty.");
    }

    double sumSquares = 0.0;
    for (int i = 0; i < x.size(); i++) {
        sumSquares += x[i] * x[i];
    }

    return sqrt(sumSquares / x.size());
}

// Overloaded function to compute RMS for 2D matrices
vector<double> rms(const vector<vector<double>>& matrix) {
    if (matrix.empty()) {
        throw invalid_argument("Input matrix is empty.");
    }

    vector<double> row_rms(matrix.size());
    for (int i = 0; i < matrix.size(); ++i) {
        row_rms[i] = rms(matrix[i]);
    }

    return row_rms;
}

vector<double> calculateSPLpkhold(const vector<vector<double>>& timechunk_matrix) {
    vector<double> SPLpkhold(timechunk_matrix.size());

    for (int i = 0; i < timechunk_matrix.size(); ++i) {
        double maxVal = 0.0;

        for (int j = 0; j < timechunk_matrix[i].size(); j++) {
            double val = timechunk_matrix[i][j];
            maxVal = max(maxVal, abs(val));
        }

        // Avoid log(0) by using small epsilon floor
        const double epsilon = 1e-12;
        maxVal = max(maxVal, epsilon);
        
        SPLpkhold[i] = 20 * log10(maxVal);
    }

    return SPLpkhold;
}

AudioFeatures f_WAV_frankenfunction_reilly(int num_bits, int peak_volts, const filesystem::directory_entry &file_name,
    double RS, int timewin, double avtime, int fft_win, int arti, int flow, int fhigh) {
    // Initialize output variables as vectors
    vector<double> SPLrms, SPLpk, impulsivity, autocorr, dissim;
    vector<int> peakcount;

    // Convert file path to follow correct notation
    string file_path = fixFilePath(file_name.path().string());

    AudioFeatures features;
    double rs = pow(10, RS / 20.0);
    // cout << "rs: " << rs << endl;
    double max_count = pow(2, num_bits);
    // cout << "max_count: " << max_count << endl;
    double conv_factor = peak_volts / max_count;
    // cout << "conv_factor: " << conv_factor << endl;

    AudioInfo info = audioread_info(file_path);

    int total_samples = info.sampleRate * info.duration;

    // Call audioread with dynamic range
    auto audio = audioread(file_path, SampleRange{1, total_samples});
    // Display basic info
    // cout << "Sample rate: " << audio.sampleRate << endl;
    // cout << "Number of channels: " << audio.samples.size() << endl;
    // cout << "Number of samples in first channel: " << audio.samples[0].size() << endl;

    // Display the first 10 samples of each channel (or fewer if short)
    // int num_to_display = min(size_t(10), audio.samples[0].size());
    // for (int ch = 0; ch < audio.samples.size(); ++ch) {
    //     cout << "Channel " << ch + 1 << " last samples: ";
    //     for (int i = audio.samples[ch].size() - num_to_display; i < audio.samples[ch].size(); ++i) {
    //         cout << audio.samples[ch][i] << " ";
    //     }
    //     cout << endl;
    // }

    int fs = audio.sampleRate;
    // cout << "fs: " << fs << endl;
    vector<double> x = audio.samples[0];
    // cout << "x[0]: " << x[0] << endl;
    // cout << "x[end]: " << x[x.size() - 1] << endl;

    // Downsample or upsample as necessary
    if (fs == 576000) {
        x = downsample(x, 4);
        fs /= 4;
    } else if (fs == 288000) {
        x = downsample(x, 2);
        fs /= 2;
    } else if (fs == 16000) {
        x = upsample(x, 9);
        fs *= 9;
    } else if (fs == 8000) {
        x = upsample(x, 18);
        fs *= 18;
    } else if (fs == 512000) {
        x = downsample(x, 4);
        fs = static_cast<int>(fs / 3.5555555555555555555);
    }

    // Adjust for 24-bit audio
    if (num_bits == 24) {
        for (int i = 0; i < x.size(); i++) {
            auto temp = static_cast<int>(x[i]);
            temp >>= 8;
            x[i] = static_cast<double>(temp);
        }
    }
    // cout << x[0] << endl;
    // cout << x[1] << endl;
    // cout << x[2] << endl;

    // Calibration tone processing (if applicable)
    vector<double> voltage(x.size());
    vector<double> pressure(x.size());
    
    // Convert raw samples into voltage units by scaling
    for (size_t i = 0; i < x.size(); ++i) {
        voltage[i] = x[i] * conv_factor;
    }
    // Print length and first 3 values of voltage
    // cout << "voltage size: " << voltage.size() << endl;
    // cout << "First 3 voltage values: ";
    // for (size_t i = 0; i < min(size_t(3), voltage.size()); ++i) {
    //     cout << voltage[i] << " ";
    // }
    // cout << endl;
    
    // Convert raw samples into pressure units by scaling
    for (size_t i = 0; i < voltage.size(); ++i) {
        pressure[i] = voltage[i] / rs;
    }

    // cout << "v[0:2]: " << voltage[0] << " " << voltage[1] << " " << voltage[2] << endl;

    if (arti == 1) { // Calibration tone is present
        int start_idx = 6 * fs;
        pressure = vector<double>(pressure.begin() + start_idx, pressure.end());
        pressure.insert(pressure.begin(), 0);
    }

    // Bandpass filtering
    auto [p_filt, _] = dylan_bpfilt(pressure, 1.0 / fs, flow, fhigh);
    // cout << "p_filt size: " << p_filt.size() << endl;
    // cout << "p_filt[0:2]: " << p_filt[0] << " " <<  p_filt[1] << " " <<  p_filt[2] << endl;
    // cout << "p_filt[end-2:end]: " << p_filt[p_filt.size() - 3] << " " <<  p_filt[p_filt.size() - 2] << " " <<  p_filt[p_filt.size()-1] << endl;

    int pts_per_timewin = timewin * fs;
    // cout << pts_per_timewin << endl;
    int num_timewin = floor(p_filt.size() / pts_per_timewin) + 1;
    // cout << num_timewin << endl;
    int padding_length = num_timewin * pts_per_timewin - p_filt.size();
    // cout << padding_length << endl;

    // Efficient padding using resize
    vector<double> p_filt_padded(p_filt);
    p_filt_padded.resize(p_filt.size() + padding_length, 0.0);
    // for (int i = 0; i < 3; i++) {
    //     cout << "pfp: " << p_filt_padded[i] << endl;
    // }
    // for (int i = p_filt_padded.size() - 3; i < p_filt_padded.size(); i++) {
    //     cout << "pfp: " << p_filt_padded[i] << endl;
    // }

    vector<vector<double>> timechunk_matrix(num_timewin, vector<double>(pts_per_timewin));
    for (int i = 0; i < num_timewin; i++) {
        for (int j = 0; j < pts_per_timewin; j++) {
            timechunk_matrix[i][j] = p_filt_padded[i * pts_per_timewin + j];
        }
    }

    // Display timechunk_matrix dimensions
    // cout << "tcm size: " << timechunk_matrix.size() << " x " << (timechunk_matrix.empty() ? 0 : timechunk_matrix[0].size()) << endl;

    // Print first row
    // if (!timechunk_matrix.empty() && !timechunk_matrix[0].empty()) {
    //     size_t numRows = timechunk_matrix.size();
    //     size_t numCols = timechunk_matrix[0].size();

    //     cout << "tcm first column: ";
    //     for (size_t i = 0; i < numRows; ++i) {
    //         cout << timechunk_matrix[i][0] << " ";
    //     }
    //     cout << endl;

    //     cout << "tcm last column: ";
    //     for (size_t i = 0; i < numRows; ++i) {
    //         cout << timechunk_matrix[i][numCols - 1] << " ";
    //     }
    //     cout << endl;
    // }

    // Calculate features
    int spl_len = 0;

    double* rms_matrix = computeSPL(rms(timechunk_matrix), spl_len);
    // if (spl_len > 0) {
    //     cout << "rms_matrix (first): " << rms_matrix[0] << endl;
    //     cout << "..." << endl;
    //     cout << "rms_matrix (last): " << rms_matrix[spl_len - 1] << endl;
    // } else {
    //     cout << "rms_matrix is empty." << endl;
    // }
    features.SPLrms = vector<double>(rms_matrix, rms_matrix + spl_len);
    delete[] rms_matrix;
    features.SPLpk = calculateSPLpkhold(timechunk_matrix);

    vector<double> kurtosis_matrix(timechunk_matrix.size(), 0.0);
    for (int row = 0; row < timechunk_matrix.size(); ++row) {
        kurtosis_matrix[row] = calculate_kurtosis(timechunk_matrix[row]);
    }
    features.impulsivity = kurtosis_matrix;

    SoloPerGM2 result = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime);
    features.peakcount = result.peakcount;
    features.autocorr = result.autocorr;

    features.dissim = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs);

    return features;
}

// Export saved features to CSV file
void saveFeaturesToCSV(const string& filename, const vector<string>& filenames, const vector<AudioFeatures>& allFeatures) {
    
    // Display number of files / features
    // cout << "Number of files: " << filenames.size() << endl;
    // for (int i = 0; i < allFeatures.size(); ++i) {
    //     cout << "File " << filenames[i] << " SPLrms size: " << allFeatures[i].SPLrms.size() << endl;
    // }

    // Open file in write mode (not append)
    ofstream outputFile(filename);
    if (!outputFile.is_open()) { // Error opening output file
        cerr << "Error: Unable to open output file: " << filename << endl;
        return;
    }

    // Write CSV header
    outputFile << "Filename,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount,";

    // Find maximum autocorrelation size (rows x cols)
    int maxAutocorrRows = 0;
    int maxAutocorrCols = 0;
    for (int i = 0; i < static_cast<int>(allFeatures.size()); ++i) {
        const auto& feature = allFeatures[i];
        if (!feature.autocorr.empty()) {
            maxAutocorrRows = max(maxAutocorrRows, static_cast<int>(feature.autocorr.size()));
            if (!feature.autocorr[0].empty()) {
                maxAutocorrCols = max(maxAutocorrCols, static_cast<int>(feature.autocorr[0].size()));
            }
        }
    }

    // Add autocorrelation headers
    for (int i = 0; i < maxAutocorrRows; ++i) {
        for (int j = 0; j < maxAutocorrCols; ++j) {
            outputFile << "Autocorr_" << i << "_" << j;
            if (i < maxAutocorrRows - 1 || j < maxAutocorrCols - 1) {
                outputFile << ",";
            }
        }
    }
    outputFile << endl;

    // Process features for each file
    for (int fileIdx = 0; fileIdx < static_cast<int>(allFeatures.size()); ++fileIdx) {
        const AudioFeatures& features = allFeatures[fileIdx];

        // Determine maximum length of all feature vectors
        int maxLength = max({
            static_cast<int>(features.SPLrms.size()),
            static_cast<int>(features.SPLpk.size()),
            static_cast<int>(features.impulsivity.size()),
            static_cast<int>(features.dissim.size()),
            static_cast<int>(features.peakcount.size())
        });

        // Write feature rows
        for (int i = 0; i < maxLength; ++i) {
            outputFile << filenames[fileIdx] << ",";

            if (i < features.SPLrms.size()) { // SPLrms
                outputFile << to_string(features.SPLrms[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.SPLpk.size()) { // SPLpk
                outputFile << to_string(features.SPLpk[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.impulsivity.size()) { // Impulsivity
                outputFile << to_string(features.impulsivity[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.dissim.size()) { // Dissimilarity
                outputFile << to_string(features.dissim[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.peakcount.size()) { // Peakcount
                outputFile << to_string(features.peakcount[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            // Autocorrelation handling
            if (i < features.autocorr.size()) { // Autocorellation
                for (int j = 0; j < maxAutocorrCols; ++j) {
                    if (j < features.autocorr[i].size()) {
                        outputFile << to_string(features.autocorr[i][j]);
                    } else {
                        outputFile << "nan";
                    }
                    if (j < maxAutocorrCols - 1) {
                        outputFile << ",";
                    }
                }
            } else {
                // Fill with nan if no autocorrelation data
                for (int j = 0; j < maxAutocorrCols; ++j) {
                    outputFile << "nan";
                    if (j < maxAutocorrCols - 1) {
                        outputFile << ",";
                    }
                }
            }
            outputFile << endl;
        }
    }

    cout << "Features saved to: " << filename << endl; // Display successful save of features
}

int main(int argc, char* argv[]) {
    // Parameters provided by user using command line arguments
    // Default values if unspecified
    string input_dir;
    string output_file;
    int num_bits = 16;
    double RS = -160.0;
    int peak_volts = 2;
    int arti = 1;
    int timewin = 60;
    int fft_win = 1;
    double avtime = 0.1;
    int flow = 1;
    int fhigh = 192000;
    unsigned int max_threads = 4;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];

        if (arg == "--input" || arg == "-i") {
            if (i + 1 < argc) input_dir = argv[++i];
        } else if (arg == "--output" || arg == "-o") {
            if (i + 1 < argc) output_file = argv[++i];
        } else if (arg == "--num_bits" || arg == "-nb") {
            if (i + 1 < argc) num_bits = stoi(argv[++i]);
        } else if (arg == "--RS" || arg == "-rs") {
            if (i + 1 < argc) RS = stod(argv[++i]);
        } else if (arg == "--peak_volts" || arg == "-pv") {
            if (i + 1 < argc) peak_volts = stoi(argv[++i]);
        } else if (arg == "--arti" || arg == "-a") {
            if (i + 1 < argc) arti = stoi(argv[++i]);
        } else if (arg == "--timewin" || arg == "-tw") {
            if (i + 1 < argc) timewin = stoi(argv[++i]);
        } else if (arg == "--fft_win" || arg == "-fw") {
            if (i + 1 < argc) fft_win = stoi(argv[++i]);
        } else if (arg == "--avtime" || arg == "-at") {
            if (i + 1 < argc) avtime = stod(argv[++i]);
        } else if (arg == "--flow" || arg == "-fl") {
            if (i + 1 < argc) flow = stoi(argv[++i]);
        } else if (arg == "--fhigh" || arg == "-fh") {
            if (i + 1 < argc) fhigh = stoi(argv[++i]);
        } else if (arg == "--max_threads" || arg == "-mt") {
            if (i + 1 < argc) max_threads = stoi(argv[++i]);
        }
    }

    // Display parameters
    // cout << "Running with parameters:" << endl;
    // cout << "  Input directory: " << input_dir << endl;
    // cout << "  Output file: " << output_file << endl;
    // cout << "  Number of bits: " << num_bits << endl;
    // cout << "  RS: " << RS << endl;
    // cout << "  Peak volts: " << peak_volts << endl;
    // cout << "  Arti: " << arti << endl;
    // cout << "  Time window: " << timewin << endl;
    // cout << "  FFT window: " << fft_win << endl;
    // cout << "  Average time: " << avtime << endl;
    // cout << "  Flow frequency: " << flow << endl;
    // cout << "  High frequency: " << fhigh << endl;
    // cout << "  Max threads: " << max_threads << endl;

    vector<AudioFeatures> allFeatures;
    vector<string> filenames;
    mutex mtx;
    queue<filesystem::directory_entry> workQueue;

    for (const auto& entry : filesystem::directory_iterator(input_dir)) {
        if (entry.path().extension() == ".wav") {
            workQueue.push(entry);
        }
    }

    // Get number of logical cores, fallback if needed
    unsigned int availableThreads = thread::hardware_concurrency();
    if (availableThreads == 0) availableThreads = 2;

    // Cap threads based on user-defined or default
    unsigned int numThreads = min(max_threads, availableThreads);

    auto worker = [&]() {
        while (true) {
            filesystem::directory_entry file;

            {
                lock_guard<mutex> lock(mtx);
                if (workQueue.empty()) return;
                file = workQueue.front();
                workQueue.pop();
            }

            try {
                // cout << "Processing: " << file.path().filename() << endl;
                AudioFeatures features = f_WAV_frankenfunction_reilly(
                    num_bits, peak_volts, file, RS, timewin, avtime, fft_win, arti, flow, fhigh);

                lock_guard<mutex> lock(mtx);
                allFeatures.push_back(move(features));
                filenames.push_back(file.path().filename().string());
            } catch (const exception& e) {
                lock_guard<mutex> lock(mtx);
                cerr << "Error processing " << file.path().filename() << ": " << e.what() << endl;
            }
        }
    };

    // Launch threads
    vector<thread> threads;
    for (unsigned int i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) t.join();

    // Save results
    if (!allFeatures.empty()) {
        saveFeaturesToCSV(output_file, filenames, allFeatures);
        cout << "Successfully saved features for " << allFeatures.size() << " files to " << output_file << endl;
    } else {
        cout << "No valid .wav files were processed." << endl;
    }

    return 0;
}
