#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <sndfile.h>
#include <stdexcept>
#include <complex>
#include <numeric>
#include <fftw3.h>
#include <cstring>
#include <cstdint>
#include <thread>
#include <mutex>

using namespace std;

// Declare structs
struct SampleRange {
    int first;
    int second;
    
    SampleRange(int start = 1, int end = -1) : first(start), second(end) {}
};

// Struct for bandpass filter return
struct BandpassFilterResult {
    vector<double> filtered_timeseries;
    vector<double> amplitude_spectrum;

    // Constructor for convenient initialization
    BandpassFilterResult(const vector<double>& ts, const vector<double>& spec) 
        : filtered_timeseries(ts), amplitude_spectrum(spec) {}
};

struct CorrelationResult {
    vector<double> correlation_values;
    vector<double> lags;
};

struct AudioFeatures {
    vector<double> SPLrms;
    vector<double> SPLpk;
    vector<double> impulsivity;
    vector<int> peakcount;
    vector<vector<double>> autocorr;
    vector<double> dissim;
};

struct AudioData {
    vector<vector<double>> samples; // 2D array: samples[channel][sample]
    int sampleRate;
};

struct AudioInfo {
    int sampleRate;
    double duration;
};

class AudioReader {
public:
    static vector<double> readAudio(const string& filename, int& sampleRate, SampleRange range = {}) {
        SNDFILE* file;
        SF_INFO fileInfo = {};
        
        file = sf_open(filename.c_str(), SFM_READ, &fileInfo);
        if (!file) {
            throw runtime_error("Error opening file: " + string(sf_strerror(file)));
        }

        sampleRate = fileInfo.samplerate;
        int totalSamples = fileInfo.frames * fileInfo.channels;

        if (range.second == -1 || range.second > totalSamples) {
            range.second = totalSamples;
        }
        if (range.first < 1 || range.first > range.second) {
            sf_close(file);
            throw invalid_argument("Invalid sample range.");
        }

        int firstSample = range.first - 1;
        int sampleCount = range.second - range.first + 1;
        int frameStart = firstSample / fileInfo.channels;
        int framesToRead = sampleCount / fileInfo.channels;

        vector<double> buffer(framesToRead * fileInfo.channels);
        
        sf_seek(file, frameStart, SEEK_SET);
        sf_readf_double(file, buffer.data(), framesToRead);
        sf_close(file);

        return buffer;
    }
};

string fixFilePath(const string& path) {
    // Replaces "\\" in file path with "/"
    string fixed_path = path;
    replace(fixed_path.begin(), fixed_path.end(), '\\', '/');
    return fixed_path;
}

// Function to read audio file with options similar to MATLAB's audioread
AudioData audioread(const string& filename, pair<int, int> range = {1, -1}, const string& datatype = "int16") {
    SNDFILE* file;
    SF_INFO sfinfo;

    // Open the audio file
    file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw runtime_error("Error opening audio file.");
    }

    int numChannels = sfinfo.channels;
    int totalSamples = sfinfo.frames;

    // Handle range parameter
    int startSample = max(0, range.first - 1);
    int endSample = (range.second == -1) ? totalSamples : min(range.second, totalSamples);
    int numSamples = endSample - startSample;

    if (numSamples <= 0) {
        sf_close(file);
        throw runtime_error("Invalid sample range.");
    }

    // Read samples as int16_t
    vector<int16_t> interleavedSamples(numSamples * numChannels);
    sf_seek(file, startSample, SEEK_SET);
    sf_readf_short(file, interleavedSamples.data(), numSamples);
    sf_close(file);

    // Deinterleave and convert to double
    vector<vector<double>> samples(numChannels, vector<double>(numSamples));
    for (int i = 0; i < numSamples; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) {
            samples[ch][i] = static_cast<double>(interleavedSamples[i * numChannels + ch]);
        }
    }

    return AudioData{samples, sfinfo.samplerate};
}

AudioInfo audioread_info(const string& file_path) {
    // Declare the SF_INFO structure
    SF_INFO sfInfo = {0};

    // Open the audio file
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sfInfo);

    if (!file) {
        throw runtime_error("Error opening audio file: " + file_path);
    }

    // Get the sample rate and the number of frames
    int sampleRate = sfInfo.samplerate;
    int numFrames = sfInfo.frames;

    // Calculate the duration in seconds
    float duration = static_cast<float>(numFrames) / sampleRate;

    // Close the file after reading the info
    sf_close(file);

    // Return the gathered information
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

// Function to implement fftshift (Manually shifting zero-frequency to center)
vector<double> fftshift(const vector<double>& data) {
    int n = data.size();
    vector<double> shifted(n);
    int mid = n / 2;
    rotate_copy(data.begin(), data.begin() + mid, data.end(), shifted.begin());
    return shifted;
}

// Bandpass filter function
BandpassFilterResult dylan_bpfilt(
    const vector<double>& ts, double samint, double flow, double fhigh) {

    int npts = ts.size();
    double reclen = npts * samint;

    // Allocate and fill input
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_plan plan = fftw_plan_dft_1d(npts, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    for (int i = 0; i < npts; ++i) {
        in[i][0] = ts[i];
        in[i][1] = 0.0;
    }

    fftw_execute(plan);  // out holds the FFT result

    // Construct frequency vector and apply fftshift
    vector<double> freq(npts);
    for (int i = 0; i < npts; ++i) {
        freq[i] = (-npts / 2.0 + i) / reclen;
    }
    freq = fftshift(freq);

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
        if (abs(freq[i]) < flow || abs(freq[i]) > fhigh) {
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

    fftw_execute(ifft_plan);  // in now has the IFFT result

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

// Function to compute SPL values
vector<double> computeSPL(const vector<double>& rms_values) {
    constexpr double eps = 1e-12;
    vector<double> spl_values;
    spl_values.reserve(rms_values.size());

    for (int i = 0; i < rms_values.size(); i++) {
        double rms = rms_values[i];
        spl_values.push_back(20 * log10(max(rms, eps)));
    }

    return spl_values;
}

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

CorrelationResult correl_5(const vector<double>& ts1, const vector<double>& ts2, int lags, int offset) {
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

    return CorrelationResult{P, nlags};
}

pair<vector<int>, vector<vector<double>>> f_solo_per_GM2(
    const vector<double>& p_filt_input, double fs, double timewin, double avtime) {
    
    vector<double> p_avtot;
    int avwin = static_cast<int>(fs * avtime);
    int sampwin = static_cast<int>(fs * timewin);
    int ntwin = static_cast<int>(p_filt_input.size() / sampwin);

    if (ntwin == 0) {
        throw runtime_error("Not enough data for even one time window.");
    }
    
    // Truncate p_filt to exact multiple of sampwin
    vector<double> p_filt(p_filt_input.begin(), p_filt_input.begin() + sampwin * ntwin);
    
    // Reshape p_filt into a 2D structure (sampwin x ntwin)
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
    
    // Convert p_av to p_avtot (flatten to a 1D vector)
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
        for (int i = 1; i < acorr[zz].size() - 1; i++) {
            // Simple peak detection: higher than neighbors and prominence > 0.5
            if (acorr[zz][i] > acorr[zz][i-1] && acorr[zz][i] > acorr[zz][i+1]) {
                double prominence = min(acorr[zz][i] - acorr[zz][i-1], acorr[zz][i] - acorr[zz][i+1]);
                if (prominence > 0.5) {
                    peak_count++;
                }
            }
        }
        
        pkcount[zz] = peak_count;
    }
    
    return {pkcount, acorr};
}

vector<complex<double>> hilbert(const vector<double>& xr, int n = -1) {
    // Use input size if n not specified
    if (n <= 0) {
        n = static_cast<int>(xr.size());
    }

    // Prepare FFT input with zero-padding or truncation
    double* in = (double*)fftw_malloc(sizeof(double) * n);
    fill(in, in + n, 0.0);  // Zero-initialize
    copy(xr.begin(), xr.begin() + min(xr.size(), static_cast<size_t>(n)), in);

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

void processAnalyticSignal(const vector<complex<double>>& analytic1,
    vector<double>& at1) {
    at1.resize(analytic1.size());

    vector<double> abs_analytic1(analytic1.size());
    transform(analytic1.begin(), analytic1.end(), abs_analytic1.begin(),
        [](const complex<double>& val) { return abs(val); });

        double sum_abs = 0.0;
        for (int i = 0; i < abs_analytic1.size(); i++) {
            sum_abs += abs_analytic1[i];
        }

    if (sum_abs != 0.0) {
        transform(abs_analytic1.begin(), abs_analytic1.end(), at1.begin(),
            [sum_abs](double val) { return val / sum_abs; });
    } else {
        fill(at1.begin(), at1.end(), 0.0);
    }
}

// Function to calculate the sum of all elements in a vector
double sumVector(const vector<double>& vec) {
    double sum = 0.0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}

// Function to calculate the absolute value of each element in a vector
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

    // Allocate FFTW memory outside the loop
    fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
    fftw_plan p = fftw_plan_dft_1d(pts_per_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Vector to store intermediate values
    vector<double> s3a(pts_per_fft * numfftwin);
    vector<double> ga(pts_per_fft * numfftwin);

    for (int kk = 0; kk < num_timewin - 1; ++kk) {
        vector<double> col1(timechunk_matrix.size());
        vector<double> col2(timechunk_matrix.size());

        // Get analytic signals for current and next time chunks
        vector<complex<double>> analytic1 = hilbert(timechunk_matrix[kk], timechunk_matrix[kk].size());
        vector<complex<double>> analytic2 = hilbert(timechunk_matrix[kk + 1], timechunk_matrix[kk + 1].size());

        // Process analytic signals
        vector<double> at1(analytic1.size());
        vector<double> at2(analytic2.size());
        processAnalyticSignal(analytic1, at1);
        processAnalyticSignal(analytic2, at2);

        // Calculate Dt (difference in analytic signals)
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

        // Normalize the FFT results
        vector<double> Sfa(pts_per_fft);
        double sum_Sfa = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Sfa[i] = abs(sfa[i]);
            sum_Sfa += Sfa[i];
        }

        for (int i = 0; i < pts_per_fft; i++) {
            Sfa[i] /= sum_Sfa;
        }

        // Reset and repeat for next time chunk
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

        // Normalize the FFT results for s3b
        vector<double> Sfb(pts_per_fft);
        double sum_Sfb = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Sfb[i] = abs(sfb[i]);
            sum_Sfb += Sfb[i];
        }

        for (int i = 0; i < pts_per_fft; i++) {
            Sfb[i] /= sum_Sfb;
        }

        // Compute the dissimilarity (Df)
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

vector<double> audioread(const string& filename, int& sampleRate) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(EXIT_FAILURE);
    }

    char chunkId[4];
    file.read(chunkId, 4); // "RIFF"
    file.ignore(4);        // Chunk size
    file.read(chunkId, 4); // "WAVE"
    file.read(chunkId, 4); // "fmt "

    if (strncmp(chunkId, "fmt ", 4) != 0) {
        cerr << "Error: Invalid WAV file (no fmt chunk)" << endl;
        exit(EXIT_FAILURE);
    }

    int subchunk1Size;
    file.read(reinterpret_cast<char*>(&subchunk1Size), 4);
    int audioFormat;
    file.read(reinterpret_cast<char*>(&audioFormat), 2);
    int numChannels;
    file.read(reinterpret_cast<char*>(&numChannels), 2);
    int sampleRateRaw;
    file.read(reinterpret_cast<char*>(&sampleRateRaw), 4);
    sampleRate = static_cast<int>(sampleRateRaw);
    file.ignore(6); // Byte rate + block align
    int bitsPerSample;
    file.read(reinterpret_cast<char*>(&bitsPerSample), 2);

    if (audioFormat != 1 && audioFormat != 3) {
        cerr << "Error: Only PCM (1) or IEEE float (3) formats supported" << endl;
        exit(EXIT_FAILURE);
    }

    // Skip extra fmt bytes if any
    if (subchunk1Size > 16) {
        file.ignore(subchunk1Size - 16);
    }

    // Find "data" subchunk
    while (true) {
        file.read(chunkId, 4);
        int chunkSize;
        file.read(reinterpret_cast<char*>(&chunkSize), 4);

        if (strncmp(chunkId, "data", 4) == 0) {
            break;
        } else {
            file.ignore(chunkSize); // Skip other chunks
        }
    }

    vector<double> audioData;

    if (bitsPerSample == 16) {
        vector<int> buffer;
        int sample;
        while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
            buffer.push_back(sample);
        }
        audioData.reserve(buffer.size() / numChannels);
        for (int i = 0; i < buffer.size(); i += numChannels) {
            audioData.push_back(buffer[i] / 32768.0); // Normalize to [-1, 1]
        }
    } else if (bitsPerSample == 32 && audioFormat == 3) {
        vector<float> buffer;
        float sample;
        while (file.read(reinterpret_cast<char*>(&sample), sizeof(sample))) {
            buffer.push_back(sample);
        }
        audioData.reserve(buffer.size() / numChannels);
        for (int i = 0; i < buffer.size(); i += numChannels) {
            audioData.push_back(static_cast<double>(buffer[i])); // First channel
        }
    } else if (bitsPerSample == 24) {
        vector<int> buffer(3);
        while (file.read(reinterpret_cast<char*>(buffer.data()), 3)) {
            int sample = (buffer[2] << 16) | (buffer[1] << 8) | buffer[0];
            // Sign-extend 24-bit to 32-bit
            if (sample & 0x800000) {
                sample |= 0xFF000000;
            }
            audioData.push_back(sample / 8388608.0); // Normalize to [-1, 1]
            for (int i = 1; i < numChannels; ++i) {
                file.ignore(3); // Skip other channels
            }
        }
    }

    return audioData;
}

// Function to compute RMS (Root Mean Square) of a vector
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

// Overloaded function to compute RMS for a 2D matrix
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

        // Avoid log(0) by using a small epsilon floor
        const double epsilon = 1e-12;
        maxVal = max(maxVal, epsilon);
        
        SPLpkhold[i] = 20 * log10(maxVal);
    }

    return SPLpkhold;
}

AudioFeatures f_WAV_frankenfunction_reilly(int num_bits, int peak_volts, const filesystem::directory_entry &file_name,
    double RS, int timewin, double avtime, int fft_win, int arti, int flow, int fhigh) {
    // Initialize output variables as cell arrays
    vector<double> SPLrms, SPLpk, impulsivity, autocorr, dissim;
    vector<int> peakcount;

    // Convert and fix the file path
    string file_path = fixFilePath(file_name.path().string());

    AudioFeatures features;
    double rs = pow(10, RS / 20.0);
    double max_count = pow(2, num_bits);
    double conv_factor = peak_volts / max_count;

    AudioInfo info = audioread_info(file_path);
    vector<double> audioDataA = audioread(file_name.path().string(), info.sampleRate);

    int total_samples = info.sampleRate * info.duration;

    // Call audioread with dynamic range
    auto audio = audioread(file_path, make_pair(1, total_samples), "double");

    int fs = audio.sampleRate;
    vector<double> x = audio.samples[0];

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

    // Calibration tone processing (if applicable)
    vector<double> voltage(x.size());
    vector<double> pressure(x.size());
    
    transform(
        x.begin(), x.end(), voltage.begin(),
        [conv_factor](double sample) {
            return sample * conv_factor;
        }
    );
    
    transform(
        voltage.begin(), voltage.end(), pressure.begin(),
        [rs](double sample) {
            return sample / rs;
        }
    );

    if (arti == 1) {
        int start_idx = 6 * fs;
        pressure = vector<double>(pressure.begin() + start_idx, pressure.end());
        pressure.insert(pressure.begin(), 0);
    }

    // Bandpass filtering
    auto [p_filt, filtspec1] = dylan_bpfilt(pressure, 1.0 / fs, flow, fhigh);

    int pts_per_timewin = timewin * fs;
    int num_timewin = floor(p_filt.size() / pts_per_timewin) + 1;
    int padding_length = num_timewin * pts_per_timewin - p_filt.size();

    // Efficient padding using resize
    vector<double> p_filt_padded(p_filt);
    p_filt_padded.resize(p_filt.size() + padding_length, 0.0);

    vector<vector<double>> timechunk_matrix(num_timewin, vector<double>(pts_per_timewin));
    for (int i = 0; i < num_timewin; i++) {
        for (int j = 0; j < pts_per_timewin; j++) {
            timechunk_matrix[i][j] = p_filt_padded[i * pts_per_timewin + j];
        }
    }

    // Calculate features
    features.SPLrms = computeSPL(rms(timechunk_matrix));
    features.SPLpk = calculateSPLpkhold(timechunk_matrix);

    vector<double> kmat(timechunk_matrix.size(), 0.0);
    for (int row = 0; row < timechunk_matrix.size(); ++row) {
        kmat[row] = calculate_kurtosis(timechunk_matrix[row]);
    }
    features.impulsivity = kmat;

    auto result = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime);
    features.peakcount = get<0>(result);
    features.autocorr = get<1>(result);

    features.dissim = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs);

    return features;
}

// Helper function to save features to a CSV file
void saveFeaturesToCSV(const string& filename, const vector<string>& filenames, 
    const vector<AudioFeatures>& allFeatures) {
    
    // Debug: Print number of files and features
    cout << "Number of files: " << filenames.size() << endl;
    for (int i = 0; i < allFeatures.size(); ++i) {
        cout << "File " << filenames[i] << " SPLrms size: " << allFeatures[i].SPLrms.size() << endl;
    }

    // Open file in write mode (not append)
    ofstream outputFile(filename);
    if (!outputFile.is_open()) {
        cerr << "Error: Unable to open output file: " << filename << endl;
        return;
    }

    // Write CSV header
    outputFile << "Filename,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount,";

    // Find maximum autocorrelation size (rows x cols)
    size_t maxAutocorrRows = 0;
    size_t maxAutocorrCols = 0;
    for (int i = 0; i < allFeatures.size(); i++) {
        const auto& feature = allFeatures[i];
        if (!feature.autocorr.empty()) {
            maxAutocorrRows = max(maxAutocorrRows, feature.autocorr.size());
            if (!feature.autocorr[0].empty()) {
                maxAutocorrCols = max(maxAutocorrCols, feature.autocorr[0].size());
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

    // Process each file's features
    for (int fileIdx = 0; fileIdx < allFeatures.size(); ++fileIdx) {
        const AudioFeatures& features = allFeatures[fileIdx];

        // Determine maximum length among all feature vectors
        int maxLength = max({
            features.SPLrms.size(),
            features.SPLpk.size(),
            features.impulsivity.size(),
            features.dissim.size(),
            features.peakcount.size()
        });

        // Write feature rows
        for (int i = 0; i < maxLength; ++i) {
            outputFile << filenames[fileIdx] << ",";

            if (i < features.SPLrms.size()) {
                outputFile << to_string(features.SPLrms[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.SPLpk.size()) {
                outputFile << to_string(features.SPLpk[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.impulsivity.size()) {
                outputFile << to_string(features.impulsivity[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.dissim.size()) {
                outputFile << to_string(features.dissim[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            if (i < features.peakcount.size()) {
                outputFile << to_string(features.peakcount[i]) << ",";
            } else {
                outputFile << "nan,";
            }
            // Autocorrelation handling
            if (i < features.autocorr.size()) {
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

    cout << "Features saved to: " << filename << endl;
}

int main(int argc, char* argv[]) {
    string input_dir = "C:/Users/rakos/OneDrive/Desktop/MATLAB Code/Audio/runThisInput/long_recording";
    string output_file = "C:/Users/rakos/OneDrive/Desktop/MATLAB Code/Audio/runThisOutput/cpp_output.csv";

    if (argc > 1) {
        input_dir = argv[1];
    }
    if (argc > 2) {
        output_file = argv[2];
    }

    int num_bits = 16;
    double RS = -178.3;
    int peak_volts = 2;
    int arti = 1;
    int timewin = 60;
    int fft_win = 1;
    double avtime = 0.1;
    int flow = 1;
    int fhigh = 192000;

    vector<AudioFeatures> allFeatures;
    vector<string> filenames;

    mutex mtx;
    vector<thread> threads;

    auto dirIt = filesystem::directory_iterator(input_dir);
    vector<filesystem::directory_entry> entries(dirIt, filesystem::directory_iterator{});

    for (int i = 0; i < entries.size(); i++) {
        const auto& file_dir = entries[i];
        if (file_dir.path().extension() == ".wav") {
            threads.emplace_back([&, file_dir]() {
                try {
                    cout << "Processing: " << file_dir.path().filename() << endl;
                    AudioFeatures features = f_WAV_frankenfunction_reilly(
                        num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh);

                    lock_guard<mutex> lock(mtx);
                    allFeatures.push_back(move(features));
                    filenames.push_back(file_dir.path().filename().string());
                } catch (const exception& e) {
                    lock_guard<mutex> lock(mtx);
                    cerr << "Error processing " << file_dir.path().filename() << ": " << e.what() << endl;
                }
            });
        }
    }

    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    if (!allFeatures.empty()) {
        saveFeaturesToCSV(output_file, filenames, allFeatures);
        cout << "Successfully saved features for " << allFeatures.size() << " files to " << output_file << endl;
    } else {
        cout << "No valid .wav files were processed." << endl;
    }

    return 0;
}
