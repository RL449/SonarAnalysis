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

using namespace std;

class AudioReader {
    public:
        static std::vector<double> readAudio(const std::string& filename, int& sampleRate, std::pair<int, int> range = {1, -1}, const std::string& dataType = "double") {
            SNDFILE* file;
            SF_INFO fileInfo;
            
            fileInfo.format = 0;
            file = sf_open(filename.c_str(), SFM_READ, &fileInfo);
            if (!file) {
                throw std::runtime_error("Error opening file: " + std::string(sf_strerror(file)));
            }
    
            sampleRate = fileInfo.samplerate;
            int totalSamples = fileInfo.frames * fileInfo.channels;
            
            if (range.second == -1 || range.second > totalSamples) {
                range.second = totalSamples;
            }
            if (range.first < 1 || range.first > range.second) {
                sf_close(file);
                throw std::invalid_argument("Invalid sample range.");
            }
            
            int samplesToRead = range.second - range.first + 1;
            std::vector<double> buffer(samplesToRead);
            
            sf_seek(file, range.first - 1, SEEK_SET);
            sf_readf_double(file, buffer.data(), samplesToRead / fileInfo.channels);
            
            sf_close(file);
            return buffer;
        }
    };
    

struct AudioFeatures {
    std::vector<double> SPLrms;
    std::vector<double> SPLpk;
    std::vector<double> impulsivity;
    std::vector<int> peakcount;
    std::vector<std::vector<double>> autocorr;
    std::vector<double> dissim;
};

std::string fixFilePath(const std::string& path) {
    // Replaces "\\" in file path with "/"
    std::string fixed_path = path;
    std::replace(fixed_path.begin(), fixed_path.end(), '\\', '/');
    return fixed_path;
}

struct AudioData {
    std::vector<std::vector<double>> samples; // 2D array: samples[channel][sample]
    int sampleRate;
};

// Function to read audio file with options similar to MATLAB's audioread
AudioData audioread(const std::string& filename, std::pair<int, int> range = {1, -1}, const std::string& datatype = "int16") {
    SNDFILE* file;
    SF_INFO sfinfo;

    // Open the audio file
    file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw std::runtime_error("Error opening audio file.");
    }

    int numChannels = sfinfo.channels;
    int totalSamples = sfinfo.frames;

    // Handle range parameter
    int startSample = std::max(0, range.first - 1);
    int endSample = (range.second == -1) ? totalSamples : std::min(range.second, totalSamples);
    int numSamples = endSample - startSample;

    if (numSamples <= 0) {
        sf_close(file);
        throw std::runtime_error("Invalid sample range.");
    }

    // Read samples as int16_t
    std::vector<int16_t> interleavedSamples(numSamples * numChannels);
    sf_seek(file, startSample, SEEK_SET);
    sf_readf_short(file, interleavedSamples.data(), numSamples);
    sf_close(file);

    // Deinterleave data and convert to double
    std::vector<std::vector<double>> samples(numChannels, std::vector<double>(numSamples));
    for (int i = 0; i < numSamples; ++i) {
        for (int ch = 0; ch < numChannels; ++ch) {
            samples[ch][i] = static_cast<double>(interleavedSamples[i * numChannels + ch]);
        }
    }

    return {samples, sfinfo.samplerate};
}

struct AudioInfo {
    int sampleRate;
    double duration;
};

AudioInfo audioread_info(const std::string& file_path) {
    // Declare the SF_INFO structure
    SF_INFO sfInfo = {0};

    // Open the audio file
    SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sfInfo);

    if (!file) {
        throw std::runtime_error("Error opening audio file: " + file_path);
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

std::vector<double> downsample(const std::vector<double>& x, int factor) {
    if (factor <= 0) throw std::invalid_argument("Factor must be positive");
    std::vector<double> result;
    for (size_t i = 0; i < x.size(); i += factor) {
        result.push_back(x[i]);
    }
    return result;
}

std::vector<double> upsample(const std::vector<double>& x, int factor) {
    if (factor <= 0) throw std::invalid_argument("Factor must be positive");
    std::vector<double> result(x.size() * factor);
    for (size_t i = 0; i < x.size(); ++i) {
        result[i * factor] = x[i];
    }
    return result;
}

// Function to implement fftshift (Manually shifting zero-frequency to center)
std::vector<double> fftshift(const std::vector<double>& data) {
    int n = data.size();
    std::vector<double> shifted(n);
    int mid = n / 2;
    
    std::rotate_copy(data.begin(), data.begin() + mid, data.end(), shifted.begin());
    return shifted;
}

// Bandpass filter function
std::pair<std::vector<double>, std::vector<double>> dylan_bpfilt(
    const std::vector<double>& ts, double samint, double flow, double fhigh) {
    
    int npts = ts.size();
    double reclen = npts * samint;

    // Allocate FFTW arrays
    fftw_complex *in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_plan plan = fftw_plan_dft_1d(npts, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Fill input array for FFT
    for (int i = 0; i < npts; ++i) {
        in[i][0] = ts[i];  // Real part
        in[i][1] = 0.0;    // Imaginary part (zero for real signals)
    }

    // Perform FFT
    fftw_execute(plan);

    // Convert output into a std::vector of std::complex
    std::vector<std::complex<double>> spec(npts);
    for (int i = 0; i < npts; ++i) {
        spec[i] = std::complex<double>(out[i][0], out[i][1]);
    }

    // Compute amplitude and phase spectra
    std::vector<double> aspec(npts), pspec(npts);
    for (int i = 0; i < npts; ++i) {
        aspec[i] = std::abs(spec[i]);
        pspec[i] = std::arg(spec[i]);
    }

    // Generate frequency array
    std::vector<double> freq(npts);
    for (int i = 0; i < npts; ++i) {
        freq[i] = (-npts / 2.0 + i) / reclen;
    }
    freq = fftshift(freq);  // Apply fftshift

    // Apply frequency cutoff
    if (fhigh == 0) {
        fhigh = 1.0 / (2.0 * samint);
    }

    // Identify frequencies in the desired range
    std::vector<int> ifr;
    for (int i = 0; i < npts; ++i) {
        if (std::abs(freq[i]) >= flow && std::abs(freq[i]) <= fhigh) {
            ifr.push_back(i);
        }
    }

    // Initialize filtered spectrum
    std::vector<std::complex<double>> filtspec2(npts, {0.0, 0.0});

    // Apply bandpass filtering
    for (int i : ifr) {
        filtspec2[i] = std::polar(aspec[i], pspec[i]);  // Convert back to complex
    }

    // Prepare arrays for inverse FFT
    fftw_complex *in_ifft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_complex *out_ifft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * npts);
    fftw_plan ifft_plan = fftw_plan_dft_1d(npts, in_ifft, out_ifft, FFTW_BACKWARD, FFTW_ESTIMATE);

    // Copy filtered spectrum into FFTW input array
    for (int i = 0; i < npts; ++i) {
        in_ifft[i][0] = filtspec2[i].real();
        in_ifft[i][1] = filtspec2[i].imag();
    }

    // Perform inverse FFT
    fftw_execute(ifft_plan);

    // Extract the real part of the result
    std::vector<double> tsfilt(npts);
    for (int i = 0; i < npts; ++i) {
        tsfilt[i] = out_ifft[i][0] / npts;  // Normalize
    }

    // Free FFTW memory
    fftw_destroy_plan(plan);
    fftw_destroy_plan(ifft_plan);
    fftw_free(in);
    fftw_free(out);
    fftw_free(in_ifft);
    fftw_free(out_ifft);

    // Return both filtered time series and filtered spectrum
    return {tsfilt, aspec};
}

// Function to compute SPL values
std::vector<double> computeSPL(const std::vector<double>& rms_values) {
    std::vector<double> spl_values;
    for (double rms : rms_values) {
        spl_values.push_back(20 * std::log10(rms));
    }
    return spl_values;
}

double calculate_kurtosis(const std::vector<double>& data) {
    // Step 1: Calculate the mean of the data
    double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

    // Step 2: Calculate the variance and fourth moment
    double variance = 0.0;
    double fourth_moment = 0.0;

    for (double val : data) {
        double diff = val - mean;
        variance += diff * diff;
        fourth_moment += diff * diff * diff * diff;
    }

    variance /= data.size();
    fourth_moment /= data.size();

    // Step 3: Calculate kurtosis (subtract 3 for excess kurtosis)
    double kurtosis = (fourth_moment / (variance * variance)); // double kurtosis = (fourth_moment / (variance * variance)) - 3;
    return kurtosis;
}

std::pair<std::vector<double>, std::vector<double>> correl_5(const std::vector<double>& ts1, const std::vector<double>& ts2, int lags, int offset) {
    std::vector<double> P(lags + 1);
    std::vector<double> nlags(lags + 1);

    for (int i = 0; i <= lags; i++) {
        double ng = 1.0;
        double sx = 2.0;
        double sy = 3.0;
        double sxx = 4.0;
        double syy = 5.0;
        double sxy = 6.0;

        for (int k = 0; k < (ts1.size() - (i + offset)); k++) {
            double x = ts1[k];
            double y = ts2[k + (i + offset)];

            // Check for NaN values (using isnan for C++)
            if (!std::isnan(x) && !std::isnan(y)) {
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
                ng += 1.0;
            }
        }

        double covar1 = (sxy / ng) - ((sx / ng) * (sy / ng));
        double denom1 = sqrt((sxx / ng) - pow(sx / ng, 2));
        double denom2 = sqrt((syy / ng) - pow(sy / ng, 2));
        P[i] = covar1 / (denom1 * denom2);
        nlags[i] = i;
    }

    return {P, nlags};
}

std::pair<std::vector<int>, std::vector<std::vector<double>>> f_solo_per_GM2(
    const std::vector<double>& p_filt_input, double fs, double timewin, double avtime) {
    
    std::vector<double> p_avtot;
    double avwin = fs * avtime;
    double sampwin = fs * timewin;
    
    int ntwin = floor(p_filt_input.size() / sampwin); // Number of time windows
    
    // Truncate p_filt to exact multiple of sampwin
    std::vector<double> p_filt(p_filt_input.begin(), p_filt_input.begin() + sampwin * ntwin);
    
    // Reshape p_filt into a 2D structure (sampwin x ntwin)
    std::vector<std::vector<double>> p_filt_reshaped(ntwin, std::vector<double>(sampwin));
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
    std::vector<std::vector<double>> p_av;
    
    for (int jj = 0; jj < ntwin; jj++) {
        // Reshape into averaging windows
        std::vector<std::vector<double>> avwinmatrix(numavwin, std::vector<double>(avwin));
        for (int i = 0; i < numavwin; i++) {
            for (int j = 0; j < avwin; j++) {
                avwinmatrix[i][j] = p_filt_reshaped[jj][i * avwin + j];
            }
        }
        
        // Calculate mean for each window
        std::vector<double> p_avi(numavwin);
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
    for (const auto& row : p_av) {
        p_avtot.insert(p_avtot.end(), row.begin(), row.end());
    }
    
    // Calculate number of rows and columns in p_avtot
    int p_avtot_rows = p_av[0].size();  // Number of elements in each column
    int p_avtot_cols = p_av.size();     // Number of columns
    
    std::vector<std::vector<double>> acorr(p_avtot_cols);
    std::vector<int> pkcount(p_avtot_cols);
    
    for (int zz = 0; zz < p_avtot_cols; zz++) {
        // Extract column zz from p_avtot
        std::vector<double> column_zz(p_avtot_rows);
        for (int i = 0; i < p_avtot_rows; i++) {
            column_zz[i] = p_av[zz][i];
        }
        
        // Compute correlation
        auto corr_result = correl_5(column_zz, column_zz, p_avtot_rows * 0.7, 0);
        acorr[zz] = corr_result.first;
        
        // Find peaks
        int peak_count = 0;
        for (size_t i = 1; i < acorr[zz].size() - 1; i++) {
            // Simple peak detection: higher than neighbors and prominence > 0.5
            if (acorr[zz][i] > acorr[zz][i-1] && acorr[zz][i] > acorr[zz][i+1]) {
                double prominence = std::min(acorr[zz][i] - acorr[zz][i-1], acorr[zz][i] - acorr[zz][i+1]);
                if (prominence > 0.5) {
                    peak_count++;
                }
            }
        }
        
        pkcount[zz] = peak_count;
    }
    
    return {pkcount, acorr};
}

std::vector<std::complex<double>> hilbert(const std::vector<double>& xr, int n = -1) {
    // If n is not specified, use the length of xr
    if (n <= 0) {
        n = xr.size();
    }

    // Initialize the result vector
    std::vector<std::complex<double>> x(n);
    
    // Prepare the FFT input (padded or truncated as needed)
    double* in = (double*) fftw_malloc(sizeof(double) * n);
    
    // Copy the input into the FFT input array (with zero-padding or truncation)
    if (xr.size() <= n) {
        std::memcpy(in, xr.data(), sizeof(double) * xr.size());
        std::memset(in + xr.size(), 0, sizeof(double) * (n - xr.size()));
    } else {
        std::memcpy(in, xr.data(), sizeof(double) * n);
    }
    
    // Prepare the FFT output
    fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    
    // Create and execute FFT plan
    fftw_plan p = fftw_plan_dft_r2c_1d(n, in, out, FFTW_ESTIMATE);
    fftw_execute(p);
    
    // Modify FFT output for Hilbert transform
    // For even n, h = [1, 2, 2, ..., 2, 1, 0, 0, ..., 0]
    // For odd n, h = [1, 2, 2, ..., 2, 0, 0, ..., 0]
    if (n % 2 == 0) {  // even
        // DC component (index 0) stays as is
        out[0][0] = out[0][0];  // real part
        out[0][1] = 0;          // imaginary part
        
        // Double the positive frequencies (except DC and Nyquist)
        for (int i = 1; i < n/2; i++) {
            out[i][0] *= 2;  // real part
            out[i][1] *= 2;  // imaginary part
        }
        
        // Nyquist frequency stays as is (index n/2)
        // (for FFTW, the Nyquist frequency is at index n/2)
        
        // Zero out the negative frequencies
        for (int i = n/2 + 1; i < n; i++) {
            out[i][0] = 0;
            out[i][1] = 0;
        }
    } else {  // odd
        // DC component (index 0) stays as is
        out[0][0] = out[0][0];  // real part
        out[0][1] = 0;          // imaginary part
        
        // Double the positive frequencies
        for (int i = 1; i <= n/2; i++) {
            out[i][0] *= 2;  // real part
            out[i][1] *= 2;  // imaginary part
        }
        
        // Zero out the negative frequencies
        for (int i = n/2 + 1; i < n; i++) {
            out[i][0] = 0;
            out[i][1] = 0;
        }
    }
    
    // Prepare the IFFT output
    fftw_complex* ifft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n);
    
    // Create and execute IFFT plan
    fftw_plan p_inv = fftw_plan_dft_1d(n, out, ifft_out, FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p_inv);
    
    // Copy the result to the output vector
    for (int i = 0; i < n; i++) {
        // FFTW doesn't normalize, so we divide by n
        x[i] = std::complex<double>(ifft_out[i][0] / n, ifft_out[i][1] / n);
    }
    
    // Clean up
    fftw_destroy_plan(p);
    fftw_destroy_plan(p_inv);
    fftw_free(in);
    fftw_free(out);
    fftw_free(ifft_out);
    
    return x;
}

void processAnalyticSignal(const std::vector<std::complex<double>>& analytic1) {
    std::vector<double> abs_analytic1(analytic1.size());
    
    // Compute absolute values of analytic1
    std::transform(analytic1.begin(), analytic1.end(), abs_analytic1.begin(),
                   [](std::complex<double> val) { return std::abs(val); });

    // Compute sum of absolute values
    double sum_abs_analytic1 = std::accumulate(abs_analytic1.begin(), abs_analytic1.end(), 0.0);

    // Normalize to create at1
    std::vector<double> at1(analytic1.size());
    if (sum_abs_analytic1 != 0) { // Prevent division by zero
        std::transform(abs_analytic1.begin(), abs_analytic1.end(), at1.begin(),
                       [sum_abs_analytic1](double val) { return val / sum_abs_analytic1; });
    }
}

// Function to calculate the sum of all elements in a vector
double sumVector(const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0);
}

// Function to calculate the absolute value of each element in a vector
std::vector<double> absVector(const std::vector<std::complex<double>>& vec) {
    std::vector<double> result(vec.size());
    for (size_t i = 0; i < vec.size(); i++) {
        result[i] = std::abs(vec[i]);
    }
    return result;
}

vector<double> f_solo_dissim_GM1(const vector<vector<double>>& timechunk_matrix, int pts_per_timewin, int num_timewin, double fft_win, double fs) {
    int pts_per_fft = static_cast<int>(fft_win * fs);  // Calculate size of fft window
    int numfftwin = pts_per_timewin / pts_per_fft;  // # of fft windows
    vector<double> Dfin, D;
    
    for (int kk = 0; kk < num_timewin - 1; ++kk) {
        std::vector<double> col1(timechunk_matrix.size());
        std::vector<double> col2(timechunk_matrix.size());

        vector<complex<double>> analytic1 = hilbert(timechunk_matrix[kk], timechunk_matrix[kk].size());
        vector<complex<double>> analytic2 = hilbert(timechunk_matrix[kk + 1], timechunk_matrix[kk + 1].size());

        vector<double> at1(analytic1.size());
        vector<double> at2(analytic2.size());

        // Calculate the absolute values of all elements in analytic1
        std::vector<double> abs_analytic1(analytic1.size());
        for (size_t i = 0; i < analytic1.size(); ++i) {
            abs_analytic1[i] = std::abs(analytic1[i]);
        }
        
        // Calculate the sum of absolute values
        double sum_abs = std::accumulate(abs_analytic1.begin(), abs_analytic1.end(), 0.0);
        
        // Normalize the absolute values
        for (size_t i = 0; i < analytic1.size(); ++i) {
            at1[i] = abs_analytic1[i] / sum_abs;
        }

        processAnalyticSignal(analytic1);
        processAnalyticSignal(analytic2);

        // Calculate at2
        std::vector<double> absAnalytic2 = absVector(analytic2);
        double sumAbsAnalytic2 = sumVector(absAnalytic2);
        
        for (size_t i = 0; i < analytic2.size(); i++) {
            at2[i] = absAnalytic2[i] / sumAbsAnalytic2;
        }

        // Calculate Dt
        std::vector<double> diff(at1.size());
        for (size_t i = 0; i < at1.size(); i++) {
            diff[i] = std::abs(at1[i] - at2[i]);
        }
        double sumDiff = sumVector(diff);
        double Dt = sumDiff / 2.0;
        
        std::vector<double> s3a(pts_per_fft * numfftwin);
        for (int i = 0; i < pts_per_fft * numfftwin; i++) {
            s3a[i] = timechunk_matrix[kk][i];
        }

        fftw_complex* in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
        fftw_complex* out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
        fftw_plan p = fftw_plan_dft_1d(pts_per_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        std::vector<double> ga(pts_per_fft * numfftwin);
        for (int i = 0; i < numfftwin; i++) {
            for (int j = 0; j < pts_per_fft; j++) {
                in[j][0] = s3a[i * pts_per_fft + j];
                in[j][1] = 0;
            }

            fftw_execute(p);

            for (int j = 0; j < pts_per_fft; j++) {
                ga[i * pts_per_fft + j] = std::sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) / pts_per_fft;
            }
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        std::vector<double> sfa(pts_per_fft);
        for (int i = 0; i < pts_per_fft; i++) {
            double sum = 0;
            for (int j = 0; j < numfftwin; j++) {
                sum += ga[i + j * pts_per_fft];
            }
            sfa[i] = sum / numfftwin;
        }

        std::vector<double> Sfa(pts_per_fft);
        double sum_Sfa = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Sfa[i] = std::abs(sfa[i]);
            sum_Sfa += Sfa[i];
        }

        for (int i = 0; i < pts_per_fft; i++) {
            Sfa[i] /= sum_Sfa;
        }

        std::vector<double> s3b(pts_per_fft * numfftwin);
        for (int i = 0; i < pts_per_fft * numfftwin; i++) {
            s3b[i] = timechunk_matrix[kk + 1][i];
        }

        in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
        out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * pts_per_fft);
        p = fftw_plan_dft_1d(pts_per_fft, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

        std::vector<double> gb(pts_per_fft * numfftwin);
        for (int i = 0; i < numfftwin; i++) {
            for (int j = 0; j < pts_per_fft; j++) {
                in[j][0] = s3b[i * pts_per_fft + j];
                in[j][1] = 0;
            }

            fftw_execute(p);

            for (int j = 0; j < pts_per_fft; j++) {
                gb[i * pts_per_fft + j] = std::sqrt(out[j][0] * out[j][0] + out[j][1] * out[j][1]) / pts_per_fft;
            }
        }

        fftw_destroy_plan(p);
        fftw_free(in);
        fftw_free(out);

        std::vector<double> sfb(pts_per_fft);
        for (int i = 0; i < pts_per_fft; i++) {
            double sum = 0;
            for (int j = 0; j < numfftwin; j++) {
                sum += gb[i + j * pts_per_fft];
            }
            sfb[i] = sum / numfftwin;
        }

        std::vector<double> Sfb(pts_per_fft);
        double sum_Sfb = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Sfb[i] = std::abs(sfb[i]);
            sum_Sfb += Sfb[i];
        }

        for (int i = 0; i < pts_per_fft; i++) {
            Sfb[i] /= sum_Sfb;
        }

        double Df = 0;
        for (int i = 0; i < pts_per_fft; i++) {
            Df += std::abs(Sfb[i] - Sfa[i]);
        }
        Df /= 2;

        double Di = Dt * Df;
        D.push_back(Di);
    }

    return D;
}

std::vector<double> audioread(const std::string& filename, int& sampleRate) {
    // Open the audio file
    SF_INFO sfInfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfInfo);

    if (!file) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    // Extract sample rate
    sampleRate = sfInfo.samplerate;
    int numChannels = sfInfo.channels;
    sf_count_t numFrames = sfInfo.frames;

    // Read the audio data
    std::vector<double> audioData(numFrames * numChannels);
    sf_readf_double(file, audioData.data(), numFrames);

    // Close the file
    sf_close(file);

    return audioData;
}

// Function to compute RMS (Root Mean Square) of a vector
double rms(const std::vector<double>& x) {
    if (x.empty()) {
        throw std::invalid_argument("Input vector is empty.");
    }

    double sumSquares = std::accumulate(x.begin(), x.end(), 0.0, [](double sum, double val) {
        return sum + val * val;
    });

    return std::sqrt(sumSquares / x.size());
}

// Overloaded function to compute RMS for a 2D matrix
std::vector<double> rms(const std::vector<std::vector<double>>& matrix) {
    if (matrix.empty()) {
        throw std::invalid_argument("Input matrix is empty.");
    }

    std::vector<double> row_rms(matrix.size());
    for (size_t i = 0; i < matrix.size(); ++i) {
        row_rms[i] = rms(matrix[i]);
    }

    return row_rms;
}

std::vector<double> calculateSPLpkhold(const std::vector<std::vector<double>>& timechunk_matrix) {
    std::vector<double> SPLpkhold(timechunk_matrix.size());
    
    for (size_t i = 0; i < timechunk_matrix.size(); ++i) {
        double maxVal = 0.0;
        
        // Find the maximum absolute value in this row
        for (size_t j = 0; j < timechunk_matrix[i].size(); ++j) {
            maxVal = std::max(maxVal, std::abs(timechunk_matrix[i][j]));
        }
        
        // Convert to dB
        SPLpkhold[i] = 20 * std::log10(maxVal);
    }
    
    return SPLpkhold;
}

AudioFeatures f_WAV_frankenfunction_reilly(int num_bits, int peak_volts, const std::filesystem::directory_entry &file_name,
                                           double RS, int timewin, double avtime, int fft_win, int arti, int flow, int fhigh) {
    // Initialize output variables as cell arrays
    std::vector<double> SPLrms;
    std::vector<double> SPLpk;
    std::vector<double> impulsivity;
    std::vector<int> peakcount;
    std::vector<double> autocorr;
    std::vector<double> dissim;

    // Convert and fix the file path
    std::string file_path = fixFilePath(file_name.path().string());

    AudioFeatures features;
    double rs = pow(10, RS / 20.0);
    double max_count = pow(2, num_bits);
    double conv_factor = peak_volts / max_count;

    AudioInfo info = audioread_info(file_path);
    std::vector<double> audioDataA = audioread(file_name.path().string(), info.sampleRate);
        
    // Assuming total_samples is computed as before:
    size_t total_samples = info.sampleRate * info.duration;
        
    // Now call audioread with a dynamic range
    auto audio = audioread(file_path, std::make_pair(1, total_samples), "double");

    std::vector<double> audioData = AudioReader::readAudio(file_path, audio.sampleRate);
    
    int fs = audio.sampleRate;
    std::vector<double> x = audio.samples[0];
        
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

    if (num_bits == 24) {
        for (auto& value : x) {
            auto temp = static_cast<int>(value);
            temp >>= 8;
            value = static_cast<double>(temp);
        }
    }

    std::vector<double> v(x.size());
    std::vector<double> p(x.size());
    std::vector<double> pout;
    std::transform(x.begin(), x.end(), v.begin(), [conv_factor](double sample) { return sample * conv_factor; });

    p.resize(v.size());
    std::transform(v.begin(), v.end(), p.begin(), [rs](double sample) { return sample / rs; });
    
    if (arti == 1) {  // Calibration tone is present
        size_t start_idx = 6 * fs;
        p = std::vector<double>(p.begin() + start_idx, p.end());
        p.insert(p.begin(), 0);
    }
    
    pout.insert(pout.end(), p.begin(), p.end());
    
    // Perform bandpass filtering
    auto [p_filt, filtspec1] = dylan_bpfilt(pout, 1.0 / fs, flow, fhigh);

    int pts_per_timewin = timewin * fs;
    int num_timewin = floor(p_filt.size()/pts_per_timewin) + 1;
    int padding_length = num_timewin * pts_per_timewin - p_filt.size();

    std::vector<double> p_filt_padded = p_filt;  // Copy original data
    p_filt_padded.insert(p_filt_padded.end(), padding_length, 0.0);  // Append zeros
    
    std::vector<std::vector<double>> timechunk_matrix(num_timewin, std::vector<double>(pts_per_timewin));
    for (int i = 0; i < num_timewin; i++) {
        for (int j = 0; j < pts_per_timewin; j++) {
            timechunk_matrix[i][j] = p_filt_padded[i * pts_per_timewin + j];
        }
    }
    
    std::vector<double> rms_matrix = rms(timechunk_matrix);  // Compute RMS values
    std::vector<double> SPLrmshold = computeSPL(rms_matrix);  // Compute SPL RMS hold
    features.SPLrms = SPLrmshold;
    std::vector<double> SPLpkhold = calculateSPLpkhold(timechunk_matrix);
    features.SPLpk = SPLpkhold;

    // Number of columns (assuming all rows have the same number of columns)
    size_t num_columns = timechunk_matrix[0].size();

    // Create a vector to store the kurtosis for each row
    std::vector<double> kmat(timechunk_matrix.size(), 0.0);

    // Compute kurtosis for each row of the timechunk_matrix
    for (size_t row = 0; row < timechunk_matrix.size(); ++row) {
        std::vector<double> row_data;
        
        // Collect all values in the current row
        for (size_t col = 0; col < num_columns; ++col) {
            row_data.push_back(timechunk_matrix[row][col]);
        }
        
        // Calculate kurtosis for the current row
        kmat[row] = calculate_kurtosis(row_data);
    }
    features.impulsivity = kmat;

    std::pair<std::vector<int>, std::vector<std::vector<double>>> result = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime);
    
    std::vector<int> pkcount = std::get<0>(result);
    features.peakcount = pkcount;
    std::vector<std::vector<double>> acorr = std::get<1>(result);
    features.autocorr = acorr;

    std::vector<double> Dfin = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs);
    features.dissim = Dfin;

    return features;
}

// Helper function to save features to a CSV file
void saveFeaturesToCSV(const std::string& filename, const std::vector<std::string>& filenames, 
    const std::vector<AudioFeatures>& allFeatures) {
    // Debug: Print number of files and features
    std::cout << "Number of files: " << filenames.size() << std::endl;
    for (size_t i = 0; i < allFeatures.size(); ++i) {
        std::cout << "File " << filenames[i] << " SPLrms size: " << allFeatures[i].SPLrms.size() << std::endl;
    }

    // Open file in write mode (not append)
    std::ofstream outputFile(filename);

    if (!outputFile.is_open()) {
        std::cerr << "Error: Unable to open output file: " << filename << std::endl;
        return;
    }

    // Write CSV header
    outputFile << "Filename,SPLrms,SPLpk,Impulsivity,Dissimilarity,PeakCount,";

    // Determine max size of autocorrelation for header
    size_t maxAutocorrRows = 0;
    size_t maxAutocorrCols = 0;

    for (const auto& feature : allFeatures) {
        if (!feature.autocorr.empty()) {
            maxAutocorrRows = std::max(maxAutocorrRows, feature.autocorr.size());
            if (!feature.autocorr[0].empty()) {
                maxAutocorrCols = std::max(maxAutocorrCols, feature.autocorr[0].size());
            }
        }
    }

    // Add autocorrelation headers
    for (size_t i = 0; i < maxAutocorrRows; ++i) {
        for (size_t j = 0; j < maxAutocorrCols; ++j) {
            outputFile << "Autocorr_" << i << "_" << j;
            if (i < maxAutocorrRows - 1 || j < maxAutocorrCols - 1) {
                outputFile << ",";
            }
        }
    }
    outputFile << std::endl;

    // Process each file
    for (size_t fileIdx = 0; fileIdx < allFeatures.size(); ++fileIdx) {
        const AudioFeatures& features = allFeatures[fileIdx];

        // Determine the maximum length among different feature vectors
        size_t maxLength = std::max({
            features.SPLrms.size(),
            features.SPLpk.size(),
            features.impulsivity.size(),
            features.dissim.size(),
            features.peakcount.size()
        });

        // Write each row of data
        for (size_t i = 0; i < maxLength; ++i) {
            // Write filename
            outputFile << filenames[fileIdx] << ",";
            outputFile << (i < features.SPLrms.size() ? std::to_string(features.SPLrms[i]) : "0") << ",";  // SPLrms
            outputFile << (i < features.SPLpk.size() ? std::to_string(features.SPLpk[i]) : "0") << ",";  // SPLpk
            outputFile << (i < features.impulsivity.size() ? std::to_string(features.impulsivity[i]) : "0") << ",";  // Impulsivity
            outputFile << (i < features.dissim.size() ? std::to_string(features.dissim[i]) : "0") << ",";  // Dissimilarity
            outputFile << (i < features.peakcount.size() ? std::to_string(features.peakcount[i]) : "0") << ",";  // Peak Count

            // Autocorrelation
            if (i < features.autocorr.size()) {
                for (size_t j = 0; j < maxAutocorrCols; ++j) {
                    if (j < features.autocorr[i].size()) {
                        outputFile << features.autocorr[i][j];
                    } else {
                        outputFile << "0";
                    }
                    if (j < maxAutocorrCols - 1) {
                        outputFile << ",";
                    }
                }
            } else {
                // Fill with zeros if no autocorrelation data
                for (size_t j = 0; j < maxAutocorrCols; ++j) {
                    outputFile << "0";
                    if (j < maxAutocorrCols - 1) {
                        outputFile << ",";
                    }
                }
            }
            outputFile << std::endl;
        }
    }

    std::cout << "Features saved to: " << filename << std::endl;
}

int main() {
    // Calibration information
    std::string input_dir = "/home/csg/rjl1050/CCOM/matlab_2_cpp_agate/wav_recordings";
    std::string output_dir = "/home/csg/rjl1050/CCOM/matlab_2_cpp_agate/wav_recordings_cpp.csv";

    // Parameters
    int num_bits = 16;
    double RS = -178.3;     // BE SURE TO CHANGE FOR EACH HYDROPHONE
                            // sensitivity is based on hydrophone, not recorder

    int peak_volts = 2;
    int arti = 1;         // Make 1 if calibration tone present

    // Analysis options
    int timewin = 60;
    int fft_win = 1;
    double avtime = 0.1;
    int flow = 1;
    int fhigh = 192000;

    AudioFeatures features;
    std::vector<AudioFeatures> allFeatures;
    std::vector<std::string> filenames;

    // Variables to store analysis results
    std::vector<double> SPLrms, SPLpk, impulsivity, autocorr, dissim;
    std::vector<int> peakcount;

    // Iterate through directory entries
    for (const auto& file_dir : std::filesystem::directory_iterator(input_dir)) {
        // Process each file
        AudioFeatures features = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh);
        
        // Store the features and filename
        allFeatures.push_back(features);
        filenames.push_back(file_dir.path().filename().string());
        
    }

    // Save all features to the output file
    if (!allFeatures.empty()) {
        saveFeaturesToCSV(output_dir, filenames, allFeatures);
        std::cout << "Successfully saved features for " << allFeatures.size() << " files." << std::endl;
    } else {
        std::cout << "No files were processed." << std::endl;
    }

    return 0;
}
