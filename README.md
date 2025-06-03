This program has been converted from MATLAB code (written by Grant Milne and Dylan Wilford) into C++. It is designed to use multithreading to make data processing more time-efficient and available on Windows platforms, without requiring a MATLAB license.

This program reads from a directory of .wav files and produces a .csv file containing the following columns:
Filename
Year
Month
Day
Hour
Minute
SegmentDuration
Sound pressure level root mean square
Sound pressure level peak
Impulsivity
Dissimilarity
Peak count
Autocorrelation

The user specified the following parameters:
Input directory
Output directory
Bit depth
Reference sensitivity (dB)
Peak volts
Presence of a calibration tone (1 if present, 0 if not present)
Time window length (seconds)
Fast Fourier Transform window (seconds)
Averaging window time (seconds)
Low frequency cutoff (Hz)
High frequency cutoff (Hz)
Maximum number of threads
Downsampling factor (No downsampling if unspecified)
Omission of incomplete minutes (No ommission if unspecified)

File names in the input directory are assumed to be on contain a timestamp with the format "YYYYMMDD_HHMMSS".
File names without this format will generate invalid outputted date/time columns.

The required libraries for this program are sndfile and fftw3.

Compilation:
g++ -o main_original.exe main_original.cpp -IC:\path\to\fftw3.h\and\sndfile.h -LC:\path\to\fftw3.lib\and\and\sndfile.lib -lsndfile -lfftw3

Run program:
main_original.exe --input "C:\path\to\input\directory" --output "C:\path\to\output\directory\output_filename.csv" --num_bits # --RS -# --peak_volts # --arti # --timewin # --fft_win # --avtime # --flow # --fhigh # --max_threads # --omit_partial_minute (optional)

This is fully compatible with gcc version 11.4.0 and partially compatible with gcc version 14.2.0.

