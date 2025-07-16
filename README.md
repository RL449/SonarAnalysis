This program has been converted from MATLAB code (written by Grant Milne and Dylan Wilford) into C++. It is designed to use multithreading to make data processing more time-efficient and available on Windows platforms, without requiring a MATLAB license.
Note: Save this as a file that allows non-ASCII characters, i.e. Unicode (UTF-8 without signature) - Codepage 65001 in Visual Studio.

This program reads from a directory of .wav files and produces a .csv file containing the following columns:

- Filename

- Year

- Month

- Day

- Hour

- Minute

- SegmentDuration

- Sound pressure level root mean square

- Sound pressure level peak

- Impulsivity

- Dissimilarity

- Peak count

- Autocorrelation


The user specifies the following parameters:

- Input directory

- Output directory

- Bit depth

- Reference sensitivity (dB)

- Peak volts

- Presence of a calibration tone (1 if present, 0 if not present)

- Time window length (seconds)

- Fast Fourier Transform window (seconds)

- Averaging window time (seconds)

- Low frequency cutoff (Hz)

- High frequency cutoff (Hz)

OptionalParameters
- Maximum number of threads

- Downsampling factor

- Omission of incomplete minutes

File names in the input directory are assumed to be on contain a timestamp with the format "YYYYMMDD_HHMMSS".
File names without this format will generate "NaN" date/time columns.

Ensure these libraries are available and properly linked:
- sndfile

- fftw3

- cufft

- cudart

Compilation:
nvcc -std=c++17 --extended-lambda -Xcompiler "/MT" -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -o main_original.exe main_original.cu -I"C:\path\to\vcpkg\installed\x64-windows\include" -L"C:\path\to\vcpkg\installed\x64-windows\lib" -I"C:\path\to\NVIDIA GPU Computing Toolkit\CUDA\version number\include" -L"C:\path\to\NVIDIA GPU Computing Toolkit\CUDA\version number\lib\x64" -lsndfile -lfftw3 -lcufft -lcudart --Wno-deprecated-gpu-targets -Xlinker /SUBSYSTEM:CONSOLE

Run program:
main_original.exe --input "C:\path\to\input\directory" --output "C:\path\to\output\directory\output_filename.csv" --num_bits # --RS -# --peak_volts # --arti # --timewin # --fft_win # --avtime # --flow # --fhigh # --max_threads # --downsample # --omit_partial_minute

