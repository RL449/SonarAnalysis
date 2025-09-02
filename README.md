This program has been converted from MATLAB code (written by Grant Milne and Dylan Wilford) into C++. It is designed to use multithreading to make data processing more time-efficient and available on Windows platforms, without requiring a MATLAB license. For devices without an NVIDIA GPU, the .cpp file is available. For devices with an NVIDIA GPU, the .cu file is an option for faster data processing.

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


The user specifies the following required parameters:

- Input directory

- Output file name with full directory

- Peak voltage


The following additional parameters are optional (with default settings):

- Bit depth (16)

- Reference sensitivity in dB (-178.3)

- Duration of excess noise at start of recording in seconds (0.0)

- Time window length in seconds (60)

- Fast Fourier Transform window in seconds (1)

- Averaging window time in seconds (0.1)

- Low frequency cutoff in Hz (1)

- High frequency cutoff in Hz (16000)

- Maximum number of threads (All threads on device)

- Downsampling factor (-1: No downsampling)

- Omission of incomplete minutes (false)

- Inclusion of processing output (no inclusion)

File names in the input directory are assumed to be on contain a timestamp with the format "YYYYMMDD_HHMMSS" or "XXXX.YYMMDDHHMMSS".
File names without this format will generate "NaN" date/time columns.

Ensure these libraries are available and properly linked:
- sndfile

- fftw3

- cufft

- cudart


CPP:


Compilation:
g++ -o main_original.exe main_original.cpp -IC:\Users\rlessard\Desktop\vcpkg\installed\x64-windows\include -LC:\path\to\vcpkg\installed\x64-windows\lib -lsndfile -lfftw3

Compilation Example:
g++ -o main_original.exe main_original.cpp -IC:\Users\rlessard\Desktop\vcpkg\installed\x64-windows\include -LC:\Users\rlessard\Desktop\vcpkg\installed\x64-windows\lib -lsndfile -lfftw3

Program Execution:
main_original.exe --input "C:\path\to\input\directory" --output "C:\path\to\output\directory\output_filename.csv" --num_bits # --RS -# --peak_volts # --arti_len # --timewin # --fft_win # --avtime # --flow # --fhigh # --max_threads # --downsample # --omit_partial_minute --debug_output 1

Program Execution Example:
main_original.exe --input "C:\Users\rlessard\Desktop\ValDrixAudio" --output "C:\Users\rlessard\Desktop\runThisOutput\cpp_output_ValDrixAudio.csv" --num_bits 24 --RS -178.3 --peak_volts 2 --arti_len 0.0 --timewin 60 --fft_win 1 --avtime 0.1 --flow 1 --fhigh 128000 --max_threads 5 --downsample 2 --omit_partial_minute --debug_output 1


CUDA

Compilation:
nvcc -std=c++17 --extended-lambda -Xcompiler "/MT" -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -o main_original.exe main_original.cu -I"C:\path\to\vcpkg\installed\x64-windows\include" -L"C:\path\to\vcpkg\installed\x64-windows\lib" -I"C:\path\to\NVIDIA GPU Computing Toolkit\CUDA\version number\include" -L"C:\path\to\NVIDIA GPU Computing Toolkit\CUDA\version number\lib\x64" -lsndfile -lfftw3 -lcufft -lcudart --Wno-deprecated-gpu-targets -Xlinker /SUBSYSTEM:CONSOLE

Compilation Example:
nvcc -std=c++17 --extended-lambda -Xcompiler "/MT" -DCCCL_IGNORE_DEPRECATED_CPP_DIALECT -o main_original.exe main_original.cu -I"C:\Users\rlessard\Desktop\vcpkg\installed\x64-windows\include" -L"C:\Users\rlessard\Desktop\vcpkg\installed\x64-windows\lib" -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64" -lsndfile -lfftw3 -lcufft -lcudart --Wno-deprecated-gpu-targets -Xlinker /SUBSYSTEM:CONSOLE

Program Execution:
main_original.exe --input "C:\path\to\input\directory" --output "C:\path\to\output\directory\output_filename.csv" --num_bits # --RS -# --peak_volts # --arti_len # --timewin # --fft_win # --avtime # --flow # --fhigh # --max_threads # --downsample # --omit_partial_minute --debug_output 1

Program Execution Example:
main_original.exe --input "C:\Users\rlessard\Desktop\ValDrixAudio" --output "C:\Users\rlessard\Desktop\runThisOutput\cpp_output_ValDrixAudio.csv" --num_bits 24 --RS -178.3 --peak_volts 2 --arti_len 0.0 --timewin 60 --fft_win 1 --avtime 0.1 --flow 1 --fhigh 128000 --max_threads 5 --downsample 2 --omit_partial_minute --debug_output 1

Notes:

It is assumed that the date/time in file names is the time at which recording began.
