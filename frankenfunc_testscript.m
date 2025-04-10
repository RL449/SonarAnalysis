
clc; clear all; close all;

%calibration information

input_dir = 'C:\Users\rlessard\Desktop\5593 organized\0406_0531\0406_0531'; % 'C:\Users\rlessard\Desktop\runThisInput\long_recording'; % 'C:\Users\rlessard\Desktop\0406_0531_bda_recordings\24_hr_after_sonar_exposure'; % 'C:\Users\rlessard\Desktop\before_during_after_sonar_10minclip'; % '\\BOOBY\Public\incoming\SoundTrap\5593\0601_0719_10min'; % 'D:\5593\0601_0813_10min\During_Exposure'; % 'C:\Users\rlessard\Desktop\5593 organized\0406_0531' % '\\BOOBY\Public\incoming\SoundTrap\5593\0807_0813\0813';
output_dir = 'C:\Users\rlessard\Desktop\before_during_after_sonar_output\20240409_updated\after_ulth'; % 'C:\Users\rlessard\Desktop\runThisOutput\long_recording'; % 'C:\Users\rlessard\Desktop\0406_0531_bda_recordings\24_hr_output\0406_0531\24_hr_after'; % 'C:\Users\rlessard\Desktop\runThisOutput\runThisMATLAB'; % 'C:\Users\rlessard\Desktop\before_during_after_sonar_output\20240611\24hrafter_ulth'; % 'C:\Users\rlessard\Desktop\SoundscapeCodeDesktop\DataOutput\192khz\low_mid_high_ulth\0601_0719high_new'; % 'C:\Users\rlessard\Desktop\SoundscapeCodeDesktop\DataOutput\96khz\low_mid_high_ulth\0406_0531_high';
file_dir = dir(input_dir);
% file_dir = dir('C:\Users\rlessard\Desktop\SoundscapeCodeDesktop\PracticeSounds');

num_bits = 16; 
RS = -178.3;                %BE SURE TO CHANGE FOR EACH HYDROPHONE, sensitivity
                            %is based on hydrophone, not recorder
peak_volts = 2;
arti = 1;                   %make 1 if calibration tone present

%analysis options
timewin = 60; % 58               %length of time window in seconds for analysis bins
fft_win = 1; % 1                %length of fft window in minutes
avtime = .1; % .1
flow = 10000; % 50
fhigh = 96000; % 300

[SPLrms, SPLpk,impulsivity, peakcount, autocorr, dissim] = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win,arti,flow,fhigh);
save(output_dir,'SPLrms', 'SPLpk','impulsivity', 'peakcount', 'autocorr', 'dissim');
