%{
function [SPLrms, SPLpk,impulsivity, peakcount, autocorr, dissim] = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win,arti,flow,fhigh) 
%f_WAV_frankenfunction: accepts acoustic data, hydrophone features, and
%processing preferences as inputs then outputs calibrated soundscape metrics
%Inputs:
%num_bits = bit rate of hydrophone
%peak_volts = voltage of the recorder, peak to peak
%file_dir = the directory of files intended for processing
%RS = hydrophone sensitivity
%timewin = size of time windows in seconds for calculation of soundscape metrics
%avtime = averaging time duration in seconds for autocorrelation measurements
%fft_win = size of time window in minutes over which fft is performed 
%arti = enter 1 if there are artifacts such as calibration tones at the
%begginings of recordings
%flow = lower frequency cutoff
%fhigh = upper frequency cutoff
%year = 4 characters (i.e. 2021) - determines whether to process 10 min or
%30 min samples 
%
%Outputs:
%SPLrms = matrix of root mean square sound pressure level, each column is a sound file, each row is 1 min
%of data
%SPLpk = matrix of peak SPL (the highest SPL in a sample), each column is a sound file, each row is 1 min
%of data
%impulsivity = uses kurtosis function to measure impulsive sounds
%peakcount = a count of the # of times that the autocorrelation threshold is
%exceeded 
%autocorr = a matrix of autocorrelation calculations, each column is 1 min
%of data, each row is the autocorrelation value calculated at .1 sec time
%dissim = a matrix of the amount of uniformity between each 1 min of data
%compared to the following minute of data, should have n-1 rows where n is
%the number of minutes of data
profile on

file_dir = file_dir(3:end);
num_files = length(file_dir);

p = []; %empty variable
pout = [];
SPLrms = {}; %empty variable for SPLrms allows func to build from 0
SPLpk = {};
impulsivity = {};
peakcount = {};
autocorr = {};
dissim = {};

%acmax = autocorr(1:end);
%acmax(1:end) = 0;
%acmin = autocorr(1:end);
%acmin(1:end) = 0;
%acmean = autocorr(1:end);
%acmean(1:end) = 0;
%acmedian = autocorr(1:end);
%acmedian(1:end) = 0;


    tic %placing a tic at the start of the loop and toc at the end gives an estimate of the time taken to complete each loop
    for ii = 1:num_files %ii is an index value - completes 1 loop for every file until all files are analyzed
        ii %lists ii as a variable to tell you every time it completes a loop
        
        for kk = ii   
            kk
            disp(" out of " + num_files)
            
            properties(file_dir);
            
            filename = [file_dir(kk).folder '\' file_dir(kk).name]; %file_dir is a structure - this creates a filename for audioread

            rs = (10^(RS/20)); %convert from log to linear sensitivity
            max_count = 2^(num_bits); 
            conv_factor = peak_volts/max_count; %volts per bit

            [x,fs] = audioread(filename,'native');
            % disp("x" + x(1:5))
            % disp("x" + x(end - 4:end))
            
            %x = detrend(x,0); %removes DC offset, unnecessary because we
            %filter below 10 Hz
            %downsample by factor of 4.5 if sample rate is 576 kHz for a new fs of
            %128 kHz

            % fs

            if fs == 576000
                x = downsample(x,4);
                fs = fs/4;
            elseif fs == 288000
                x = downsample(x,2);
                fs = fs/2;
            elseif fs == 16000
                x = upsample(x,9);
                fs = fs*9;
            elseif fs == 512000
                roundNum = 4; % 3.5555555555555
                x = downsample(x,roundNum);
                fs = fs/3.5555555555555555555;
            end 

             if num_bits == 24
                x = bitshift(x,-8);            %Bitcount; accounts for audioread casting of 24 bit to 32 bit (zeroes behind)
             end

            v = double(x)*conv_factor;        %converts to volts, double prevents numbers from wrapping back to 0
            % disp("v size: " + size(v))
            % disp("v: " + v(1:10))
            p = v./rs;                         % voltage to pressure
            % disp("p size: " + size(p))
            % disp("p: " + p(1:5))

            if arti == 1 
                p = p(6*fs:end);               %trims first 4 sec of recording
                % disp("p 1: " + p(1:5))
            end

            % disp("v: " + v(1:20))
            % disp("p: " + p(1:20))

            pout = [pout;p];                   %make this so the new p gets added at end of original p
            % disp("pout size: " + size(pout))
            % disp("pout: " + pout(1:5))
            [a, b] = size(pout);
            % disp("pout middle: " + pout(a / 2 - 2:a / 2 + 2))
            % disp("pout last: " + pout(end - 2:end))
        end
        
        p = [];

        % disp("pout size: " + size(pout))

        p_filt = dylan_bpfilt(pout,1/fs,flow,fhigh);
        % [pfA, pfB] = size(p_filt)
        % disp("p_filt: " + p_filt(1:3))
        % disp("p_filt: " + p_filt(end-2:end))
                
        pout = [];
       
        pts_per_timewin = timewin * fs;  %number of samples per time window - set window * 576 kHz sample rate
        
        %if numsamp_timewin > 30          %sets a cutoff at 30 min of data analysis
            %numsamp_timewin = 30;
        %end
        num_timewin = floor(length(p_filt)/pts_per_timewin) + 1; %number of time windows contained in the sound file
        
        % Pad the signal to accommodate the extra time window
        padding_length = num_timewin * pts_per_timewin - length(p_filt);
        p_filt_padded = [p_filt; zeros(padding_length, 1)];
        
        % trimseries = p_filt(1:(pts_per_timewin*num_timewin)); %trims the
        % time series to make it fit into matrix columns % This code has
        % been commented out to so allow full 60 second minutes instead of
        % 58 second "minutes"
        
        
        
        % disp("trimseries: " + trimseries(1:3))
        % disp("trimseries: " + trimseries(end - 2:end))

        % Display the last 5 points of the first time window
        % disp(['trimseries pptw: ', num2str(trimseries(pts_per_timewin - 4:pts_per_timewin + 5)')]);

        % Display the last 5 points of the last time window
        % disp(['trimseries ntw: ', num2str(trimseries(end - 9:end)')]);

        % [tsPPTW, tsPPTW2] =  size(pts_per_timewin)
        % [tsTS, ~] =  size(trimseries)

        % disp("trimseries pptw: " + trimseries(tsPPTW - 4:tsPPTW))
        % disp("trimseries: " + trimseries(tsTS - 4:tsTS))

        % [sizeA, ~] = size(trimseries);

        % timechunk_matrix = reshape(trimseries,[pts_per_timewin,num_timewin]); %shapes trimmed series into sample# per time chunk rows x number of time windows in file columns
        timechunk_matrix = reshape(p_filt_padded, [pts_per_timewin, num_timewin]);
        % disp("p_filt_padded: " + p_filt_padded(1:10))
        % disp("pts_per_timewin: " + pts_per_timewin)
        % disp("num_timewin: " + num_timewin)
        
        % disp("tcm size " + size(timechunk_matrix))
        % disp("tcm: " + timechunk_matrix(1,:))
        % disp("tcm: " + timechunk_matrix(2,:))
        % disp("tcm: " + timechunk_matrix(end,:))

        %Amplitude
        % rms_matrix = rms(timechunk_matrix); %calculates the rms pressure of the matrix
        rms_matrix = rms(timechunk_matrix);
        % disp("rmsm size: " + size(rms_matrix))
        % disp("rms_matrix: " + rms_matrix)

        SPLrmshold = 20*log10(rms_matrix); %log transforms the rms pressure
        % disp("SPLrmshold shape: " + size(SPLrmshold))
        % disp("SPLrmshold: " + SPLrmshold)

        % disp("abs tcm shape: " + size(abs(timechunk_matrix)))
        % disp("abs: " + abs(timechunk_matrix(1:10)))
        % disp("l10 tcm shape: " + size(log10(abs(timechunk_matrix))))
        % disp("l10: " + log10(abs(timechunk_matrix(1:10))))
        % disp("l10_20 tcm shape: " + size(20 * log10(abs(timechunk_matrix))))
        % disp("l10_20: " + 20 * log10(abs(timechunk_matrix(1:10))))
        
        % disp("pk hold size: " + size(max(20*log10(abs(timechunk_matrix)))))
        
        % disp("tcm shape: " + size(timechunk_matrix))
        % disp("tcm: " + timechunk_matrix(1,:))
        % disp("tcm: " + timechunk_matrix(2,:))
        % disp("tcm: " + timechunk_matrix(end,:))
        
        
        SPLpkhold = max(20*log10(abs(timechunk_matrix))); %identifies the peak in rms pressure
        % disp(20*log10(abs(timechunk_matrix)))
        
        
        % disp("SPLpkhold: " + SPLpkhold)

        SPLrms = [SPLrms SPLrmshold']; %this var SPLrms is the outputted rms matrix
        SPLpk = [SPLpk SPLpkhold']; %generates the pk matrix
        %this means the function will give two output matrices [SPLrms SPLpk]
        %so you need to call them both when you use the function 

        %Impulsivity
        % disp("timechunk_matrix: " + timechunk_matrix(1:3))
        % disp("timechunk_matrix: " + timechunk_matrix(end - 2:end))
        
        % kmat = kurtosis(timechunk_matrix)
        
        % disp("tcm: " + timechunk_matrix(1,:))
        % disp("tcm: " + timechunk_matrix(2,:))
        % disp("tcm: " + timechunk_matrix(end,:))
        
        kmat = kurtosis(timechunk_matrix);
        % disp("kmat: " + kmat)
        
        impulsivity = [impulsivity kmat'];

        % disp("fs: " + fs)
        % disp("timewin: " + timewin)
        % disp("avtime: " + avtime)
        % disp("p_filt_padded: " + p_filt_padded(end-9:end))
        % disp("p_filt_padded: " + p_filt_padded(end-2:end))
        
        %Periodicity
        [pkcount, acorr] = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime);
        % disp("pkcount: " + pkcount)
        % disp("acorr: " + acorr)
        

        peakcount = [peakcount pkcount'];
        autocorr = horzcat(autocorr,acorr);
        
        acmax = max(autocorr);
        acmin = min(autocorr);
        acmean = mean(autocorr);
        acmedian = median(autocorr);

        %D-index
        % disp("tcm: " + timechunk_matrix(1,:))
        % disp("tcm: " + timechunk_matrix(2,:))
        % disp("tcm: " + timechunk_matrix(end,:))
        % disp(newline)
        % disp("pts_per_timewin: " + pts_per_timewin)
        % disp("num_timewin: " + num_timewin)
        % disp("fft_win: " + fft_win)
        % disp("fs: " + fs)
        
        [Dfin] = f_solo_dissim_GM1(timechunk_matrix,pts_per_timewin,num_timewin,fft_win,fs);
        dissim = [dissim Dfin];
        
    toc
    
    %autocmax = acmax / length(file_dir)
    %autocmin = acmin / length(file_dir)
    %autocmean = acmean / length(file_dir)
    %autocmedian = acmedian / length(file_dir)
    
    end
    
%}

function [SPLrms, SPLpk, impulsivity, peakcount, autocorr, dissim] = f_WAV_frankenfunction_reilly(num_bits, peak_volts, file_dir, RS, timewin, avtime, fft_win, arti, flow, fhigh) 
    
    % disp(num_bits)
    % disp(peak_volts)
    % disp(file_dir)
    % disp(RS)
    % disp(timewin)
    % disp(avtime)
    % disp(fft_win)
    % disp(arti)
    % disp(flow)
    % disp(fhigh)

    % Initialize output variables as cell arrays
    SPLrms = {};
    SPLpk = {};
    impulsivity = {};
    peakcount = {};
    autocorr = {};
    dissim = {};

    file_dir = file_dir(3:end);
    num_files = length(file_dir);

    p = []; 
    pout = [];

    for ii = 1:num_files
        ii
        tic
        for kk = ii   
            kk
            disp(" out of " + num_files)
            
            filename = [file_dir(kk).folder '\' file_dir(kk).name];

            rs = (10^(RS/20));
            max_count = 2^(num_bits);
            conv_factor = peak_volts/max_count;

            [x,fs] = audioread(filename,'native');
            % disp("fs:" + fs);
            % disp("x size:" + size(x))
            % disp("x:" + x(1:10));
            % disp("...");
            % disp("x:" + x(end-9:end));
            
            
            if fs == 576000
                x = downsample(x,4);
                fs = fs/4;
            elseif fs == 288000
                x = downsample(x,2);
                fs = fs/2;
            elseif fs == 16000
                x = upsample(x,9);
                fs = fs*9;
            elseif fs == 8000
                x = upsample(x,18);
                fs = fs*18;
            elseif fs == 512000
                x = downsample(x,4);
                fs = fs/3.5555555555555555555;
            end
            
            % disp("fs:" + fs);
            % disp("x:" + x(1:3));
            % disp("...");
            % disp("x:" + x(end-2:end));

            if num_bits == 24
                x = bitshift(x,-8);            
            end
            
            % disp("x:" + x(end-19:end))
            
            v = double(x)*conv_factor;
            % disp("v:" + v(1:20))
            % disp(size(v))
            p = v./rs;
            % disp("p:" + p(1:20))
            % [dimA, dimB] = size(p);
            % disp("p(-20:)" + p(dimA - 19: dimA))
            % disp(size(p))

            if arti == 1
                p = p(6*fs:end);
            end
            % disp("p:" + p(1:10))

            pout = [pout;p];
            [dimA, dimB] = size(pout);
            % disp("pout size: " + dimA + " " + dimB)
            % disp("pout:" + pout(1:20))
            % disp("...")
            % disp("pout:" + pout(end-19:end))
        end
        
        p = [];

        p_filt = dylan_bpfilt(pout,1/fs,flow,fhigh);
        % disp("p_filt size: " + size(p_filt))
        % disp("p_filt:" + p_filt(1:3))
        % disp("...")
        % disp(p_filt(end-2:end))
        
        pout = [];
       
        pts_per_timewin = timewin * fs;  
        num_timewin = floor(length(p_filt)/pts_per_timewin) + 1; 
        
        padding_length = num_timewin * pts_per_timewin - length(p_filt);
        p_filt_padded = [p_filt; zeros(padding_length, 1)];
        % disp("p_filt_padded" + p_filt_padded(1,:))
        
        timechunk_matrix = reshape(p_filt_padded, [pts_per_timewin, num_timewin]);
        % disp("tcm_matrix" + timechunk_matrix(1,:))
        
        rms_matrix = rms(timechunk_matrix);
        SPLrmshold = 20*log10(rms_matrix); 
        SPLpkhold = max(20*log10(abs(timechunk_matrix))); 

        kmat = kurtosis(timechunk_matrix);
        
        [pkcount, acorr] = f_solo_per_GM2(p_filt_padded, fs, timewin, avtime);

        [Dfin] = f_solo_dissim_GM1(timechunk_matrix, pts_per_timewin, num_timewin, fft_win, fs);
        
        % Store results in cell arrays
        SPLrms{end+1} = SPLrmshold';
        SPLpk{end+1} = SPLpkhold';
        impulsivity{end+1} = kmat';
        peakcount{end+1} = pkcount';
        autocorr{end+1} = acorr;
        dissim{end+1} = Dfin;
        
        toc
    end
    
    % Adjust arrays to consistent size and convert to matrices
    SPLrms = adjust_and_convert(SPLrms);
    SPLpk = adjust_and_convert(SPLpk);
    impulsivity = adjust_and_convert(impulsivity);
    peakcount = adjust_and_convert(peakcount);
    autocorr = adjust_and_convert(autocorr);
    dissim = adjust_and_convert(dissim);
end

function adjusted_matrix = adjust_and_convert(cell_array)
    % Find the maximum size
    sizes = cellfun(@size, cell_array, 'UniformOutput', false);
    max_size = max(cell2mat(sizes'));
    
    % Pad or trim each array to match the maximum size
    for i = 1:length(cell_array)
        current_size = size(cell_array{i});
        if any(current_size ~= max_size)
            padded_array = zeros(max_size);
            padded_array(1:current_size(1), 1:current_size(2)) = cell_array{i};
            cell_array{i} = padded_array;
        end
    end
    
    % Convert to matrix
    adjusted_matrix = cell2mat(cell_array);
end
