function [pkcount, acorr] = f_solo_per_GM2(p_filt,fs,timewin,avtime)    

    % disp("fs is " + fs)

    p_av = [];
    p_avtot = [];
    avwin = fs * avtime;
    % disp("avwin is " + avwin)

   
    sampwin = fs * timewin;
    % disp("sampwin is " + sampwin)
    
    % disp("Length of p_filt " + length(p_filt))
    ntwin = floor(length(p_filt)/sampwin); % Number of minutes
    % disp("ntwin is " + ntwin)

    p_filt = p_filt(1:sampwin*ntwin);
    % disp("1:10: " + p_filt(1:10))
    % disp("p_filt: " + p_filt(57500000-9:57500000))
    % disp("p_filt: " + p_filt(57504005-4:57504005-4))
    % disp("p_filt: " + p_filt(57550000-9:57550000))
    % disp("p_filt: " + p_filt(1:10))
    % disp("p_filt: " + p_filt(end-9:end))
    
    % point94338 = 72202533
    % point94339 = 72203298
    % point943395 = 72203681
    % point94340 = 72204063;
    % disp("p_filt: " + p_filt(point94340 + 1:point94340 + 10))
    p_filt = reshape(p_filt, sampwin, ntwin);
    % disp("p_filt shape is " + size(p_filt))
    % disp("p_filt: " + p_filt(1,:))
    % disp("p_filt: " + p_filt(end,:))
    % disp("p_filt: " + p_filt(2,:))
    % disp("p_filt: " + p_filt(end-9:end))
    
    % disp("p_filt: " + p_filt(size(p_filt) - 7:size(p_filt)))

    p_filt = p_filt.^2;
    % disp("p_filt shape is " + size(p_filt))
    % disp("p_filt: " + p_filt(1,:))
    % disp("p_filt: " + p_filt(2,:))
    % disp("p_filt: " + p_filt(3,:))
    % disp("p_filt: " + p_filt(4,:))
    % disp("p_filt: " + p_filt(end,:))
    % disp("p_filt: " + p_filt(1:15, 1:10))
    % disp("p_filt 1: " + p_filt(:, 1))
    % disp("p_filt 2: " + p_filt(:, 2))
    % disp("p_filt 3: " + p_filt(:, 3))
    
    % disp("p_filt: " + p_filt(end-9:end))
    
    % [pfA, pfB] = size(p_filt)
    % disp("p_filt: " + p_filt(1:3))
    % disp("p_filt: " + p_filt(end-2:end))

    numavwin = length(p_filt(:,1))/avwin;
    p_av = [];
    
    for jj = 1:ntwin
        
        avwinmatrix = reshape(p_filt(:,jj),[avwin numavwin]);
        % disp("avwinmatrix shape is " + size(avwinmatrix))
        
        % disp("avwinmatrix: " + avwinmatrix(jj,:))
        % disp("avwinmatrix: " + avwinmatrix(end,:))
        % disp("avwinmatrix: " + avwinmatrix(jj, :))
        
        % disp("p_filt: " + p_filt(1,:))
        % disp("p_filt: " + p_filt(end,:))
        % disp("p_f: " + p_filt(jj, :))
        
        p_avi = mean(avwinmatrix)';
        % disp("p_avi 1: " + p_avi(1))
        % disp("p_avi end: " + p_avi(end))
        % disp(newline)
        
        p_av = [p_av p_avi];
        % disp("p_av: " + p_av)
        % disp("p_av end: " + p_av(end,:))
    end
    % disp("p_av size: " + size(p_av))
    % disp("p_av: " + p_av)
    % disp("p_av: " + p_av(2,:))
    % disp("p_av: " + p_av(end,:))

    p_avtot = [p_avtot p_av];
    % disp("p_avtot is " + p_avtot(1:10))
    % disp("p_avtot is " + p_avtot(end-9:end))

    for zz = 1:size(p_avtot,2)

        acorr(:,zz) = correl_5(p_avtot(:,zz), p_avtot(:,zz), size(p_avtot,1)*.7, 0);
        [pks, ~] = findpeaks(acorr(:,zz),'minpeakprominence',.5);
        pkcount(zz) = length(pks);

    end
    % disp("pkcount: " + pkcount)
    % disp("acorr: " + acorr)
end
