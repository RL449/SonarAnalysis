function [Dfin] = f_solo_dissim_GM1(timechunk_matrix,pts_per_timewin,num_timewin,fft_win,fs)
    % disp("tcm: " + timechunk_matrix(1,:))
    % disp("tcm: " + timechunk_matrix(2,:))
    % disp("tcm: " + timechunk_matrix(end,:))
    % disp(newline)
    % disp("pts_per_timewin: " + pts_per_timewin)
    % disp("num_timewin: " + num_timewin)
    % disp("fft_win: " + fft_win)
    % disp("fs: " + fs)


    pts_per_fft = fft_win * fs;                    %calc size fft window          pts_per_fft = fft_win * fs;
    numfftwin = floor(pts_per_timewin/pts_per_fft);   % # of fft windows
    
    Dfin = [];
    D = [];
    Di = [];
    
    for kk = 1:num_timewin - 1
        % analytic1 = hilbert(timechunk_matrix(:,kk));
        
        analytic1 = hilbert(timechunk_matrix(:,kk));
        % disp("a1: " + analytic1(1,:))
        
        % analytic2 = hilbert(timechunk_matrix(:,kk+1));
        
        analytic2 = hilbert(timechunk_matrix(:,kk+1));
        % disp("a2: " + analytic2(1,:))
        % disp(newline)
        
        at1 = abs(analytic1)/sum(abs(analytic1));
        % disp("at1 size: " + size(at1))
        % disp("at1: " + at1(1, :))
        at2 = abs(analytic2)/sum(abs(analytic2));
        % disp("at2 size: " + size(at2))
        % disp("at2: " + at2(1, :))
        
        Dt = sum(abs(at1 - at2))/2;
        
        s3a = timechunk_matrix(:,kk);
        s3a = s3a(1:pts_per_fft*numfftwin);
        % disp("s3a shape: " + size(s3a))
        s3a = reshape(s3a,[pts_per_fft numfftwin]);
        % disp("s3a: " + s3a(1,:))
        % disp("s3a: " + s3a(2,:))
        % disp("s3a: " + s3a(end,:))
        
        % disp("abs 1 fft: " + abs(fft(s3a(1,:))))
        % disp("abs end fft: " + abs(fft(s3a(end,:))))
        
        % disp(fft(s3a(1,:)))
        % disp(abs(fft(s3a(1,:))))
        % disp(size(s3a,1))
        
        ga = abs(fft(s3a))/size(s3a,1);
        % disp("ga shape: " + size(ga))
        % disp("ga1: " + ga(1,:))
        % disp("ga2: " + ga(2,:))
        % disp("gaend: " + ga(end,:))
        sfa = mean(ga,2);
        % disp("sfa: " + sfa(1:10))
        Sfa = abs(sfa)./sum(abs(sfa));
        % disp("Sfa: " + Sfa(1:10))
        
        s3b = timechunk_matrix(:,kk+1);
        % disp("s3b: " + s3b(1,:))
        s3b = s3b(1:pts_per_fft*numfftwin);
        % disp("s3b: " + s3b(1,:))
        % disp("s3b: " + s3b(end,:))
        % disp(newline)
        s3b = reshape(s3b,[pts_per_fft numfftwin]);
        % disp("s3b shape: " + size(s3b))
        % disp("s3b: " + s3b(1,:))
        % disp("s3b: " + s3b(end,:))
        
        gb = abs(fft(s3b))/size(s3b,1);
        % disp("gb shape: " + size(gb))
        % disp("gb: " + gb(1,:))
        % disp("gb: " + gb(end,:))
        
        sfb = mean(gb,2);
        % disp("sfb: " + sfb(1:10))
        Sfb = abs(sfb)./sum(abs(sfb));
        % disp("Sfb: " + Sfb(1:10))
        
        Df = sum(abs(Sfb - Sfa))/2;
        
        Di = Dt * Df;
        
        D = [D Di];

    end
    
    Dfin = [Dfin D'];
    end
    

