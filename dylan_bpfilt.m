function [tsfilt, filtspec1] = dylan_bpfilt(ts, samint, flow, fhigh)

%%function [tsfilt] = my_bp_filter(ts, samint, flow, fhigh);
%
% inputs:	ts		= the time series
%		samint		= sampling interval
%		flow		= low frequency cut off
%		fhigh		= high frequency cut off  (zero if use nyquist)
%
% outputs	tsfilt		= filtered time series

% disp("ts len:" + size(ts))
% disp("ts:" + ts(1:20))
% disp("samint:" + samint)
% disp("flow:" + flow)
% disp("fhigh:" + fhigh)

npts = length(ts);
% disp("ts(2): " + ts(2))
reclen = npts*samint;

spec = fft(ts,npts);
% disp("spec(1:5): " + spec(1:5))
aspec = abs(spec);
% disp("aspec(1:5): " + aspec(1:5))
pspec = atan2(imag(spec), real(spec));
% disp("pspec(1:5): " + pspec(1:5))
freq = fftshift((-npts/2:npts/2-1)/reclen);
freqSize = length(freq);
% disp("freqSize: " + freqSize)

if fhigh == 0
	fhigh = 1/(2*samint);
end

ifr = find(abs(freq) >= flow & abs(freq) <= fhigh);
[~, ifrSizeB] = size(ifr);
% disp("ifr size: " + ifrSizeB)

% for i = 1:10
%     disp("ifr(" + i + ")" + ifr(i))
% end
% disp("")
% disp("First 100: ")
% for j = 1:100
%     disp("ifr(" + j + ")" + ifr(j))
% end
% 
% disp("Last 100: ")
% for j = ifrSizeB - 100:ifrSizeB
%     disp("ifr(" + j + ")" + ifr(j))
% end

% disp("Middle Values: ")
% for j = ifrSizeB / 2 - 2:ifrSizeB / 2 + 2
%     disp("ifr(" + j + ")" + ifr(j))
% end

disp("")
% for j = ifrSizeB - 5:ifrSizeB
%     disp("ifr(" + j + ")" + ifr(j))
% end
% for j = ifrSizeB - 3:ifrSizeB
%     disp("ifr(" + j + ")" + ifr(j))
% end

filtspec2 = zeros(size(spec));
% disp("filtspec2 size: " + size(filtspec2))

rspec = zeros(size(spec));
% disp("rspec size: " + size(rspec))

ispec = zeros(size(spec));
% disp("ispec size: " + size(ispec))

rspec(ifr) = aspec(ifr).*cos(pspec(ifr));
% disp("rspec 100: ")
% for j = 1:100
%     disp("rspec(" + j + "): " + rspec(j))
% end

ispec(ifr) = aspec(ifr).*sin(pspec(ifr));
% disp("ispec 100: ")
% for j = 1:100
%     disp("ispec(" + j + "): " + ispec(j))
% end

filtspec2 = rspec + i*ispec;

filtspec1 = abs(filtspec2(1:npts/2+1));
filtspec1(2:end-1) = 2*filtspec1(2:end-1);

% disp("filtspec1 100: ")
% for j = 1:100
%     disp("filtspec1(" + j + "): " + filtspec1(j))
% end

spec = [];
ifr = [];
freq = [];
aspec = [];
pspec = [];
rspec = [];
ispec = [];
rspec = [];
filtspec1 = [];
tsfilt = real(ifft(filtspec2, npts));
% disp("tsfilt 100: ")
% for j = 1:100
%     disp("tsfilt(" + j + "): " + tsfilt(j))
% end

return;
