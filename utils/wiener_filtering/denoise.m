function xhat_all = denoise(y, fs, nfft, noverlap)
    if nargin < 3
        noiseLengthSec = 3.0;
    end

    if nargin < 4
        nfft = 4096;
    end

    if nargin < 5
        noverlap = nfft/2;
    end

% seg = 131072;
% nfft = 32768;
% ratio = 2
% Input SNR (dB)min    	median 	mean   	max    	std    
% -11.90	1.54	0.43	10.92	6.71
% Output SNR (dB)min    	median 	mean   	max    	std    
% -11.79	3.24	2.70	11.50	4.89
% Change in SNR (dB)min    	median 	mean   	max    	std    
% -4.87	1.68	2.27	11.27	3.01

% seg = 131072;
% nfft = 65536;
% ratio = 1
%     Input SNR (dB)min    	median 	mean   	max    	std    
% -11.90	1.54	0.43	10.92	6.71
% Output SNR (dB)min    	median 	mean   	max    	std    
% -11.79	3.30	2.62	11.68	5.01
% Change in SNR (dB)min    	median 	mean   	max    	std    
% -4.15	1.67	2.18	11.24	2.78

% baseline
% -11.79  3.12  2.98  11.39  4.25

    seg = 131072; 
    % The sample rate is 44100, 131072 is about 3 seconds.
    % The rest segment use the original weiner filtering.
    % We estimate the noise profile respectively for each segment
    % We calculate the zero crossing rate in each segemnt and
    % find out the frame with max zero crossing number.
    l_ori = length(y);
    xhat_all = zeros(l_ori,1);
    n=fix(l_ori/seg);
    for ind=0:n-1
        x=y(1+ind*seg:ind*seg+seg);

        wlen=32768; 
        hop=16384;   
        ratio = 2;
 
        win=hanning(wlen);                      
        X=enframe(x,win,hop)';     
        fn=size(X,2);               
        zcr1=zeros(1,fn);                
        for i=1:fn
            z=X(:,i);                    
            for j=1: (wlen- 1)       
                 if z(j)* z(j+1)< 0       
                     zcr1(i)=zcr1(i)+1;  
                 end
            end
        end
        zcr1_0=zcr1(1:2:end);
        zcr1_1=zcr1(2:2:end);        
        l = fix(fn/(ratio*2));
        zccalc0 = zeros(1,l); 
        zccalc1 = zeros(1,l); 
        for i=1:l
            for j=1:ratio
                zccalc0(i)=zccalc0(i)+zcr1_0(ratio*(i-1)+j);
                zccalc1(i)=zccalc1(i)+zcr1_1(ratio*(i-1)+j);
            end
        end
        m0 = max(zccalc0);
        m1 = max(zccalc1);
        if m0>m1
            for k=1:l
                if zccalc0(k)==m0
                    index = k;
                end
            end
            t= wlen*ratio*(index-1)+1;
        else
            for k=1:l
                if zccalc1(k)==m1
                    index = k;
                end
            end
            t= hop+ wlen*ratio*(index-1)+1;
        end
        y = y(:); 
        noise_profile = y(t:t+wlen*ratio);
        xhat = wienerFilter(x, noise_profile, nfft, noverlap, fs);
        xhat_all(1+ind*seg:ind*seg+seg) = xhat;
    end
    remain = y(1+n*seg:end);
    noiseLengthSampl = floor(noiseLengthSec * fs);
    noise_profile_ori = y(1:noiseLengthSampl);
%     if length(remain)<length(noise_profile)
%         xhat_all(1+n*seg:end) = remain;
%     else
    xhat = wienerFilter(remain, noise_profile_ori, nfft, noverlap, fs);
    xhat_all(1+n*seg:end) = xhat;
%     end
    
end