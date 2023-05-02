function xhat = denoise_specsub(y, fs, noiseLengthSec, nfft, noverlap)
    if nargin < 3
        noiseLengthSec = 3.0;
    end

    if nargin < 4
        nfft = 4096;
    end

    if nargin < 5
        noverlap = nfft/2;
    end

x=y(:,1).';  %remove the beginning of the sample
Nx=length(x);

apriori_SNR=0;  %select 0 for aposteriori SNR estimation and 1 for apriori
alpha=0.05;      %only used if apriori_SNR=1
beta1=0.5;
beta2=1;
lambda=3;

%STFT parameters
NFFT=4096;
window_length=round(0.031*fs); 
window=hamming(window_length);
window = window(:);
overlap=floor(0.45*window_length); %number of windows samples without overlapping

%Signal parameters
t_min=0.1;    %interval for learning the noise
t_max=3.00;   %spectrum (in second)

%construct spectrogram 
[S,F,T] = spectrogram(x+i*eps,window,window_length-overlap,NFFT,fs); %put a short imaginary part to obtain two-sided spectrogram
[Nf,Nw]=size(S);

%Noisy spectrum extraction
t_index=find(T>t_min & T<t_max);
absS_noise=abs(S(:,t_index)).^2;
noise_spectrum=mean(absS_noise,2); %average spectrum of the noise 
noise_specgram=repmat(noise_spectrum,1,Nw);

%Estimate SNR
absS=abs(S).^2;
SNR_est=max((absS./noise_specgram)-1,0); 
if apriori_SNR==1
    SNR_est=filter((1-alpha),[1 -alpha],SNR_est);
end  

%Compute Attenuation Map
an_lk=max((1-lambda*((1./(SNR_est+1)).^beta1)).^beta2,0); 
STFT=an_lk.*S;

%Compute Inverse STFT
ind=mod((1:window_length)-1,Nf)+1;
output_signal=zeros((Nw-1)*overlap+window_length,1);

for indice=1:Nw %Overlapp add technique
    left_index=((indice-1)*overlap) ;
    index=left_index+[1:window_length];
    temp_ifft=real(ifft(STFT(:,indice),NFFT));
    output_signal(index)= output_signal(index)+temp_ifft(ind).*window;
end
xhat = [output_signal;y(length(output_signal)+1:end)];
end