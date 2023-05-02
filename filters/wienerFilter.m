function xhat = wienerFilter(y, n, nfft, noverlap, fs)
    len = length(y);
    Syy = pwelch(y, hann(nfft, "periodic"), noverlap, nfft, fs, "twosided");
    Snn = pwelch(n, hann(nfft, "periodic"), noverlap, nfft, fs, "twosided");
    Hspec = max(0, 1 - abs(Snn./Syy));
    h = ifft(Hspec);
    xhat = fftfilt(h, y);
    xhat = xhat(1:len);
end