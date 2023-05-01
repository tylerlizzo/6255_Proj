function r = computeSnr(cleanSig, estimSig)
    errorSig = estimSig - cleanSig;

    r = snr(cleanSig, errorSig);
end