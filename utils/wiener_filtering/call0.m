% voicebox
% addpath(genpath('D:\MATLAB\toolbox\voicebox'))
% save path
tic;
addpath(genpath("./src"));

devDataPath = "./data/dev";

devDataFolders = ls(devDataPath);
devDataFolders = devDataFolders(3:end, :);

nDevSamples = size(devDataFolders, 1);
% nDevSamples=10;
initSnr = NaN(nDevSamples, 1);
finalSnr = NaN(nDevSamples, 1);

for i = 1:nDevSamples
    disp(i)
    noisyFile = fullfile(devDataPath, devDataFolders(i, :), "noisy.wav");
    cleanFile = fullfile(devDataPath, devDataFolders(i, :), "clean.wav");

    [noisySig, fs] = audioread(noisyFile);
    [cleanSig, fsc] = audioread(cleanFile);

    assert(fs == fsc);

%     estimSig = denoise(noisySig, fs);
    estimSig = denoise_final0(noisySig, fs);
    
    initSnr(i) = computeSnr(cleanSig, noisySig);
    finalSnr(i) = computeSnr(cleanSig, estimSig);    
end

deltaSnr = finalSnr - initSnr;

fprintf("Input SNR (dB)");
descStats(initSnr);



fprintf("Output SNR (dB)");
descStats(finalSnr);

fprintf("Change in SNR (dB)");
descStats(deltaSnr);

toc
% figure;
% boxplot([initSnr, finalSnr], ...
%     'Notch', 'on', ...
%     'Labels', ["input", "baseline"] ...
% );
% title("SNR Distribution");
% xlabel("Algorithm");
% ylabel("SNR (dB)");
% grid;
% 
% figure;
% boxplot(deltaSnr, ...
%     'Notch', 'on', ...
%     'Labels', "baseline" ...
% );
% title("SNR Improvement Distribution");
% xlabel("Algorithm");
% ylabel("SNR Improvement (dB)");
% grid;

