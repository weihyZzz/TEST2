function [sep, label] = ILRMA_GWPE(mix, N, option)
%% init parameter
n_fft = option.MNMF_fftSize;
shift_size = option.MNMF_shiftSize;

%% GWPE
run_gwpe = option.run_gwpe;
if run_gwpe == 1
    fft_config.frame_len = n_fft;
    fft_config.frame_shift = shift_size;
    fft_config.fft_len = fft_config.frame_len;
    % GWPE config
    gwpe_config.K = 30; % default 50
    gwpe_config.delta = 2; % default 2
    gwpe_config.iterations = 10;
    % GWPE dereverb
    [X_FTM, window] = GWPE(mix, gwpe_config, fft_config);
else
    [X_FTM, window] = STFT(mix, n_fft, shift_size, 'hamming');
end

%% load spectrum
signalScale = sum(mean(mean(abs(X_FTM).^2, 3), 2), 1);
X_FTM = X_FTM ./ signalScale; % signal scaling

%% ILRMA
ILRMA_type = option.ILRMA_type;
ILRMA_nb = option.ILRMA_nb;
ILRMA_it = option.ILRMA_it;
ILRMA_normalize = option.ILRMA_normalize;
ILRMA_drawConv = option.ILRMA_drawConv;
[Y_FTN, ~, ~] = ILRMA(X_FTM,ILRMA_type,ILRMA_it,ILRMA_nb,ILRMA_drawConv,ILRMA_normalize); % N x M x F
% Y_FTN: estimated multisource signals in time-frequency domain (frequency bin x time frame x source)
fprintf('\nILRMA_GWPE done.\n');
%% inverse STFT
separated_spec = Y_FTN .* signalScale; % F x T x N
sep = ISTFT(separated_spec, shift_size, window, size(mix, 1));
% label setting
label = cell(1, N);
for k = 1 : N
    label{k} = 'target';
end

end