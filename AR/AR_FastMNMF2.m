function [sep, label] = AR_FastMNMF2(mix, N, option)
%%% AR_FastMNMF2 %%%
% Reference:
%    [1]Autoregressive Fast Multichannel Nonnegative Matrix Factorization for Joint Blind Source Separation and Dereverberation
%    There are some differences between the code and the refernence article.
%
%    The joint blind souce separation and dereverberation method that integrates FastMNMF2 with the AR reverberation model.
%
%    X_FTM: the observed complex spectrogram
%    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
%    P_FMM: the matrix obtained by multiplying diagonalizer and AR filter
%    G_NM: diagonal elements of the diagonalized SCMs
%    W_NFK: basis vectors
%    H_NKT: activations
%    PSD_NFT: power spectral densities
%    Px_power_FTM: power spectra of P_FMM times X_FTM
%    Y_FTM: sum of (PSD_NFT x G_NM) over all sources

%% init parameter
g_eps = 5e-2;
EPS = 1e-10;
mic_index = option.MNMF_refMic;
K = option.MNMF_nb;
n_fft = option.MNMF_fftSize;
shift_size = option.MNMF_shiftSize;
n_iter = option.MNMF_it;
n_iter_init = option.AR_init_it;
n_tap_AR = option.AR_tap; 
n_delay_AR = option.AR_delay;
init_SCM = char(option.AR_init_SCM);
interval_norm = option.AR_internorm;

%%  load spectrum
[X_FTM, window] = STFT(mix, n_fft,shift_size, 'hamming');
signalScale = sum(mean(mean(abs(X_FTM).^2, 3), 2), 1);
X_FTM = X_FTM ./ signalScale; % signal scaling
[F, T, M] = size(X_FTM); % fftSize/2+1 x time frames x mics
Xbar_FxTxMLa = zeros(F, T, M * (n_tap_AR + 1)); % size: F x T x M*(tap+1) ????FastMNMF????tap??X_FTM
Xbar_FxTxMLa(:, :, 1 : M) = X_FTM;
for i = 1 : n_tap_AR
    Xbar_FxTxMLa(:, n_delay_AR + i : T, M * i + 1 : M * i + M) = X_FTM(:, 1 : T + 1 - i - n_delay_AR, :);
end
XXbar_FxTxMLaxMLa = zeros(F, T, M * (n_tap_AR + 1), M * (n_tap_AR + 1));
for m1 = 1 : M * (n_tap_AR + 1)
    for m2 = 1 : M * (n_tap_AR + 1)
        XXbar_FxTxMLaxMLa(:, :, m1, m2) = Xbar_FxTxMLa(:, :, m1) .* conj(Xbar_FxTxMLa(:, :, m2));
    end
end

%% solve
% init source model
W_NFK = max(rand(N, F, K), eps);
H_NKT = max(rand(N, K, T), eps);

% init spatial model
start_idx = 1;
switch init_SCM
    case 'circular'
        Q_FMM = permute(repmat(eye(M), [1, 1, F]), [3, 1, 2]);
        G_NM = ones(N, M) * g_eps;
        for m = 1 : M
            G_NM(mod(m - 1, N) + 1, m) = 1;
        end
    case 'obs'
        XX_FMM = zeros(F, M, M);
        Q_FMM = zeros(F, M, M);
        G_NM = ones(N, M) / M;
        for f = 1 : F
            tmp_MM = squeeze(X_FTM(f, :, :)).' * conj(squeeze(X_FTM(f, :, :)));
            XX_FMM(f, :, :) = tmp_MM;
            [vec, val] = eig(tmp_MM);
            if f == 1
                [eig_val, index] = sort(diag(val)); % descend??????
                G_NM(1, :) = eig_val;
            else
                [~, index] = sort(diag(val)); % descend??????
            end
            Q_FMM(f, :, :) = vec(:, index)';   
        end        
    case 'ILRMA'
        ILRMA_type = option.ILRMA_type;
        ILRMA_nb = option.ILRMA_nb;
        ILRMA_it = option.ILRMA_it;
        ILRMA_normalize = option.ILRMA_normalize;
        ILRMA_drawConv = option.ILRMA_drawConv;
        [Y_FTN,~,ILRMA_W] = ILRMA(X_FTM,ILRMA_type,ILRMA_it,ILRMA_nb,ILRMA_drawConv,ILRMA_normalize); % N x M x F
        separated_spec_power = squeeze(mean(abs(Y_FTN),[1 2]));
        Q_FMM = permute(ILRMA_W,[3,1,2]);
        G_NM = ones(N,M) * 1e-2;
        for n = 1:N
            [~,max_index] = max(separated_spec_power);
            G_NM(n,max_index) = 1;
            separated_spec_power(max_index) = 0;
        end
    otherwise
        error('init_SCM should be circular, obs or ILRMA.\n')
end
[P_FxMxMLa, W_NFK, H_NKT, G_NM, Px_FTM, Px_power_FTM, PSD_NFT, Y_FTM] ...
    = normalize(Q_FMM, P_FxMxMLa, W_NFK, H_NKT, G_NM, Xbar_FxTxMLa, EPS);

%% update loop
fprintf('Iteration:    ');
for it = start_idx : n_iter
    fprintf('\b\b\b\b%4d', it);
    % update W
    tmp1_NFT = zeros(N, F, T);
    tmp2_NFT = zeros(N, F, T);
    for n = 1 : N 
        tmp1_NFT(n, :, :) = squeeze(sum(permute(Px_power_FTM ./ (Y_FTM .^ 2), [3, 1, 2]) .* (G_NM(n, :).'), 1));
        tmp2_NFT(n, :, :) = squeeze(sum(permute(1 ./ Y_FTM, [3, 1, 2]) .* (G_NM(n, :).'), 1));
    end
    W_numerator = zeros(N, F, K);
    W_denominator = zeros(N, F, K);
    for n = 1 : N
        W_numerator(n, :, :) = squeeze(tmp1_NFT(n, :, :)) * squeeze(H_NKT(n, :, :)).'; % numerator of (13) in [1]
        W_denominator(n, :, :) = squeeze(tmp2_NFT(n, :, :)) * squeeze(H_NKT(n, :, :)).'; % denominator of (13) in [1]
    end
    W_NFK = W_NFK .* sqrt(W_numerator ./ W_denominator); % (13) in [1]
    % calculate PSD
    [PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS);
    % calculate Y
    [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);

    % update H
    tmp1_NFT = zeros(N, F, T);
    tmp2_NFT = zeros(N, F, T);
    for n = 1 : N 
        tmp1_NFT(n, :, :) = squeeze(sum(permute(Px_power_FTM ./ (Y_FTM .^ 2), [3, 1, 2]) .* (G_NM(n, :).'), 1));
        tmp2_NFT(n, :, :) = squeeze(sum(permute(1 ./ Y_FTM, [3, 1, 2]) .* (G_NM(n, :).'), 1));
    end
    H_numerator = zeros(N, K, T);
    H_denominator = zeros(N, K, T);
    for n = 1 : N
        H_numerator(n, :, :) = squeeze(W_NFK(n, :, :)).' * squeeze(tmp1_NFT(n, :, :)); % numerator of (14) in [1]
        H_denominator(n, :, :) = squeeze(W_NFK(n, :, :)).' * squeeze(tmp2_NFT(n, :, :)); % denominator of (14) in [1]
    end
    H_NKT = H_NKT .* sqrt(H_numerator ./ H_denominator); % (14) in [1]
    % calculate PSD
    [PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS);
    % calculate Y
    [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);

    % update G
    G_numerator = zeros(N, M);
    G_denominator = zeros(N, M);
    for n = 1 : N
        G_numerator(n, :) = squeeze(sum(squeeze(PSD_NFT(n, :, :)) .* (Px_power_FTM ./ (Y_FTM .^ 2)), [1 2])); % numerator of (15) in [1]
        G_denominator(n, :) = squeeze(sum(squeeze(PSD_NFT(n, :, :)) .* (1 ./ Y_FTM), [1 2])); % denominator of (15) in [1]
    end
    G_NM = G_NM .* sqrt(G_numerator ./ G_denominator); % (15) in [1]
    % calculate Y
    [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);

    % update P
    for m = 1 : M
        tmp_FxMLaxMLa = squeeze(mean(XXbar_FxTxMLaxMLa ./ Y_FTM(:, :, m), 2));
        for f = 1 : F
            Vinv_MLtxMLt = inv(squeeze(tmp_FxMLaxMLa(f, :, :))); % (10) in [1]
            tmp_MM = inv(squeeze(P_FxMxMLa(f, :, 1 : M)));
            u_M = tmp_MM(:, m);
            denominator  = sqrt(u_M' * Vinv_MLtxMLt(1 : M, 1 : M) * u_M); % ??V??ʽ10?ķ??Ӳ??ֹ???ת????????
            P_FxMxMLa(f, m, :) = conj((Vinv_MLtxMLt(:, 1 : M) * u_M) ./ denominator); % (11)(12) in [1] ȡ????????Ϊ??V??ʽ10?ķ?ĸy_ftm???ڼ??㹫ʽ12ʱ??Ҫ??һ?θ?????
        end
        Q_FMM = P_FxMxMLa(:, :, 1 : M);
    end
    
    % judge whether to normalize
    if mod(it, interval_norm) == 0
        % normalize
        [P_FxMxMLa, W_NFK, H_NKT, G_NM, Px_FTM, Px_power_FTM, PSD_NFT, Y_FTM] ...
            = normalize(Q_FMM, P_FxMxMLa, W_NFK, H_NKT, G_NM, Xbar_FxTxMLa, EPS);
    else
        % calculate Px
        [Px_FTM, Px_power_FTM] = calculate_Px(P_FxMxMLa, Xbar_FxTxMLa);
    end    
end
fprintf('\nAR-FastMNMF2 done.\n');

%% separate
% calcluate Y_NFTM & Y_FTM
Y_NFTM = zeros(N, F, T, M);
for m = 1 : M
    Y_NFTM(:, :, :, m) = PSD_NFT .* G_NM(:, m);
end
Y_FTM = squeeze(sum(Y_NFTM, 1));
% calculate Px
[Px_FTM, ~] = calculate_Px(P_FxMxMLa, Xbar_FxTxMLa);
Qinv_FMM = zeros(F, M, M);
for f = 1 : F
    Qinv_FMM(f, :, :) = inv(squeeze(Q_FMM(f, :, :)));
end
separated_spec = zeros(N, F, T);
for n = 1 : N
    separated_spec(n, :, :) = sum(Qinv_FMM(:, mic_index, :) .* (Px_FTM ./ Y_FTM) .* squeeze(Y_NFTM(n, :, :, :)), 3);
end

%% inverse STFT
separated_spec = permute(separated_spec .* signalScale, [2, 3, 1]); % F x T x N
sep = ISTFT(separated_spec, shift_size, window, size(mix, 1));
% label setting
label = cell(1, N);
for k = 1 : N
    label{k} = 'target';
end
end

%% normalize
function [P_FxMxMLa, W_NFK, H_NKT, G_NM, Px_FTM, Px_power_FTM, PSD_NFT, Y_FTM] ...
    = normalize(Q_FMM, P_FxMxMLa, W_NFK, H_NKT, G_NM, Xbar_FxTxMLa, EPS)
[~, M] = size(G_NM);
% (16) in [1]
phi_F = real(sum((Q_FMM .* conj(Q_FMM)), [2 3])) / M; % cal trace, left part of (16) in [1]
P_FxMxMLa = P_FxMxMLa ./ sqrt(phi_F); % (16).1 in [1]
W_FNK = permute(W_NFK, [2, 1, 3]);
W_NFK = permute(W_FNK ./ phi_F, [2, 1, 3]); % (16).2 in [1]
% (17) in [1]
mu_N = sum(G_NM, 2); % left part of (17) in [1]
G_NM = G_NM ./ mu_N; % (17).1 in [1]
W_NFK = W_NFK .* mu_N; % (17).2 in [1]
% (18) in [1]
nu_NK = squeeze(sum(W_NFK, 2)); % left part of (18) in [1]
W_NKF = permute(W_NFK, [1, 3, 2]);
W_NFK = permute(W_NKF ./ nu_NK, [1, 3, 2]); % (18).1 in [1]
H_NKT = H_NKT .* nu_NK; % (18).2 in [1]
% calculate Px
[Px_FTM, Px_power_FTM] = calculate_Px(P_FxMxMLa, Xbar_FxTxMLa);
% calculate PSD
[PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS);
% calculate Y
[Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);
end

%% calculate Px 
function [Px_FTM, Px_power_FTM] = calculate_Px(P_FxMxMLa, Xbar_FxTxMLa)
% P_FxMxMLa and Xbar_FxTxMLa ????L֮֡ǰ?ź?
[F, M, ~] = size(P_FxMxMLa);
[~, T, ~] = size(Xbar_FxTxMLa);
Px_FTM = zeros(F, T, M);
for f = 1 : F
    Px_FTM(f, :, :) = squeeze(Xbar_FxTxMLa(f, :, :)) * squeeze(P_FxMxMLa(f, :, :)).'; % P_FxMxMLa includes Q_FMM
end
Px_power_FTM = abs(Px_FTM) .^ 2; % similar to the formula under (9) in [1]
end

%% calculate PSD
function [PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS)
[N, F, ~] = size(W_NFK);
[~, ~, T] = size(H_NKT);
PSD_NFT = zeros(N, F, T);
for n = 1 : N
    PSD_NFT(n, :, :) = squeeze(W_NFK(n, :, :)) * squeeze(H_NKT(n, :, :)) + EPS; % (3) in [1]
end
end

%% calculate Y
function [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS)
[~, F, T] = size(PSD_NFT);
[~, M] = size(G_NM);
Y_FTM = zeros(F, T, M);
for m = 1 : M
    Y_FTM(:, :, m) = squeeze(sum(PSD_NFT .* G_NM(:, m), 1)) + EPS; % under (6) in [1]
end
end
