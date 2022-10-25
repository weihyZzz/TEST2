function [sep, label] = FastMNMF2_nopart(mix, N, option)
%% FastMNMF2_nopart.m %%
%  FastMNMF2 without partition z version
%% init parameter
g_eps = 5e-2;
EPS = 1e-10;
mic_index = option.MNMF_refMic;
K = option.MNMF_nb;
n_fft = option.MNMF_fftSize;
shift_size = option.MNMF_shiftSize;
n_iter = option.MNMF_it;
init_SCM = char(option.AR_init_SCM);
interval_norm = option.AR_internorm;

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

%%  load spectrum
% [X_FTM, window] = STFT(mix, n_fft,shift_size, 'hamming');
signalScale = sum(mean(mean(abs(X_FTM).^2, 3), 2), 1);
X_FTM = X_FTM ./ signalScale; % signal scaling
[F, T, M] = size(X_FTM); % fftSize/2+1 x time frames x mics
XX_FTMM = zeros(F, T, M, M);
for m1 = 1 : M
    for m2 = 1 : M
        XX_FTMM(:, :, m1, m2) = X_FTM(:, :, m1) .* conj(X_FTM(:, :, m2));
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
                [eig_val, index] = sort(diag(val)); % descend降序？
                G_NM(1, :) = eig_val;
            else
                [~, index] = sort(diag(val)); % descend降序？
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
% normalize
[Q_FMM, W_NFK, H_NKT, G_NM, Qx_FTM, Qx_power_FTM, PSD_NFT, Y_FTM] ...
    = normalize(Q_FMM, W_NFK, H_NKT, G_NM, X_FTM, EPS);

% pre calculate cost function
% cost = zeros(n_iter + 1, 1);
% cost(1) = cost_cal(Qx_power_FTM, Y_FTM, Q_FMM);

%% update loop
fprintf('Iteration:    ');
for it = start_idx : n_iter
    fprintf('\b\b\b\b%4d', it);
   
    % update W
    tmp1_NFT = zeros(N, F, T);
    tmp2_NFT = zeros(N, F, T);
    for n = 1 : N 
        tmp1_NFT(n, :, :) = squeeze(sum(permute(Qx_power_FTM ./ (Y_FTM .^ 2), [3, 1, 2]) .* (G_NM(n, :).'), 1));
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
        tmp1_NFT(n, :, :) = squeeze(sum(permute(Qx_power_FTM ./ (Y_FTM .^ 2), [3, 1, 2]) .* (G_NM(n, :).'), 1));
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
        G_numerator(n, :) = squeeze(sum(squeeze(PSD_NFT(n, :, :)) .* (Qx_power_FTM ./ (Y_FTM .^ 2)), [1 2])); % numerator of (15) in [1]
        G_denominator(n, :) = squeeze(sum(squeeze(PSD_NFT(n, :, :)) .* (1 ./ Y_FTM), [1 2])); % denominator of (15) in [1]
    end
    G_NM = G_NM .* sqrt(G_numerator ./ G_denominator); % (15) in [1]
    % calculate Y
    [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);

    % update Q
    for m = 1 : M
        tmp_FxMxM = squeeze(mean(XX_FTMM ./ Y_FTM(:, :, m), 2));
        for f = 1 : F
            tmp_MM = inv(squeeze(Q_FMM(f, :, :)));
            u_M = tmp_MM(:, m);
            denominator  = sqrt(u_M' / squeeze(tmp_FxMxM(f, :, :)) * u_M); % 求V公式10的分子部分共轭转置是自身
            Q_FMM(f, m, :) = conj((squeeze(tmp_FxMxM(f, :, :)) \ u_M) ./ denominator); % (11)(12) in [1] 取共轭是因为求V公式10的分母y_ftm，在计算公式12时需要做一次复共轭
        end
    end

    % judge whether to normalize
    if mod(it, interval_norm) == 0
        % normalize
        [Q_FMM, W_NFK, H_NKT, G_NM, Qx_FTM, Qx_power_FTM, PSD_NFT, Y_FTM] ...
            = normalize(Q_FMM, W_NFK, H_NKT, G_NM, X_FTM, EPS);
    else
        % calculate Qx 
        [Qx_FTM, Qx_power_FTM] = calculate_Qx(Q_FMM, X_FTM);
    end
    
    % calculate cost function
%     cost(it + 1) = cost_cal(Qx_power_FTM, Y_FTM, Q_FMM);
end
fprintf('\nFastMNMF2_my done.\n');

%% separate
% calcluate Y_NFTM & Y_FTM
Y_NFTM = zeros(N, F, T, M);
for m = 1 : M
    Y_NFTM(:, :, :, m) = PSD_NFT .* G_NM(:, m);
end
Y_FTM = squeeze(sum(Y_NFTM, 1));
% calculate Qx 
[Qx_FTM, ~] = calculate_Qx(Q_FMM, X_FTM);
Qinv_FMM = zeros(F, M, M);
for f = 1 : F
    Qinv_FMM(f, :, :) = inv(squeeze(Q_FMM(f, :, :)));
end
separated_spec = zeros(N, F, T);
for n = 1 : N
    separated_spec(n, :, :) = sum(Qinv_FMM(:, mic_index, :) .* (Qx_FTM ./ Y_FTM) .* squeeze(Y_NFTM(n, :, :, :)), 3);
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
function [Q_FMM, W_NFK, H_NKT, G_NM, Qx_FTM, Qx_power_FTM, PSD_NFT, Y_FTM] ...
    = normalize(Q_FMM, W_NFK, H_NKT, G_NM, X_FTM, EPS)
[~, M] = size(G_NM);

phi_F = real(sum((Q_FMM .* conj(Q_FMM)), [2 3])) / M;
Q_FMM = Q_FMM ./ sqrt(phi_F);
W_FNK = permute(W_NFK, [2, 1, 3]);
W_NFK = permute(W_FNK ./ phi_F, [2, 1, 3]);

mu_N = sum(G_NM, 2);
G_NM = G_NM ./ mu_N;
W_NFK = W_NFK .* mu_N;

nu_NK = squeeze(sum(W_NFK, 2));
W_NKF = permute(W_NFK, [1, 3, 2]);
W_NFK = permute(W_NKF ./ nu_NK, [1, 3, 2]);
H_NKT = H_NKT .* nu_NK;
% calculate Qx
[Qx_FTM, Qx_power_FTM] = calculate_Qx(Q_FMM, X_FTM);
% calculate PSD
[PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS);
% calculate Y
[Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);
end

%% calculate Qx 
function [Qx_FTM, Qx_power_FTM] = calculate_Qx(Q_FMM, X_FTM)
% the formula under (15) in [1]
[F, T, M] = size(X_FTM);
Qx_FTM = zeros(F, T, M);
for f = 1 : F
    Qx_FTM(f, :, :) = squeeze(X_FTM(f, :, :)) * squeeze(Q_FMM(f, :, :)).';
end
Qx_power_FTM = abs(Qx_FTM) .^ 2;
end

%% calculate PSD
function [PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS)
[N, F, ~] = size(W_NFK);
[~, ~, T] = size(H_NKT);
PSD_NFT = zeros(N, F, T);
for n = 1 : N
    PSD_NFT(n, :, :) = squeeze(W_NFK(n, :, :)) * squeeze(H_NKT(n, :, :)) + EPS;
end
end

%% calculate Y
function [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS)
[~, F, T] = size(PSD_NFT);
[~, M] = size(G_NM);
Y_FTM = zeros(F, T, M);
for m = 1 : M
    Y_FTM(:, :, m) = squeeze(sum(PSD_NFT .* G_NM(:, m), 1)) + EPS;
end
end

%% calculate cost function
function [cost] = cost_cal(Qx_power_FTM, Y_FTM, Q_FMM)
[F, T, ~] = size(Y_FTM);
logQ = 0;
for f = 1 : F
    logQ = logQ + log(abs(det(squeeze(Q_FMM(f, :, :)))));
end
cost = -sum(Qx_power_FTM ./ Y_FTM + log(Y_FTM), 'all') + 2 * T * logQ;
end