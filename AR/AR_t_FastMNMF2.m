function [sep, label] = AR_t_FastMNMF2(mix, N, option)
%%% AR_t_FastMNMF2 %%%
% Reference:
%    [1]Autoregressive Fast Multichannel Nonnegative Matrix Factorization for Joint Blind Source Separation and Dereverberation
%    [2]Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices
%       for Blind Source Separation
%    [3]Joint-Diagonalizability-Constrained Multichannel Nonnegative Matrix Factorization Based on Multivariate Complex 
%       Student's t-distribution
%
%    The joint blind souce separation and dereverberation method that integrates t_FastMNMF2 with the AR reverberation model.
%
%    X_FTM: the observed complex spectrogram
%    Q_FMM: diagonalizer that converts SCMs to diagonal matrices
%    G_NM: diagonal elements of the diagonalized SCMs
%    W_NFK: basis vectors
%    H_NKT: activations
%    PSD_NFT: power spectral densities   
%    Y_FTM: sum of (PSD_NFT x G_NM) over all sources
%    v: the degree of freedom parameter

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
v = option.tMNMF_v;

%%  load spectrum
[X_FTM, window] = STFT(mix, n_fft,shift_size, 'hamming');
signalScale = sum(mean(mean(abs(X_FTM).^2, 3), 2), 1);
X_FTM = X_FTM ./ signalScale; % signal scaling
[F, T, M] = size(X_FTM); % fftSize/2+1 x time frames x mics
Xbar_FxTxML = zeros(F, T, M * n_tap_AR); % (23) in [1]
for i = 1 : n_tap_AR
    Xbar_FxTxML(:, n_delay_AR + i : T, M * i - 1 : M * i) = X_FTM(:, 1 : T + 1 - i - n_delay_AR, :);
end
XXbar_FxTxMLxML = zeros(F, T, M * n_tap_AR, M * n_tap_AR);
for m1 = 1 : M * n_tap_AR
    for m2 = 1 : M * n_tap_AR
        XXbar_FxTxMLxML(:, :, m1, m2) = Xbar_FxTxML(:, :, m1) .* conj(Xbar_FxTxML(:, :, m2));
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
% calculate PSD
[PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS);
% calculate Y
[Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);
% calculate D
alpha_FT = max(rand(F, T), eps);
[D_FTM] = calculate_D(Q_FMM, X_FTM, Y_FTM, Xbar_FxTxML, XXbar_FxTxMLxML, alpha_FT);
% normalize
[Q_FMM, W_NFK, H_NKT, G_NM, Qd_FTM, Qd_power_FTM, PSD_NFT, Y_FTM] ...
    = normalize(Q_FMM, W_NFK, H_NKT, G_NM, D_FTM, EPS);

% pre calculate cost function
% cost = zeros(n_iter + 1, 1);
% cost(1) = cost_cal(Qd_power_FTM, Y_FTM, Q_FMM, v);
%% update loop
fprintf('Iteration:    ');
for it = start_idx : n_iter
    fprintf('\b\b\b\b%4d', it);

    % calculate the numerator of (10)
    DD_FxTxMxM = zeros(F, T, M, M);
    for m1 = 1 : M
        for m2 = 1 : M
            DD_FxTxMxM(:, :, m1, m2) = D_FTM(:, :, m1) .* conj(D_FTM(:, :, m2));
        end
    end

    % update W and calculate alpha
    alpha_FT = (2 * M + v) ./ (v + 2 * sum(Qd_power_FTM ./ Y_FTM, 3)); % (24) in [3]
    tmp1_NFT = zeros(N, F, T);
    tmp2_NFT = zeros(N, F, T);
    for n = 1 : N 
        tmp1_NFT(n, :, :) = squeeze(sum(permute(Qd_power_FTM ./ (Y_FTM .^ 2) .* alpha_FT, [3, 1, 2]) .* (G_NM(n, :).'), 1));
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

    % update H and calculate alpha
    alpha_FT = (2 * M + v) ./ (v + 2 * sum(Qd_power_FTM ./ Y_FTM, 3)); % (24) in [3]
    tmp1_NFT = zeros(N, F, T);
    tmp2_NFT = zeros(N, F, T);
    for n = 1 : N 
        tmp1_NFT(n, :, :) = squeeze(sum(permute(Qd_power_FTM ./ (Y_FTM .^ 2) .* alpha_FT, [3, 1, 2]) .* (G_NM(n, :).'), 1));
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

    % update G and calculate alpha
    alpha_FT = (2 * M + v) ./ (v + 2 * sum(Qd_power_FTM ./ Y_FTM, 3)); % (24) in [3]
    G_numerator = zeros(N, M);
    G_denominator = zeros(N, M);
    for n = 1 : N
        G_numerator(n, :) = squeeze(sum(squeeze(PSD_NFT(n, :, :)) .* alpha_FT .* (Qd_power_FTM ./ (Y_FTM .^ 2)), [1 2])); % numerator of (15) in [1]
        G_denominator(n, :) = squeeze(sum(squeeze(PSD_NFT(n, :, :)) .* (1 ./ Y_FTM), [1 2])); % denominator of (15) in [1]
    end
    G_NM = G_NM .* sqrt(G_numerator ./ G_denominator); % (15) in [1]
    % calculate Y
    [Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);

    % update Q and calculate alpha
    alpha_FT = (2 * M + v) ./ (v + 2 * sum(Qd_power_FTM ./ Y_FTM, 3)); % (24) in [3]
    for m = 1 : M
        tmp_FxMxM = squeeze(mean(DD_FxTxMxM .* alpha_FT ./ Y_FTM(:, :, m), 2)); % (10) in [1]
        for f = 1 : F
            tmp_MM = inv(squeeze(Q_FMM(f, :, :)));
            u_M = tmp_MM(:, m);
            denominator  = sqrt(u_M' / squeeze(tmp_FxMxM(f, :, :)) * u_M); % 求V公式10的分子部分共轭转置是自身
            Q_FMM(f, m, :) = conj((squeeze(tmp_FxMxM(f, :, :)) \ u_M) ./ denominator); % (11)(12) in [1] 取共轭是因为求V公式10的分母y_ftm，在计算公式12时需要做一次复共轭
        end
    end

    % calculate alpha, update B and calculate D
    [~, Qd_power_FTM] = calculate_Qd(Q_FMM, D_FTM);
    alpha_FT = (2 * M + v) ./ (v + 2 * sum(Qd_power_FTM ./ Y_FTM, 3)); % (24) in [3]
    [D_FTM] = calculate_D(Q_FMM, X_FTM, Y_FTM, Xbar_FxTxML, XXbar_FxTxMLxML, alpha_FT);

    % judge whether to normalize
    if mod(it, interval_norm) == 0
        % normalize
        [Q_FMM, W_NFK, H_NKT, G_NM, Qd_FTM, Qd_power_FTM, PSD_NFT, Y_FTM] ...
            = normalize(Q_FMM, W_NFK, H_NKT, G_NM, D_FTM, EPS);
    else
        % calculate Qd 
        [Qd_FTM, Qd_power_FTM] = calculate_Qd(Q_FMM, D_FTM);
    end
    
    % calculate cost function
%     cost(it + 1) = cost_cal(Qd_power_FTM, Y_FTM, Q_FMM, v);
end
fprintf('\nAR-t-FastMNMF2 done.\n');

%% separate
% calcluate Y_NFTM & Y_FTM
Y_NFTM = zeros(N, F, T, M);
for m = 1 : M
    Y_NFTM(:, :, :, m) = PSD_NFT .* G_NM(:, m);
end
Y_FTM = squeeze(sum(Y_NFTM, 1));
% calculate Qd 
[Qd_FTM, ~] = calculate_Qd(Q_FMM, D_FTM);
Qinv_FMM = zeros(F, M, M);
for f = 1 : F
    Qinv_FMM(f, :, :) = inv(squeeze(Q_FMM(f, :, :)));
end
separated_spec = zeros(N, F, T);
for n = 1 : N
    separated_spec(n, :, :) = sum(Qinv_FMM(:, mic_index, :) .* (Qd_FTM ./ Y_FTM) .* squeeze(Y_NFTM(n, :, :, :)), 3);
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
function [Q_FMM, W_NFK, H_NKT, G_NM, Qd_FTM, Qd_power_FTM, PSD_NFT, Y_FTM] ...
    = normalize(Q_FMM, W_NFK, H_NKT, G_NM, D_FTM, EPS)
[~, M] = size(G_NM);
% (16) in [1]
phi_F = real(sum((Q_FMM .* conj(Q_FMM)), [2 3])) / M; % cal trace, left part of (16) in [1]
Q_FMM = Q_FMM ./ sqrt(phi_F); % (16).1 in [1]
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
% calculate Qd
[Qd_FTM, Qd_power_FTM] = calculate_Qd(Q_FMM, D_FTM);
% calculate PSD
[PSD_NFT] = calculate_PSD(W_NFK, H_NKT, EPS);
% calculate Y
[Y_FTM] = calculate_Y(PSD_NFT, G_NM, EPS);
end

%% calculate D
function [D_FTM] = calculate_D(Q_FMM, X_FTM, Y_FTM, Xbar_FxTxML, XXbar_FxTxMLxML, alpha_FT)
% the definition of Q_FMM is under the formula (4)
[F, T, M] = size(X_FTM);
D_FTM = zeros(F, T, M);
[~, ~, ML] = size(Xbar_FxTxML);
for f = 1 : F
    psi_tmp_MMLxM = zeros(M * ML, M);
    phi_tmp_MMLxMMLxM = zeros(M * ML, M * ML, M);
    for m = 1 : M
        QH_MM = squeeze(Q_FMM(f, :, :))';
        q_M = QH_MM(:, m);
        % right part of (25) in [1]
%         psi_right_ML = sum((conj(squeeze(X_FTM(f, :, :))) * q_M) ./ Y_FTM(f, : ,m).' .* squeeze(Xbar_FxTxML(f, :, :)), 1)';
        psi_right_ML = sum((conj(squeeze(X_FTM(f, :, :))) * q_M) ./ Y_FTM(f, : ,m).' .* alpha_FT(f, :).' .* squeeze(Xbar_FxTxML(f, :, :)), 1)';
        % right part of (26) in [1]
%         phi_right_MLxML = squeeze(sum(squeeze(XXbar_FxTxMLxML(f, :, :, :)) ./ Y_FTM(f, :, m).' , 1)).';
        phi_right_MLxML = squeeze(sum(squeeze(XXbar_FxTxMLxML(f, :, :, :)) ./ Y_FTM(f, :, m).' .* alpha_FT(f, :).', 1)).';
        % left part of (26) in [1]
        QQ_MM = q_M * q_M';
        % (25) in [1]
        psi_tmp_MMLxM(:, m) = kron(q_M, psi_right_ML);
        % (26) in [1]
        phi_tmp_MMLxMMLxM(:, :, m) = kron(QQ_MM, phi_right_MLxML);
    end
    psi_tmp_MML = sum(psi_tmp_MMLxM, 2);
    phi_tmp_MMLxMML = sum(phi_tmp_MMLxMMLxM, 3);
    % (27) in [1]
    bhat_MML = phi_tmp_MMLxMML \ psi_tmp_MML;
%     for t = 1 : T
%         % (22) in [1]
%         Xhat_MxMML = kron(eye(M), squeeze(Xbar_FxTxML(f, t, :)).');
%         % (19) in [1]
%         D_FTM(f, t, :) = squeeze(X_FTM(f, t, :)) - Xhat_MxMML * bhat_MML;
%     end
    % (22) in [1]
    Xhat_MTxMML = kron(eye(M), squeeze(Xbar_FxTxML(f, :, :)));
    % (19) in [1]
    D_FTM(f, :, :) = squeeze(X_FTM(f, :, :)) - reshape(Xhat_MTxMML * bhat_MML, [T, M]);
end
end

%% calculate Qd 
function [Qd_FTM, Qd_power_FTM] = calculate_Qd(Q_FMM, D_FTM)
% the formula under (9) in [1]
[F, T, M] = size(D_FTM);
Qd_FTM = zeros(F, T, M);
for f = 1 : F
    Qd_FTM(f, :, :) = squeeze(D_FTM(f, :, :)) * squeeze(Q_FMM(f, :, :)).';
end
Qd_power_FTM = abs(Qd_FTM) .^ 2;
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

%% calculate cost function
function [cost] = cost_cal(Qd_power_FTM, Y_FTM, Q_FMM, v)
[F, T, M] = size(Y_FTM);
logQ = 0;
for f = 1 : F
    logQ = logQ + log(abs(det(squeeze(Q_FMM(f, :, :)))));
end
cost = -sum((v + 2 * M) / 2 * log(1 + 2 / v * sum(Qd_power_FTM ./ Y_FTM, 3)) + sum(log(Y_FTM), 3), 'all') + 2 * T * logQ;
end