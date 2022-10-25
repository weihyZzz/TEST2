function [x_out,label] = MLDR_online(mix,option)
%% online Maximum-Likelihood Distortionless Response Beamforming with SVE
%% Reference:
%  [1] Convolutional Maximum-Likelihood Distortionless Response Beamforming with Steering Vector Estimation for Robust Speech Recognition
%% Initialization
fft_size = option.MLDR_fft_size;
shift_size = option.MLDR_shift_size;
epsilon1 = option.MLDR_epsilon1; % robust estimation of lambda_tilde
zeta = option.MLDR_zeta; % diagonal loading
beta = option.MLDR_beta; % moving average weight of lambda_tilde
rho = option.MLDR_rho; % scaling factor to compensate for overestimation
epsilon1_hat = option.MLDR_epsilon1_hat; % robust estimation of Y
gamma_first5 = option.MLDR_gamma_first5; % gamma for first five frames
gamma_others = option.MLDR_gamma_others; % gamma for other frames
% Short-time Fourier transform
[X_FTM, window] = STFT(mix,fft_size,shift_size,'hamming');
[F,T,M] = size(X_FTM); % F x T x M
x_MFT = permute(X_FTM,[3,1,2]); % M x F x T
% t=0 initialization
lambda_F = zeros(F,1);
h_MF = ones(M,F);
w_MF = zeros(M,F);
Rx_MMF = zeros(M,M,F);
Rn_MMF = zeros(M,M,F);
psi_MMF = repmat(eye(M).*zeta, [1,1,F]); 
%% online iteration
fprintf('Frame:    ');
lambda_hat_FT = zeros(F,T);
Y_FT = zeros(F,T);
for t = 1:T    
    fprintf('\b\b\b\b%4d', t); 
    lambda_pre_F = lambda_F; h_pre_MF = h_MF; w_pre_MF = w_MF;
    Rx_pre_MMF = Rx_MMF; Rn_pre_MMF = Rn_MMF; psi_pre_MMF = psi_MMF;
    if t <= 5
        gamma = gamma_first5;
    else
        gamma = gamma_others;
    end
    Y_tilde_square_F = zeros(F,1);
    for f = 1:F
        Y_tilde_square_F(f) = abs(w_pre_MF(:,f)' * x_MFT(:,f,t)).^2; % (18) of [1]
    end
    lambda_hat_FT(:,t) = max(Y_tilde_square_F, epsilon1_hat); % (27) of [1] robust estimation of Y
    Rx_MMF = cal_Rx(Rx_pre_MMF, x_MFT(:,:,t), gamma, t, M, F); % (25) of [1]
    Rn_MMF = cal_Rn(Rn_pre_MMF, x_MFT(:,:,t), gamma, t, M, F, lambda_hat_FT, epsilon1_hat); % (26) of [1]
    Rs_MMF = Rx_MMF - rho .* Rn_MMF; % (24) of [1] rho is a scaling factor to compensate for overestimation
    h_MF = cal_h(h_pre_MF, Rs_MMF, M, F); % (28-30) of [1]
    lambda_tilde_F = max((beta .* lambda_pre_F + (1-beta) .* Y_tilde_square_F), epsilon1); % (19) of [1]
    psi_MMF = cal_psi(psi_pre_MMF, x_MFT(:,:,t), lambda_tilde_F, gamma, M, F); % (20) of [1]
    w_MF = cal_w(psi_MMF, h_MF, M, F); % (21) of [1]
    for f = 1:F
        Y_FT(f,t) = w_MF(:,f)' * x_MFT(:,f,t); % (22) of [1]
    end
    lambda_F = beta .* lambda_pre_F + (1-beta) .* abs(Y_FT(:,t)); % (23) of [1]
end
fprintf(' MLDR online done.\n');
x_out = ISTFT(Y_FT, shift_size, window, size(mix,1));

label = cell(1);
label{1} = 'target';

end

function Rx_MMF = cal_Rx(Rx_pre_MMF, x_MF, gamma, t, M, F)
% update Rx (25) of [1]
Rx_MMF = zeros(M,M,F);
nume = 0;
for m = 1:t-1
    nume = nume + gamma.^(t-m); % numerator of the left part in (25) of [1]
end
deno = nume + 1; % denominator in (25) of [1]
for f = 1:F
    Rx_MMF(:,:,f) = nume / deno .* Rx_pre_MMF(:,:,f) + x_MF(:,f) * x_MF(:,f)' ./ deno;
end
end

function Rn_MMF = cal_Rn(Rn_pre_MMF, x_MF, gamma, t, M, F, lambda_hat_FT, epsilon1_hat)
% update Rn (26) of [1]
Rn_MMF = zeros(M,M,F);
lambda_hat_FT = max(lambda_hat_FT, epsilon1_hat); % robust estimation of lambda_hat
nume_F = zeros(F,1);
for m = 1:t-1
    nume_F = nume_F + gamma.^(t-m) ./ lambda_hat_FT(:,m); % numerator of the left part in (26) of [1]
end
deno_F = nume_F + 1 ./ lambda_hat_FT(:,t); % denominator in (26) of [1]
for f = 1:F
    Rn_MMF(:,:,f) = nume_F(f) / deno_F(f) .* Rn_pre_MMF(:,:,f) + x_MF(:,f) * x_MF(:,f)' ./ lambda_hat_FT(f,t) ./ deno_F(f);
end
end

function h_MF = cal_h(h_pre_MF, Rs_MMF, M, F)
% update h (28)(29)(30) of [1]
h_MF = zeros(M,F);
for f = 1:F
    h_tilde_M = h_pre_MF(:,f); % (28) of [1] ?
    tmp_M = Rs_MMF(:,:,f) * h_tilde_M; % nume of (29) of [1]
    h_bar_M = tmp_M ./ norm(tmp_M); % (29) of [1]
    h_MF(:,f) = h_bar_M ./ h_bar_M(1); % (30) of [1]
end
end

function psi_MMF = cal_psi(psi_pre_MMF, x_MF, lambda_tilde_F, gamma, M, F)
% update psi (20) of [1]
psi_MMF = zeros(M,M,F);
for f = 1:F
    psi_MMF(:,:,f) = (psi_pre_MMF(:,:,f) - (psi_pre_MMF(:,:,f)*x_MF(:,f)*x_MF(:,f)'*psi_pre_MMF(:,:,f)) ./ ...
        (gamma*lambda_tilde_F(f)+x_MF(:,f)'*psi_pre_MMF(:,:,f)*x_MF(:,f))) ./ gamma;
end
end

function w_MF = cal_w(psi_MMF, h_MF, M, F)
% update w (21) of [1]
w_MF = zeros(M,F);
for f = 1:F
    w_MF(:,f) = psi_MMF(:,:,f) * h_MF(:,f) ./ (h_MF(:,f)' * psi_MMF(:,:,f) * h_MF(:,f)); % (21) of [1]
end
end