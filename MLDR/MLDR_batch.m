function [x_out,label] = MLDR_batch(mix,option)
%% batch Maximum-Likelihood Distortionless Response Beamforming with SVE
%% Reference:
%  [1] Convolutional Maximum-Likelihood Distortionless Response Beamforming with Steering Vector Estimation for Robust Speech Recognition
%  [2] A Beamforming Algorithm Based on Maximum Likelihood of a Complex Gaussian Distribution With Time-Varying Variances for Robust Speech Recognition
%% Initialization
fft_size = option.MLDR_fft_size;
shift_size = option.MLDR_shift_size;
iteration = option.MLDR_iteration;
delta = option.MLDR_delta; % diagonal loading
epsilon = option.MLDR_epsilon; % flooring parameter of Rx_tilde
epsilon_hat = option.MLDR_epsilon_hat; % robust estimation of lambda
moving_lambda = option.MLDR_moving_lambda; % 是否采用moving average的方式更新lambda
moving_average = option.MLDR_moving_average; % average的范围 往前和往后移动的帧数
% Short-time Fourier transform
[X_FTM, window] = STFT(mix,fft_size,shift_size,'hamming');
[F,T,M] = size(X_FTM); % F x T x M

% Obtain time-frequency-wise spatial covariance matrices
XX_FTMM = zeros(F,T,M,M);
lambda_FT = zeros(F,T);
x_MFT = permute(X_FTM,[3,1,2]); % M x F x T
for f = 1:F
    for t = 1:T
        XX_FTMM(f,t,:,:) = x_MFT(:,f,t) * x_MFT(:,f,t)'; % observed spatial covariance matrix in each time-frequency slot
        lambda_FT(f,t) = (x_MFT(:,f,t)' * x_MFT(:,f,t)) ./ M;
    end
end
Rx_FMM = squeeze(mean(XX_FTMM,2)); % (8) of [1]
%% Iterative update
fprintf('Iteration:    ');
for iter = 1:iteration
    fprintf('\b\b\b\b%4d', iter); 
    Rn_FMM = cal_Rn(XX_FTMM, lambda_FT, epsilon_hat); % (10) of [1]
    Rs_FMM = Rx_FMM - Rn_FMM; % (7) of [1]
    h_MF = cal_h(Rs_FMM, F, M); % principal eigenvector of Rs_FMM
    Rx_tilde_FMM = squeeze(sum(XX_FTMM./max(lambda_FT,epsilon), 2)); % (6)(50) of [1]
    w_MF = cal_w(Rx_tilde_FMM, h_MF, F, M, delta); % (5) of [1]
    lambda_tmp_FT = zeros(F,T);
    for f = 1:F
        for t = 1:T
            lambda_tmp_FT(f,t) = abs(w_MF(:,f)' * x_MFT(:,f,t)).^2; % (4) of [1]
        end
    end  
    lambda_FT = lambda_tmp_FT;
    if moving_lambda % (17) of [2]
        lambda_tmp1_FT = zeros(F,T);
        for t = 1:T
            if t <= moving_average
                lambda_tmp1_FT(:,t) = mean(lambda_tmp_FT(:,1:t+moving_average),2);
            elseif t >= T+1-moving_average
                lambda_tmp1_FT(:,t) = mean(lambda_tmp_FT(:,t-moving_average:end),2);
            else
                lambda_tmp1_FT(:,t) = mean(lambda_tmp_FT(:,t-moving_average:t+moving_average),2);
            end
        end
        lambda_FT = lambda_tmp1_FT;
    end      
end
fprintf(' MLDR batch done.\n');
Y_FT = zeros(F,T);
for f = 1:F
    for t = 1:T
        Y_FT(f,t) = w_MF(:,f)' * x_MFT(:,f,t);
    end
end
x_out = ISTFT(Y_FT, shift_size, window, size(mix,1));

label = cell(1);
label{1} = 'target';

end

function Rn_FMM = cal_Rn(XX_FTMM, lambda_FT, epsilon_hat)
% update Rn (10) of [1]
lambda_FT = max(lambda_FT, epsilon_hat); % robust estimation of lambda in (10) of [1] page.7
Rn_tmp_FMM = squeeze(sum(XX_FTMM ./ lambda_FT, 2));
Rn_FMM = 1 ./ sum(1./lambda_FT, 2) .* Rn_tmp_FMM;
end

function h_MF = cal_h(Rs_FMM, F, M)
% calculate steering vector h
h_MF = zeros(M,F);
for f = 1:F
    [vec,val] = eig(squeeze(Rs_FMM(f,:,:)));
    [~,index] = sort(diag(val),'descend');
    h_MF(:,f) = vec(:,index(1));
end
end

function w_MF = cal_w(Rx_tilde_FMM, h_MF, F, M, delta)
% calculate w (5) of [1]
w_MF = zeros(M,F);
Rx_tilde_MMF = permute(Rx_tilde_FMM, [2,3,1]);
for f = 1:F
    w_MF(:,f) = (Rx_tilde_MMF(:,:,f)+eye(M)*delta)\h_MF(:,f) ./ (h_MF(:,f)'/(Rx_tilde_MMF(:,:,f)+eye(M)*delta)*h_MF(:,f)); % robust estimation of w (49) of [1]
end
end