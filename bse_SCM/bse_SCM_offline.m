function [sep,label] = bse_SCM_offline(mix,ns,option)
% [1] Blind Speech Extraction Based on Rank-Constrained Spatial Covariance Matrix Estimation With Multivariate Generalized Gaussian Distribution
MNMF_nb = option.MNMF_nb;
MNMF_it = option.MNMF_it;
MNMF_fftSize = option.MNMF_fftSize; 
MNMF_shiftSize = option.MNMF_shiftSize;
MNMF_drawConv = option.MNMF_drawConv;
alpha = option.SCM_alpha;
beta = 1e-16;
rou = 0.5;
% rou=option.rou;

delta = option.MNMF_delta; % to avoid numerical conputational instability

% Short-time Fourier transform
[X, window] = STFT(mix,MNMF_fftSize,MNMF_shiftSize,'hamming');
signalScale = sum(mean(mean(abs(X).^2,3),2),1);
X = X./signalScale; % signal scaling
[I,J,M] = size(X); % fftSize/2+1 x time frames x mics

% Obtain time-frequency-wise spatial covariance matrices
XX = zeros(I,J,M,M);
x = permute(X,[3,1,2]); % M x I x J
for i = 1:I
    for j = 1:J
        XX(i,j,:,:) = x(:,i,j)*x(:,i,j)' + eye(M)*delta; % observed spatial covariance matrix in each time-frequency slot
    end
end

% ILRMA initialization
ILRMA_type = option.ILRMA_type;
ILRMA_nb = option.ILRMA_nb;
ILRMA_it = option.ILRMA_it;
ILRMA_normalize = option.ILRMA_normalize;
ILRMA_dlratio = option.ILRMA_dlratio;
ILRMA_drawConv = option.ILRMA_drawConv;
[~, ~, T, V, Z, W] = ILRMA_bse(X,ILRMA_type,ILRMA_it,ILRMA_nb,ILRMA_drawConv,ILRMA_normalize); % N x M x I
% W ： N x M x I
Af = zeros(M,ns,I); % M x N x I
A = zeros(I,ns,M,M); % I x N x M x M
for i = 1:I
    Af(:,:,i) = inv(conj(W(:,:,i)));
end
AA = zeros(M,M,I);
for i = 1:I
%     for n = 1:ns
            AA(:,:,i) = Af(:,1,i) * Af(:,1,i)';
%     end
end
%% SCM initialization
  % rt
 rt_IJ = zeros(I,J);
for i = 1:I
    for j = 1:J
        for src = 1:ns           
            for k = 1:MNMF_nb
                rt_IJ(i,j) = rt_IJ(i,j) + Z(src,k)*T(i,k)*V(k,j);% (18) of [1]
            end
        end%size(rt)  I x J 
    end
end
  % Rn_hat nt=1 :第一个是源
  wx=zeros(ns,I,J);y_hat=zeros(ns,I,J);Rn_tmp=zeros(M,M,I,J);
for i = 1:I
    for j = 1:J
        for n = 1:ns
            wx(n,i,j) = W(n,:,i)*x(:,i,j);
        end
        wx(1,i,j) = 0;
        y_hat(:,i,j) = Af(:,:,i) * wx(:,i,j);% (16) of [1]
        Rn_tmp(:,:,i,j) = y_hat(:,i,j)*y_hat(:,i,j)';% (15) of [1]
    end
end
Rn_hat_MMI = squeeze(1/J * sum(Rn_tmp,4));% M x M x I   
  % rn
  rn_IJ = zeros(I,J);
for i = 1:I
    for j = 1:J
        rn_IJ(i,j)=1/M * y_hat(:,i,j)' * pinv(Rn_tmp(:,:,i,j)) * y_hat(:,i,j);% (19) of [1]
    end
end  

%% offline SCM
[rt_IJ,rn_IJ,Rx,Rn] = SCM_offline(x,MNMF_it,Af,rt_IJ,rn_IJ,Rn_hat_MMI,alpha,beta,rou);        
% Multichannel Wiener filtering  
Y = zeros(I,J,M,ns);
for i = 1:I
    Rx_inv = inv(Rx(:,:,i));
    for j = 1:J
        Y(i,j,:,1) = rt_IJ(i,j) * AA(:,:,i) * Rx_inv * x(:,i,j);
     %   Y(i,j,:,2) = rn(i,j) * Rn(:,:,i) * Rx_inv * x(:,i,j);
    end
end

% Inverse STFT for each source
Y = Y.*signalScale; % signal rescaling
for src = 1
    sep(:,:,src) = ISTFT(Y(:,:,:,src), MNMF_shiftSize, window, size(mix,1));
end

label = cell(1,ns);
for k = 1:ns
    label{k} = 'target';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%