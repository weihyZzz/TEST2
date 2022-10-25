function [sep,label,cost] = t_MNMF_bss_offline(mix,ns,option)
% Ref.[1] Unsupervised Speech Enhancement Based on Multichannel NMF-Informed Beamforming for Noise-Robust Automatic Speech Recognition
MNMF_nb = option.MNMF_nb;
MNMF_it = option.MNMF_it;
MNMF_fftSize = option.MNMF_fftSize; 
MNMF_shiftSize = option.MNMF_shiftSize;
MNMF_drawConv = option.MNMF_drawConv;
ILRMA_init = option.ILRMA_init;
trial = option.tMNMF_trial;
v = option.tMNMF_v;
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
if ILRMA_init == 1
    ILRMA_type = option.ILRMA_type;
    ILRMA_nb = option.ILRMA_nb;
    ILRMA_it = option.ILRMA_it;
    ILRMA_normalize = option.ILRMA_normalize;
    ILRMA_dlratio = option.ILRMA_dlratio;
    ILRMA_drawConv = option.ILRMA_drawConv;
    [~,~,W] = ILRMA(X,ILRMA_type,ILRMA_it,ILRMA_nb,ILRMA_drawConv,ILRMA_normalize); % N x M x I
    Gf = zeros(M,ns,I); % M x N x I
    G = zeros(I,ns,M,M); % I x N x M x M
    for i = 1:I
        Gf(:,:,i) = inv(conj(W(:,:,i)));
    end
    for i = 1:I
        for n = 1:ns
            C = Gf(:,n,i)*Gf(:,n,i)';
            C = (C+C')/2 + eye(M)*ILRMA_dlratio;
            G(i,n,:,:) = C/trace(C);
        end
    end
    % t-MNMF
    [Xhat,T,V,G,Z,cost] = t_MNMF_offline(XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,v,trial,G);
else              
% t-MNMF
    [Xhat,T,V,G,Z,cost] = t_MNMF_offline(XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,v,trial);
end
%% Multichannel Wiener filtering
Y = zeros(I,J,M,ns);
Xhat = permute(Xhat,[3,4,1,2]); % M x M x I x J
for i = 1:I
    for j = 1:J
        for src = 1:ns
            ys = 0;
            for k = 1:MNMF_nb
                ys = ys + Z(k,src)*T(i,k)*V(k,j);
            end
            Y(i,j,:,src) = ys * squeeze(G(i,src,:,:))/Xhat(:,:,i,j)*x(:,i,j); % M x 1
        end%size(Y)  I x J x M x N 
    end
end

% Inverse STFT for each source
Y = Y.*signalScale; % signal rescaling
for src = 1:ns
    sep(:,:,src) = ISTFT(Y(:,:,:,src), MNMF_shiftSize, window, size(mix,1));
end

label = cell(1,ns);
for k = 1:ns
    label{k} = 'target';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%