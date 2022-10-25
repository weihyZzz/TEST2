function [sep,label] = bss_multichannelNMF_online(mix,ns,option)
MNMF_nb = option.MNMF_nb;
MNMF_it = option.MNMF_it;
MNMF_first_batch = option.MNMF_first_batch;
MNMF_batch_size = option.MNMF_batch_size;
MNMF_rho = option.MNMF_rho;
MNMF_fftSize = option.MNMF_fftSize; 
MNMF_shiftSize = option.MNMF_shiftSize;
ILRMA_init = option.ILRMA_init;

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
    [~,~,W] = ILRMA(X(:,1:MNMF_first_batch,:),ILRMA_type,ILRMA_it,ILRMA_nb,ILRMA_drawConv,ILRMA_normalize); % N x M x I
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
    % Online Multichannel NMF
    Y = online_multichannelNMF(x,XX,ns,MNMF_nb,MNMF_it,MNMF_batch_size,MNMF_first_batch,MNMF_rho,G);
else
    Y = online_multichannelNMF(x,XX,ns,MNMF_nb,MNMF_it,MNMF_batch_size,MNMF_first_batch,MNMF_rho);
end
% % Y = zeros(I,J,M,2);
% % for i = 1:I
% %     for j = 1:J
% %         Wij1 = reshape(W1(i,j,:,:), [M, M]);
% %         Wij2 = reshape(W2(i,j,:,:), [M, M]);
% %         Xij = reshape(X(i,j,:), [M, 1]);
% %         Y(i,j,:,1) = Wij1' * Xij;
% %         Y(i,j,:,2) = Wij2' * Xij;
% %     end
% % end
Y = Y.*signalScale; % signal rescaling
for src = 1:ns
    sep(:,:,src) = ISTFT(Y(:,:,:,src), MNMF_shiftSize, window, size(mix,1));
end
label = cell(1,ns);
for k = 1:ns
    label{k} = 'target';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%