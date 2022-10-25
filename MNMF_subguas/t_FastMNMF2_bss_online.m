function [sep,label] = t_FastMNMF2_bss_online(mix,ns,option)
% ns: source_num
% M : mic_num
% I : frequence bin
% J : time-frame bin
%Ref.[1] Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation

MNMF_nb = option.MNMF_nb;
MNMF_it = option.MNMF_it;
MNMF_first_batch = option.MNMF_first_batch;
MNMF_batch_size = option.MNMF_batch_size;
MNMF_rho = option.MNMF_rho;
MNMF_fftSize = option.MNMF_fftSize; 
MNMF_shiftSize = option.MNMF_shiftSize;
MNMF_drawConv = option.MNMF_drawConv;
v = option.tMNMF_v;
trial = option.tMNMF_trial;

delta = option.MNMF_delta; % to avoid numerical conputational instability

% Short-time Fourier transform
[X, window] = STFT(mix,MNMF_fftSize,MNMF_shiftSize,'hamming');
signalScale = sum(mean(mean(abs(X).^2,3),2),1);
averageScale = mean(sum(sum(abs(X).^2,3),1),2);scalelap = averageScale/signalScale.^2;
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

%% offline FastMNMF
    [Y] = t_FastMNMF2_online(X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,v,trial,MNMF_batch_size,MNMF_first_batch,MNMF_rho,scalelap);
% %%% qxº∆À„
% QX = zeros(I,J,M);
% for ii = 1:I
%     QX(ii,:,:) = squeeze(X_hat(ii,:,:)) * squeeze(Q(ii,:,:)).';% I* J* M
% end
% %% Multichannel Wiener filtering
% Y = zeros(I,J,M,ns);
% for i = 1:I
%     for j = 1:J
%         for src = 1:ns
%             ys = 0;
%             for k = 1:MNMF_nb
%                 ys = ys + T(src,i,k)*V(src,k,j);
%             end
%             Y(i,j,:,src) = inv(squeeze(Q(i,:,:)))*diag(ys * squeeze(G(i,src,:))./( squeeze(Xhat(i,j,:)) +eps))* squeeze(QX(i,j,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
%         end%size(Y)  I x J x M x N 
%     end
% end

% Inverse STFT for each source
Y = Y.*signalScale; % signal rescaling
for src = 1:ns
    sep(:,:,src) = ISTFT(Y(:,:,:,src), MNMF_shiftSize, window, size(mix,1));
end

label = cell(1,ns);
for k = 1:ns
    label{k} = 'target';
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%