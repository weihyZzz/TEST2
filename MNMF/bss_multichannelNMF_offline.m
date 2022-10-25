function [sep,label] = bss_multichannelNMF_offline(mix,ns,option)
MNMF_nb = option.MNMF_nb;
MNMF_it = option.MNMF_it;
MNMF_fftSize = option.MNMF_fftSize; 
MNMF_shiftSize = option.MNMF_shiftSize;
MNMF_drawConv = option.MNMF_drawConv;
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
    % offline Multichannel NMF
    [Xhat,T,V,H,Z,~] = multichannelNMF(XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,G);
else
% offline Multichannel NMF
    [Xhat,T,V,H,Z,~] = multichannelNMF(XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv);
end
%% Ñ¡MIC²Ù×÷ 
Pft = zeros(I,J,M,M);Qft = zeros(I,J,M,M);Wft = zeros(I,J,M,M);um = eye(M);
for mm = 1:M*M
    Hmm1 = H(:,1,mm); Pft(:,:,mm) = ((Hmm1*Z(:,1)').*T)*V; % (35) of [1]
    Hmm2 = H(:,2:end,mm); Qft(:,:,mm) = ((Hmm2*Z(:,2:end)').*T)*V; % (36) of [1]
end 
for i=1:I
    for j=1:J
        for m=1:M
            Wft(i,j,m,:) = inv(squeeze(Pft(i,j,:,:) + Qft(i,j,:,:))) * squeeze(Pft(i,j,:,:)) * um(:,m);        
        end
    end
end
for i=1:I
    for j=1:J
        for m=1:M
            nume(i,j,m) = squeeze(Wft(i,j,m,:))' * squeeze(Pft(i,j,:,:)) * squeeze(Wft(i,j,m,:)); 
            deno(i,j,m) = squeeze(Wft(i,j,m,:))' * squeeze(Qft(i,j,:,:)) * squeeze(Wft(i,j,m,:));
        end
    end
end
for m=1:M
    fraction(m) = sum(sum(nume(:,:,m),1),2)/sum(sum(deno(:,:,m),1),2);
end
[~,refmic] = max(fraction);
refmic


% Multichannel Wiener filtering
Y = zeros(I,J,M,ns);
Xhat = permute(Xhat,[3,4,1,2]); % M x M x I x J
for i = 1:I
    for j = 1:J
        for src = 1:ns
            ys = 0;
            for k = 1:MNMF_nb
                ys = ys + Z(k,src)*T(i,k)*V(k,j);
            end
            Y(i,j,:,src) = ys * squeeze(H(i,src,:,:))/Xhat(:,:,i,j)*x(:,i,j); % M x 1
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