function [sep,label] = subguas_FastMNMF_bss_offline(mix,ns,option)
% ns: source_num
% M : mic_num
% I : frequence bin
% J : time-frame bin
%Ref.[1] Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation

MNMF_nb = option.MNMF_nb;
MNMF_it = option.MNMF_it;
MNMF_fftSize = option.MNMF_fftSize; 
MNMF_shiftSize = option.MNMF_shiftSize;
MNMF_drawConv = option.MNMF_drawConv;

MNMF_first_batch = option.MNMF_first_batch;
MNMF_batch_size = option.MNMF_batch_size;
MNMF_rho = option.MNMF_rho;

delta = option.MNMF_delta; % to avoid numerical conputational instability

% Short-time Fourier transform
[X, window] = STFT(mix,MNMF_fftSize,MNMF_shiftSize,'hamming');
signalScale = sum(mean(mean(abs(X).^2,3),2),1);
X = X./signalScale; % signal scaling
[I,J,M] = size(X); % fftSize/2+1 x time frames x mics

averageScale = mean(sum(sum(abs(X).^2,3),1),2);scalelap = averageScale/signalScale.^2;

% Obtain time-frequency-wise spatial covariance matrices
XX = zeros(I,J,M,M);
x = permute(X,[3,1,2]); % M x I x J
for i = 1:I
    for j = 1:J
        XX(i,j,:,:) = x(:,i,j)*x(:,i,j)' + eye(M)*delta; % observed spatial covariance matrix in each time-frequency slot
    end
end
%% 子块
spec_coeff_num = MNMF_fftSize / 2 + 1;

global SubBlockSize; global SB_ov_Size;
parti = option.parti;
partisize = option.partisize;
if parti  % 子块算法初始化
    block_size = SubBlockSize;        block_overlap = floor(SubBlockSize*SB_ov_Size);
    block_starts = 1:block_size - block_overlap  :spec_coeff_num - block_size - 1;
    for n = 1:length(block_starts)
        partition_index{n} = block_starts(n):block_starts(n) + block_size - 1;
    end
else
    partition_index = {1:spec_coeff_num * partisize};
end
partition_size = cellfun(@(x) length(x), partition_index);

par1.num = length(partition_index);     par1.size = partition_size;     par1.index = partition_index;    par1.contrast = {'blank'};
par1.contrast_derivative =  {'blank'};

par2.num = length(partition_index);     par2.size = partition_size;    par2.index = partition_index;     par2.contrast =  {'blank'};
par2.contrast_derivative =  {'blank'};

if option.mix_model == 0 % 对于不用混合模型的情况，可以加入源3和源4
    par3.num = length(partition_index);    par3.size = partition_size;    par3.index = partition_index;    par3.contrast = {'blank'};
    par3.contrast_derivative =  {'blank'};
    
    par4.num = length(partition_index);    par4.size = partition_size;    par4.index = partition_index;    par4.contrast = {'blank'};
    par4.contrast_derivative =  {'blank'};
    
    partition = {par1, par2, par3, par4};
else
    partition = {par1, par2};
end
option.partition = partition;
parti = option.parti; % 是否对载波进行分块（子块）
select = option.select; % 是否选择子块
thFactor = option.thFactor; % 子块选择阈值
partition = option.partition; % 子块选择阈值
%%% select threshold
YY = reshape(XX,[I,J, M^2]);
YE_mean = zeros(1,M);
for m = 1:M
    YE_mean(:,m) = mean(mean(abs(YY(:,:,m^2))));
    YE_TH = YE_mean * thFactor; % select threshold
end
[spec_indices, par_select] = selectpar(partition, select, parti, YY, ns, I, YE_TH); % select & partition initalize
if ~select
    spec_indices = {[1:1:spec_coeff_num]}; % select off par.index{n}
end
option.spec_indices = spec_indices;
%% online subguass Multichannel NMF
%     [Xhat,T,V,G,Q,~] = subguas_FastMNMF_online(X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,option);
    [Y] = subguas_FastMNMF_online(X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,MNMF_batch_size,MNMF_first_batch,MNMF_rho,scalelap,option);
%     [Xhat,T,V,G,Z,Q,~] = subguas_FastMNMF(X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv);

% Inverse STFT for each source
Y = Y.*signalScale; % signal rescaling
for src = 1:ns
    sep(:,:,src) = ISTFT(Y(:,:,:,src), MNMF_shiftSize, window, size(mix,1));
end

label = cell(1,ns);
for k = 1:ns
    label{k} = 'target';
end
%%% Xhat %%%
end
function [ Xhat ] = local_Xhat( T, V, G, Z, I, J, M,Q)
Xhat = zeros(I,J,M,M); %N=M; G_joint = zeros(N,M,I);
for i=1:I 
    for src = 1:size(G,2)
        G_hat(i,src,:,:) = inv(squeeze(Q(i,:,:))) *diag(squeeze(G(i,src,:)))*inv(squeeze(Q(i,:,:))');
    end
end
for mm = 1:2*M
    Gmm = G_hat(:,:,mm); % I x N
    Xhat(:,:,mm) = ((Gmm*Z').*T)*V; % (17) of [2] & (40) of [1]
end %size(Xhat)      size(Gmm*Z')                     size(T)   size(V)
    % I * J * M      (I x N)*(N x K) =I(fre)*K        I*K       K*J(time frame)
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%