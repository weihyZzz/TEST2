function [Yhat] = mpdf_FastMNMF_online(X,XX,N,K,maxIt,drawConv,v,trial,batch_size,batch1_size,rho,scalelap,pro,G_FNM,T_NFK,V_NKT,Q_NFMM)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)  X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,v,trial
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2] Joint-Diagonalizability-Constrained Multichannel  Nonnegative Matrix Factorization Based on Multivariate Complex Student’s t-distribution%% see also% http://d-kitamura.net
% [3] Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation 

% [syntax]
%   [T,V,H,Z,cost] = multichannelNMF(XX,N)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T,V)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T,V,H)
%   [T,V,H,Z,cost] = multichannelNMF(XX,N,K,it,drawConv,T,V,H,Z)
%
% [inputs]
%        XX: input 4th-order tensor (time-frequency-wise covariance matrices) (F x T x M x M)
%         N: number of sources
%         K: number of bases (default: ceil(T/10))
%        it: number of iteration (default: 300)
%  drawConv: plot cost function values in each iteration or not (true or false)
%         T: initial basis matrix (F x K)
%         V: initial activation matrix (K x T)
%         G: initial spatial covariance tensor (F x N x M )
%         Z: initial partitioning matrix (K x N)
%         Q: diagonalizer (F x M x M)
% [outputs]
%      Xhat: output 4th-order tensor reconstructed by T, V, H, and Z (F x T x M x M)
%         T: basis matrix (F x K)
%         V: activation matrix (K x T)
%         G: spatial covariance tensor (F x N x M x M)
%         Z: partitioning matrix (K x N)
%         Q: diagonalizer (F x M x M)
%      cost: convergence behavior of cost function in multichannel NMF (maxIt+1 x 1)

% Check errors and set default values
[F,T,M,M] = size(XX);
delta = 0.001; % to avoid numerical conputational instability
if size(XX,3) ~= size(XX,4)
    error('The size of input tensor is wrong.\n');
end
if (nargin < 4)
    K = ceil(T/10);
end
if (nargin < 5)
    maxIt = 300;
end
if (nargin < 6)
    drawConv = false;
end
if (nargin < 7)
    v = 5;
end
if (nargin < 8)
    trial = 10;
end
if (nargin < 9)
    batch_size = 4;
end
if (nargin < 10)
    batch1_size = 80;
end
if (nargin < 11)
     rho = 0.9; % default 0.9
end
if (nargin < 12)
     scalelap = 500/1000^2;
end
if (nargin < 13)
    pro = ones(1,N);% F x M x M
end
if (nargin < 14)
    ginit = 2;%1-随机初始化；2-对角初始化；3-Circular初始化；
    if ginit == 1
        G_FNM = randn(F,N,M);
    elseif ginit == 2
        ita = 0;%1e-2;
%         initg_f = ita * ones(M) - diag(diag(ita * ones(M))) + eye(M);%  (40) of [2] 
        initg_f = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));zeros(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为0
        %initg_f  = [eye(N);zeros(N,M-N)];% (41) of [2]
        G_FNM = repmat(initg_f,[1,1,F]);
        G_FNM = permute(G_FNM,[3,1,2]); % F x N x M  "N=M"  
    elseif ginit == 3
%     G_init = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));zeros(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为0
%     G_init = [eye(N),zeros(N,M-N)];% (41) of [2]
        ita = 1e-2;
        initg_f1 = ita * ones(N) - diag(diag(ita * ones(N))) + eye(N);% (41) of [2] 
        initg_f2 = ita * ones(mod(M,N)) - diag(diag(ita * ones(mod(M,N)))) + eye(mod(M,N));%  (41) of [2] 
        G_init = [reshape(repmat(initg_f1,1,1,fix(M/N)),[N,N*fix(M/N)]),[initg_f2;ita * ones(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为1e-2
        G_FNM = repmat(G_init,[1,1,F]);
        G_FNM = permute(G_FNM,[3,1,2]); % F x N x M  "M>N"
    end
 end
if (nargin < 15)
    T_NFK = max(rand(N,F,K),eps);% F *K
        %% 狄利克雷实现
%     shape = 2;shape_F = ones(1,F) * shape;
%     T_NFK = zeros(N,F,K);
%     for k = 1:K
%         T_NFK(:,:,k) = drchrnd(shape_F,N); %(78) of [4]
%     end
 
end
if (nargin < 16)
%     V_NKT = max(rand(N,K,T),eps);% K *T
     V_NKT = max(rand(N,K,batch_size),eps);
            %% gamma实现
%     shape = 2;power_observation = mean(abs(X).^2,'all');
%     V_NKT = max(gamrnd(shape,power_observation*F*M/(shape*N*K),[N K T]),eps);%(79) of [4]第二个参数为逆尺度参数，是文章中的倒数与python版本一致

end
% if (nargin < 12)
%     varZ = 0.01;
%     Z = varZ*rand(K,N) + 1/N;
%     Z = max( Z./sum(Z,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
% end
if (nargin < 17)
%     Q_FMM = repmat(eye(M),1,1,F);
%     Q_FMM = permute(Q_FMM,[3,1,2]); % F x M x M
    Q_NFMM = repmat(eye(M),1,1,N,F);
    Q_NFMM = permute(Q_NFMM,[3,4,1,2]); % F x M x M
end
%% Detection
detection=1;average_prenergy = scalelap;speech_flag = zeros(T,1);%% 设定语音帧的能量阈值
for frame = 1:T
    detect_range1 = 1:round(0.25*F);detect_range2=round(0.75*F):F;
    front_energy = sum(sum(abs(X(detect_range1,frame,:)).^2));
    back_energy = sum(sum(abs(X(detect_range2,frame,:)).^2));% 上一帧帧内低频与高频的能量
    currentframe_energy = sum(sum(abs(X(:,frame,:)).^2));% 上一帧的总能量
    if front_energy > 50*back_energy  && currentframe_energy > average_prenergy
        speech_flag(frame) = 1;
    end
end
%% [Q_NFMM, G_FNM, T_NFK, V_NKT] = normalize(Q_NFMM, G_FNM, T_NFK, V_NKT, M);
% Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N); % (6) of [2] initial model tensor F *M *M 
%% Iterative update
    cost = 0;
%     for it = 1:trial
%         fprintf('\b\b\b\b%4d', it);
%         [ Yhat_FTM, T_NFK, V_NKT] = init_local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM, v);
%     end
%     fprintf(' t-fastMNMF trial_init done.\n');fprintf('\n');
%% gradualinit 未改
    gradualinit = 0;initIt = 10;
    if gradualinit
        fprintf('initIteration:    ');
        for it = 1:initIt
            fprintf('\b\b\b\b%4d', it);
            [ Yhat_FTM, T_NFK, V_NKT, G_FNM, Q_NFMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_NFMM);
        end
        T_NFK = max(rand(N,F,K),eps);%K = K + 2; N *F *K
        V_NKT = max(rand(N,K,T),eps);% N *K *T  
%         Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N); % (17) of [2] initial model tensor F *M *M 
    end
    fprintf('\n');fprintf('Iteration:    ');
%% main iterate
    Yhat = zeros(F,T,M,N);
    batch_num = T - batch1_size + 1; % num of mini-batch
    fprintf('mini_batch_num:    ');
    for j = batch1_size:T
        fprintf('\b\b\b\b%4d', j);   
        if j == batch1_size
            Tjhat = zeros(N,F,K); Tnumehat = zeros(N,F,K); Tdenohat = zeros(N,F,K);
            Gjhat = zeros(F,N,M); Gnumehat = zeros(F,N,M); Gdenohat = zeros(F,N,M);Qj_NFMM = Q_NFMM;
            Vj_NKT = max(rand(N,K,batch1_size),eps);Tj_NFK = T_NFK;%V_NKT;max(rand(N,K,batch1_size),eps);
            XX_hat = XX(:,1:batch1_size,:,:);X_hat = X(:,1:batch1_size,:);
            Jj = batch1_size;
        else
            Tjhat = Tj_NFK; Tnumehat = Tnume; Tdenohat = Tdeno;
            Gjhat = Gj_FNM; Gnumehat = Gnume; Gdenohat = Gdeno;
            Vj_NKT = V_NKT;Jj = batch_size;Tj_NFK = T_NFK;%max(rand(N,K,batch_size),eps);
%             XX_hat = XX(:,j-batch_size+1:j,:,:);
%             X_hat = X(:,j-batch_size+1:j,:);
%% Detection
            XX_hatin = zeros(F,batch_size,M,M);X_hatin = zeros(F,batch_size,M); flag=0;speech_num = sum(speech_flag(1:j));
            if speech_num < batch_size/2
                XX_hatin = XX(:,j-batch_size+1:j,:,:);X_hatin = X(:,j-batch_size+1:j,:);
            else
                XX_hatin(:,batch_size,:,:) = XX(:,j,:,:);X_hatin(:,batch_size,:) = X(:,j,:);
                for jj = j-1:-1:1 
                    if speech_flag(jj) == 1
                        flag = flag+1;
                        XX_hatin(:,batch_size-flag,:,:) = XX(:,jj,:,:);X_hatin(:,batch_size-flag,:,:) = X(:,jj,:);
                    if flag == batch_size-1 break;end
                    end
                end
            end  
            XX_hat = XX_hatin;X_hat = X_hatin;
        end
         Gj_FNM = G_FNM;
        Yhat_FTM = local_Yhat( Tj_NFK, Vj_NKT, Gj_FNM, F, Jj, M ,N);% (6) of [2]
    % Iterative update
        for it = 1:maxIt
%             fprintf('\b\b\b\b%4d', it);
            [ Yhat_FTM, Tj_NFK, Vj_NKT, Gj_FNM, Qj_NFMM, Tnume, Tdeno, Gnume, Gdeno ] = ...
            local_iterativeUpdate( X_hat,XX_hat, Yhat_FTM, Tj_NFK, Vj_NKT, Gj_FNM, F, Jj, K, N, M, Qj_NFMM, v, Tjhat, Gjhat, Tnumehat, Tdenohat, Gnumehat, Gdenohat, rho ,pro);    
        end
    QX = zeros(N,F,Jj,M);
    for ns = 1:N
    for ii = 1:F
        QX(ns,ii,:,:) = squeeze(X_hat(ii,:,:)) * squeeze(Qj_NFMM(ns,ii,:,:)).';% F* T* M
    end 
    end
    % Multichannel Wiener filtering
        if j == batch1_size
            for f = 1:F   
                for t = 1:Jj       
                    for src = 1:N            
                        ys = 0;           
                        for k = 1:K               
                            ys = ys + Tj_NFK(src,f,k)*Vj_NKT(src,k,t); %lamda in (19) of [3]       
                        end                    
%                         Yhat(f,t,:,src) = ys * squeeze(G(f,src,:,:))/Yhat_IJM(:,:,f,t)*X(:,f,t); % (54) of [2] M x 1       
%                         Yhat(i,j,:,src) = inv(squeeze(Q(src,i,:,:)))*diag(ys * squeeze(G(i,src,:))./( squeeze(Xhat(i,j,:)) +eps))* squeeze(QX(src,i,j,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
                        Yhat(f,t,:,src) = inv(squeeze(Qj_NFMM(src,f,:,:)))*diag(ys * squeeze(Gj_FNM(f,src,:))./( squeeze(Yhat_FTM(f,t,:)) +eps))* squeeze(QX(src,f,t,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
                    end              
                end          
            end
        else
            for f = 1:F    
               for src = 1:N                 
                    ys = 0;            
                    for k = 1:K                                      
                        ys = ys + Tj_NFK(src,f,k)*Vj_NKT(src,k,batch_size); %lamda in (19) of [3]                               
                    end                
                    Yhat(f,j,:,src) = inv(squeeze(Qj_NFMM(src,f,:,:)))*diag(ys * squeeze(Gj_FNM(f,src,:))./( squeeze(Yhat_FTM(f,batch_size,:)) +eps)) * squeeze(QX(src,f,batch_size,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
%                     Yhat(f,t,:,src) = inv(squeeze(Qj_NFMM(src,f,:,:)))*diag(ys * squeeze(Gj_FNM(f,src,:))./( squeeze(Yhat_FTM(f,t,:)) +eps))* squeeze(QX(src,f,t,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
               end                      
            end
        end
    end 
fprintf(' mpdfFastMNMF_online done.\n');
end

%%% Xhat %%%
function [ Yhat_FTM ] = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N)
Yhat_FTM = zeros(F,T,M); TV_NFT = zeros(N,F,T);%N=M; G_joint = zeros(N,M,F);
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;
end
for f = 1:F
    Yhat_FTM(f,:,:) = squeeze(TV_NFT(:,f,:)).'* squeeze(G_FNM(f,:,:)); % under 1 line of (7) in [2] & (40) of [1]
end
end

%%% Cost function %%% 没改
function [ cost ] = local_cost( X, Yhat_FTM, F, T, M ,Q_FMM)
QX_power = local_QX_power( Q_FMM, X, F, T, M);% F T M
sumq = 0;
for f = 1:F 
    sumq = sumq + log(det(squeeze(Q_FMM(f,:,:))*squeeze(Q_FMM(f,:,:))')); 
end
temp1 = squeeze(sum(sum(sum(QX_power./Yhat_FTM + log(Yhat_FTM)))));
cost = -temp1 + T * sumq; % (15) of [2]
end
% %%% initIterative update %%% 没改
% function [ Yhat_FTM, T_NFK, V_NKT] = init_local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM,v)
% 
% %%%%% Update T %%%%%% F * K 
% QX_power = local_QX_power( Q_FMM, X, F, T, M);
% QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]分母里 QX_power/Xhat
% alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
% QY_FTM = (QX_power./(Yhat_FTM).^2);
% [Tnume ,Tdeno] = local_Tfrac( alpha, QY_FTM, Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N); % F x K         % % (30) of [2]分子更新
% T_NFK = T_NFK.*max(sqrt(Tnume./Tdeno),eps); % (30) of [2] 
%  
% Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M , N); % (23) in [2]
% 
% %%%%% Update V %%%%%% K * T
% QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]分母里 QX_power/Xhat
% alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
% QY_FTM = (QX_power./(Yhat_FTM).^2);
% [Vnume ,Vdeno] = local_Vfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N); % F x K         % % (33) of [2]
% V_NKT = V_NKT.*max(sqrt(Vnume./Vdeno),eps); % (33) of [2]
% 
% % [Q, G, T, V] = normalize(Q, G, T, V, M);
% Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M , N); % (23) in [2]
% 
% end

%%% Iterative update %%%
% function [ Yhat_FTM, T_NFK, V_NKT, G_FNM, Q_NFMM] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_NFMM,v)
function [ Yhat_FTM, T_NFK, V_NKT, G_FNM, Q_NFMM, Tnume, Tdeno, Gnume, Gdeno] = local_iterativeUpdate( X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M, Q_NFMM, v, That, Ghat, Tnumehat, Tdenohat, Gnumehat, Gdenohat, rho ,pro)

%%%%% Update T %%%%%% F * K 
QX_power = local_QX_power( Q_NFMM, X, F, T, M, N);% N F T M
% QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]分母里 QX_power/Xhat
QX_power_Xhat = squeeze(sum(squeeze(QX_power(1,:,:,:))./ (Yhat_FTM + eps),3));% (24) in [2]分母里 QX_power/Xhat, 源1用t分布QX_power(1,:,:)
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
% QY_FTM = (QX_power./(Yhat_FTM).^2);
for n=1:N QY_NFTM(n,:,:,:) = (squeeze(QX_power(n,:,:,:))./(Yhat_FTM + eps).^2);end
% [Tnume ,Tdeno] = local_Tfrac( alpha, QY_FTM, Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N); % F x K         % % (30) of [2]分子更新
[Tnume ,Tdeno] = local_Tfrac( alpha, QY_NFTM, Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N, pro); % F x K         % % (30) of [2]分子更新
% T_NFK = T_NFK.*max(sqrt(Tnume./Tdeno),eps); % (30) of [2] 
T_NFK = sqrt(Fjaxa(T_NFK, Tnume, That, Tnumehat, rho) ./ (Fjx(Tdeno, Tdenohat, rho)+eps)); % (15) of [2]
 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N ); % (23) in [2]

%%%%% Update V %%%%%% K * T
% QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]分母里 QX_power/Xhat
QX_power_Xhat = squeeze(sum(squeeze(QX_power(1,:,:,:))./ (Yhat_FTM + eps),3));% (24) in [2]分母里 QX_power/Xhat, 源1用t分布QX_power(1,:,:)
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
% QY_FTM = (QX_power./(Yhat_FTM).^2);
for n=1:N QY_NFTM(n,:,:,:) = (squeeze(QX_power(n,:,:,:))./(Yhat_FTM + eps).^2);end
% [Vnume ,Vdeno] = local_Vfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N); % F x K         % % (33) of [2]
[Vnume ,Vdeno] = local_Vfrac( alpha, QY_NFTM, Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N, pro); % F x K         % % (33) of [2]
V_NKT = V_NKT.*max(sqrt(Vnume./Vdeno),eps); % (33) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N ); % (23) in [2]

%%%%% Update G %%%%%
% QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]分母里 QX_power/Xhat
QX_power_Xhat = squeeze(sum(squeeze(QX_power(1,:,:,:))./ (Yhat_FTM + eps),3));% (24) in [2]分母里 QX_power/Xhat, 源1用t分布QX_power(1,:,:)
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
% QY_FTM = (QX_power./(Yhat_FTM).^2);
for n=1:N QY_NFTM(n,:,:,:) = (squeeze(QX_power(n,:,:,:))./(Yhat_FTM + eps).^2);end
% [Gnume ,Gdeno] = local_Gfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, K); % F x K         % % (35) of [2]
[Gnume ,Gdeno] = local_Gfrac( alpha, QY_NFTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, K, pro); % F x K         % % (35) of [2]
% G_FNM = G_FNM.*max(sqrt(Gnume./Gdeno),eps)+ 1e-10; %  (35) of [2]
G_FNM = sqrt(Fjaxa(G_FNM, Gnume, Ghat, Gnumehat, rho) ./ (Fjx(Gdeno, Gdenohat, rho)+eps))+ 1e-10; % (15) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N ); % (23) in [2]

%%%%% Update Q %%%%%
% QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]分母里 QX_power/Xhat
QX_power_Xhat = squeeze(sum(squeeze(QX_power(1,:,:,:))./ (Yhat_FTM + eps),3));% (24) in [2]分母里 QX_power/Xhat, 源1用t分布QX_power(1,:,:)
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
V_NFMMM  = local_V_FMMM( alpha, Q_NFMM, XX, Yhat_FTM, F, T, M, N, pro);
Q_NFMM     = local_Q( V_NFMMM, Q_NFMM, F, T, M, N);% (20,21) of [2]

% normalize Q，G, T, V
% [Q_NFMM, G_FNM, T_NFK, V_NKT] = normalize(Q_NFMM, G_FNM, T_NFK, V_NKT, M);

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N ); % (23) in [2]
end

%% normalize Q，G, T, V  未改
function [Q_FMM, G_FNM, T_NFK, V_NKT] = normalize(Q_FMM, G_FNM, T_NFK, V_NKT, M)
QQ = real(sum(sum(Q_FMM.*conj(Q_FMM),2),3)/M); % F *1
Q_FMM = Q_FMM./sqrt(QQ);% (26) in [3]
G_FNM = G_FNM./QQ ;% (26) in [3]

G_sum = sum(real(G_FNM),3);% F * N * 1 
G_FNM = G_FNM./G_sum ;%  (27) in [3]
T_NFK = T_NFK .* squeeze(sum(G_sum,2)).';%size( squeeze(sum(G_sum,2)).') =1 *F %sum(G_sum,2);% N * F * K  
 
T_sum = sum(T_NFK,2);% N *1* K
T_NFK = T_NFK./T_sum;%  (28) in [2]
V_NKT = V_NKT .* squeeze(T_sum);%  (28) in [2] size(squeeze(T_sum)) M*K
end

%%% Tfrac %%%
function [ Tnume, Tdeno ] = local_Tfrac( alpha, QY_NFTM,Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N, pro)
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Tnume = zeros(N,F,K);Tdeno = zeros(N,F,K);  % QY_IJM =|q*x|^2 * y.^-2 即是（31）of [2]除了g的部分
% alphaQY_FTM = alpha.*  QY_FTM;% F * T * M
% for f = 1:F
%     GQY_NFT(:,f,:) = squeeze(G_FNM(f,:,:)) * squeeze(alphaQY_FTM(f,:,:)).';% 分子中的 alpha * beta (30) of [2] size(G)  % F * N * M 
% end
alphaQY_NFTM = zeros(N,F,T,M);
for i = 1:N
    if(pro(i)==1)
        alphaQY_NFTM(i,:,:,:) = alpha.*  squeeze(QY_NFTM(i,:,:,:));% N * F * T * M
    elseif(pro(i)==2)
        alphaQY_NFTM(i,:,:,:) = squeeze(QY_NFTM(i,:,:,:));% N * F * T * M
    else
    end
end
for n = 1:N
    for f = 1:F
    GQY_NFT(n,f,:) = squeeze(G_FNM(f,n,:)).' * squeeze(alphaQY_NFTM(n,f,:,:)).';% 分子中的 alpha * beta (30) of [2] size(G)  % F * N * M 
    end
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = squeeze(G_FNM(:,n,:)) ./ squeeze(Yhat_FTM(:,t,:) + eps);   % (32) of [2] 
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));% 
for n=1:N
    Tnume(n,:,:) = squeeze(GQY_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; % 分子(30) of [2] 
    Tdeno(n,:,:) = squeeze(GYhat_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; % 分母(30) of [2] 
end
end 

%%% Vfrac %%%
function [ Vnume, Vdeno ] = local_Vfrac( alpha, QY_NFTM,Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N, pro) % QY_IJM =|q*x|^2 * y.^-2 即是（31）of [2]除了g的部分
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Vnume = zeros(N,K,T);Vdeno = zeros(N,K,T); %size(XX)  % F * T * M 
% alphaQY_FTM = alpha.*  QY_FTM;% F * T * M
% for f = 1:F
%     GQY_NFT(:,f,:) = squeeze(G_FNM(f,:,:)) * squeeze(alphaQY_FTM(f,:,:)).';% 分子中的 alpha * beta (33) of [2] size(G)  % F * N * M 
% end
% for i=1:size(alpha,1) alphaQY_NFTM(i,:,:,:) = squeeze(alpha(i,:,:)).*  QY_FTM;end% N * F * T * M
alphaQY_NFTM = zeros(N,F,T,M);
for i = 1:N
    if(pro(i)==1)
        alphaQY_NFTM(i,:,:,:) = alpha.*  squeeze(QY_NFTM(i,:,:,:));% N * F * T * M
    elseif(pro(i)==2)
        alphaQY_NFTM(i,:,:,:) = squeeze(QY_NFTM(i,:,:,:));% N * F * T * M
    else
    end
end
for n = 1:N
    for f = 1:F
    GQY_NFT(n,f,:) = squeeze(G_FNM(f,n,:)).' * squeeze(alphaQY_NFTM(n,f,:,:)).';% 分子中的 alpha * beta (30) of [2] size(G)  % F * N * M 
    end
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = squeeze(G_FNM(:,n,:)) ./ squeeze(Yhat_FTM(:,t,:) + eps);    % (32) of [2]  
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Vnume(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GQY_NFT(n,:,:)); % 分子(33) of [2] 
    Vdeno(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GYhat_NFT(n,:,:)); % 分母(33) of [2] 
end
end

% %%% Zfrac %%%
% function [ Znume, Zdeno ] = local_Zfrac( alpha, QY_IJM,Yhat_IJM, T, V, G, F, T, K, M ,N) % QY_IJM =|q*x|^2 * y.^-2 即是（31）of [2]除了g的部分
% GQY_NIJ = zeros(N,F,T);GYhat_NIJM = zeros(N,F,T,M);Znume_frac = zeros(N,F,K);Zdeno_frac = zeros(N,F,K); %size(XX)  % F * T * M 
% alphaQY_IJM = alpha .*  QY_IJM;% F * T * M
% for i = 1:F
%     GQY_NIJ(:,i,:) = squeeze(G(i,:,:)) * squeeze(alphaQY_IJM(i,:,:)).';
% end
% for n=1:N
%     for j=1:T
%     GYhat_NIJM(n,:,j,:) = squeeze(G(:,n,:)) ./ squeeze(Yhat_IJM(:,j,:));    
%     end
% end
% GYhat_NIJ = squeeze(sum(GYhat_NIJM,4));
% for n=1:N
%     Znume_frac(n,:,:) = squeeze(GQY_NIJ(n,:,:))* V.'; % N * F * K
%     Zdeno_frac(n,:,:) = squeeze(GYhat_NIJ(n,:,:))* V.'; 
% end
% Znume = squeeze(sum(permute(Znume_frac,[2,3,1]) .* T,1));%size(T):F,K size(V):K,T
% Zdeno = squeeze(sum(permute(Zdeno_frac,[2,3,1]) .* T,1));%size(T):F,K size(V):K,T
% end

%%% Gfrac %%% 
function [ Gnume, Gdeno ] = local_Gfrac( alpha, QY_NFTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, K, pro) % QY_IJM =|q*x|^2 * y.^-2 即是（31）of [2]除了g的部分
TV_NFT = zeros(N,F,T);Gnume = zeros(F,N,M);Gdeno = zeros(F,N,M);Gdeno_NFTM  = zeros(N,F,T,M);%size(XX)  % F * T * M
% alphaQY_FTM = alpha.*  QY_FTM;% F * T * M   size(T):F,K size(V):K,T
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;% 添加eps
end
% for f = 1:F
%     Gnume(f,:,:) = squeeze(TV_NFT(:,f,:)) * squeeze(alphaQY_FTM(f,:,:));% 分子中t*v*alpha in (35) of [2] 
% end
% for i=1:size(alpha,1) alphaQY_NFTM(i,:,:,:) = squeeze(alpha(i,:,:)).*  QY_FTM;end% N * F * T * M
alphaQY_NFTM = zeros(N,F,T,M);
for i = 1:N
    if(pro(i)==1)
        alphaQY_NFTM(i,:,:,:) = alpha.*  squeeze(QY_NFTM(i,:,:,:));% N * F * T * M
    elseif(pro(i)==2)
        alphaQY_NFTM(i,:,:,:) = squeeze(QY_NFTM(i,:,:,:));% N * F * T * M
    else
    end
end
for n = 1:N
    for f = 1:F
    Gnume(f,n,:) = squeeze(TV_NFT(n,f,:)).' * squeeze(alphaQY_NFTM(n,f,:,:));% 分子中t*v*alpha in (35) of [2] 
    end
end
for n = 1:N
    for m = 1:M
    Gdeno_NFTM(n,:,:,m) = squeeze(TV_NFT(n,:,:)) ./ squeeze(Yhat_FTM(:,:,m) + eps); % 分母 in (35) of [2]    
    end
end
Gdeno = permute(squeeze(sum(Gdeno_NFTM,3)),[2,1,3]);
end

%%%% V_FMMM %%%
function [V_NFMMM]  = local_V_FMMM( alpha, Q_NFMM, XX, Yhat_FTM, F, T, M, N, pro)
V_NFTMMM = zeros(N,F,T,M,M,M);%alphaXX = alpha .* XX ;
% alphaXX = squeeze(mean(alpha,1)) .* XX ;%对不同自由度系数取均值mean(alpha,1)alpha(2,:,:)
alphaXX = alpha .* XX ;%对不同自由度系数取均值mean(alpha,1)alpha(2,:,:)
for m = 1:M
    for i = 1:N
        if(pro(i)==1)
            V_NFTMMM(i,:,:,m,:,:) = alphaXX./ Yhat_FTM(:,:,m);% N * F * T * M
        elseif(pro(i)==2)
            V_NFTMMM(i,:,:,m,:,:) = XX./ Yhat_FTM(:,:,m);% N * F * T * M
        else
        end
    end
end
V_NFMMM = squeeze(mean(V_NFTMMM,3));
end

%%% QX_power %%%
function [ QX_power ] = local_QX_power( Q_NFMM, X, F, T, M, N)
QX_power = zeros(N,F,T,M);
for n=1:N
for f = 1:F
    QX_power(n,f,:,:) = abs(squeeze(X(f,:,:)) * squeeze(Q_NFMM(n,f,:,:)).').^2;% F* T* M
end
end
end

%%% Q  update%%%
function [ Q_NFMM ] = local_Q( V_NFMMM, Q_NFMM, F, T, M, N)
ekm = eye(M);
for n=1:N
for f = 1:F
    for m=1:M
        V_temp = squeeze(V_NFMMM(n,f,m,:,:));
        q_fm = (squeeze(Q_NFMM(n,f,:,:))*V_temp)\ekm(:,m);%  M * 1  inv(a)*ekm(:,1)=a\ekm(:,1)  % (20) of [2]       
        q_fm = q_fm / (sqrt(q_fm' * V_temp * q_fm) + eps);% 1 * M  % (21) of [2] 
        Q_NFMM(n,f,m,:) = conj(q_fm); 
    end
end
end
end
%% % Fj(x) %%%
function Fx = Fjx( X, Xhat, rho ) % (50) of [1]
Fx = X + rho .* Xhat;
end

%%% Fj(a,x,a) %%%
function Fx = Fjaxa( A, X, Ahat, Xhat, rho ) % (51) of [1] scalar 
Fx = A.*X.*A + rho.*Ahat.*Xhat.*Ahat;
end

%%% Fj(A,X,A) %%%
function Fx = FjAXA( A, X, Ahat, Xhat, rho ) % (51) of [1] matrix
Fx = A*X*A + rho.*Ahat*Xhat*Ahat;
end

%% Dirichlet Distribution %
function r = drchrnd(a,n)
p = length(a);
r = gamrnd(repmat(a,n,1),1,n,p);%gamrnd(A,B,M,N)A:shape;B:scale;M.N:矩阵维度
r = r ./ repmat(sum(r,2),1,p);
%充分利用了dirichlet distribution和gamma分布之间的关系。dirichlet distribution可以看作是多个gamma(ai,1)的乘积（包括除）。
%同时利用了gamma的分布的一个重要性质，xi~gamma(ai,b)分布，则sum(xi)~gamma(sum(ai),b)分布。
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%