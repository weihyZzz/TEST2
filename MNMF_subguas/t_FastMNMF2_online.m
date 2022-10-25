function [Yhat] = t_FastMNMF2_online(X,XX,N,K,maxIt,drawConv,v,trial,batch_size,batch1_size,rho,scalelap,G_NM,T_NFK,V_NKT,Q_FMM)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)  X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,v,trial
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2] Joint-Diagonalizability-Constrained Multichannel  Nonnegative Matrix Factorization Based on Multivariate Complex Student��s t-distribution%% see also% http://d-kitamura.net
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
    ginit = 2;%1-�����ʼ����2-�Խǳ�ʼ����3-Circular��ʼ����
    if ginit == 1
        G_NM = randn(N,M);
    elseif ginit == 2
        ita = 0;%1e-2;
        G_NM = ita * ones(M) - diag(diag(ita * ones(M))) + eye(M);%  (40) of [2] 
    elseif ginit == 3
%     G_init = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));zeros(N-mod(M,N),mod(M,N))]];% (41) of [2] g�г���1��Ԫ��Ϊ0
%     G_init = [eye(N),zeros(N,M-N)];% (41) of [2]
        ita = 1e-2;
        initg_f1 = ita * ones(N) - diag(diag(ita * ones(N))) + eye(N);% (41) of [2] 
        initg_f2 = ita * ones(mod(M,N)) - diag(diag(ita * ones(mod(M,N)))) + eye(mod(M,N));%  (41) of [2] 
        G_NM = [reshape(repmat(initg_f1,1,1,fix(M/N)),[N,N*fix(M/N)]),[initg_f2;ita * ones(N-mod(M,N),mod(M,N))]];% (41) of [2] g�г���1��Ԫ��Ϊ1e-2
    end
 end
if (nargin < 14)
    T_NFK = max(rand(N,F,K),eps);% F *K
        %% ��������ʵ��
%     shape = 2;shape_F = ones(1,F) * shape;
%     T_NFK = zeros(N,F,K);
%     for k = 1:K
%         T_NFK(:,:,k) = drchrnd(shape_F,N); %(78) of [4]
%     end
end
if (nargin < 15)
%     V = max(rand(N,K,T),eps)
    V_NKT = max(rand(N,K,batch_size),eps);
        %% gammaʵ��
%     shape = 2;power_observation = mean(abs(X).^2,'all');
%     V_NKT = max(gamrnd(shape,power_observation*F*M/(shape*N*K),[N K batch_size]),eps);%(79) of [4]�ڶ�������Ϊ��߶Ȳ������������еĵ�����python�汾һ��

end
% if (nargin < 12)
%     varZ = 0.01;
%     Z = varZ*rand(K,N) + 1/N;
%     Z = max( Z./sum(Z,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
% end
if (nargin < 16)
    Q_FMM = repmat(eye(M),1,1,F);
    Q_FMM = permute(Q_FMM,[3,1,2]); % F x M x M
end

if sum(size(T_NFK) ~= [N,F,K]) || sum(size(V_NKT) ~= [N,K,batch_size]) || sum(size(G_NM) ~= [N,M]) %|| sum(size(Z) ~= [K,N])
    error('The size of input initial variable is incorrect.\n');
end
%% Detection
%% Detection
detection=1;average_prenergy = scalelap;speech_flag = zeros(T,1);%% �趨����֡��������ֵ
for frame = 1:T
    detect_range1 = 1:round(0.25*F);detect_range2=round(0.75*F):F;
    front_energy = sum(sum(abs(X(detect_range1,frame,:)).^2));
    back_energy = sum(sum(abs(X(detect_range2,frame,:)).^2));% ��һ֡֡�ڵ�Ƶ���Ƶ������
    currentframe_energy = sum(sum(abs(X(:,frame,:)).^2));% ��һ֡��������
    if front_energy > 50*back_energy  && currentframe_energy > average_prenergy
        speech_flag(frame) = 1;
    end
end
%% Iterative update
fprintf('Iteration:    ');
    for it = 1:trial
        fprintf('\b\b\b\b%4d', it);V1 = max(rand(N,K,batch1_size),eps);
        [Q_FMM, G_NM, T1, V1] = normalize(Q_FMM, G_NM, T_NFK, V1, M);
        Yhat_FTM = local_Yhat( T_NFK, V1, G_NM, F, batch1_size, M, N); % (17) of [2] initial model tensor F *M *M 
        [ ~, T1, V1] = init_local_iterativeUpdate(X(:,1:batch1_size,:), XX, Yhat_FTM, T1, V1, G_NM, F, batch1_size, K, N, M ,Q_FMM, v);
    end
    fprintf(' t-fastMNMF trial_init done.\n');fprintf('\n');fprintf('Iteration:    ');
    
%     for it = 1:maxIt
%         fprintf('\b\b\b\b%4d', it);
%         [ Yhat_IJM, T, V, G, Q ] = local_iterativeUpdate(X, XX, Yhat_IJM, T, V, G, F, T, K, N, M ,Q,v);
%     end
    Yhat = zeros(F,T,M,N);
    batch_num = T - batch1_size + 1; % num of mini-batch
    fprintf('mini_batch_num:    ');
    for j = batch1_size:T
        fprintf('\b\b\b\b%4d', j);   
        if j == batch1_size
            Tjhat = zeros(N,F,K); Tnumehat = zeros(N,F,K); Tdenohat = zeros(N,F,K);
            Gjhat = zeros(N,M); Gnumehat = zeros(N,M); Gdenohat = zeros(N,M);Qj_FMM = Q_FMM;
            Vj_NKT = V1;Tj_NFK = T1;%max(rand(N,K,batch1_size),eps);
            XX_hat = XX(:,1:batch1_size,:,:);X_hat = X(:,1:batch1_size,:);
            Jj = batch1_size;
        else
            Tjhat = Tj_NFK; Tnumehat = Tnume; Tdenohat = Tdeno;
            Gjhat = Gj_NM; Gnumehat = Gnume; Gdenohat = Gdeno;
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
         Gj_NM = G_NM;
        Yhat_FTM = local_Yhat( Tj_NFK, Vj_NKT, Gj_NM, F, Jj, M ,N);% (6) of [2]
    % Iterative update
        for it = 1:maxIt
%             fprintf('\b\b\b\b%4d', it);
            [ Yhat_FTM, Tj_NFK, Vj_NKT, Gj_NM, Qj_FMM, Tnume, Tdeno, Gnume, Gdeno ] = ...
            local_iterativeUpdate( X_hat,XX_hat, Yhat_FTM, Tj_NFK, Vj_NKT, Gj_NM, F, Jj, K, N, M, Qj_FMM, v, Tjhat, Gjhat, Tnumehat, Tdenohat, Gnumehat, Gdenohat, rho );    
        end
    QX = zeros(F,Jj,M);
    for ii = 1:F
        QX(ii,:,:) = squeeze(X_hat(ii,:,:)) * squeeze(Qj_FMM(ii,:,:)).';% F* T* M
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
                        Yhat(f,t,:,src) = inv(squeeze(Qj_FMM(f,:,:)))*diag(ys * Gj_NM(src,:).'./( squeeze(Yhat_FTM(f,t,:)) +eps))* squeeze(QX(f,t,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
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
                    Yhat(f,j,:,src) = inv(squeeze(Qj_FMM(f,:,:)))*diag(ys *  Gj_NM(src,:).'./( squeeze(Yhat_FTM(f,batch_size,:)) +eps)) * squeeze(QX(f,batch_size,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
               end                      
            end
        end
    end

fprintf(' tFastMNMF done.\n');
end

%%% Xhat %%%
function [ Yhat_FTM ] = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M, N)
Yhat_FTM = zeros(F,T,M); TV_NFT = zeros(N,F,T);%N=M;
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;
end
for f = 1:F
    Yhat_FTM(f,:,:) = squeeze(TV_NFT(:,f,:)).'* G_NM; % under 1 line of (31) in [2] & (40) of [1]
end 
end

%%% Cost function %%% 
function [ cost ] = local_cost( X, Yhat_FTM, F, T, M ,Q_FMM)
QX_power = local_QX_power( Q_FMM, X, F, T, M);% F T M
sumq = 0;
for f = 1:F 
    sumq = sumq + log(det(squeeze(Q_FMM(f,:,:))*squeeze(Q_FMM(f,:,:))')); 
end
temp1 = squeeze(sum(sum(sum(QX_power./Yhat_FTM + log(Yhat_FTM)))));
cost = -temp1 + T * sumq; % (15) of [2]
end
%%% initIterative update %%%
function [ Yhat_FTM, T_NFK, V_NKT] = init_local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_NM, F, T, K, N, M ,Q_FMM,v)

%%%%% Update T %%%%%% F * K 
QX_power = local_QX_power( Q_FMM, X, F, T, M);
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]��ĸ�� QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Tnume ,Tdeno] = local_Tfrac( alpha, QY_FTM, Yhat_FTM, V_NKT, G_NM, F, T, K, M ,N); % F x K         % % (30) of [2]���Ӹ���
T_NFK = T_NFK.*max(sqrt(Tnume./Tdeno),eps); % (30) of [2] 
 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M , N); % (23) in [2]

%%%%% Update V %%%%%% K * T
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]��ĸ�� QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Vnume ,Vdeno] = local_Vfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, G_NM, F, T, K, M ,N); % F x K         % % (33) of [2]
V_NKT = V_NKT.*max(sqrt(Vnume./Vdeno),eps); % (33) of [2]

% [Q, G, T, V] = normalize(Q, G, T, V, M);
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M , N); % (23) in [2]

end

%%% Iterative update %%%
function [ Yhat_FTM, T_NFK, V_NKT, G_NM, Q_FMM, Tnume, Tdeno, Gnume, Gdeno] = local_iterativeUpdate( X, XX, Yhat_FTM, T_NFK, V_NKT, G_NM, F, T, K, N, M, Q_FMM, v, That, Ghat, Tnumehat, Tdenohat, Gnumehat, Gdenohat, rho )

%%%%% Update T %%%%%% F * K 
QX_power = local_QX_power( Q_FMM, X, F, T, M);
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]��ĸ�� QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Tnume ,Tdeno] = local_Tfrac( alpha, QY_FTM, Yhat_FTM, V_NKT, G_NM, F, T, K, M ,N); % F x K         % % (20) of [2]���Ӹ���
T_NFK = sqrt(Fjaxa(T_NFK, Tnume, That, Tnumehat, rho) ./ Fjx(Tdeno, Tdenohat, rho)); % (15) of [2]
% T = T.*max(sqrt(Tnume./Tdeno),eps); % (20) of [2] 
 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M, N ); % (23) in [2]

%%%%% Update V %%%%%% K * T
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]��ĸ�� QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Vnume ,Vdeno] = local_Vfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, G_NM, F, T, K, M ,N); % F x K         % % (33) of [2]
V_NKT = V_NKT.*max(sqrt(Vnume./Vdeno),eps); % (21) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M, N ); % (23) in [2]

%%%%% Update G %%%%%
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]��ĸ�� QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Gnume ,Gdeno] = local_Gfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, K); % F x K         % % (35) of [2]
G_NM = sqrt(Fjaxa(G_NM, Gnume, Ghat, Gnumehat, rho) ./ Fjx(Gdeno, Gdenohat, rho))+ 1e-10; % (15) of [2]
% G = G.*max(sqrt(Gnume./Gdeno),eps)+ 1e-10; %  (22) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M, N ); % (23) in [2]

%%%%% Update Q %%%%%
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_FTM,3));% (24) in [2]��ĸ�� QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% F * T (24) in [2]
V_FMMM  = local_V_FMMM( alpha, Q_FMM, XX, Yhat_FTM, F, T, M);
Q_FMM     = local_Q( V_FMMM, Q_FMM, F, T, M);% (36,37) of [2]

% normalize Q��G, T, V
[Q_FMM, G_NM, T_NFK, V_NKT] = normalize(Q_FMM, G_NM, T_NFK, V_NKT, M);

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M, N ); % (23) in [2]
end

%% normalize Q��G, T, V  
function [Q_FMM, G_NM, T_NFK, V_NKT] = normalize(Q_FMM, G_NM, T_NFK, V_NKT, M)
QQ = real(sum(sum(Q_FMM.*conj(Q_FMM),2),3)/M); % F *1
Q_FMM = Q_FMM./sqrt(QQ);% line 1 of (37) in [2]
T_NFK = T_NFK./QQ.' ;%  line 2 of (37) in [2]

G_sum = sum(real(G_NM),2);%  N * 1 
G_NM = G_NM./G_sum ;%  line 1 of (38) in [2]
T_NFK = T_NFK .* G_sum;%  line 2 of (38) in [2]
% 
T_sum = sum(T_NFK,2);% N *1* K
T_NFK = T_NFK./T_sum;%  (28) in [2]
V_NKT = V_NKT .* squeeze(T_sum);%  (28) in [2] size(squeeze(T_sum)) M*K
end

%%% Tfrac %%%
function [ Tnume, Tdeno ] = local_Tfrac( alpha, QY_FTM,Yhat_FTM, V_NKT, G_NM, F, T, K, M ,N)
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Tnume = zeros(N,F,K);Tdeno = zeros(N,F,K);  % QY_IJM =|q*x|^2 * y.^-2 ���ǣ�31��of [2]����g�Ĳ���
alphaQY_IJM = alpha.*  QY_FTM;% F * T * M
for f = 1:F
    GQY_NFT(:,f,:) = G_NM * squeeze(alphaQY_IJM(f,:,:)).';% �����е� alpha * beta (30) of [2] size(G)  % F * N * M 
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = G_NM(n,:) ./ squeeze(Yhat_FTM(:,t,:));   % (32) of [2] 
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));% 
for n=1:N
    Tnume(n,:,:) = squeeze(GQY_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; % ����(30) of [2] 
    Tdeno(n,:,:) = squeeze(GYhat_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; % ��ĸ(30) of [2] 
end
end 

%%% Vfrac %%%
function [ Vnume, Vdeno ] = local_Vfrac( alpha, QY_FTM,Yhat_FTM, T_NFK, G_NM, F, T, K, M ,N) % QY_IJM =|q*x|^2 * y.^-2 ���ǣ�31��of [2]����g�Ĳ���
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Vnume = zeros(N,K,T);Vdeno = zeros(N,K,T); %size(XX)  % F * T * M 
alphaQY_FTM = alpha.*  QY_FTM;% F * T * M
for f = 1:F
    GQY_NFT(:,f,:) = G_NM * squeeze(alphaQY_FTM(f,:,:)).';% �����е� alpha * beta (33) of [2] size(G)  % F * N * M 
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = G_NM(n,:) ./ squeeze(Yhat_FTM(:,t,:));    % (32) of [2]  
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Vnume(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GQY_NFT(n,:,:)); % ����(33) of [2] 
    Vdeno(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GYhat_NFT(n,:,:)); % ��ĸ(33) of [2] 
end
end

% %%% Zfrac %%%
% function [ Znume, Zdeno ] = local_Zfrac( alpha, QY_IJM,Yhat_IJM, T, V, G, F, T, K, M ,N) % QY_IJM =|q*x|^2 * y.^-2 ���ǣ�31��of [2]����g�Ĳ���
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
function [ Gnume, Gdeno ] = local_Gfrac( alpha, QY_FTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, K) % QY_IJM =|q*x|^2 * y.^-2 ���ǣ�31��of [2]����g�Ĳ���
TV_NFT = zeros(N,F,T);Gnumetmp = zeros(F,N,M);Gdeno = zeros(F,N,M);Gdeno_NFTM  = zeros(N,F,T,M);%size(XX)  % F * T * M
alphaQY_FTM = alpha.*  QY_FTM;% F * T * M   size(T):F,K size(V):K,T
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;% ���eps
end
for f = 1:F
    Gnumetmp(f,:,:) = squeeze(TV_NFT(:,f,:)) * squeeze(alphaQY_FTM(f,:,:));% ������t*v*alpha in (35) of [2] 
end
Gnume = squeeze(sum(Gnumetmp,1));
for n = 1:N
    for m = 1:M
    Gdeno_NFTM(n,:,:,m) = squeeze(TV_NFT(n,:,:)) ./ squeeze(Yhat_FTM(:,:,m)); % ��ĸ in (35) of [2]    
    end
end
Gdeno = squeeze(sum(sum(Gdeno_NFTM,3),2));
end

%%%% V_FMMM %%%
function [V_FMMM]  = local_V_FMMM( alpha, Q_FMM, XX, Yhat_FTM, F, T, M)
V_FTMMM = zeros(F,T,M,M,M);alphaXX = alpha .* XX ;
for m = 1:M
    V_FTMMM(:,:,m,:,:) = alphaXX./ Yhat_FTM(:,:,m);
end
V_FMMM = squeeze(mean(V_FTMMM,2));
end

%%% QX_power %%%
function [ QX_power ] = local_QX_power( Q_FMM, X, F, T, M)
QX_power = zeros(F,T,M);
for ii = 1:F
    QX_power(ii,:,:) = abs(squeeze(X(ii,:,:)) * squeeze(Q_FMM(ii,:,:)).').^2;% F* T* M
end
end

%%% Q  update%%%
function [ Q_FMM ] = local_Q( V_FMMM, Q_FMM, F, T, M)
ekm = eye(M);
for f = 1:F
    for m=1:M
        V_temp = squeeze(V_FMMM(f,m,:,:));
        q_fm = (squeeze(Q_FMM(f,:,:))*V_temp)\ekm(:,m);%  M * 1  inv(a)*ekm(:,1)=a\ekm(:,1)        
        q_fm = q_fm / sqrt(q_fm' * V_temp * q_fm);% 1 * M
        Q_FMM(f,m,:) = conj(q_fm); 
    end
end
end
%%% Fj(x) %%%
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
r = gamrnd(repmat(a,n,1),1,n,p);%gamrnd(A,B,M,N)A:shape;B:scale;M.N:����ά��
r = r ./ repmat(sum(r,2),1,p);
%���������dirichlet distribution��gamma�ֲ�֮��Ĺ�ϵ��dirichlet distribution���Կ����Ƕ��gamma(ai,1)�ĳ˻�������������
%ͬʱ������gamma�ķֲ���һ����Ҫ���ʣ�xi~gamma(ai,b)�ֲ�����sum(xi)~gamma(sum(ai),b)�ֲ���
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%