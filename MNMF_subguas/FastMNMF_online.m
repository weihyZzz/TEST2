function [Yhat] = FastMNMF_online(X,XX,N,K,maxIt,drawConv,batch_size,batch1_size,rho, scalelap ,option,G_FNM,T_NFK,V_NKT,Q_FMM)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2] Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation 
% [3] Unsupervised Speech Enhancement Based on Multichannel NMF-Informed Beamforming for Noise-Robust Automatic Speech Recognition
% [4] Semi-Supervised Multichannel Speech Enhancement With a Deep Speech Prior
% see also% http://d-kitamura.net
%
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
spec_indices = option.spec_indices;
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
%%%%%%%%%
if (nargin < 7)
    batch_size = 4;
end
if (nargin < 8)
    batch1_size = 80;
end
if (nargin < 9)
     rho = 0.9; % default 0.9
end
if (nargin < 10)
     scalelap = 500/1000.^2; % default 0.9
end
%%%%%%%%
if (nargin < 12)
    ginit = 2;%1-随机初始化；2-对角初始化；3-Circular初始化；
    if ginit == 1
        G_FNM = randn(F,N,M);
    elseif ginit == 2
        ita = 0;%1e-2;
%         initg_f = ita * ones(M) - diag(diag(ita * ones(M))) + eye(M);%  (40) of [2] 
        initg_f = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));ones(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为0
        G_FNM = repmat(initg_f,[1,1,F]);
        G_FNM = permute(G_FNM,[3,1,2]); % F x N x M  "N=M"  
    elseif ginit == 3
%     G_init = [eye(N),zeros(N,M-N)];% (41) of [2]
        ita = 1e-2;
        initg_f1 = ita * ones(N) - diag(diag(ita * ones(N))) + eye(N);% (41) of [2] 
        initg_f2 = ita * ones(mod(M,N)) - diag(diag(ita * ones(mod(M,N)))) + eye(mod(M,N));%  (41) of [2] 
        G_init = [reshape(repmat(initg_f1,1,1,fix(M/N)),[N,N*fix(M/N)]),[initg_f2;ita * ones(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为1e-2
        G_FNM = repmat(G_init,[1,1,F]);
        G_FNM = permute(G_FNM,[3,1,2]); % F x N x M  "M>N"
    end
end
if (nargin < 13)
    T_NFK = max(rand(N,F,K),eps);% N *F *K
    %% 狄利克雷实现
%     shape = 2;shape_F = ones(1,F) * shape;
%     T_NFK = zeros(N,F,K);
%     for k = 1:K
%         T_NFK(:,:,k) = drchrnd(shape_F,N); %(78) of [4]
%     end
    
end
if (nargin < 14)
%     V = max(rand(N,K,T),eps);% N *K *T
    V_NKT = max(rand(N,K,batch_size),eps);
    %% gamma实现
%     shape = 2;power_observation = mean(abs(X).^2,'all');
%     V_NKT = max(gamrnd(shape,power_observation*F*M/(shape*N*K),[N K T]),eps);%(79) of [4]第二个参数为逆尺度参数，是文章中的倒数与python版本一致
end
if (nargin < 15)
    Q_FMM = repmat(eye(M),1,1,F);
    Q_FMM = permute(Q_FMM,[3,1,2]); % F x M x M
end
if sum(size(T_NFK) ~= [N,F,K]) || sum(size(V_NKT) ~= [N,K,batch_size]) || sum(size(G_FNM) ~= [F,N,M]) 
    error('The size of input initial variable is incorrect.\n');
end
x = permute(X,[3,1,2]); 
% [Q, G, T, V] = normalize(Q, G, T, V, M);
% Yhat_IJM = local_Yhat( T, V, G, F, T, M ,N); % (17) of [2] initial model tensor F *M *M 
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
%% Iterative update
fprintf('Iteration:    ');
    batch_num = T - batch1_size + 1; % num of mini-batch
    fprintf('mini_batch_num:    ');
     Yhat = zeros(F,T,M,N);%if mod(T-batch1_size,batch_size)==0 J_index = [batch1_size:batch_size:T];else J_index = [batch1_size:batch_size:T,T]; end
    for j = batch1_size:T
        fprintf('\b\b\b\b%4d', j);   
        if j == batch1_size
            Tjhat = zeros(N,F,K); Tnumehat = zeros(N,F,K); Tdenohat = zeros(N,F,K);
            Gjhat = zeros(F,N,M); Gnumehat = zeros(F,N,M); Gdenohat = zeros(F,N,M);
            Vj_NKT = max(rand(N,K,batch1_size),eps);Qj_FMM = Q_FMM;
            XX_hat = XX(:,1:batch1_size,:,:);X_hat = X(:,1:batch1_size,:);
            Jj = batch1_size;
        else
            Tjhat = Tj_NFK; Tnumehat = Tnume; Tdenohat = Tdeno;
            Gjhat = Gj_FNM; Gnumehat = Gnume; Gdenohat = Gdeno;
%             Tjhat = zeros(N,F,K); Tnumehat = zeros(N,F,K); Tdenohat = zeros(N,F,K);
%             Gjhat = zeros(F,N,M); Gnumehat = zeros(F,N,M); Gdenohat = zeros(F,N,M);
            Vj_NKT = V_NKT;Jj = batch_size;%
%             XX_hat = XX(:,j-batch_size+1:j,:,:);
%             X_hat = X(:,j-batch_size+1:j,:);
 %% Detection
            XX_hatin = zeros(F,batch_size,M,M);X_hatin = zeros(F,batch_size,M); flag=0;speech_num = sum(speech_flag(1:j));
            if speech_num < fix(batch_size/2)
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
        Tj_NFK = T_NFK; Gj_FNM = G_FNM;%Qj = Q;
%         [Qj, Gj, Tj, Vj] = normalize(Qj, Gj, Tj, Vj, M);
        Yhat_FTM = local_Yhat( Tj_NFK, Vj_NKT, Gj_FNM, F, Jj, M ,N, spec_indices{1});
    % Iterative update
        for it = 1:maxIt
%             fprintf('\b\b\b\b%4d', it);
            [ Yhat_FTM, Tj_NFK, Vj_NKT, Gj_FNM, Qj_FMM, Tnume, Tdeno, Gnume, Gdeno ] = ...
            local_iterativeUpdate( X_hat, XX_hat, Yhat_FTM, Tj_NFK, Vj_NKT, Gj_FNM, F, Jj, K, N, M, Qj_FMM, Tjhat, Gjhat, Tnumehat, Tdenohat, Gnumehat, Gdenohat, rho, spec_indices{1},option );    
        end
    % Multichannel Wiener filtering
        if j == batch1_size
            for f = 1:F   
                for t = 1:Jj       
                    for src = 1:N            
                        ys = 0;           
                        for k = 1:K               
                            ys = ys + Tj_NFK(src,f,k)*Vj_NKT(src,k,t); % lamda in (19) of [2]           
                        end                    
%                         Yhat(f,t,:,src) = ys * squeeze(G(f,src,:,:))/Yhat_IJM(:,:,f,t)*X(:,f,t); % (54) of [2] M x 1       
                        Yhat(f,t,:,src) = inv(squeeze(Qj_FMM(f,:,:)))*diag(ys * squeeze(Gj_FNM(f,src,:))./ squeeze(Yhat_FTM(f,t,:)+eps))* squeeze(Qj_FMM(f,:,:)) *x(:,f,t);%squeeze(QX(f,t,:));% % M x 1 (19) of [2]
                    end              
                end          
            end
        else
            for f = 1:F    
%                 for t = 1:Jj
                    for src = 1:N                 
                    ys = 0;            
                    for k = 1:K                                      
%                         ys = ys + Tj(src,f,k)*Vj(src,k,t); %  lamda in (19) of [2]                             
                        ys = ys + Tj_NFK(src,f,k)*Vj_NKT(src,k,Jj); %  lamda in (19) of [2]                             
                    end                
%                     Yhat(f,j-Jj+t,:,src) = inv(squeeze(Qj(f,:,:)))*diag(ys * squeeze(Gj(f,src,:))./squeeze(Yhat_IJM(f,t,:))) * squeeze(Qj(f,:,:)) *x(:,f,j-Jj+t);%* squeeze(QX(f,batch_size,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
                    Yhat(f,j,:,src) = inv(squeeze(Qj_FMM(f,:,:)))*diag(ys * squeeze(Gj_FNM(f,src,:))./squeeze(Yhat_FTM(f,Jj,:)+eps)) * squeeze(Qj_FMM(f,:,:)) *x(:,f,j);%* squeeze(QX(f,batch_size,:));%squeeze(Q(i,:,:)) *x(:,i,j); % M x 1 (19) of [2]
                    end                      
%                 end
            end
        end
    end
fprintf(' FastMNMF_online done.\n');
end

%%% Xhat %%%
function [ Yhat_FTM ] = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N, indice)
Yhat_FTM = zeros(F,T,M); TV_NFT = zeros(N,F,T);%N=M; G_joint = zeros(N,M,F);
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;
end
for i = indice
    Yhat_FTM(i,:,:) = squeeze(TV_NFT(:,i,:)).'* squeeze(G_FNM(i,:,:)); % under 1 line of (15) in [2] & (40) of [1]
end %size(squeeze(TV_NIJ(:,i,:)).') T*M size(squeeze(G(i,:,:))) M*N  size(Yhat(i,:,:)) 1*T*N
end

%%% Cost function %%%未改
function [ cost ] = local_cost( X, Yhat_FTM, F, T, M ,Q_FMM)
QX_power = local_QX_power( Q_FMM, X, F, T, M, indice);% F T M
sumq = 0;
for f = 1:F 
    sumq = sumq + log(det(squeeze(Q_FMM(f,:,:))*squeeze(Q_FMM(f,:,:))')); 
end
temp1 = squeeze(sum(sum(sum(QX_power./Yhat_FTM + log(Yhat_FTM)))));
cost = -temp1 + T * sumq; % (15) of [2]
end

%%% Iterative update %%%  
function [ Yhat_FTM, T_NFK, V_NKT, G_FNM, Q_FMM, Tnume_NFK, Tdeno_NFK, Gnume_FNM, Gdeno_FNM] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM, That_NFK, Ghat_FNM, Tnumehat_NFK, Tdenohat_NFK, Gnumehat_FNM, Gdenohat_FNM, rho, indice,option)

%%%%% Update T %%%%%% N *F *K
QX_power = local_QX_power( Q_FMM, X, F, T, M, indice);
QY_FTM = (QX_power./(Yhat_FTM).^2);% |q*x|^2 * y.^-2 in (20) of [2]
[Tnume_NFK ,Tdeno_NFK] = local_Tfrac( QY_FTM, Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N, indice); % F x K         % % (20) of [2]分子更新
T_NFK = sqrt(Fjaxa(T_NFK, Tnume_NFK, That_NFK, Tnumehat_NFK, rho) ./ Fjx(Tdeno_NFK, Tdenohat_NFK, rho)); % (43) of [3]
% T = T.*max(sqrt(Tnume./Tdeno),eps); % (20) of [2] 
 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % under 1 line of (15) in [2]

%%%%% Update V %%%%%% N *K *T
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Vnume_NKT ,Vdeno_NKT] = local_Vfrac( QY_FTM, Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N, indice); % F x K         % % (21) of [2]分子更新
V_NKT = V_NKT.*max(sqrt(Vnume_NKT./Vdeno_NKT),eps); % (21) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % under 1 line of (15) in [2]

%%%%% Update G %%%%%
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Gnume_FNM ,Gdeno_FNM] = local_Gfrac( QY_FTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, indice); % F x K         % % (22) of [2]分子更新
G_FNM = sqrt(Fjaxa(G_FNM, Gnume_FNM, Ghat_FNM, Gnumehat_FNM, rho) ./ Fjx(Gdeno_FNM, Gdenohat_FNM, rho))+ 1e-10; % 类比(43) of [3]
% G = G.*max(sqrt(Gnume./Gdeno),eps) + 1e-10; % (22) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % under 1 line of (15) in [2]

%%%%% Update Q %%%%%
V_FMMM  = local_V_FMMM( Q_FMM, XX, Yhat_FTM, F, T, M);% (23) of [2]
Q_FMM     = local_Q( V_FMMM, Q_FMM, F, T, M, indice, option);% (24,25) of [2]

% normalize Q，G, T, V
[Q_FMM, G_FNM, T_NFK, V_NKT] = normalize(Q_FMM, G_FNM, T_NFK, V_NKT, M);

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % under 1 line of (15) in [2]
end

%% normalize Q，G, T, V
function [Q_FMM, G_FNM, T_NFK, V_NKT] = normalize(Q_FMM, G_FNM, T_NFK, V_NKT, M)
QQ = real(sum(sum(Q_FMM.*conj(Q_FMM),2),3)/M); % F *1
Q_FMM = Q_FMM./sqrt(QQ);% (26) in [2]
G_FNM = G_FNM./QQ ;% (26) in [2]

G_sum = sum(real(G_FNM),3);% F * N * 1 
G_FNM = G_FNM./G_sum ;%  (27) in [2]
T_NFK = T_NFK .* squeeze(sum(G_sum,2)).';%size( squeeze(sum(G_sum,2)).') =1 *F; % N * F * K  %  (27) in [2]
% 
T_sum = sum(T_NFK,2);% N *1* K
T_NFK = T_NFK./T_sum;%  (28) in [2]
V_NKT = V_NKT .* squeeze(T_sum);%  (28) in [2] size(squeeze(T_sum)) M*K
end

%%% Tfrac %%%
function [ Tnume, Tdeno ] = local_Tfrac( QY_FTM,Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N, indice)
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Tnume = zeros(N,F,K);Tdeno = zeros(N,F,K); % QY_IJM =|q*x|^2 * y.^-2
for f = indice
    GQY_NFT(:,f,:) = squeeze(G_FNM(f,:,:)) * squeeze(QY_FTM(f,:,:)).';%size(G)=F * N * M ； 分子的 g * |q*x|^2 * y.^-2 in (20) of [2]
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = squeeze(G_FNM(:,n,:)) ./ squeeze(Yhat_FTM(:,t,:)+eps); %分母的 g * y.^-1 in (20) of [2]   
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Tnume(n,:,:) = squeeze(GQY_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; %分子 in (20) of [2]
    Tdeno(n,:,:) = squeeze(GYhat_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; %分母 in (20) of [2]
end
end 

%%% Vfrac %%%
function [ Vnume, Vdeno ] = local_Vfrac( QY_FTM,Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N,indice) % QY_IJM =|q*x|^2 * y.^-2
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Vnume = zeros(N,K,T);Vdeno = zeros(N,K,T); %size(XX)  % F * T * M 
for f = indice
    GQY_NFT(:,f,:) = squeeze(G_FNM(f,:,:)) * squeeze(QY_FTM(f,:,:)).';%分子的 g * |q*x|^2 * y.^-2 in (21) of [2]
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = squeeze(G_FNM(:,n,:)) ./ squeeze(Yhat_FTM(:,t,:)+eps); %分母的 g * y.^-1 in (21) of [2]         
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Vnume(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GQY_NFT(n,:,:)); %分子 in (21) of [2]
    Vdeno(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GYhat_NFT(n,:,:)); %分母 in (21) of [2]
end
end 

%%% Gfrac %%%
function [ Gnume, Gdeno ] = local_Gfrac( QY_FTM,Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, indice) % QY_IJM =|q*x|^2 * y.^-2
TV_NFT = zeros(N,F,T);Gnume = zeros(F,N,M);Gdeno = zeros(F,N,M);Gdeno_NFTM  = zeros(N,F,T,M);%size(XX)  % F * T * M
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;% 添加eps % w * h in (22) of [2]
end
for f = indice
    Gnume(f,:,:) = squeeze(TV_NFT(:,f,:)) * squeeze(QY_FTM(f,:,:));%分子的 w * h  * |q*x|^2 * y.^-2 in (22) of [2]
end
for n = 1:N
    for m = 1:M
    Gdeno_NFTM(n,:,:,m) = squeeze(TV_NFT(n,:,:)) ./ squeeze(Yhat_FTM(:,:,m)+eps);  %分母的 w * h  *  y.^-1 in (22) of [2]   
    end
end
Gdeno = permute(squeeze(sum(Gdeno_NFTM,3)),[2,1,3]);
end

%%%% V_FMMM %%%
function [V_FMMM]  = local_V_FMMM( Q_FMM, XX, Yhat_FTM, F, T, M)
V_FTMMM = zeros(F,T,M,M,M);
for m = 1:M
    V_FTMMM(:,:,m,:,:) = XX./ (Yhat_FTM(:,:,m)+eps);% (23) of [2]
end
V_FMMM = squeeze(mean(V_FTMMM,2));
end

%%% QX_power %%%
function [ QX_power ] = local_QX_power( Q_FMM, X, F, T, M, indice)
QX_power = zeros(F,T,M);
for f = indice
    QX_power(f,:,:) = abs(squeeze(X(f,:,:)) * squeeze(Q_FMM(f,:,:)).').^2;% F* T* M
end %size(squeeze(X(ii,:,:))) T*M size(QX_power(ii,:,:)) 1 *T*M
end

%%% Q  update%%%
function [ Q_FMM ] = local_Q( V_FMMM, Q_FMM, F, T, M, indice, option)
global epsilon_start_ratio;global epsilon;global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;global SubBlockSize; global SB_ov_Size;
ekm = eye(M);
for f = indice%1:F
    for m=1:M
        if option.prior(m) == 1
            fac_a = option.fac_a;
            hf = option.hf;
            delta_f = option.deltaf;
            V_temp = squeeze(V_FMMM(f,m,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (DOA_tik_ratio * eye(M) + DOA_Null_ratio *squeeze(hf(:,f,m))*squeeze(hf(:,f,m)).') * fac_a/delta_f^2;%Hf维度是M*F*N，但M=N,所以应该用hf(:,f,m)
        else
            V_temp = squeeze(V_FMMM(f,m,:,:));
        end
%         V_temp = squeeze(V_FMMM(f,m,:,:));
        q_fm = (squeeze(Q_FMM(f,:,:)) * V_temp) \ ekm(:,m);%  M * 1  inv(a)*ekm(:,1)=a\ekm(:,1) ； (24) of [2]             
        q_fm = q_fm / (sqrt(q_fm' * V_temp * q_fm)+eps);% 1 * M；  (25) of [2]
        Q_FMM(f,m,:) =conj(q_fm); 
    end
end
end
%%% Fj(x) %%%
function Fx = Fjx( X, Xhat, rho ) % (50) of [3]
Fx = X + rho .* Xhat;
end

%%% Fj(a,x,a) %%%
function Fx = Fjaxa( A, X, Ahat, Xhat, rho ) % (51) of [3] scalar 
Fx = A.*X.*A + rho.*Ahat.*Xhat.*Ahat;
end

%%% Fj(A,X,A) %%%
function Fx = FjAXA( A, X, Ahat, Xhat, rho ) % (51) of [3] matrix
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