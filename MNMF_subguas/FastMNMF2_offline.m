function [Yhat_FTM,T_NFK,V_NKT,G_NM,Q_FMM,cost] = FastMNMF2_offline(X,XX,N,K,maxIt,drawConv,G_NM,T_NFK,V_NKT,Q_FMM)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2] Fast Multichannel Nonnegative Matrix Factorization with Directivity-Aware Jointly-Diagonalizable Spatial Covariance Matrices for Blind Source Separation 
%% see also% http://d-kitamura.net
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
    ginit = 2;%1-随机初始化；2-对角初始化；3-Circular初始化；
    if ginit == 1
        G_NM = randn(N,M);
    elseif ginit == 2
        ita = 0;%1e-2;
        G_NM = ita * ones(M) - diag(diag(ita * ones(M))) + eye(M);%  (40) of [2] 
    elseif ginit == 3
%     G_init = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));zeros(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为0
%     G_init = [eye(N),zeros(N,M-N)];% (41) of [2]
        ita = 1e-2;
        initg_f1 = ita * ones(N) - diag(diag(ita * ones(N))) + eye(N);% (41) of [2] 
        initg_f2 = ita * ones(mod(M,N)) - diag(diag(ita * ones(mod(M,N)))) + eye(mod(M,N));%  (41) of [2] 
        G_NM = [reshape(repmat(initg_f1,1,1,fix(M/N)),[N,N*fix(M/N)]),[initg_f2;ita * ones(N-mod(M,N),mod(M,N))]];% (41) of [2] g中除了1的元素为1e-2
    end
 end
if (nargin < 8)
    T_NFK = max(rand(N,F,K),eps);% N *F *K
        %% 狄利克雷实现
%     shape = 2;shape_F = ones(1,F) * shape;
%     T_NFK = zeros(N,F,K);
%     for k = 1:K
%         T_NFK(:,:,k) = drchrnd(shape_F,N); %(78) of [4]
%     end
 
end
if (nargin < 9)
    V_NKT = max(rand(N,K,T),eps);% N *K *T
        %% gamma实现
%     shape = 2;power_observation = mean(abs(X).^2,'all');
%     V_NKT = max(gamrnd(shape,power_observation*F*M/(shape*N*K),[N K T]),eps);%(79) of [4]第二个参数为逆尺度参数，是文章中的倒数与python版本一致

end
if (nargin < 10)
    Q_FMM = repmat(eye(M),1,1,F);
    Q_FMM = permute(Q_FMM,[3,1,2]); % F x M x M
end
if sum(size(T_NFK) ~= [N,F,K]) || sum(size(V_NKT) ~= [N,K,T]) || sum(size(G_NM) ~= [N,M]) 
    error('The size of input initial variable is incorrect.\n');
end
[Q_FMM, G_NM, T_NFK, V_NKT] = normalize(Q_FMM, G_NM, T_NFK, V_NKT, M);
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M ,N); % (17) of [2] initial model tensor F *M *M 
%% Iterative update

if ( drawConv == true )
    cost = zeros( maxIt+1, 1 );
    cost(1) = local_cost( X, Yhat_FTM, F, T, M ,Q_FMM); % initial cost value
    fprintf('Iteration:    ');
   for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_FTM, T_NFK, V_NKT, G_NM, Q_FMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_NM, F, T, K, N, M ,Q_FMM);
        cost(it+1) = local_cost( X, Yhat_FTM, F, T, M ,Q_FMM);
    end
    figure;
    plot( (0:maxIt), -cost );grid on;%hold on;semilogy( (0:maxIt), cost );
    set(gca,'FontName','Times','FontSize',16);
    xlabel('Number of iterations','FontName','Arial','FontSize',16);
    ylabel('Value of cost function','FontName','Arial','FontSize',16);
else
    cost = 0;
%% gradualinit
    gradualinit = 0;initIt = 10;
    if gradualinit
        fprintf('initIteration:    ');
        for it = 1:initIt
            fprintf('\b\b\b\b%4d', it);
            [ Yhat_FTM, T_NFK, V_NKT, G_NM, Q_FMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_NM, F, T, K, N, M ,Q_FMM);
        end
        T_NFK = max(rand(N,F,K),eps);%K = K + 2; N *F *K
        V_NKT = max(rand(N,K,T),eps);% N *K *T  
%         Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N); % (17) of [2] initial model tensor F *M *M 
    end
    fprintf('\n');fprintf('Iteration:    ');
%% main iterate
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_FTM, T_NFK, V_NKT, G_NM, Q_FMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_NM, F, T, K, N, M ,Q_FMM);
    end
end

fprintf(' FastMNMF2 done.\n');
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

%%% Iterative update %%%
function [ Yhat_FTM, T_NFK, V_NKT, G_NM, Q_FMM] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_NM, F, T, K, N, M ,Q_FMM)

%%%%% Update T %%%%%% N *F *K
QX_power = local_QX_power( Q_FMM, X, F, T, M);
QY_FTM = (QX_power./(Yhat_FTM).^2);% |q*x|^2 * y.^-2 in (34) of [2]
[Tnume ,Tdeno] = local_Tfrac( QY_FTM, Yhat_FTM, V_NKT, G_NM, F, T, K, M ,N); % F x K         % % (34) of [2]更新
T_NFK = T_NFK.*max(sqrt(Tnume./Tdeno),eps); % (34) of [2] 
 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M ,N); % under 1 line of (31) in [2]

%%%%% Update V %%%%%% N *K *T
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Vnume ,Vdeno] = local_Vfrac( QY_FTM, Yhat_FTM, T_NFK, G_NM, F, T, K, M ,N); % F x K         % % (35) of [2]更新
V_NKT = V_NKT.*max(sqrt(Vnume./Vdeno),eps); % (35) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M ,N); % under 1 line of (31) in [2]

%%%%% Update G %%%%%
QY_FTM = (QX_power./(Yhat_FTM).^2);
[Gnume ,Gdeno] = local_Gfrac( QY_FTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N); % F x K         % % (36) of [2]更新
G_NM = G_NM.*max(sqrt(Gnume./Gdeno),eps) + 1e-10; % (36) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M ,N); % under 1 line of (31) in [2]

%%%%% Update Q %%%%%
V_FMMM  = local_V_FMMM( Q_FMM, XX, Yhat_FTM, F, T, M);% (23) of [2]
Q_FMM     = local_Q( V_FMMM, Q_FMM, F, T, M);% (24,25) of [2]

% normalize Q，G, T, V
[Q_FMM, G_NM, T_NFK, V_NKT] = normalize(Q_FMM, G_NM, T_NFK, V_NKT, M);

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_NM, F, T, M ,N); % under 1 line of (31) in [2]
end

%% normalize Q，G, T, V
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
function [ Tnume, Tdeno ] = local_Tfrac( QY_FTM,Yhat_FTM, V_NKT, G_NM, F, T, K, M ,N)
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Tnume = zeros(N,F,K);Tdeno = zeros(N,F,K); % QY_IJM =|q*x|^2 * y.^-2
for f = 1:F
    GQY_NFT(:,f,:) = G_NM * squeeze(QY_FTM(f,:,:)).';%size(G)= N * M ； 分子的 g * |q*x|^2 * y.^-2 in (34) of [2]
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = G_NM(n,:) ./ squeeze(Yhat_FTM(:,t,:)); %分母的 g * y.^-1 in (34) of [2]   
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Tnume(n,:,:) = squeeze(GQY_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; %分子 in (34) of [2]
    Tdeno(n,:,:) = squeeze(GYhat_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; %分母 in (34) of [2]
end
end 

%%% Vfrac %%%
function [ Vnume, Vdeno ] = local_Vfrac( QY_FTM,Yhat_FTM, T_NFK, G_NM, F, T, K, M ,N) % QY_IJM =|q*x|^2 * y.^-2
GQY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Vnume = zeros(N,K,T);Vdeno = zeros(N,K,T); %size(XX)  % F * T * M 
for f = 1:F
    GQY_NFT(:,f,:) = G_NM * squeeze(QY_FTM(f,:,:)).';%分子的 g * |q*x|^2 * y.^-2 in (21) of [2]
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = G_NM(n,:) ./ squeeze(Yhat_FTM(:,t,:)); %分母的 g * y.^-1 in (21) of [2]      
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Vnume(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GQY_NFT(n,:,:));  %分子 in (21) of [2]
    Vdeno(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GYhat_NFT(n,:,:));  %分母 in (21) of [2]
end
end 

%%% Gfrac %%%
function [ Gnume, Gdeno ] = local_Gfrac( QY_FTM,Yhat_FTM, T_NFK, V_NKT, F, T, M ,N) % QY_IJM =|q*x|^2 * y.^-2
TV_NFT = zeros(N,F,T);Gnumetmp = zeros(F,N,M);Gdeno = zeros(N,M);Gdeno_NFTM  = zeros(N,F,T,M);%size(XX)  % F * T * M
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;% 添加eps；% w * h in (22) of [2]
end
for f = 1:F
    Gnumetmp(f,:,:) = squeeze(TV_NFT(:,f,:)) * squeeze(QY_FTM(f,:,:));%分子的 w * h  * |q*x|^2 * y.^-2 in (22) of [2]
end
Gnume = squeeze(sum(Gnumetmp,1));
for n = 1:N
    for m = 1:M
    Gdeno_NFTM(n,:,:,m) = squeeze(TV_NFT(n,:,:)) ./ squeeze(Yhat_FTM(:,:,m)); %分母的 w * h  *  y.^-1 in (22) of [2]   
    end
end
Gdeno = squeeze(sum(sum(Gdeno_NFTM,3),2));
end

%%%% V_FMMM %%%
function [V_FMMM]  = local_V_FMMM( Q, XX, Yhat_FTM, F, T, M)
V_FTMMM = zeros(F,T,M,M,M);
for m = 1:M
    V_FTMMM(:,:,m,:,:) = XX./ Yhat_FTM(:,:,m);% (23) of [2]
end
V_FMMM = squeeze(mean(V_FTMMM,2));
end

%%% QX_power %%%
function [ QX_power ] = local_QX_power( Q_FMM, X, F, T, M)
QX_power = zeros(F,T,M);
for f = 1:F
    QX_power(f,:,:) = abs(squeeze(X(f,:,:)) * squeeze(Q_FMM(f,:,:)).').^2;% F* T* M
end %size(squeeze(X(f,:,:))) T*M size(QX_power(f,:,:)) 1 *T*M
end

%%% Q  update%%%
function [ Q_FMM ] = local_Q( V_FMMM, Q_FMM, F, T, M)
ekm = eye(M);
for f = 1:F
    for m=1:M
        V_temp = squeeze(V_FMMM(f,m,:,:));
        q_fm = (squeeze(Q_FMM(f,:,:)) * V_temp) \ ekm(:,m);%  M * 1  inv(a)*ekm(:,1)=a\ekm(:,1) ； (24) of [2]       
        q_fm = q_fm / sqrt(q_fm' * V_temp * q_fm);% 1 * M；  (25) of [2]
        Q_FMM(f,m,:) =conj(q_fm); 
    end
end
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