function [Yhat_FTM,T_NFK,V_NKT,G_FNM,Q_FMM,cost] = subguas_FastMNMF_offline(X,XX,N,K,maxIt,drawConv,option,G_FNM,T_NFK,V_NKT,Q_FMM)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2] Joint-Diagonalizability-Constrained Multichannel Nonnegative Matrix Factorization Based on Multivariate Complex Sub-Gaussian Distribution
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
%% Check errors and set default values
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
% if (nargin < 7)
if (nargin < 8)
    ginit = 2;%1-�����ʼ����2-�Խǳ�ʼ����3-Circular��ʼ����
    if ginit == 1
        G_FNM = randn(F,N,M);
    elseif ginit == 2
        ita = 0;%1e-2;
%         initg_f = ita * ones(M) - diag(diag(ita * ones(M))) + eye(M);%  (40) of [2]
        initg_f = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));zeros(N-mod(M,N),mod(M,N))]];% (41) of [2] g�г���1��Ԫ��Ϊ0
        %initg_f  = [eye(N);zeros(N,M-N)];% (41) of [2]

        G_FNM = repmat(initg_f,[1,1,F]);
        G_FNM = permute(G_FNM,[3,1,2]); % F x N x M  "N=M"  
    elseif ginit == 3
%     G_init = [reshape(repmat(eye(N),1,1,fix(M/N)),[N,N*fix(M/N)]),[eye(mod(M,N));zeros(N-mod(M,N),mod(M,N))]];% (41) of [2] g�г���1��Ԫ��Ϊ0
%     G_init = [eye(N),zeros(N,M-N)];% (41) of [2]
        ita = 1e-2;
        initg_f1 = ita * ones(N) - diag(diag(ita * ones(N))) + eye(N);% (41) of [2] 
        initg_f2 = ita * ones(mod(M,N)) - diag(diag(ita * ones(mod(M,N)))) + eye(mod(M,N));%  (41) of [2] 
        G_init = [reshape(repmat(initg_f1,1,1,fix(M/N)),[N,N*fix(M/N)]),[initg_f2;ita * ones(N-mod(M,N),mod(M,N))]];% (41) of [2] g�г���1��Ԫ��Ϊ1e-2
        G_FNM = repmat(G_init,[1,1,F]);
        G_FNM = permute(G_FNM,[3,1,2]); % F x N x M  "M>N"
    end
end
% if (nargin < 8)
if (nargin < 9)
    T_NFK = max(rand(N,F,K),eps);% N *F *K
        %% ��������ʵ��
%     shape = 2;shape_F = ones(1,F) * shape;
%     T_NFK = zeros(N,F,K);
%     for k = 1:K
%         T_NFK(:,:,k) = drchrnd(shape_F,N); %(78) of [4]
%     end
 
end
% if (nargin < 9)
if (nargin < 10)
    V_NKT = max(rand(N,K,T),eps);% N *K *T
        %% gammaʵ��
%     shape = 2;power_observation = mean(abs(X).^2,'all');
%     V_NKT = max(gamrnd(shape,power_observation*F*M/(shape*N*K),[N K T]),eps);%(79) of [4]�ڶ�������Ϊ��߶Ȳ������������еĵ�����python�汾һ��

end
% if (nargin < 10)
if (nargin < 11)
    Q_FMM = repmat(eye(M),1,1,F);
    Q_FMM = permute(Q_FMM,[3,1,2]); % F x M x M
end
if sum(size(T_NFK) ~= [N,F,K]) || sum(size(V_NKT) ~= [N,K,T]) || sum(size(G_FNM) ~= [F,N,M]) 
    error('The size of input initial variable is incorrect.\n');
end
%%
beta = option.sub_beta;
[Q_FMM, G_FNM, T_NFK, V_NKT] = normalize(Q_FMM, G_FNM, T_NFK, V_NKT, M);
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, spec_indices{1}); % (17) of [2] initial model tensor F *M *M 

QX_power = local_QX_power( Q_FMM, X, F, T, M, spec_indices{1});
phi = local_phi( QX_power, Yhat_FTM, F, T, M, beta ,spec_indices{1}); % under 1 line of (15) in [2]

%% Iterative update

if ( drawConv == true )% û��
    cost = zeros( maxIt+1, 1 );
    cost(1) = local_cost( X, Yhat_FTM, F, T, M ,Q_FMM); % initial cost value
    fprintf('Iteration:    ');
   for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_FTM, phi, T_NFK, V_NKT, G_FNM, Q_FMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, phi, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM);
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
            [ Yhat_FTM, phi,  T_NFK, V_NKT, G_FNM, Q_FMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM, spec_indices{1},option);
        end
        T_NFK = max(rand(N,F,K),eps);%K = K + 2; N *F *K
        V_NKT = max(rand(N,K,T),eps);% N *K *T  
%         Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N); % (17) of [2] initial model tensor F *M *M 
    end
    fprintf('\n');fprintf('Iteration:    ');
%% Main iterate
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_FTM, phi, T_NFK, V_NKT, G_FNM, Q_FMM ] = local_iterativeUpdate(X, XX, Yhat_FTM, phi, beta, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM, spec_indices{1},option);
    end
end

fprintf(' subguass_FastMNMF done.\n');
end

%%% Yhat %%%
function [ Yhat_FTM ] = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M, N, indice)
Yhat_FTM = zeros(F,T,M); TV_NFT = zeros(N,F,T);%N=M; G_joint = zeros(N,M,F);
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;
end
for f = indice%1:F
    Yhat_FTM(f,:,:) = squeeze(TV_NFT(:,f,:)).'* squeeze(G_FNM(f,:,:)); % under 1 line of (15) in [2] & (40) of [1]
end 
end

%%% PHI %%%
function [ phi ] = local_phi( QX_power, Yhat_FTM, F, T, M, beta ,indice)
phi = zeros(F,T,M); qxYhat = zeros(F,T);%N=M; G_joint = zeros(N,M,F);
for m = 1:M
    qxYhat = qxYhat + squeeze(QX_power(:,:,m))./squeeze(Yhat_FTM(:,:,m)+eps) + 1e-10;
end
for f = indice%1:F
    phi(f,:,:) = squeeze(QX_power(f,:,:).* qxYhat(f,:).^((beta-2)/2)); % (18) in [2]
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
function [ Yhat_FTM, phi, T_NFK, V_NKT, G_FNM, Q_FMM] = local_iterativeUpdate(X, XX, Yhat_FTM, phi, beta, T_NFK, V_NKT, G_FNM, F, T, K, N, M ,Q_FMM,indice,option)

%% %%% Update T %%%%%% N *F *K
QX_power = local_QX_power( Q_FMM, X, F, T, M, indice);
phiY_FTM = (phi./(Yhat_FTM+eps).^2);% phi * y.^-2 in (19) of [2]
[Tnume ,Tdeno] = local_Tfrac( phiY_FTM, Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N, indice); % F x K         % % (19) of [2]����
T_NFK = T_NFK.*max(((beta/2).*Tnume./Tdeno).^(2/(beta+2)),eps); % (19) of [2] 
 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % (17) in [2]
phi = local_phi( QX_power, Yhat_FTM, F, T, M, beta ,indice); % (18) in [2]

%% %%% Update V %%%%%% N *K *T
phiY_FTM = (phi./(Yhat_FTM+eps).^2);% phi * y.^-2 in (19) of [2]
[Vnume ,Vdeno] = local_Vfrac( phiY_FTM, Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N, indice); % F x K         % % (21) of [2]����
V_NKT = V_NKT.*max(((beta/2).*Vnume./Vdeno).^(2/(beta+2)),eps); % (21) of [2]

Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % (17) in [2]
phi = local_phi( QX_power, Yhat_FTM, F, T, M, beta ,indice); % (18) in [2]

%% %%% Update G %%%%%
  %%��׼����
phiY_FTM = (phi./(Yhat_FTM+eps).^2);% phi * y.^-2 in (19) of [2]
[Gnume ,Gdeno] = local_Gfrac( phiY_FTM, Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, indice); % F x K         % % (22) of [2]����
G_FNM = G_FNM.*max(((beta/2).*Gnume./Gdeno).^(2/(beta+2)),eps) + 1e-10; % (22) of [2]
%% riccati solution ��������׼�����ٶ�Ҫ���ܶ�
% QY_FTM = (QX_power./(Yhat_FTM).^2);
% d_Y_FTM = 1./Yhat_FTM;
% G_FNM = local_RiccatiSolver( QY_FTM, d_Y_FTM, T_NFK, V_NKT, G_FNM, F, T, N, M ); % riccati solution
% 
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % under 1 line of (15) in [2]

%% %%% Update Q %%%%%
% V_FMMM  = local_V_FMMM( Q_FMM, XX, Yhat_FTM, F, T, M);% (23) of [2]
% Q_FMM     = local_Q( V_FMMM, Q_FMM, F, T, M, indice,option);% (24,25) of [2]
% QX    = local_QX( Q_FMM, X, I, J, M);
r_ijm = local_r_ijm( QX_power, Yhat_FTM, beta, F, T, M);% (33) of [2]
U_im  = local_U_im( QX_power, XX, r_ijm, beta, F, T, M);% (34) of [2]
[B_im, SUM2]  = local_B_im( Q_FMM, QX_power, U_im, XX, r_ijm, beta, F, T, M, indice);% (35) of [2]

Q_FMM  = local_Q( QX_power, X, Q_FMM, B_im, SUM2, r_ijm, beta, F, T, M, indice);% (36,37) of [2]

%% normalize Q��G, T, V
[Q_FMM, G_FNM, T_NFK, V_NKT] = normalize(Q_FMM, G_FNM, T_NFK, V_NKT, M);
Yhat_FTM = local_Yhat( T_NFK, V_NKT, G_FNM, F, T, M ,N, indice); % under 1 line of (15) in [2]

QX_power = local_QX_power( Q_FMM, X, F, T, M, indice);
phi = local_phi( QX_power, Yhat_FTM, F, T, M, beta ,indice); % (18) in [2]
end

%% normalize Q��G, T, V
function [Q_FMM, G_FNM, T_NFK, V_NKT] = normalize(Q_FMM, G_FNM, T_NFK, V_NKT, M)
QQ = real(sum(sum(Q_FMM.*conj(Q_FMM),2),3)/M); % F *1
Q_FMM = Q_FMM./sqrt(QQ);% (26) in [2]
           
G_sum = sum(real(G_FNM),3);% F * N * 1 
G_FNM = G_FNM./G_sum ;%  (27) in [2]
T_NFK = T_NFK .* squeeze(sum(G_sum,2)).';%size( squeeze(sum(G_sum,2)).') =1 *F;sum(G_sum,2);% N * F * K  %  (27) in [2]
% 
T_sum = sum(T_NFK,2);% N *1* K
T_NFK = T_NFK./T_sum;%  (28) in [2]
V_NKT = V_NKT .* squeeze(T_sum);%  (28) in [2] size(squeeze(T_sum)) M*K
end

%% QX_power %%%
function [ QX_power ] = local_QX_power( Q_FMM, X, F, T, M, indice)
QX_power = zeros(F,T,M);
for f = indice% 1:F
    QX_power(f,:,:) = abs(squeeze(X(f,:,:)) * squeeze(Q_FMM(f,:,:)).');% F* T* M
end %size(squeeze(X(f,:,:))) T*M size(QX_power(f,:,:)) 1 *T*M
% for f = indice% 1:F
%     for m=1:M
%     QX_power1(f,:,m) = abs(squeeze(X(f,63:66,:)) * squeeze(conj(Q_FMM(f,m,:))));% F* T* M
%     sum(QX_power1(f,:,m)-QX_power(f,:,m),'all')
%     end 
% end
end

%% Tfrac %%%
function [ Tnume, Tdeno ] = local_Tfrac( phiY_FTM,Yhat_FTM, V_NKT, G_FNM, F, T, K, M ,N, indice)
GphiY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Tnume = zeros(N,F,K);Tdeno = zeros(N,F,K); % QY_IJM =|q*x|^2 * y.^-2
for f = indice %1:F
    GphiY_NFT(:,f,:) = squeeze(G_FNM(f,:,:)) * squeeze(phiY_FTM(f,:,:)).';%size(G)=F * N * M �� ���ӵ� g * |q*x|^2 * y.^-2 in (20) of [2]
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = squeeze(G_FNM(:,n,:)) ./ squeeze(Yhat_FTM(:,t,:)+eps); %��ĸ�� g * y.^-1 in (20) of [2]   
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Tnume(n,:,:) = squeeze(GphiY_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; %���� in (20) of [2]
    Tdeno(n,:,:) = squeeze(GYhat_NFT(n,:,:)) * squeeze(V_NKT(n,:,:)).'; %��ĸ in (20) of [2]
end
end

%%% Vfrac %%%
function [ Vnume, Vdeno ] = local_Vfrac( phiY_FTM,Yhat_FTM, T_NFK, G_FNM, F, T, K, M ,N, indice) % QY_IJM =|q*x|^2 * y.^-2
GphiY_NFT = zeros(N,F,T);GYhat_NFTM = zeros(N,F,T,M);Vnume = zeros(N,K,T);Vdeno = zeros(N,K,T); %size(XX)  % F * T * M 
for f =  indice%1:F
    GphiY_NFT(:,f,:) = squeeze(G_FNM(f,:,:)) * squeeze(phiY_FTM(f,:,:)).';%���ӵ� g * |q*x|^2 * y.^-2 in (21) of [2]
end
for n=1:N
    for t=1:T
    GYhat_NFTM(n,:,t,:) = squeeze(G_FNM(:,n,:)) ./ squeeze(Yhat_FTM(:,t,:)+eps); %��ĸ�� g * y.^-1 in (21) of [2]      
    end
end
GYhat_NFT = squeeze(sum(GYhat_NFTM,4));
for n=1:N
    Vnume(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GphiY_NFT(n,:,:));  %���� in (21) of [2]
    Vdeno(n,:,:) = squeeze(T_NFK(n,:,:)).' * squeeze(GYhat_NFT(n,:,:));  %��ĸ in (21) of [2]
end
end 

%%% Gfrac %%%
function [ Gnume, Gdeno ] = local_Gfrac(phiY_FTM,Yhat_FTM, T_NFK, V_NKT, F, T, M ,N, indice) % QY_IJM =|q*x|^2 * y.^-2
TV_NFT = zeros(N,F,T);Gnume = zeros(F,N,M);Gdeno = zeros(F,N,M);Gdeno_NFTM  = zeros(N,F,T,M);%size(XX)  % F * T * M
for n = 1:N
    TV_NFT(n,:,:) = squeeze(T_NFK(n,:,:))*squeeze(V_NKT(n,:,:)) + 1e-10;% ���eps��% w * h in (22) of [2]
end
for f = indice%1:F
    Gnume(f,:,:) = squeeze(TV_NFT(:,f,:)) * squeeze(phiY_FTM(f,:,:));%���ӵ� w * h  * |q*x|^2 * y.^-2 in (22) of [2]
end
for n = 1:N
    for m = 1:M
    Gdeno_NFTM(n,:,:,m) = squeeze(TV_NFT(n,:,:)) ./ squeeze(Yhat_FTM(:,:,m)+eps); %��ĸ�� w * h  *  y.^-1 in (22) of [2]   
    end
end
Gdeno = permute(squeeze(sum(Gdeno_NFTM,3)),[2,1,3]);
end

%%% r_ijm %%%
function [ r_ijm ] = local_r_ijm(QX_power, Yhat_FTM, beta, F, T, M)
r_ijm = zeros(F,T,M); 
for mm = 1:M
    r_ijm(:,:,mm) = QX_power(:,:,mm).^(1/2-1/beta).* Yhat_FTM(:,:,mm).^(1/beta).* (sum(QX_power./(Yhat_FTM+eps),3).^(1/beta-1/2));% (33) of [2]
end
end
%%% U_im %%%
function [ U_im ] = local_U_im(QX_power, XX, r_ijm, beta, F, T, M)
U_im = zeros(F,M,M,M); % ά�ȱ仯ΪI *M *M *M
MU1 = sqrt(QX_power.^(2-beta/2).* r_ijm.^beta);% I* J* M
for m=1:M
    U_im(:,m,:,:) = sum( XX./ (MU1(:,:,m)+eps),2);% (34) of [2]
end
end

%%% B_im %%%
function [ B_im ,SUM2] = local_B_im( Q_FMM, QX_power, U_im, XX, r_ijm, beta, F, T, M, indice)
SUM1 = zeros(F,M,M,M);SUM2 = zeros(F,M,M,M); SUM3 = zeros(F,M,M,M);
for i =  indice
    for m=1:M
    SUM1(i,m,:,:) = squeeze(Q_FMM(i,m,:))'*squeeze(U_im(i,m,:,:))*squeeze(Q_FMM(i,m,:)).*squeeze(U_im(i,m,:,:));% I* M* M* M % (35)of [2] ��һ��
    SUM3(i,m,:,:) = squeeze(U_im(i,m,:,:))*squeeze(Q_FMM(i,m,:))*(squeeze(U_im(i,m,:,:))*squeeze(Q_FMM(i,m,:)))';% (35)of [2] ������
    end
end
for m=1:M
    QX_r_ijm(:,:,m) = QX_power(:,:,m).^(beta/2-1)./(r_ijm(:,:,m).^beta+eps);
    SUM2(:,m,:,:) = sum( XX.* QX_r_ijm(:,:,m),2);% ���ʱ��ά�ȵĹ�һ��
end % (35)of [2] �ڶ���
% SUM2(:,m,:,:) = sum( XX.* (QX.^(beta-2)./(r_ijm(:,:,m)+eps).^beta),2);end % (35)of [2] �ڶ���
B_im = SUM2 + SUM1 - SUM3;% (35)of [2] I* M* M* M
end

%%% Q  update%%%
function [ Q_FMM ] = local_Q( QX_power, X, Q_FMM, B_im, SUM2, r_ijm, beta, F, T, M, indice)
ekm = eye(M);
for f =  indice
    for m=1:M
        B_temp = squeeze(B_im(f,m,:,:));
        SUM2_temp = squeeze(SUM2(f,m,:,:));
        q_fm = (squeeze(Q_FMM(f,:,:))*squeeze(B_temp)) \ ekm(:,m);%  M * 1 (36)of [2] 
%         qX = abs(reshape(repmat(q_im,1,J),[J,M]).*squeeze(X(i,:,:)));%  J *  M
%         qX_r_ijm = qX.^beta./ (squeeze(r_ijm(i,:,:)).^beta+eps);%  J *  M
        q_fm  = q_fm / (((beta/2/T) .* q_fm' * SUM2_temp * q_fm).^(1/beta)+eps);% (37)of [2]
        Q_FMM(f,m,:)  = conj(q_fm) ; 
%         tmp = q_fm .* sum(squeeze(((2*T/beta) ./ sum(abs(squeeze(X(f,:,:)) * q_fm).^(beta)./squeeze(r_ijm(f,:,m)).'.^beta,1)).^(1/beta)+eps),1);        
%         Q_FMM(f,m,:) = tmp;
    end
end
% %%%% V_FMMM %%%
% function [V_FMMM]  = local_V_FMMM( Q, XX, Yhat_FTM, F, T, M)
% V_FTMMM = zeros(F,T,M,M,M);
% for m = 1:M
%     V_FTMMM(:,:,m,:,:) = XX./ (Yhat_FTM(:,:,m)+eps);% (23) of [2]
% end
% V_FMMM = squeeze(mean(V_FTMMM,2));
% end
% 


% %%% Q  update%%%
% function [ Q_FMM ] = local_Q( V_FMMM, Q_FMM, F, T, M, indice, option)
% %%
% global epsilon_start_ratio;global epsilon;global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;global SubBlockSize; global SB_ov_Size;
% ekm = eye(M);
% for f = indice%1:F
%     for m=1:M
%         if option.prior(m) == 1
%             fac_a = option.fac_a;
%             hf = option.hf;
%             delta_f = option.deltaf;
%              V_temp = squeeze(V_FMMM(f,m,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
%                      + (DOA_tik_ratio * eye(M) + DOA_Null_ratio *squeeze(hf(:,f,m))*squeeze(hf(:,f,m)).') * fac_a/delta_f^2;%Hfά����M*F*N����M=N,����Ӧ����hf(:,f,m)
%         else
%              V_temp = squeeze(V_FMMM(f,m,:,:));
%         end
%         q_fm = (squeeze(Q_FMM(f,:,:)) * V_temp) \ ekm(:,m);%  M * 1  inv(a)*ekm(:,1)=a\ekm(:,1) �� (24) of [2]       
%          q_fm = q_fm / (sqrt(q_fm' * V_temp * q_fm)+eps);% 1 * M��  (25) of [2]
%          Q_FMM(f,m,:) =conj(q_fm); 
%     end
% end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%