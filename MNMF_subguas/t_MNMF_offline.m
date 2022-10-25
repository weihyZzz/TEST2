function [Xhat_FTMM,T_FK,V_KT,G_FNMM,Z_KN,cost] = t_MNMF_offline(X,N,K,maxIt,drawConv,v,trial,G_FNMM,T_FK,V_KT,Z_KN)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2]  STUDENT’S T MULTICHANNEL NONNEGATIVE MATRIX FACTORIZATION FOR BLIND SOURCE SEPARATION

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
%         H: initial spatial covariance tensor (F x N x M x M)
%         Z: initial partitioning matrix (K x N)
%
% [outputs]
%      Xhat: output 4th-order tensor reconstructed by T, V, H, and Z (F x T x M x M)
%         T: basis matrix (F x K)
%         V: activation matrix (K x T)
%         H: spatial covariance tensor (F x N x M x M)
%         Z: partitioning matrix (K x N)
%      cost: convergence behavior of cost function in multichannel NMF (maxIt+1 x 1)
%

% Check errors and set default values
global MNMF_p_norm;
[F,T,M,M] = size(X);
if size(X,3) ~= size(X,4)
    error('The size of input tensor is wrong.\n');
end
if (nargin < 3)
    K = ceil(T/10);
end
if (nargin < 4)
    maxIt = 300;
end
if (nargin < 5)
    drawConv = false;
end
if (nargin < 6)
    v = 5;
end
if (nargin < 7)
    trial = 20;
end
if (nargin < 8)
    G_FNMM = repmat(sqrt(eye(M)/M),[1,1,F,N]);
%     G = repmat(eye(M)/M,[1,1,F,N]);
    G_FNMM = permute(G_FNMM,[3,4,1,2]); % F x N x M x M
end
if (nargin < 9)
    T_FK = max(rand(F,K),eps);
end
if (nargin < 10)
    V_KT = max(rand(K,T),eps);
end
if (nargin < 11)
    varZ = 0.01;
    Z_KN = varZ*rand(K,N) + 1/N;
    Z_KN = max( Z_KN./sum(Z_KN,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
%     Z = bsxfun(@rdivide, Z, sum(Z,2)); % for prior R2016b
end
if sum(size(T_FK) ~= [F,K]) || sum(size(V_KT) ~= [K,T]) || sum(size(G_FNMM) ~= [F,N,M,M]) || sum(size(Z_KN) ~= [K,N])
    error('The size of input initial variable is incorrect.\n');
end
Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M ); % initial model tensor
% size(Xhat) =F *T *M*M
% Iterative update
fprintf('initIteration:    ');
if ( drawConv == true )
    cost = zeros( maxIt+1, 1 );
    cost(1) = local_cost( X, Xhat_FTMM, F, T, M ,v); % initial cost value
    for it = 1:trial
        fprintf('\b\b\b\b%4d', it);
        [ Xhat_FTMM, T_FK, V_KT] = init_iterativeUpdate( X, Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, K, N, M ,v);        
    end
fprintf(' t-MNMF trial_init done.\n');fprintf('\n');fprintf('Iteration:    ');
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN ] = local_iterativeUpdate( X, Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, K, N, M ,v);
        cost(it+1) = local_cost( X, Xhat_FTMM, F, T, M ,v);
    end
%     figure;
%     plot( (0:maxIt), cost );grid on%semilogy( (0:maxIt), cost );
%     set(gca,'FontName','Times','FontSize',16);
%     xlabel('Number of iterations','FontName','Arial','FontSize',16);
%     ylabel('Value of cost function','FontName','Arial','FontSize',16);
%     fprintf(' t-MNMF done.\n');
%     gcf_temp = strcat('D:\github-code\IVATEST\gcf_save\',filenameTmp(:,1:end-4),'.jpg');
%     saveas(gcf, gcf_temp); %保存当前窗口的图像
%     close(figure(gcf));

else
    cost = 0;
    for it = 1:trial
        fprintf('\b\b\b\b%4d', it);
        [ Xhat_FTMM, T_FK, V_KT] = init_iterativeUpdate( X, Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, K, N, M ,v);        
    end
fprintf(' t-MNMF trial_init done.\n');fprintf('\n');fprintf('Iteration:    ');
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN ] = local_iterativeUpdate( X, Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, K, N, M ,v);
    end
end
fprintf(' t-MNMF done\n');
end

%%% Cost function %%%
function [ cost ] = local_cost( X, Xhat, F, T, M ,v)
invXhat = local_inverse( Xhat, F, T, M );
XinvXhat = local_multiplication( X, invXhat, F, T, M );
trXinvXhat = local_trace( XinvXhat, F, T, M );
detXhat = local_det( Xhat, F, T, M );% detXhat
cost = log(real(detXhat))+(M+v/2)*log(1+2/v*real(trXinvXhat)); % (9) of [2],对概率密度函数取负对数,(9)的负数;
cost = sum(cost(:));
end
function [ Xhat_FTMM, T_FK, V_KT] = init_iterativeUpdate( X, Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, K, N, M ,v)
global MNMF_p_norm;
%%%%% Update T %%%%%
invXhat_FTMM = local_inverse( Xhat_FTMM, F, T, M );% X_inv:F x T x M x M
invXhatXinvXhat_FTMM = local_multiplicationXYX( invXhat_FTMM, X, F, T, M );% X_inv* X* X_inv :F x T x M x M
tr_invXhatX_FT = local_mul_tr(invXhat_FTMM, X);%F x T 
auxvar_FT = (2 * M + v)./(v + 2 * tr_invXhatX_FT);% (2 * M + v)./(v + 2 * trace(MU1))::F x T
auxinvXhatXinvXhat_FTMM = auxvar_FT.*invXhatXinvXhat_FTMM;%local_multiplication( auxvar, invXhatXinvXhat, F, T, M);% F x T x M x M
Tnume = local_Tfrac( auxinvXhatXinvXhat_FTMM, V_KT, Z_KN, G_FNMM, F, K, M ); % F x K % (15) of [2]分子
Tdeno = local_Tfrac( invXhat_FTMM, V_KT, Z_KN, G_FNMM, F, K, M ); % F x K         % (15) of [2]
T_FK = T_FK.*max((Tnume./Tdeno).^MNMF_p_norm,eps); % (15) of [2]
Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update V %%%%%
invXhat_FTMM = local_inverse( Xhat_FTMM, F, T, M );
invXhatXinvXhat_FTMM = local_multiplicationXYX( invXhat_FTMM, X, F, T, M );
tr_invXhatX_FT = local_mul_tr(invXhat_FTMM, X);%F x T
auxvar_FT = (2 * M + v)./(v + 2 * tr_invXhatX_FT);% (2 * M + v)./(v + 2 * trace(MU1))::F x T
auxinvXhatXinvXhat_FTMM = auxvar_FT.*invXhatXinvXhat_FTMM;%local_multiplication( auxvar, invXhatXinvXhat, F, T, M);% F x T x M x M
Vnume = local_Vfrac( auxinvXhatXinvXhat_FTMM, T_FK, Z_KN, G_FNMM, T, K, M); % K x T
Vdeno = local_Vfrac( invXhat_FTMM, T_FK, Z_KN, G_FNMM, T, K, M ); % K x T
V_KT = V_KT.*max((Vnume./Vdeno).^MNMF_p_norm,eps); % (16) of [2]

Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M );     
end
%%% Iterative update %%%
function [ Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN ] = local_iterativeUpdate( X, Xhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, K, N, M ,v)
%%%%% Update T %%%%%
global MNMF_p_norm;
invXhat_FTMM = local_inverse( Xhat_FTMM, F, T, M );% X_inv:F x T x M x M
invXhatXinvXhat_FTMM = local_multiplicationXYX( invXhat_FTMM, X, F, T, M );% X_inv* X* X_inv :F x T x M x M
tr_invXhatX_FT = local_mul_tr(invXhat_FTMM, X);%F x T 
auxvar_FT = (2 * M + v)./(v + 2 * tr_invXhatX_FT);% (2 * M + v)./(v + 2 * trace(MU1))::F x T
auxinvXhatXinvXhat_FTMM = auxvar_FT.*invXhatXinvXhat_FTMM;%local_multiplication( auxvar, invXhatXinvXhat, F, T, M);% F x T x M x M
Tnume = local_Tfrac( auxinvXhatXinvXhat_FTMM, V_KT, Z_KN, G_FNMM, F, K, M ); % F x K % (15) of [2]分子
Tdeno = local_Tfrac( invXhat_FTMM, V_KT, Z_KN, G_FNMM, F, K, M ); % F x K         % (15) of [2]
T_FK = T_FK.*max((Tnume./Tdeno).^MNMF_p_norm,eps); % (15) of [2]
Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update V %%%%%
invXhat_FTMM = local_inverse( Xhat_FTMM, F, T, M );
invXhatXinvXhat_FTMM = local_multiplicationXYX( invXhat_FTMM, X, F, T, M );
tr_invXhatX_FT = local_mul_tr(invXhat_FTMM, X);%F x T
auxvar_FT = (2 * M + v)./(v + 2 * tr_invXhatX_FT);% (2 * M + v)./(v + 2 * trace(MU1))::F x T
auxinvXhatXinvXhat_FTMM = auxvar_FT.*invXhatXinvXhat_FTMM;%local_multiplication( auxvar, invXhatXinvXhat, F, T, M);% F x T x M x M
Vnume = local_Vfrac( auxinvXhatXinvXhat_FTMM, T_FK, Z_KN, G_FNMM, T, K, M); % K x T
Vdeno = local_Vfrac( invXhat_FTMM, T_FK, Z_KN, G_FNMM, T, K, M ); % K x T
V_KT = V_KT.*max((Vnume./Vdeno).^MNMF_p_norm,eps); % (16) of [2]

Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update Z %%%%%
invXhat_FTMM = local_inverse( Xhat_FTMM, F, T, M );
invXhatXinvXhat_FTMM = local_multiplicationXYX( invXhat_FTMM, X, F, T, M );
tr_invXhatX_FT = local_mul_tr(invXhat_FTMM, X);%F x T
auxvar_FT = (2 * M + v)./(v + 2 * tr_invXhatX_FT);% (2 * M + v)./(v + 2 * trace(MU1))::F x T
auxinvXhatXinvXhat_FTMM = auxvar_FT.*invXhatXinvXhat_FTMM;%local_multiplication( auxvar, invXhatXinvXhat, F, T, M);% F x T x M x M
Znume = local_Zfrac( auxinvXhatXinvXhat_FTMM, T_FK, V_KT, G_FNMM, K, N, M ); % K x N
Zdeno = local_Zfrac( invXhat_FTMM, T_FK, V_KT, G_FNMM, K, N, M ); % K x N
Z_KN = Z_KN.*sqrt(Znume./Zdeno); % (17) of [2]
Z_KN = max( Z_KN./sum(Z_KN,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update H %%%%%
invXhat_FTMM = local_inverse( Xhat_FTMM, F, T, M );
invXhatXinvXhat_FTMM = local_multiplicationXYX( invXhat_FTMM, X, F, T, M );
tr_invXhatX_FT = local_mul_tr(invXhat_FTMM, X);%F x T
auxvar_FT = (2 * M + v)./(v + 2 * tr_invXhatX_FT);% (2 * M + v)./(v + 2 * trace(MU1))::F x T
auxinvXhatXinvXhat_FTMM = auxvar_FT.*invXhatXinvXhat_FTMM;% local_multiplication( auxvar, invXhatXinvXhat, F, T, M);% F x T x M x M
G_FNMM = local_RiccatiSolver( auxinvXhatXinvXhat_FTMM, invXhat_FTMM, T_FK, V_KT, G_FNMM, Z_KN, F, T, N, M ); % (18) of [2]
Xhat_FTMM = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M );
end

%%% Xhat %%%
function [ Xhat_FTMM ] = local_Xhat( T_FK, V_KT, G_FNMM, Z_KN, F, T, M )
Xhat_FTMM = zeros(F,T,M,M);
for mm = 1:M*M
    Gmm_FN = G_FNMM(:,:,mm); % F x N
    Xhat_FTMM(:,:,mm) = ((Gmm_FN*Z_KN').*T_FK)*V_KT; % (40) of [1]
end %size(Xhat)      size(Hmm*Z')   size(Z')          size(T) size(V)
% F * T * M * M      (F x L)*  ( L*K) =F (fre)*K      F*K       K*T(time)
end
%%% multiplication & trace
function [tr_MU1] = local_mul_tr(X, Y)
[F,T,M,~] = size(X);
MU1 = local_multiplication( X, Y, F, T, M);% F * T * M * M
tr_MU1 = local_trace( MU1,F, T, M);% F * T
end
%%% Tfrac %%%
function [ Tfrac ] = local_Tfrac( X_FTMM, V_KT, Z_KN, G_FNMM, F, K, M )% (15) of [2]
Tfrac = zeros(F,K); %size(X)  % F * T * M * M
for mm = 1:M*M
    Tfrac = Tfrac + real( (( X_FTMM(:,:,mm))*V_KT').*(conj(G_FNMM(:,:,mm))*Z_KN') ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end %size(X(:,:,mm)) F * T x size(V') T * K = F*K  
end %(42) of [1] size(H(:,:,mm))  F*L  size(Z')  K*L

%%% Vfrac %%%
function [ Vfrac ] = local_Vfrac( X_FTMM, T_FK, Z_KN, G_FNMM, T, K, M )% (16) of [2]
Vfrac = zeros(K,T);
for mm = 1:M*M
    Vfrac = Vfrac + real( ((G_FNMM(:,:,mm)*Z_KN').*T_FK)'*(X_FTMM(:,:,mm)) ); %using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end %size(H(:,:,mm)) F*L x size(Z')  L*K  =F*K  size(X(:,:,mm)) F*T
end %size( ((H(:,:,mm)*Z').*T)'*X(:,:,mm))   (F*K)'x (F*T) = K*T

%%% Zfrac %%%
function [ Zfrac ] = local_Zfrac( X_FTMM, T_FK, V_KT, G_FNMM, K, N, M)% (17) of [2]
Zfrac = zeros(K,N);
for mm = 1:M*M
    Zfrac = Zfrac + real( ((( X_FTMM(:,:,mm))*V_KT.').*T_FK)'*G_FNMM(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end % size(X(:,:,mm)) F*T size(V.') T*K  size(((X(:,:,mm)*V.').*T))  (F*T) x (T*K) =F*K
end %size(H(:,:,mm))  F*K

%%% Riccati solver %%%
%input; size(X)  F*K*M*M
%input; size(Y)  F*K*M*M
function [ G_FNMM ] = local_RiccatiSolver(X, Y, T_FK, V_KT, G_FNMM, Z_KN, F, T, N, M)%% (18) of [2]等价于（18），都是solution of Riccati equation
X = reshape(permute(X, [3 4 2 1]), [M*M, T, F]); % invXhatXinvXhat, MM x T x F
Y = reshape(permute(Y, [3 4 2 1]), [M*M, T, F]); % invXhat, MM x T x F
deltaEyeM = eye(M)*(10^(-12)); % to avoid numerical instability
for n = 1:N % Riccati equation solver described in the original paper
    for f = 1:F %size(X(:,:,i))  M*M   T
        ZTV = (T_FK(f,:).*Z_KN(:,n)')*V_KT;   %ZTV: 1 * T % size(T(i,:)) 1*K; size(Z(:,n)')  1*K size(T(i,:).*Z(:,n)')  1*K 
        % size(V) K*T
        A = reshape(Y(:,:,f)*ZTV', [M, M]); % (19) of [2] size(Y(:,:,i))  MM x T 
        B = reshape(X(:,:,f)*ZTV', [M, M]); % (20) of [2] size(X(:,:,i))  MM x T 
        Gin = reshape(G_FNMM(f,n,:,:), [M, M]);  % size(H(i,n,:,:)) 1*1*2*2
        C = Gin*B*Gin; % (47) of [1] 
        AC = [zeros(M), -1*A; -1*C, zeros(M)];% (56) of [1]
        [eigVec, eigVal] = eig(AC);
        ind = find(diag(eigVal)<0);
        FF = eigVec(1:M,ind);
        O = eigVec(M+1:end,ind);
        Gin = O/FF; % G*inv(F); % (58) of [1]
        Gin = (Gin+Gin')/2 + deltaEyeM;  %  third line below (58) of [1]
        G_FNMM(f,n,:,:) = Gin/trace(Gin);
    end
end
% for n = 1:N % Another solution of Riccati equation, which is slower than the above one. The calculation result coincides with that of the above calculation.
%     for i = 1:I
%         ZTV = (T(i,:).*Z(:,n).')*V; % 1 x J
%         A = reshape( Y(:,:,i)*ZTV.', [M, M] ); % M x M
%         B = sqrtm(reshape( X(:,:,i)*ZTV.', [M, M] )); % M x M
%         Hin = reshape( H(i,n,:,:), [M, M] );
%         Hin = Hin*B/sqrtm((B*Hin*A*Hin*B))*B*Hin; % solution of Riccati equation
%         Hin = (Hin+Hin')/2; % "+eye(M)*delta" should be added here for avoiding rank deficient in such a case
%         H(i,n,:,:) = Hin/trace(Hin);
%     end
% end
end

%%% MultiplicationXXX %%%
function [ XYX ] = local_multiplicationXYX( X, Y, I, J, M )
if M == 2
    %tic
    XYX = zeros( I, J, M, M );
    x2 = real(X(:,:,1,2).*conj(X(:,:,1,2)));
    xy = X(:,:,1,2).*conj(Y(:,:,1,2));
    ac = X(:,:,1,1).*Y(:,:,1,1);
    bd = X(:,:,2,2).*Y(:,:,2,2);
    XYX(:,:,1,1) = Y(:,:,2,2).*x2 + X(:,:,1,1).*(2*real(xy)+ac);
    XYX(:,:,2,2) = Y(:,:,1,1).*x2 + X(:,:,2,2).*(2*real(xy)+bd);
    XYX(:,:,1,2) = X(:,:,1,2).*(ac+bd+xy) + X(:,:,1,1).*X(:,:,2,2).*Y(:,:,1,2);
    XYX(:,:,2,1) = conj(XYX(:,:,1,2));
    %toc
else % slow
    %tic
    XY = local_multiplication( X, Y, I, J, M );
    XYX = local_multiplication( XY, X, I, J, M );
    %toc
end
end

%%% Multiplication %%%
function [ XY ] = local_multiplication( X, Y, I, J, M )
if M == 2
    XY = zeros( I, J, M, M );  % 2048*128*2*2
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2);
elseif M == 3
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1) + X(:,:,1,3).*Y(:,:,3,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2) + X(:,:,1,3).*Y(:,:,3,2);
    XY(:,:,1,3) = X(:,:,1,1).*Y(:,:,1,3) + X(:,:,1,2).*Y(:,:,2,3) + X(:,:,1,3).*Y(:,:,3,3);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1) + X(:,:,2,3).*Y(:,:,3,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2) + X(:,:,2,3).*Y(:,:,3,2);
    XY(:,:,2,3) = X(:,:,2,1).*Y(:,:,1,3) + X(:,:,2,2).*Y(:,:,2,3) + X(:,:,2,3).*Y(:,:,3,3);
    XY(:,:,3,1) = X(:,:,3,1).*Y(:,:,1,1) + X(:,:,3,2).*Y(:,:,2,1) + X(:,:,3,3).*Y(:,:,3,1);
    XY(:,:,3,2) = X(:,:,3,1).*Y(:,:,1,2) + X(:,:,3,2).*Y(:,:,2,2) + X(:,:,3,3).*Y(:,:,3,2);
    XY(:,:,3,3) = X(:,:,3,1).*Y(:,:,1,3) + X(:,:,3,2).*Y(:,:,2,3) + X(:,:,3,3).*Y(:,:,3,3);
elseif M == 4
    XY = zeros( I, J, M, M );
    XY(:,:,1,1) = X(:,:,1,1).*Y(:,:,1,1) + X(:,:,1,2).*Y(:,:,2,1) + X(:,:,1,3).*Y(:,:,3,1) + X(:,:,1,4).*Y(:,:,4,1);
    XY(:,:,1,2) = X(:,:,1,1).*Y(:,:,1,2) + X(:,:,1,2).*Y(:,:,2,2) + X(:,:,1,3).*Y(:,:,3,2) + X(:,:,1,4).*Y(:,:,4,2);
    XY(:,:,1,3) = X(:,:,1,1).*Y(:,:,1,3) + X(:,:,1,2).*Y(:,:,2,3) + X(:,:,1,3).*Y(:,:,3,3) + X(:,:,1,4).*Y(:,:,4,3);
    XY(:,:,1,4) = X(:,:,1,1).*Y(:,:,1,4) + X(:,:,1,2).*Y(:,:,2,4) + X(:,:,1,3).*Y(:,:,3,4) + X(:,:,1,4).*Y(:,:,4,4);
    XY(:,:,2,1) = X(:,:,2,1).*Y(:,:,1,1) + X(:,:,2,2).*Y(:,:,2,1) + X(:,:,2,3).*Y(:,:,3,1) + X(:,:,2,4).*Y(:,:,4,1);
    XY(:,:,2,2) = X(:,:,2,1).*Y(:,:,1,2) + X(:,:,2,2).*Y(:,:,2,2) + X(:,:,2,3).*Y(:,:,3,2) + X(:,:,2,4).*Y(:,:,4,2);
    XY(:,:,2,3) = X(:,:,2,1).*Y(:,:,1,3) + X(:,:,2,2).*Y(:,:,2,3) + X(:,:,2,3).*Y(:,:,3,3) + X(:,:,2,4).*Y(:,:,4,3);
    XY(:,:,2,4) = X(:,:,2,1).*Y(:,:,1,4) + X(:,:,2,2).*Y(:,:,2,4) + X(:,:,2,3).*Y(:,:,3,4) + X(:,:,2,4).*Y(:,:,4,4);
    XY(:,:,3,1) = X(:,:,3,1).*Y(:,:,1,1) + X(:,:,3,2).*Y(:,:,2,1) + X(:,:,3,3).*Y(:,:,3,1) + X(:,:,3,4).*Y(:,:,4,1);
    XY(:,:,3,2) = X(:,:,3,1).*Y(:,:,1,2) + X(:,:,3,2).*Y(:,:,2,2) + X(:,:,3,3).*Y(:,:,3,2) + X(:,:,3,4).*Y(:,:,4,2);
    XY(:,:,3,3) = X(:,:,3,1).*Y(:,:,1,3) + X(:,:,3,2).*Y(:,:,2,3) + X(:,:,3,3).*Y(:,:,3,3) + X(:,:,3,4).*Y(:,:,4,3);
    XY(:,:,3,4) = X(:,:,3,1).*Y(:,:,1,4) + X(:,:,3,2).*Y(:,:,2,4) + X(:,:,3,3).*Y(:,:,3,4) + X(:,:,3,4).*Y(:,:,4,4);
    XY(:,:,4,1) = X(:,:,4,1).*Y(:,:,1,1) + X(:,:,4,2).*Y(:,:,2,1) + X(:,:,4,3).*Y(:,:,3,1) + X(:,:,4,4).*Y(:,:,4,1);
    XY(:,:,4,2) = X(:,:,4,1).*Y(:,:,1,2) + X(:,:,4,2).*Y(:,:,2,2) + X(:,:,4,3).*Y(:,:,3,2) + X(:,:,4,4).*Y(:,:,4,2);
    XY(:,:,4,3) = X(:,:,4,1).*Y(:,:,1,3) + X(:,:,4,2).*Y(:,:,2,3) + X(:,:,4,3).*Y(:,:,3,3) + X(:,:,4,4).*Y(:,:,4,3);
    XY(:,:,4,4) = X(:,:,4,1).*Y(:,:,1,4) + X(:,:,4,2).*Y(:,:,2,4) + X(:,:,4,3).*Y(:,:,3,4) + X(:,:,4,4).*Y(:,:,4,4);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    Y = reshape(permute(Y, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    XY = zeros( M, M, I*J );
    parfor ij = 1:I*J
        XY(:,:,ij) = X(:,:,ij)*Y(:,:,ij);
    end
    XY = permute(reshape(XY, [M,M,I,J]), [3,4,1,2]); % I x J x M x M
end
end

%%% Inverse %%%
function [ invX ] = local_inverse( X, I, J, M )
if M == 2
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
    invX(:,:,1,1) = X(:,:,2,2);
    invX(:,:,1,2) = -1*X(:,:,1,2);
    invX(:,:,2,1) = conj(invX(:,:,1,2));
    invX(:,:,2,2) = X(:,:,1,1);
    invX = invX./detX; % using implicit expanion
%    invX = bsxfun(@rdivide, invX, detX); % for prior R2016b
elseif M == 3
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,2,1).*X(:,:,3,2).*X(:,:,1,3) + X(:,:,3,1).*X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,2,3) - X(:,:,3,1).*X(:,:,2,2).*X(:,:,1,3) - X(:,:,2,1).*X(:,:,1,2).*X(:,:,3,3);
    invX(:,:,1,1) = X(:,:,2,2).*X(:,:,3,3) - X(:,:,2,3).*X(:,:,3,2);
    invX(:,:,1,2) = X(:,:,1,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,3,3);
    invX(:,:,1,3) = X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,3).*X(:,:,2,2);
    invX(:,:,2,1) = X(:,:,2,3).*X(:,:,3,1) - X(:,:,2,1).*X(:,:,3,3);
    invX(:,:,2,2) = X(:,:,1,1).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,3,1);
    invX(:,:,2,3) = X(:,:,1,3).*X(:,:,2,1) - X(:,:,1,1).*X(:,:,2,3);
    invX(:,:,3,1) = X(:,:,2,1).*X(:,:,3,2) - X(:,:,2,2).*X(:,:,3,1);
    invX(:,:,3,2) = X(:,:,1,2).*X(:,:,3,1) - X(:,:,1,1).*X(:,:,3,2);
    invX(:,:,3,3) = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
    invX = invX./detX; % using implicit expanion
%    invX = bsxfun(@rdivide, invX, detX); % for prior R2016b
elseif M == 4
    invX = zeros(I,J,M,M);
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,1,1) = X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2);
    invX(:,:,1,2) = X(:,:,1,2).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,4).*X(:,:,3,2).*X(:,:,4,3);
    invX(:,:,1,3) = X(:,:,1,2).*X(:,:,2,3).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,4,3) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,4,2);
    invX(:,:,1,4) = X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3);
    invX(:,:,2,1) = X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3);
    invX(:,:,2,2) = X(:,:,1,1).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,3,3).*X(:,:,4,1);
    invX(:,:,2,3) = X(:,:,1,1).*X(:,:,2,4).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,4,3);
    invX(:,:,2,4) = X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1);
    invX(:,:,3,1) = X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) - X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1);
    invX(:,:,3,2) = X(:,:,1,1).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,2).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,3,3) = X(:,:,1,1).*X(:,:,2,2).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,4,1) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,4,2) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,4,4) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,4,1);
    invX(:,:,3,4) = X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2);
    invX(:,:,4,1) = X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
    invX(:,:,4,2) = X(:,:,1,1).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,3,1).*X(:,:,4,2) - X(:,:,1,1).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,3,2).*X(:,:,4,1);
    invX(:,:,4,3) = X(:,:,1,1).*X(:,:,2,3).*X(:,:,4,2) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,4,3) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,4,3) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,4,1) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,4,2);
    invX(:,:,4,4) = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1);
    invX = invX./detX; % using implicit expanion
%    invX = bsxfun(@rdivide, invX, detX); % for prior R2016b
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    eyeM = eye(M);
    invX = zeros(M,M,I*J);
    parfor ij = 1:I*J
            invX(:,:,ij) = X(:,:,ij)\eyeM;
    end
    invX = permute(reshape(invX, [M,M,I,J]), [3,4,1,2]); % I x J x M x M
end
end

%%% Trace %%%
function [ trX ] = local_trace( X, I, J, M )
if M == 2
    trX = X(:,:,1,1) + X(:,:,2,2);
elseif M == 3
    trX = X(:,:,1,1) + X(:,:,2,2) + X(:,:,3,3);
elseif M == 4
    trX = X(:,:,1,1) + X(:,:,2,2) + X(:,:,3,3) + X(:,:,4,4);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    trX = zeros(I*J,1);
    parfor ij = 1:I*J
            trX(ij) = trace(X(:,:,ij));
    end
    trX = reshape(trX, [I,J]); % I x J
end
end

%%% Determinant %%%
function [ detX ] = local_det( X, I, J, M )
if M == 2
    detX = X(:,:,1,1).*X(:,:,2,2) - X(:,:,1,2).*X(:,:,2,1);
elseif M == 3
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3) + X(:,:,2,1).*X(:,:,3,2).*X(:,:,1,3) + X(:,:,3,1).*X(:,:,1,2).*X(:,:,2,3) - X(:,:,1,1).*X(:,:,3,2).*X(:,:,2,3) - X(:,:,3,1).*X(:,:,2,2).*X(:,:,1,3) - X(:,:,2,1).*X(:,:,1,2).*X(:,:,3,3);
elseif M == 4
    detX = X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,4) + X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,2) + X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,3) + X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,4) + X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,4) + X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,1) + X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,2) + X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,3) + X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,1).*X(:,:,2,2).*X(:,:,3,4).*X(:,:,4,3) - X(:,:,1,1).*X(:,:,2,3).*X(:,:,3,2).*X(:,:,4,4) - X(:,:,1,1).*X(:,:,2,4).*X(:,:,3,3).*X(:,:,4,2) - X(:,:,1,2).*X(:,:,2,1).*X(:,:,3,3).*X(:,:,4,4) - X(:,:,1,2).*X(:,:,2,3).*X(:,:,3,4).*X(:,:,4,1) - X(:,:,1,2).*X(:,:,2,4).*X(:,:,3,1).*X(:,:,4,3) - X(:,:,1,3).*X(:,:,2,1).*X(:,:,3,4).*X(:,:,4,2) - X(:,:,1,3).*X(:,:,2,2).*X(:,:,3,1).*X(:,:,4,4) - X(:,:,1,3).*X(:,:,2,4).*X(:,:,3,2).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,1).*X(:,:,3,2).*X(:,:,4,3) - X(:,:,1,4).*X(:,:,2,2).*X(:,:,3,3).*X(:,:,4,1) - X(:,:,1,4).*X(:,:,2,3).*X(:,:,3,1).*X(:,:,4,2);
else % slow
    X = reshape(permute(X, [3,4,1,2]), [M,M,I*J]); % M x M x IJ
    detX = zeros(I*J,1);
    parfor ij = 1:I*J
            detX(ij) = det(X(:,:,ij));
    end
    detX = reshape(detX, [I,J]); % I x J
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%