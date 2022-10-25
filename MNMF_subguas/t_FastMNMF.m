function [Yhat_IJM,T,V,Z,G,Q,cost] = t_FastMNMF(X,XX,N,K,maxIt,drawConv,v,trial,G,T,V,Z,Q)
%% multichannelNMF: Blind source separation based on multichannel NMF
%% Coded by D. Kitamura (d-kitamura@ieee.org)
%% # Original paper
% [1] H. Sawada, H. Kameoka, S. Araki, and N. Ueda,	"Multichannel extensions of non-negative matrix factorization with complex-valued data," IEEE
% Transactions on Audio, Speech, and Language Processing, vol. 21, no. 5, pp. 971-982, May 2013.
% [2] Joint-Diagonalizability-Constrained Multichannel  Nonnegative Matrix Factorization Based on Multivariate Complex Student’s t-distribution%% see also% http://d-kitamura.net
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
%        XX: input 4th-order tensor (time-frequency-wise covariance matrices) (I x J x M x M)
%         N: number of sources
%         K: number of bases (default: ceil(J/10))
%        it: number of iteration (default: 300)
%  drawConv: plot cost function values in each iteration or not (true or false)
%         T: initial basis matrix (I x K)
%         V: initial activation matrix (K x J)
%         G: initial spatial covariance tensor (I x N x M )
%         Z: initial partitioning matrix (K x N)
%         Q: diagonalizer (I x M x M)
% [outputs]
%      Xhat: output 4th-order tensor reconstructed by T, V, H, and Z (I x J x M x M)
%         T: basis matrix (I x K)
%         V: activation matrix (K x J)
%         G: spatial covariance tensor (I x N x M x M)
%         Z: partitioning matrix (K x N)
%         Q: diagonalizer (I x M x M)
%      cost: convergence behavior of cost function in multichannel NMF (maxIt+1 x 1)

% Check errors and set default values
[I,J,M,M] = size(XX);
delta = 0.001; % to avoid numerical conputational instability
if size(XX,3) ~= size(XX,4)
    error('The size of input tensor is wrong.\n');
end
if (nargin < 4)
    K = ceil(J/10);
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
    G = repmat(eye(M),[1,1,I]);
    G = permute(G,[3,1,2]); % I x N x M  "N=M"
%     initg_f = 1e-2 * ones(M) - diag(diag(1e-2 * ones(M))) + eye(M);%  IV.C 4) in [2]
%     G = repmat(initg_f,[1,1,I]);
%     G = permute(G,[3,1,2]); % I x N x M  "N=M"  4096fftsize时会出现奇异值错误,1024点较好效果
 end
if (nargin < 10)
    T = max(rand(I,K),eps);% I *K
end
if (nargin < 11)
    V = max(rand(K,J),eps);% K *J
end
if (nargin < 12)
    varZ = 0.01;
    Z = varZ*rand(K,N) + 1/N;
    Z = max( Z./sum(Z,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
end
if (nargin < 13)
    Q = repmat(eye(M),1,1,I);
    Q = permute(Q,[3,1,2]); % I x M x M
end

if sum(size(T) ~= [I,K]) || sum(size(V) ~= [K,J]) || sum(size(G) ~= [I,N,M]) || sum(size(Z) ~= [K,N])
    error('The size of input initial variable is incorrect.\n');
end
[Q, G, T, V] = normalize(Q, G, T, V, Z, M);
Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (17) of [2] initial model tensor I *M *M 
%% Iterative update
fprintf('initIteration:    ');
if ( drawConv == true )
    cost = zeros( maxIt+1, 1 );
    cost(1) = local_cost( X, Yhat_IJM, I, J, M ,Q); % initial cost value
   for it = 1:trial
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_IJM, T, V] = init_local_iterativeUpdate(X, XX, Yhat_IJM, T, V, Z, G, I, J, K, N, M ,Q, v);
        cost(it+1) = local_cost( X, Yhat_IJM, I, J, M ,Q);
   end
fprintf(' t-fastMNMF trial_init done.\n');fprintf('\n');fprintf('Iteration:    ');
   
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_IJM, T, V, Z, G, Q ] = local_iterativeUpdate(X, XX, Yhat_IJM, T, V, Z, G, I, J, K, N, M ,Q, v);
        cost(it+1) = local_cost( X, Yhat_IJM, I, J, M ,Q);
    end
    figure;
    plot( (0:maxIt), -cost );grid on;%hold on;semilogy( (0:maxIt), cost );
    set(gca,'FontName','Times','FontSize',16);
    xlabel('Number of iterations','FontName','Arial','FontSize',16);
    ylabel('Value of cost function','FontName','Arial','FontSize',16);
else
    cost = 0;
%     for it = 1:trial
%         fprintf('\b\b\b\b%4d', it);
%         [ Yhat_IJM, T, V] = init_local_iterativeUpdate(X, XX, Yhat_IJM, T, V, Z, G, I, J, K, N, M ,Q, v);
%     end
% fprintf(' t-fastMNMF trial_init done.\n');fprintf('\n');fprintf('Iteration:    ');
    for it = 1:maxIt
        fprintf('\b\b\b\b%4d', it);
        [ Yhat_IJM, T, V, Z, G, Q ] = local_iterativeUpdate(X, XX, Yhat_IJM, T, V, Z, G, I, J, K, N, M ,Q,v);
    end
end

fprintf(' tFastMNMF done.\n');
end

%%% Xhat %%%
function [ Yhat ] = local_Yhat( T, V, Z, G, I, J, M)
Yhat = zeros(I,J,M); 
for mm = 1:M
    Gmm = G(:,:,mm); % I x N
    Yhat(:,:,mm) = ((Gmm*Z').*T)*V; % (23) of [2] & (40) of [1]
end 
% [K,N] = size(Z);
% Yhat = zeros(I,J,M); TVZ_NIJK = zeros(N,I,J,K);%N=M; G_joint = zeros(N,M,I);
% for n = 1:N
%     for k=1:K
%     TVZ_NIJK(n,:,:,k) = T * V .* Z(k,n) + 1e-10;% N * I * J * K
%     end
% end
% TVZ_NIJ=squeeze(sum(TVZ_NIJK,4));
% for i = 1:I
%     Yhat(i,:,:) = squeeze(TVZ_NIJ(:,i,:)).'* squeeze(G(i,:,:)); % under 1 line of (15) in [2] & (40) of [1]
% end
end

%%% Cost function %%% 
function [ cost ] = local_cost( X, Yhat_IJM, I, J, M ,Q)
QX_power = local_QX_power( Q, X, I, J, M);% I J M
sumq = 0;
for i = 1:I 
    sumq = sumq + log(det(squeeze(Q(i,:,:))*squeeze(Q(i,:,:))')); 
end
temp1 = squeeze(sum(sum(sum(QX_power./Yhat_IJM + log(Yhat_IJM)))));
cost = -temp1 + J * sumq; % (15) of [2]
end
%%% initIterative update %%%
function [ Yhat_IJM, T, V] = init_local_iterativeUpdate(X, XX, Yhat_IJM, T, V, Z, G, I, J, K, N, M ,Q,v)

%%%%% Update T %%%%%% I * K 
QX_power = local_QX_power( Q, X, I, J, M);
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
QY_IJM = (QX_power./(Yhat_IJM).^2);
[Tnume ,Tdeno] = local_Tfrac( alpha, QY_IJM, Yhat_IJM, V, Z, G, I, J, K, M ,N); % I x K         % % (20) of [2]分子更新
T = T.*max(sqrt(Tnume./Tdeno),eps); % (20) of [2] 
 
Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]

%%%%% Update V %%%%%% K * J
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = (2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
QY_IJM = (QX_power./(Yhat_IJM).^2);
[Vnume ,Vdeno] = local_Vfrac( alpha, QY_IJM, Yhat_IJM, T, Z, G, I, J, K, M ,N); % I x K         % % (33) of [2]
V = V.*max(sqrt(Vnume./Vdeno),eps); % (21) of [2]

Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]

end

%%% Iterative update %%%
function [ Yhat_IJM, T, V, Z, G, Q] = local_iterativeUpdate(X, XX, Yhat_IJM, T, V, Z, G, I, J, K, N, M ,Q,v)

%%%%% Update T %%%%%% I * K 
QX_power = local_QX_power( Q, X, I, J, M);
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = 1;%(2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
QY_IJM = (QX_power./(Yhat_IJM).^2);
[Tnume ,Tdeno] = local_Tfrac( alpha, QY_IJM, Yhat_IJM, V, Z, G, I, J, K, M ,N); % I x K         % % (20) of [2]分子更新
T = T.*max(sqrt(Tnume./Tdeno),eps); % (20) of [2] 
 
Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]

%%%%% Update V %%%%%% K * J
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = 1;%(2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
QY_IJM = (QX_power./(Yhat_IJM).^2);
[Vnume ,Vdeno] = local_Vfrac( alpha, QY_IJM, Yhat_IJM, T, Z, G, I, J, K, M ,N); % I x K         % % (33) of [2]
V = V.*max(sqrt(Vnume./Vdeno),eps); % (21) of [2]

Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]

%%%%% Update Z %%%%%% K * J
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = 1;%(2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
QY_IJM = (QX_power./(Yhat_IJM).^2);
[Znume ,Zdeno] = local_Zfrac( alpha, QY_IJM, Yhat_IJM, T, V, G, I, J, K, M ,N); % I x K         % % (34) of [2]
Z = Z.* sqrt(Znume./Zdeno); % (34) of [2]
Z = max( Z./sum(Z,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)

Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]

%%%%% Update G %%%%%
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = 1;%(2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
QY_IJM = (QX_power./(Yhat_IJM).^2);
[Gnume ,Gdeno] = local_Gfrac( alpha, QY_IJM, Yhat_IJM, T, V, Z, I, J, M ,N, K); % I x K         % % (35) of [2]
G = G.*max(sqrt(Gnume./Gdeno),eps)+ 1e-10; %  (22) of [2]

% G_sum = sum(real(G),3);% I * N *1 
% G = G./G_sum ;

Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]

%%%%% Update Q %%%%%
QX_power_Xhat = squeeze(sum(QX_power./ Yhat_IJM,3));% (24) in [2]分母里 QX_power/Xhat
alpha = 1;%(2 * M + v)./(v + 2 * QX_power_Xhat);% I * J (24) in [2]
V_IMMM  = local_V_IMMM( alpha, Q, XX, Yhat_IJM, I, J, M);
Q     = local_Q( V_IMMM, Q, I, J, M);% (36,37) of [2]

% normalize Q，G, T, V
[Q, G, T, V] = normalize(Q, G, T, V, Z, M);

Yhat_IJM = local_Yhat( T, V, Z, G, I, J, M ); % (23) in [2]
end

%% normalize Q，G, T, V  考虑TV归一化加入Z
function [Q, G, T, V] = normalize(Q, G, T, V, Z, M)
QQ = real(sum(sum(Q.*conj(Q),2),3)/M); % I *1
Q = Q./sqrt(QQ);
G = G./QQ ;

G_sum = sum(real(G),3);% I * N *1 
G = G./G_sum ;
T = T .* squeeze(sum(G_sum,2));%sum(G_sum,2);% I * K  ????
% 
T_sum = sum(T,1);% 1 * K
T = T./T_sum;
V = V .* squeeze(T_sum).';

% [K,N] = size(Z);[I,~] = size(T);[~,J] = size(V);TZ = zeros(N,I,K);VZ = zeros(N,K,J);
% for n=1:N
%     TZ(n,:,:) = T .* Z(:,n).';
%     VZ(n,:,:) = V .* Z(:,n);
% end
% G_sum = sum(real(G),3);% I * N * 1 
% G = G./G_sum ;
% TZ = TZ .* squeeze(sum(G_sum,2)).';%sum(G_sum,2);% N * I * K  
% % 
% TZ_sum = sum(TZ,2);% N *1* K
% TZ = TZ./TZ_sum;
% VZ = VZ .* squeeze(TZ_sum);
% 
% for n=1:N
%     T = squeeze(TZ(n,:,:)) ./ Z(:,n).';
%     V = squeeze(VZ(n,:,:)) ./ Z(:,n);
% end
end

%%% Tfrac %%%
function [ Tnume, Tdeno ] = local_Tfrac( alpha, QY_IJM,Yhat_IJM, V, Z, G, I, J, K, M ,N)
GQY_NIJ = zeros(N,I,J);GYhat_NIJM = zeros(N,I,J,M);Tnume_frac = zeros(I,K,N);Tdeno_frac = zeros(I,K,N); 
alphaQY_IJM = alpha.*  QY_IJM;% I * J * M
for i = 1:I
    GQY_NIJ(:,i,:) = squeeze(G(i,:,:)) * squeeze(alphaQY_IJM(i,:,:)).';%(31) of [2] size(G)  % I * N * M 
end
for n=1:N
    for j=1:J
    GYhat_NIJM(n,:,j,:) = squeeze(G(:,n,:)) ./ squeeze(Yhat_IJM(:,j,:));    
    end
end
GYhat_NIJ = squeeze(sum(GYhat_NIJM,4));
for n=1:N
    Tnume_frac(:,:,n) = squeeze(GQY_NIJ(n,:,:)) * V.'; 
    Tdeno_frac(:,:,n) = squeeze(GYhat_NIJ(n,:,:)) * V.'; 
end
% Tnume_frac=rand(5,3,2);Z=rand(3,2);I=5;
Tnume = squeeze(sum(permute(permute(Tnume_frac,[2,3,1]) .* Z,[3,1,2]),3));%size(Z):K,N size(Tnume ):I,K
Tdeno = squeeze(sum(permute(permute(Tdeno_frac,[2,3,1]) .* Z,[3,1,2]),3));%size(Z):K,N
% for i=1:I
%     Tnume1(i,:,:) = squeeze(Tnume_frac(i,:,:)) .* Z;%I*K*N
% end
% Tnume1 = squeeze(sum(Tnume1,3));
% Tnume1==Tnume
end 

%%% Vfrac %%%
function [ Vnume, Vdeno ] = local_Vfrac( alpha, QY_IJM,Yhat_IJM, T, Z, G, I, J, K, M ,N)
GQY_NIJ = zeros(N,I,J);GYhat_NIJM = zeros(N,I,J,M);Vnume_frac = zeros(N,K,J);Vdeno_frac = zeros(N,K,J); %size(XX)  % I * J * M 
alphaQY_IJM = alpha.*  QY_IJM;% I * J * M
for i = 1:I
    GQY_NIJ(:,i,:) = squeeze(G(i,:,:)) * squeeze(alphaQY_IJM(i,:,:)).';
end
for n=1:N
    for j=1:J
    GYhat_NIJM(n,:,j,:) = squeeze(G(:,n,:)) ./ squeeze(Yhat_IJM(:,j,:));    
    end
end
GYhat_NIJ = squeeze(sum(GYhat_NIJM,4));
for n=1:N
    Vnume_frac(n,:,:) = T.' * squeeze(GQY_NIJ(n,:,:)); % N * K * J
    Vdeno_frac(n,:,:) = T.' * squeeze(GYhat_NIJ(n,:,:)); 
end
Vnume = squeeze(sum(Vnume_frac .* Z.',1));%size(Z):K,N size(Vnume ):K,J
Vdeno = squeeze(sum(Vdeno_frac .* Z.',1));%size(Z):K,N

end

%%% Zfrac %%%
function [ Znume, Zdeno ] = local_Zfrac( alpha, QY_IJM,Yhat_IJM, T, V, G, I, J, K, M ,N)
GQY_NIJ = zeros(N,I,J);GYhat_NIJM = zeros(N,I,J,M);Znume_frac = zeros(N,I,K);Zdeno_frac = zeros(N,I,K); %size(XX)  % I * J * M 
alphaQY_IJM = alpha .*  QY_IJM;% I * J * M
for i = 1:I
    GQY_NIJ(:,i,:) = squeeze(G(i,:,:)) * squeeze(alphaQY_IJM(i,:,:)).';
end
for n=1:N
    for j=1:J
    GYhat_NIJM(n,:,j,:) = squeeze(G(:,n,:)) ./ squeeze(Yhat_IJM(:,j,:));    
    end
end
GYhat_NIJ = squeeze(sum(GYhat_NIJM,4));
for n=1:N
    Znume_frac(n,:,:) = squeeze(GQY_NIJ(n,:,:))* V.'; % N * I * K
    Zdeno_frac(n,:,:) = squeeze(GYhat_NIJ(n,:,:))* V.'; 
end
Znume = squeeze(sum(permute(Znume_frac,[2,3,1]) .* T,1));%size(T):I,K size(V):K,J
Zdeno = squeeze(sum(permute(Zdeno_frac,[2,3,1]) .* T,1));%size(T):I,K size(V):K,J
end

%%% Gfrac %%%                        
function [ Gnume, Gdeno ] = local_Gfrac( alpha, QY_IJM, Yhat_IJM, T, V, Z, I, J, M ,N, K)
TVZ_NIJK = zeros(N,I,J,K);Gnume = zeros(I,N,M);Gdeno = zeros(I,N,M);Gdeno_NIJM  = zeros(N,I,J,M);%size(XX)  % I * J * M
alphaQY_IJM = alpha.*  QY_IJM;% I * J * M   size(T):I,K size(V):K,J
for n = 1:N
    for k=1:K
    TVZ_NIJK(n,:,:,k) = T * V .* Z(k,n) + 1e-10;% N * I * J * K
    end
end
TVZ_NIJ=squeeze(sum(TVZ_NIJK,4));
for i = 1:I
    Gnume(i,:,:) = squeeze(TVZ_NIJ(:,i,:)) * squeeze(alphaQY_IJM(i,:,:));
end
for n = 1:N
    for m = 1:M
    Gdeno_NIJM(n,:,:,m) = squeeze(TVZ_NIJ(n,:,:)) ./ squeeze(Yhat_IJM(:,:,m));    
    end
end
Gdeno = permute(squeeze(sum(Gdeno_NIJM,3)),[2,1,3]);
end

%%%% V_FMMM %%%
function [V_IMMM]  = local_V_IMMM( alpha, Q, XX, Yhat_IJM, I, J, M)
V_IJMMM = zeros(I,J,M,M,M);alphaXX = alpha .* XX ;
for m = 1:M
    V_IJMMM(:,:,m,:,:) = alphaXX./ Yhat_IJM(:,:,m);
end
V_IMMM = squeeze(mean(V_IJMMM,2));
end

%%% QX_power %%%
function [ QX_power ] = local_QX_power( Q, X, I, J, M)
QX_power = zeros(I,J,M);
for ii = 1:I
    QX_power(ii,:,:) = abs(squeeze(X(ii,:,:)) * squeeze(Q(ii,:,:)).').^2;% I* J* M
end
end

%%% Q  update%%%
function [ Q ] = local_Q( V_IMMM, Q, I, J, M)
ekm = eye(M);
for i = 1:I
    for m=1:M
        V_temp = squeeze(V_IMMM(i,m,:,:));
        q_im = (squeeze(Q(i,:,:))*V_temp)\ekm(:,m);%  M * 1  inv(a)*ekm(:,1)=a\ekm(:,1)        
        q_im = q_im / sqrt(q_im' * V_temp * q_im);% 1 * M
        Q(i,m,:) = conj(q_im); 
    end
end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%