function Yhat = online_multichannelNMF(x,XX,N,K,maxIt,batch_size,batch1_size,rho,G_FNMM,drawConv,V_FK,H_KT,Z_KN)
%% online_multichannelNMF: Blind source separation based on online multichannel NMF
%% # Original paper
% [1] Unsupervised Speech Enhancement Based on Multichannel NMF-Informed Beamforming 
%     for Noise-Robust Automatic Speech Recognition
% [2] Multichannel Extensions of Non-Negative Matrix Factorization With Complex-Valued Data
%
%   [inputs]
%          XX: input 4th-order tensor (time-frequency-wise covariance matrices) (F x T x M x M)
%           N: number of sources
%           K: number of bases (default: ceil(T/10))
%          it: number of iteration (default: 300)
%    drawConv: plot cost function values in each iteration or not (true or false)
%  batch_size: initial mini-batch size (default: 4)
% batch1_size: initial first mini-batch size (default: 80)
%         rho: initial weight of latest statistics
%           V: initial basis matrix (F x K)
%           H: initial activation matrix (K x T)
%           G: initial spatial covariance matrix (F x N x M x M)
%           Z: initial partitioning matrix (K x N)
%
%   [outputs]
%           Y: output 4th-order tensor reconstructed by V, H, G, and Z (F x T x M x M)
%        cost: convergence behavior of cost function in multichannel NMF (maxIt+1 x 1)
%           W: time-variant demixing matrix (F x T x M x M)

% Check errors and set default values
[F,T,M,M] = size(XX);
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
    batch_size = 4;
end
if (nargin < 7)
    batch1_size = 80;
end
if (nargin < 8)
    rho = 0.9; % default 0.9
end
if (nargin < 9)
    G_FNMM = repmat(sqrt(eye(M)/M),[1,1,F,N]);
    G_FNMM = permute(G_FNMM,[3,4,1,2]); % F x N x M x M
end
if (nargin < 10)
    drawConv = false;
end    
if (nargin < 11)
    V_FK = max(rand(F,K),eps);
end
if (nargin < 12)
    H_KT = max(rand(K,batch_size),eps);
end
if (nargin < 13)
    varZ = 0.01;
    Z_KN = varZ*rand(K,N) + 1/N;
    Z_KN = max( Z_KN./sum(Z_KN,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
%     Z = bsxfun(@rdivide, Z, sum(Z,2)); % for prior R2016b
end
if sum(size(V_FK) ~= [F,K]) || sum(size(H_KT) ~= [K,batch_size]) || sum(size(G_FNMM) ~= [F,N,M,M]) || sum(size(Z_KN) ~= [K,N])
    error('The size of input initial variable is incorrect.\n');
end
% W1 = zeros(F,T,M,M);
% W2 = zeros(F,T,M,M);
Yhat = zeros(F,T,M,N);
batch_num = T - batch1_size + 1; % num of mini-batch
fprintf('mini_batch_num:    ');
for j = 1:batch_num
    fprintf('\b\b\b\b%4d', j);   
    if j == 1
        Vjhat = zeros(F,K); Vnumehat = zeros(F,K); Vdenohat = zeros(F,K);
        Zjhat = zeros(K,N); Znumehat = zeros(K,N); Zdenohat = zeros(K,N);
        Gjhat = zeros(F,N,M,M); phihat = zeros(F,N,M,M); psihat = zeros(F,N,M,M);
        Hj_KT = max(rand(K,batch1_size),eps);
        X = XX(:,1:batch1_size,:,:);
        Jj = batch1_size;
    else
        Vjhat = Vj_FK; Vnumehat = Vnume; Vdenohat = Vdeno;
        Zjhat = Zj_KN; Znumehat = Znume; Zdenohat = Zdeno;
        Gjhat = Gj_FNMM; phihat = phij; psihat = psij;
        Hj_KT = H_KT;
        X = XX(:,j+batch1_size-batch_size:j+batch1_size-1,:,:);
        Jj = batch_size;
    end
    Vj_FK = V_FK; Zj_KN = Z_KN; Gj_FNMM = G_FNMM; 
    Y_FTMM = local_Y( Vj_FK, Hj_KT, Gj_FNMM, Zj_KN, F, Jj, M); % initial model tensor
    % Iterative update
    for it = 1:maxIt
        [ Y_FTMM, Vj_FK, Hj_KT, Gj_FNMM, Zj_KN, Vnume, Vdeno, Znume, Zdeno, phij, psij ] = ...
            local_iterativeUpdate( X, Y_FTMM, Vj_FK, Hj_KT, Gj_FNMM, Zj_KN, F, Jj, K, N, M, Vjhat, Zjhat, Gjhat, Vnumehat, Vdenohat, Znumehat, Zdenohat, phihat, psihat, rho );    
    end
    % Multichannel Wiener filtering
    Y_FTMM = permute(Y_FTMM,[3,4,1,2]); % M x M x F x Jj
    if j == 1
        for f = 1:F   
            for t = 1:Jj       
                for src = 1:N            
                    ys = 0;           
                    for k = 1:K               
                        ys = ys + Zj_KN(k,src)*Vj_FK(f,k)*Hj_KT(k,t); % (54) of [2]           
                    end                    
                    Yhat(f,t,:,src) = ys * squeeze(Gj_FNMM(f,src,:,:))/Y_FTMM(:,:,f,t)*x(:,f,t); % (54) of [2] M x 1       
                end              
            end          
        end
    else
        for f = 1:F    
            for src = 1:N                 
                ys = 0;            
                for k = 1:K                                      
                    ys = ys + Zj_KN(k,src)*Vj_FK(f,k)*Hj_KT(k,batch_size); % (54) of [2]                              
                end                
                Yhat(f,j+batch1_size-1,:,src) = ys * squeeze(Gj_FNMM(f,src,:,:))/Y_FTMM(:,:,f,batch_size)*x(:,f,j+batch1_size-1); % (54) of [2] M x 1                
            end                      
        end
    end
    
% %     G_per = reshape(permute(Gj, [3 4 2 1]), [M*M, N, F]);
% %     What1 = zeros(F,Jj,M,M);
% %     What2 = zeros(F,Jj,M,M);
% %     for t = 1:Jj   
% %         for f = 1:F   
% %             VHZ_P = (Vj(f,:).*Hj(:,t)')*Zj(:,1); % 1 x 1              
% %             VHZ_Q = (Vj(f,:).*Hj(:,t)')*Zj(:,2:N); % 1 x (N-1)               
% %             Pft = reshape(G_per(:,1,f)*VHZ_P', [M, M]); % (35) of [1]             
% %             Qft = reshape(G_per(:,2:N,f)*VHZ_Q', [M, M]); % (36) of [1]
% %             Wft1 = (Pft+Qft)\Pft; % (8) of [1]
% %             Wft2 = (Pft+Qft)\Qft;
% %             What1(f,t,:,:) = Wft1;
% %             What2(f,t,:,:) = Wft2;
% %         end    
% %     end
% %     if j == 1
% %         W1(:,1:batch1_size,:,:) = What1;
% %         W2(:,1:batch1_size,:,:) = What2;
% %     else
% %         W1(:,j+batch1_size-1,:,:) = What1(:,batch_size,:,:);
% %         W2(:,j+batch1_size-1,:,:) = What2(:,batch_size,:,:);
% %     end
end
end

%%% Iterative update %%%
function [ Y_FTMM, V_FK, H_KT, G_FNMM, Z_KN, Vnume, Vdeno, Znume, Zdeno, phi, psi ] = local_iterativeUpdate( X, Y_FTMM, V_FK, H_KT, G_FNMM, Z_KN, F, T, K, N, M, Vhat, Zhat, Ghat, Vnumehat, Vdenohat, Znumehat, Zdenohat, phihat, psihat, rho )
global MNMF_p_norm;
%%%%% Update V %%%%%
invY_FTMM = local_inverse( Y_FTMM, F, T, M );% inverse of Y    F x T x M x M 
invYXinvY_FTMM = local_multiplicationXYX( invY_FTMM, X, F, T, M ); % invY * X * invY
Vnume = local_Vfrac( invYXinvY_FTMM, H_KT, Z_KN, G_FNMM, F, K, M ); % (41) of [1]  F x K 
Vdeno = local_Vfrac( invY_FTMM, H_KT, Z_KN, G_FNMM, F, K, M ); % (42) of [1]  F x K         
V_FK = (Fjaxa(V_FK, Vnume, Vhat, Vnumehat, rho) ./ Fjx(Vdeno, Vdenohat, rho)) .^ MNMF_p_norm; % (43) of [1]
Y_FTMM = local_Y( V_FK, H_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update H %%%%%
invY_FTMM = local_inverse( Y_FTMM, F, T, M ); % inverse of Y
invYXinvY_FTMM = local_multiplicationXYX( invY_FTMM, X, F, T, M ); % invY * X * invY
Hnume = local_Hfrac( invYXinvY_FTMM, V_FK, Z_KN, G_FNMM, T, K, M ); % K x T
Hdeno = local_Hfrac( invY_FTMM, V_FK, Z_KN, G_FNMM, T, K, M ); % K x T
H_KT = H_KT.*max((Hnume./Hdeno).^MNMF_p_norm,eps); % (30) of [1]
Y_FTMM = local_Y( V_FK, H_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update Z %%%%%
invY_FTMM = local_inverse( Y_FTMM, F, T, M ); % inverse of Y
invYXinvY_FTMM = local_multiplicationXYX( invY_FTMM, X, F, T, M ); % invY * X * invY
Znume = local_Zfrac( invYXinvY_FTMM, V_FK, H_KT, G_FNMM, K, N, M ); % (44) of [1]  K x N
Zdeno = local_Zfrac( invY_FTMM, V_FK, H_KT, G_FNMM, K, N, M ); % (45) of [1]  K x N
Z_KN = (Fjaxa(Z_KN, Znume, Zhat, Znumehat, rho) ./ Fjx(Zdeno, Zdenohat, rho)) .^ MNMF_p_norm; % (46) of [1]
Z_KN = max( Z_KN./sum(Z_KN,2), eps ); % to ensure "sum_n Z_kn = 1" (using implicit expansion)
% Z = bsxfun(@rdivide, Z, sum(Z,2)); % for prior R2016b
Y_FTMM = local_Y( V_FK, H_KT, G_FNMM, Z_KN, F, T, M );

%%%%% Update G %%%%%
invY_FTMM = local_inverse( Y_FTMM, F, T, M ); % inverse of Y
invYXinvY_FTMM = local_multiplicationXYX( invY_FTMM, X, F, T, M ); % invY * X * invY
[G_FNMM,phi,psi] = local_RiccatiSolver( invYXinvY_FTMM, invY_FTMM, V_FK, H_KT, G_FNMM, Z_KN, F, T, N, M, Ghat, phihat, psihat, rho );
Y_FTMM = local_Y( V_FK, H_KT, G_FNMM, Z_KN, F, T, M );
end

%%% Y %%%
function [ Y_FTMM ] = local_Y( V_FK, H_KT, G_FNMM, Z_KN, F, T, M )
Y_FTMM = zeros(F,T,M,M);
for mm = 1:M*M
    Gmm_FN = G_FNMM(:,:,mm); % F x N
    Y_FTMM(:,:,mm) = ((Gmm_FN*Z_KN').*V_FK)*H_KT; % (26) of [1]
end 
end

%%% Vfrac %%%
function [ Vfrac_FK ] = local_Vfrac( X_FTMM, H_KT, Z_KN, G_FNMM, F, K, M ) % (41)(42) of [1]
Vfrac_FK = zeros(F,K); %size(X_FTMM(:,:,mm))  F*T ;size(H_KT')  T*K
for mm = 1:M*M         %size(G_FNMM(:,:,mm))  F*N ;size(Z_KN')  N*K
    Vfrac_FK = Vfrac_FK + real( (X_FTMM(:,:,mm)*H_KT').*(conj(G_FNMM(:,:,mm))*Z_KN') ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Hfrac %%%
function [ Hfrac_KT ] = local_Hfrac( X_FTMM, V_FK, Z_KN, G_FNMM, T, K, M ) % (30) of [1]
Hfrac_KT = zeros(K,T);%size(G_FNMM(:,:,mm))  F*N ;size(Z_KN')  N*K
for mm = 1:M*M        %size(((G_FNMM(:,:,mm)*Z_KN').*V_FK)')   K*F; size(X_FTMM(:,:,mm))  F*T
    Hfrac_KT = Hfrac_KT + real( ((G_FNMM(:,:,mm)*Z_KN').*V_FK)'*X_FTMM(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Zfrac %%%
function [ Zfrac_KN ] = local_Zfrac( X_FTMM, V_FK, H_KT, G_FNMM, K, N, M) % (44)(45) of [1]
Zfrac_KN = zeros(K,N);%size(X_FTMM(:,:,mm))  F*T;size(H_KT')  T*K
for mm = 1:M*M        %size(((X_FTMM(:,:,mm)*H_KT.').*V_FK)')   K*F; size(G_FNMM(:,:,mm))  F*N ;
    Zfrac_KN = Zfrac_KN + real( ((X_FTMM(:,:,mm)*H_KT.').*V_FK)'*G_FNMM(:,:,mm) ); % using tr(AB') = sum_{i,j} (A_{ij}*B_{ij})
end
end

%%% Riccati solver %%%
function [ G_FNMM, phi_FNMM, psi_FNMM ] = local_RiccatiSolver( X, Y, V_FK, H_KT, G_FNMM, Z_KN, F, T, N, M, Ghat, phihat, psihat, rho )
X = reshape(permute(X, [3 4 2 1]), [M*M, T, F]); % invYXinvY, MM x T x F
Y = reshape(permute(Y, [3 4 2 1]), [M*M, T, F]); % invY, MM x T x F
phi_FNMM = zeros(F,N,M,M);
psi_FNMM = zeros(F,N,M,M);
deltaEyeM = eye(M)*(10^(-12)); % to avoid numerical instability
for n = 1:N
    for f = 1:F
        ZVH_1T = (V_FK(f,:).*Z_KN(:,n)')*H_KT; % V(i,:) 1xK, Z(:,n)' 1xK, H KxJ,  1 x T
        phi_in = reshape(X(:,:,f)*ZVH_1T', [M, M]); % (47) of [1]  size(X(:,:,i)) MMxJ, ZVH' Jx1,  MM x 1 
        psi_in = reshape(Y(:,:,f)*ZVH_1T', [M, M]); % (48) of [1]  size(Y(:,:,i)) MMxJ, ZVH' Jx1,  MM x 1
        G_in = reshape(G_FNMM(f,n,:,:), [M, M]); 
        phihat_in = reshape(phihat(f,n,:,:), [M, M]); % last batch phi
        psihat_in = reshape(psihat(f,n,:,:), [M, M]); % last batch psi
        Ghat_in = reshape(Ghat(f,n,:,:), [M, M]); % last batch G
        A = sqrtm(Fjx(psi_in,psihat_in,rho)+deltaEyeM); % (49) of [1]第一行
        B = FjAXA(G_in,phi_in,Ghat_in,phihat_in,rho);% (49) of [1]第二行中Fj(Gnf,phi_nf,Gnf)
        C = A\sqrtm(A*B*A+deltaEyeM)/A; % (49) of [1]
        C = (C+C')/2 + deltaEyeM;
        G_FNMM(f,n,:,:) = C/trace(C);
        phi_FNMM(f,n,:,:) = phi_in;
        psi_FNMM(f,n,:,:) = psi_in;
    end
end
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EOF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%