function [Y, W, D, V, C, W_hat, W_res, obj_vals] =  binaural_auxive_update_multi(X, W, D, V, C, W_hat, partition, varargin)
% IVE for online and offline BSS
% References:
%   [1] Overdetermined independent vector analysis
%   [2] adaptive blind audio source extraction supervised by dominant speaker identification using x-vectors
%   [3] R. Scheibler and N. Ono, “Independent vector analysis with more microphones than sources,” in Proc. WASPAA, 2019.
%   [4] R. Scheibler and N. Ono, "fast independent vector extraction by iterative sinr maximization"
%   [5] A Unified Bayesian View on Spatially Informed Source Separation and Extraction based on Independent Vector Analysis
option.iter_num = 10; option.inner_iter_num = 2; option.verbose = true; option.select = 1;
option.parti = 0; option.thFactor = 1/50; option.online = 0; option.project_back = 1;
if nargin > 7
    user_option = varargin{1};
    for fn = fieldnames(user_option)'
        option.(fn{1}) = user_option.(fn{1});
    end
end
%% Initalization
global Ratio_Rxx; global PowerRatio;  global GammaRatio; global epsilon;
iter_num = option.iter_num; % 外迭代
inner_iter_num = option.inner_iter_num; % 内迭代
verbose = option.verbose; % 是否观察收敛
alpha = 1; % 滑动平均因子；offline = 1，online < 1
gamma = 1;
parti = option.parti; % 是否对载波进行分块（子块）
select = option.select; % 是否选择子块
thFactor = option.thFactor; % 子块选择阈值
online = option.online; % 是否为在线
D_open = option.D_open; % 是否使用加权因子更新R
IVE_method = option.IVE_method; % IVE的方法
if online % Default online Aux-IVA parameter
    iter_num = option.iter_num; inner_iter_num = option.inner_iter_num; alpha = option.forgetting_fac; verbose = false;
    gamma = option.gamma;
end

[N, T, M] = size(X); % #freq * #frame * #mic
[~, ~] = size(W); % #freq * #mic * #source
K = 1; % IVE single source

% M>2时Xn * Xn'的快速计算方式，size(XX) = N * T * M^2，比下面方法快5倍
% 但是不太好加入diagonal loading
XXn = reshape(X, [N, T, M, 1]) .* conj(reshape(X, [N, T, 1, M]));    XX = reshape(XXn,[N, T, M^2]);
%%% 以下为快速diagonal loading代码
% Y = ones(size(XXn,1),size(XXn,2),size(XXn,3),size(XXn,4))*eps*0;
% [XY] = local_addtion(XXn, Y, M);
% XY1 = reshape(XY,[N, T, M^2]);
%%%
Cxx = cal_Cxx(X); % (10) of [2] || (P1) of [1]
%Cxx = squeeze(mean(XXn,2)); % Cxx快速计算方式, 4mic时出错，待解决
C = gamma * Cxx + (1 - gamma) * C; % online, (10) of [2]
W = W_hat(:,1:M,1);

if IVE_method == 4 % FIVE方法需要pre-whitening，令Cxx等于单位矩阵;
    Q_H = zeros(N,M,M);
    for n = 1:N
        % We will need the inverse square root of Cx
        [e_vec,e_val] = eig(squeeze(C(n,:,:)),'vector');
        Q_H = fliplr(e_vec) .* sqrt(flipud(e_val)).';
        X(n,:,:) = squeeze(X(n,:,:)) * (Q_H^-1).';
    end
    project_back = 1; % FIVE need backprojection
    XXn = reshape(X, [N, T, M, 1]) .* conj(reshape(X, [N, T, 1, M]));   
    XX = reshape(XXn,[N, T, M^2]);
end

for m = 1:M
    XE_mean(:,m) = mean(mean(abs(XX(:,:,m^2))));
    XE_TH = XE_mean * thFactor; % select threshold
end

[spec_indices,par_select] = selectpar(partition, select, parti, XX, 1, N, XE_TH); % select & partition initalize
H = zeros(N, M); % #freq * #mic
Y = zeros(N, T); % #freq * # frame * #source
Vxx = zeros(N, M, M); % #freq * #mic * #mic * #source
if verbose % callback obj_vals
    obj_vals = zeros(iter_num + 1, 1);
    obj_vals(1) = obj_mm(W_hat, D, X, V, partition, alpha);
    fprintf('iter = %d (I), obj = %f\n', 0, obj_vals(1));
end

%% Main Iteration
for iter = 1:iter_num
    % faster calculation of Y
    Y = sum(X .* permute(repmat(W,1,1,T),[1,3,2]), 3);
    % Calculate R = ||Y||^2;
    if D_open
        R = D .* abs(Y).^PowerRatio;
    else
        R = abs(Y).^PowerRatio;
    end
    % Calculate Vxx
    if select
        Vxx = cal_Vxx(R, XX, par_select{1}); % select on
    else
        Vxx = cal_Vxx(R, XX, partition{1}); % select off
    end
    for inner_iter = 1:inner_iter_num
        switch IVE_method
            case 1 % Ref.[1], IP-1，严格来说是IP-3，但是K=1时IP-1=IP-3 
                ind = 1:M;
                ind(1) = [];
                for n = 1:N % (11) of [1]
                    WW = squeeze(W_hat(n,:,:));
                    H(n,:) = -null(WW(:,ind).')';
                end  
                if option.annealing % 是否使用退火因子
                    fac_a = max(0.5-iter/iter_num, 0); % annealing factor
                else
                    fac_a = 1; % =1即等效于不退火
                end
                Vk = alpha * Vxx + (1 - alpha) * V; % online
                if option.prior(1) == 1 % 有目标源的prior, (43) & (79) of [5]
                    hf = cal_RTF(N,16000,option.mic_pos,option.theta(:,1));                  
                    delta_f = option.deltaf; 
                    [W, D] = update_w(W, Vk, H, spec_indices{1}, hf, delta_f, fac_a);                    
                else
                    [W, D] = update_w(W, Vk, H, spec_indices{1});                    
                end
                if option.prior(2) == 1 % 有BG的prior, (43) & (87) of [5]            
                    hf_bg = cal_RTF(N,16000,option.mic_pos,option.theta(:,2:end));
                    delta_bg = option.deltabg;                    
                    [W_hat] = update_What(W, W_hat, C, hf_bg, delta_bg, fac_a); % (87) of [5]
                else
                    [W_hat] = update_What(W, W_hat, C); % (17) of [1]
                end
%                 [W, D] = update_w(W, Vk, H, spec_indices{1}); % (11),(12) of [1]
                
            case 2 % Ref.[1], IP-2, for K=1
                Vk = alpha * Vxx + (1 - alpha) * V; % online
                G1 = squeeze(Vk); % Algorithm 1.7 of [1]
                for t = 1:T
                    RR(t) = norm(Y(:,t)); % Algorithm 1.6 of [1]
                end
                ck = sum(RR)/T;
                for n = 1:N
                    [u,du] = eig(squeeze(C(n,:,:)),squeeze(G1(n,:,:)),'vector'); % Algorithm 3.1 of [1]
                    [~,max_eig_index] = max(du); 
                    u = u(:,max_eig_index); % Algorithm 3.1 of [1]
                    w = u / sqrt(u' * squeeze(G1(n,:,:)) * u); % Algorithm 3.2 of [1]
                    W(n,:) = w';
                    D(n) = 1 / real(u' * squeeze(G1(n,:,:)) * u); 
                    R(n,:) = R(n,:) * ck^-1; % Algorithm 1.9 of [1]
                    W(n,:) = W(n,:) * ck^-1/2; % Algorithm 1.9 of [1]
                end

            case 3 % Ref.[2]
                Vk = alpha * Vxx + (1 - alpha) * V; % online, (9) of [2]
                A = zeros(N,M);
                for t = 1:T
                    RR(t) = norm(Y(:,t)); 
                end
                ck = sum(RR)/T;
                for n = 1:N
                    c = squeeze(C(n,:,:));
                    w = squeeze(W(n,:)).';
                    A(n,:) = c*w / (w'*c*w); % (11) of [2]
                    W(n,:) = (squeeze(Vk(n,:,:))+epsilon * eye(M)) \ squeeze(conj(A(n,:))).'; % (12) of [2]
%                     R(n,:) = R(n,:) * ck^-1; % 从[1]中copy过来的归一化方式，同case2
%                     W(n,:) = W(n,:) * ck^-1/2; % 从[1]中copy过来的归一化方式，同case2
                end
            case 4 % Ref.[4], Algorithm 1
                Vk = alpha * Vxx + (1 - alpha) * V; % online
                for n = 1:N
                    Vf = squeeze(Vk(n,:,:));
                    [rm,lambda] = eig(Vf,'vector'); % Algorithm 3.1 of [1]
                    [lambda_m, min_eig_index] = min(lambda);
                    w = lambda_m^-1/2 * rm(:,min_eig_index);
                    W(n,:) = w';
                end
        end
    end
    if verbose
        obj_vals(iter+1) = obj_mm(W_hat, D, X, V, partition, alpha);
        fprintf('iter = %d (W), obj = %f\n', iter, obj_vals(iter+1));
    end
end
%% After-Processing

% faster calculation of Y
Y = sum(X .* permute(repmat(W,1,1,T),[1,3,2]), 3);
if D_open
    R = D .* abs(Y).^PowerRatio;
else
    R = abs(Y).^PowerRatio;
end
if select
    Vxx = cal_Vxx(R, XX, par_select{1});
else
    Vxx = cal_Vxx(R, XX, partition{1});
end

V = alpha * Vxx + (1 - alpha) * V; % online

% Cal. obj function
if ~verbose
    obj_vals = obj_mm(W_hat, D, X, V, partition, 1);
end

if IVE_method == 2 || IVE_method == 3
    [W_hat] = update_What(W, W_hat, C); % (17) of [1] || Algorithm 1.11 of [1]
end

% Rescale (Normalization)，该rescale从auxIVA中继承而来，要求W_hat必须为方阵
[W_res,~] = rescale(W_hat,D); % overdetermined
if IVE_method == 4 
    W_res = W;
end
% Output Y
Y = sum(X .* permute(repmat(W_res(:,:,1),1,1,T),[1,3,2]), 3);
end

function val = obj_mm(W, D, X, V, par, alpha)
% Calculate obj function
global epsilon
[N, T, ~] = size(X);
[~,K] = size(D);
det_W = zeros(N,1);
Y = zeros(N, T, K);
G = zeros(K, 1);
for k = 1:K
    Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
    R = D(:,k) .* abs(Y(:,:,k)).^2;
    G(k) = contrast(R, par{k});
    d_hat(:,k) = abs(dot(W(:,:,k), squeeze(sum(W(:,:,k) .* V(:,:,:,k), 2)), 2));
end
G_hat = sum(dot(D, d_hat));
for n = 1:N
    det_W(n,:) = det(squeeze(W(n,:,:)));
end
val = alpha * sum(G) + (1 - alpha) * G_hat + epsilon * sum(sum(D)) - sum(sum(log(D))) ...
    - 2 * sum(log(abs(det_W)));
end

function val = contrast(R, par)
% Calculate contrast funtion
[~, T] = size(R);
G = zeros(par.num, T);
for n = 1:par.num
    indices = par.index{n};
    G(n,:) = sum(par.contrast(sum(R(indices,:), 1)), 1);
end
val = sum(mean(G, 2));
end

function [W, D] = update_w(W, V, H, indices, varargin)
% Update W (12)
global diagonal_method; global frameNum; global frameStart; global epsilon_start_ratio;
global epsilon; global Ratio_Rxx; global epsilon_ratio;    global DOA_tik_ratio; global DOA_Null_ratio;
[N, M] = size(W);
D = ones(N, 1);
if nargin > 4
    hf = varargin{1};
    delta_f = varargin{2};
    fac_a = varargin{3};
    n_t = size(hf,3);
    RTF = 1;
else
    RTF = 0;
end
if diagonal_method == 0
    if  frameNum <= frameStart
        for n = indices
            if RTF
                Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (eye(M)+hf(:,n)*hf(:,n)') / delta_f^2;
            else
                Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M);
            end
            %  Q = squeeze(V(n,:,:)) + diag([epsilon1 epsilon2  ])
            h = H(n,:)';
            w = Q \ h; %sum(abs(Q \ h-inv(Q)*h))
            % w = w / norm(w);
            w = w / sqrt(w' * Q * w);
            W(n,:) = w';
            D(n) = 1 / real(w' * Q * w);
        end
    else
        for n = indices
            if RTF
                P_t = DOA_tik_ratio * fac_a * eye(M);
                for i = n_t
                    P_t = P_t + DOA_Null_ratio * fac_a * hf(:,n,i)*hf(:,n,i)'; % (43) of [5]
                end
%                 Q = squeeze(V(n,:,:)) + epsilon_ratio*epsilon * eye(M)...
%                     + (DOA_tik_ratio*eye(M) + DOA_Null_ratio*hf(:,n)*hf(:,n)') / delta_f;
                Q = squeeze(V(n,:,:)) + P_t / delta_f^2; % (79) of [5] 
            else
                Q = squeeze(V(n,:,:)) + epsilon_ratio*epsilon * eye(M);
            end
            %  Q = squeeze(V(n,:,:)) + diag([epsilon1 epsilon2  ])
            h = H(n,:)';
            w = Q \ h; %sum(abs(Q \ h-inv(Q)*h))
            % w = w / norm(w);
            w = w / sqrt(w' * Q * w);
            W(n,:) = w';
            D(n) = 1 / real(w' * Q * w);
        end
    end
end
if diagonal_method == 1
    if RTF==0
        for n = indices
            %Q = squeeze(V(n,:,:)) + eye(M)*mean(diag(squeeze(V(n,:,:))))*(eps+Ratio_Rxx) + RTF;
            %  Xnn =Xnn+eye(M)*mean(diag(Xnn))*Ratio_Rxx; % diagonal loading facto
            mean_V = mean(diag(squeeze(V(n,:,:))));
            Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M);
            h = H(n,:)';
            w = Q \ h; %sum(abs(Q \ h-inv(Q)*h))
            % w = w / norm(w);
            w = w / sqrt(w' * Q * w); % [2] algorithm 1
            W(n,:) = w';
            D(n) = 1 / real(w' * Q * w);
        end
    end
    if RTF==1
        for n = indices
            %Q = squeeze(V(n,:,:)) + eye(M)*mean(diag(squeeze(V(n,:,:))))*(eps+Ratio_Rxx) + RTF;
            %  Xnn =Xnn+eye(M)*mean(diag(Xnn))*Ratio_Rxx; % diagonal loading facto
            mean_V = mean(diag(squeeze(V(n,:,:))));
            Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                + (mean_V*DOA_tik_ratio*eye(M)+mean_V*DOA_Null_ratio*hf(:,n)*hf(:,n)');
            h = H(n,:)';
            w = Q \ h; %sum(abs(Q \ h-inv(Q)*h))
            % w = w / norm(w);
            w = w / sqrt(w' * Q * w); % [2] algorithm 1
            W(n,:) = w';
            D(n) = 1 / real(w' * Q * w);
        end
    end
end
if diagonal_method == 2
    for n = indices
        Q = squeeze(V(n,:,:)) + diag(diag(squeeze(V(n,:,:))))*(eps+Ratio_Rxx) + RTF;
        h = H(n,:)';
        w = Q \ h; %sum(abs(Q \ h-inv(Q)*h))
        % w = w / norm(w);
        w = w / sqrt(w' * Q * w);
        W(n,:) = w';
        D(n) = 1 / real(w' * Q * w);
    end
end
end

function [W_hat,J] = update_What(W, W_hat, C, varargin)
% Update W_hat & J (12)(13) of [3]
global diagonal_method;  global DOA_tik_ratio; global DOA_Null_ratio;
global epsilon; global epsilon_ratio1; global diagonal_method; global Ratio_Rxx;
if nargin > 3
    hf = varargin{1};
    delta_bg = varargin{2};
    fac_a = varargin{3};
    n_bg = size(hf,3);
    RTF = 1;
else
    RTF = 0;
end
[N, M, K] = size(W);
J = zeros(N, M - K, K);
E1 = [eye(K) zeros(K,M-K)]; % #source * #mic
E2 = [zeros(M-K,K) eye(M-K)]; % #mic-source *  #mic
if diagonal_method==0
    if RTF
        for n = 1:N
            P_bg = DOA_tik_ratio * eye(M);
            if K == 1
                Wf = W(n,:,:).';% Wf是[3]中wf的转置，所以wf' = conj(Wf)
            else
                Wf = squeeze(W(n,:,:)); % Wf是[3]中wf的转置，所以wf' = conj(Wf)
            end
            W_hat(n,:,1:K) = W(n,:,:);
            for i = n_bg
               P_bg = P_bg + DOA_Null_ratio * hf(:,n,i)*hf(:,n,i)'; % (43) of [5]
            end
            Cf = squeeze(C(n,:,:)) + P_bg * fac_a / delta_bg^2; % (87) of [5]
            J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf) ); % (87) of [5]
            W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
        end
    else
        for n = 1:N
            if K == 1
                Wf = W(n,:,:).';% Wf是[3]中wf的转置，所以wf' = conj(Wf)
            else
                Wf = squeeze(W(n,:,:)); % Wf是[3]中wf的转置，所以wf' = conj(Wf)
            end
            W_hat(n,:,1:K) = W(n,:,:);
            Cf = squeeze(C(n,:,:));% + epsilon_ratio1*epsilon * eye(M);
            J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf) ); % (13) of [3]
            %         J(n,:,:) = (( Wf.' * Cf * E1.' ) \ ( Wf.' * Cf * E2.' ))'; % (17) of [2]
            W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
            %         J1(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf))+ epsilon_ratio1*epsilon * eye(K);
            %         W_hat1(n,1:K,(K+1):M) = conj(permute(J1(n,:,:),[1,3,2]));
            %         squeeze(J1(n,:,:))
            %         cond(squeeze(W_hat1(n,:,:)))
            %         cond(squeeze(W_hat(n,:,:)))
        end
    end
end
if diagonal_method==1
    for n = 1:N
        if K == 1
            Wf = W(n,:,:).';
        else
            Wf = squeeze(W(n,:,:));
        end
        W_hat(n,:,1:K) = W(n,:,:);
        Cf = squeeze(C(n,:,:)) + eye(M)*mean(diag(squeeze(C(n,:,:))))*(eps+Ratio_Rxx);
        J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf));
        W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
    end
end
if diagonal_method==2
    for n = 1:N
        if K == 1
            Wf = W(n,:,:).';
        else
            Wf = squeeze(W(n,:,:));
        end
        W_hat(n,:,1:K) = W(n,:,:);
        Cf = squeeze(C(n,:,:)) + diag(diag(squeeze(C(n,:,:))))*(eps+Ratio_Rxx);
        J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf) );
        W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
    end
end
end

function [W,D] = rescale(W,D)
% rescale W or W_hat (also D)
[N, M, K] = size(W);
[~, rK] = size(D);
for n = 1:N
    U = squeeze(W(n,:,:));
    d = diag(U \ eye(K));
    U = U * diag(d);
    W(n,:,:) = U;
    if M == rK
        D(n,:) = D(n,:) * diag(1 ./ abs(d).^2);
    else
        Ur = squeeze(W(n,1:rK,1:rK));
        dr = diag(Ur \ eye(rK));
        D(n,:) = D(n,:) * diag(1 ./ abs(dr).^2);
    end
end
end

function Vxx = cal_Vxx(R, XX, par)
% Calculate V (12)
[N, T] = size(R);
M = sqrt(size(XX,3));
Vxx = zeros(N, M, M);
for n = 1:par.num
    indices = par.index{n};
    N_par = par.size(n);
    F = par.contrast_derivative(sum(R(indices,:), 1));
    Z = squeeze(sum(F .* XX(indices,:,:), 2));
    Vn = permute(reshape(Z', M, M, N_par), [3 2 1]);
    Vxx(indices,:,:) = Vxx(indices,:,:) + Vn / T;
end
end

% function Cxx = cal_Cxx(XX) % 更精简更快速的Cxx计算方法（结果与旧版相同）
% [N, T, M] = size(XX);
% M = sqrt(M);
% Cxx = zeros(N, M, M);
% Z = squeeze(sum(XX, 2));
% Cn = permute(reshape(Z', M, M, N), [3 2 1]);
% Cxx = Cxx + Cn / T;
% end

function Cxx = cal_Cxx(X) % 旧版Cxx计算方法，暂时保留，(9) of [3]
[N, T, M] = size(X);
Cxx = zeros(N, M, M);
for n = 1:N
    Xf = squeeze(X(n,:,:));
    if T == 1
        % 当T=1时，由于squeeze的特性，Xf与[3]中xf相等，直接用原公式
        Cf = Xf * Xf';
    else
        % 本Xf为[3]中xf的转置，因此(xf*xf') = Xf.' * conj(Xf)
        Cf = Xf.' * conj(Xf) / T; 
    end
    Cxx(n,:,:) = Cf;
end
end

%%% 快速diagonal loading
function [ XY ] = local_addtion( X, Y, M)
if M == 2
    %XY = ones(I, J, M, M)*;  % 2048*128*2*2
    XY(:,:,1,1) = X(:,:,1,1)+Y(:,:,1,1);    
    XY(:,:,1,2) = X(:,:,1,2); % or only diagonal addition: XY(:,:,1,2) = X(:,:,1,2);
    XY(:,:,2,1) = X(:,:,2,1); % or only diagonal addition: XY(:,:,2,1) = X(:,:,2,1);
    XY(:,:,2,2) = X(:,:,2,2)+Y(:,:,2,2);
elseif M == 3
    XY(:,:,1,1) = X(:,:,1,1)+Y(:,:,1,1);
    XY(:,:,1,2) = X(:,:,1,2);
    XY(:,:,1,3) = X(:,:,1,3);
    XY(:,:,2,1) = X(:,:,2,1);
    XY(:,:,2,2) = X(:,:,2,2)+Y(:,:,2,2);
    XY(:,:,2,3) = X(:,:,2,3);
    XY(:,:,3,1) = X(:,:,3,1);
    XY(:,:,3,2) = X(:,:,3,2);
    XY(:,:,3,3) = X(:,:,3,3)+Y(:,:,3,3);
elseif M == 4
    XY(:,:,1,1) = X(:,:,1,1)+Y(:,:,1,1);
    XY(:,:,1,2) = X(:,:,1,2);
    XY(:,:,1,3) = X(:,:,1,3);
    XY(:,:,1,4) = X(:,:,1,4);
    XY(:,:,2,1) = X(:,:,2,1);
    XY(:,:,2,2) = X(:,:,2,2)+Y(:,:,2,2);
    XY(:,:,2,3) = X(:,:,2,3);
    XY(:,:,2,4) = X(:,:,2,4);
    XY(:,:,3,1) = X(:,:,3,1);
    XY(:,:,3,2) = X(:,:,3,2);
    XY(:,:,3,3) = X(:,:,3,3)+Y(:,:,3,3);
    XY(:,:,3,4) = X(:,:,3,4);
    XY(:,:,4,1) = X(:,:,4,1);
    XY(:,:,4,2) = X(:,:,4,2);
    XY(:,:,4,3) = X(:,:,4,3);
    XY(:,:,4,4) = X(:,:,4,4)+Y(:,:,4,4);
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
