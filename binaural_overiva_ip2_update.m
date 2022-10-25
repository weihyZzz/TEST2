function [Y, W, D, V, C, W_hat, W_res, obj_vals] =  binaural_overiva_ip2_update(X, W, D, V, C, W_hat, partition, varargin)
% OverIVA IP-NP and DX-BG for offline BSS
% References:
%   [1] R. Scheibler and N. Ono, "MM Algorithms for Joint Independent Subspace Analysis 
%                                 with Application to Blind Single and Multi-Source Extraction"
%   [2] A. Brendel, T. Haubner and W. Kellermann, "A Unified Bayesian View on Spatially Informed Source Separation 
%                                                  and Extraction based on Independent Vector Analysis"

% Default batch Aux-IVA parameter
option.iter_num = 10; option.inner_iter_num = 2; option.verbose = true; option.select = 1;
option.parti = 0; option.thFactor = 1/50; option.online = 0;
if nargin > 7
    user_option = varargin{1};
    for fn = fieldnames(user_option)'
        option.(fn{1}) = user_option.(fn{1});
    end
end
%% Initalization
global Ratio_Rxx; global PowerRatio; global OrthogonalIVA;

iter_num = option.iter_num; % 外迭代
verbose = option.verbose; % 是否观察收敛
alpha = 1; % 滑动平均因子；offline = 1，online < 1
gamma = 1;
parti = option.parti; % 是否对载波进行分块（子块）
select = option.select; % 是否选择子块
thFactor = option.thFactor; % 子块选择阈值
online = option.online; % 是否为在线
D_open = option.D_open; % 是否使用加权因子更新R
if online % Default online Aux-IVA parameter
    iter_num = option.iter_num; inner_iter_num = option.inner_iter_num; alpha = option.forgetting_fac; verbose = false;
    gamma = option.gamma;
end

[N, T, M] = size(X); % #freq * #frame * #mic
[~, ~, K] = size(W); % #freq * #mic * #source

% M>2时Xn * Xn'的快速计算方式，size(XX) = N * T * M^2，比下面方法快5倍
% 但是不太好加入diagonal loading
XXn = reshape(X, [N, T, M, 1]) .* conj(reshape(X, [N, T, 1, M]));    XX = reshape(XXn,[N, T, M^2]);
if M > K  % overdetermined
    Cxx = cal_Cxx(X);
%     Cxx = squeeze(mean(XXn,2)); % Cxx快速计算方式, 4mic时出错，待解决
    C = gamma * Cxx + (1 - gamma) * C; % online
    W = W_hat(:,1:M,1:K);
end

for m = 1:M
    XE_mean(:,m) = mean(mean(abs(XX(:,:,m^2))));
    XE_TH = XE_mean * thFactor; % select threshold
end
[spec_indices,par_select] = selectpar(partition, select, parti, XX, K, N, XE_TH); % select & partition initalize

H = zeros(N, M); % #freq * #mic
Y = zeros(N, T, K); % #freq * # frame * #source
Vxx = zeros(N, M, M, K); % #freq * #mic * #mic * #source

if verbose % callback obj_vals
    if M == K
        W_hat = W;
    end
    obj_vals = zeros(iter_num + 1, 1);
    obj_vals(1) = obj_mm(W_hat, D, X, V, partition, alpha);
    fprintf('iter = %d (I), obj = %f\n', 0, obj_vals(1));
end

%% Main Iteration，multi-source update
for iter = 1:2:iter_num
    for k = 1:2:K
        L = k:k+1;
        % faster calculation of Y
        Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
        Y(:,:,k+1) = sum(X .* permute(repmat(W(:,:,k+1),1,1,T),[1,3,2]), 3);
        
        if D_open
            R = permute(repmat(D(:,L),1,1,T),[1,3,2]) .* abs(Y(:,:,L)).^PowerRatio;% freq * time * 2
        else
            R = abs(Y(:,:,L)).^PowerRatio; % freq * time * 2
        end
        
        if option.annealing % 是否使用退火因子
            fac_a = max(0.5-iter/iter_num, 0); % annealing factor
        else
            fac_a = 1; % =1即等效于不退火
        end
        
        if M > K % overdetermined update BG
            if option.prior(K+1) == 1 % 有BG先验的话就使用DOA
                hf_bg = cal_RTF(N,16000,option.mic_pos,option.theta(:,K+1:end));
                delta_bg = option.deltabg;
                W_hat = update_What_NP(W, W_hat, C, hf_bg, delta_bg, fac_a); % Algorithm 1 of [1]
            else
                W_hat = update_What_NP(W, W_hat, C); % Algorithm 1 of [1]
            end
        end
        
        if select
            Vxx(:,:,:,k) = cal_Vxx(R(:,:,k), XX, par_select{k}); % select on
            Vxx(:,:,:,k+1) = cal_Vxx(R(:,:,k), XX, par_select{k+1}); % select on
        else
            Vxx(:,:,:,k) = cal_Vxx(R(:,:,k), XX, partition{k}); % select off
            Vxx(:,:,:,k+1) = cal_Vxx(R(:,:,k), XX, partition{k+1}); % select off
        end
        
        ek = zeros(M,1); ek(k) = 1; 
        ek1 = zeros(M,1); ek1(k+1) = 1;
        V_L = alpha * Vxx(:,:,:,k:k+1) + (1 - alpha) * V(:,:,:,k:k+1); % online
        % 有当前源先验的话就使用DOA
        if option.prior(k) == 1
            hf = cal_RTF(N,16000,option.mic_pos,option.theta(k,:));
            hf1 = cal_RTF(N,16000,option.mic_pos,option.theta(k+1,:));
            delta_f = option.deltaf;
            [W(:,:,k:k+1), D(:,k:k+1)] = update_w_std(W_hat, V_L, [ek ek1] ,spec_indices{k}, hf, hf1, delta_f);
        else
            [W(:,:,k:k+1), D(:,k:k+1)] = update_w_std(W_hat, V_L, [ek ek1] ,spec_indices{k});
        end       
    end
    % ## 上次修改 ##
    W_hat(:,:,1:K) = W(:,:,1:K);
    
    if OrthogonalIVA==1
        sub_carr_num = size(W,1);
        for i=1:sub_carr_num
            tmp =squeeze(W(i,:,:));
            W(i,:,:) =(tmp*tmp')^(1/2)*tmp;
        end
    end
    
    if verbose % 每次迭代显示ojb_func的值
        if M == K
            W_hat = W;
        end
        obj_vals(iter+1) = obj_mm(W_hat, D, X, V, partition, alpha);
        fprintf('iter = %d (W), obj = %f\n', iter, obj_vals(iter+1));
    end
end
%% After-Processing
for k = 1:2:K
    Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
    Y(:,:,k+1) = sum(X .* permute(repmat(W(:,:,k+1),1,1,T),[1,3,2]), 3);
    
    if D_open
        R = D(:,k) .* abs(Y(:,:,k)).^PowerRatio;
        R = D(:,k+1) .* abs(Y(:,:,k+1)).^PowerRatio;
    else
        R = abs(Y(:,:,k)).^PowerRatio;
        R = abs(Y(:,:,k+1)).^PowerRatio;
    end
    if select
        Vxx(:,:,:,k) = cal_Vxx(R, XX, par_select{k}); % select on
        Vxx(:,:,:,k+1) = cal_Vxx(R, XX, par_select{k+1}); % select on
    else
        Vxx(:,:,:,k) = cal_Vxx(R, XX, partition{k}); % select off
        Vxx(:,:,:,k+1) = cal_Vxx(R, XX, partition{k+1}); % select off
    end
end

for k=1:size(V,4)
    V1(:,:,:,k) = alpha * Vxx(:,:,:,k) + (1 - alpha) * V(:,:,:,k); % online
end
V =V1;

if ~verbose
    if M == K
        W_hat = W;
    end
    obj_vals = obj_mm(W_hat, D, X, V, partition, 1);
    %         obj_vals = []; % turn off for faster sim speed
end
if M > K
    [W_res,~] = rescale(W_hat,D); % overdetermined
else
    [W_res,~] = rescale(W,D); % determined
end
for k = 1:K
    Y(:,:,k) = sum(X .* permute(repmat(W_res(:,:,k),1,1,T),[1,3,2]), 3);
end

end

function val = obj_mm(W, D, X, V, par, alpha)
% Calculate obj function
global epsilon
[N, T, M] = size(X);
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

function [WW, D] = update_w_std(W, V, ek, indices, varargin)
% 按照JISA更新模式: Algorithm 1 of [1] 编写的W更新函数
global diagonal_method; global frameNum; global frameStart; global epsilon_start_ratio;
global epsilon; global Ratio_Rxx; global epsilon_ratio;    global DOA_tik_ratio; global DOA_Null_ratio;
[N, M, ~] = size(W); WW = zeros(N, M, 2);
Vk = V(:,:,:,1); Vk1 = V(:,:,:,2);
D = ones(N, 2);
if nargin > 4
    hf = varargin{1};
    hf1 = varargin{2};
    delta_f = varargin{3};
    RTF = 1;
else
    RTF = 0;
end
if diagonal_method == 0
    if  frameNum <= frameStart
        for n = indices
            if RTF
                Q = squeeze(Vk(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (eye(M)+hf(:,n)*hf(:,n)')/delta_f^2;
                Q1 = squeeze(Vk1(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (eye(M)+hf1(:,n)*hf1(:,n)')/delta_f^2;
            else
                Q = squeeze(Vk(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon* eye(M);
                Q1 = squeeze(Vk1(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon* eye(M);
            end
            Wf = squeeze(W(n,:,:)).';
            WVk = Wf * Q;
            WVk1 = Wf * Q1;
            Pk = WVk \ ek;
            Pk1 = WVk1 \ ek;
            Vk_hat = Pk' * Q * Pk;
            Vk1_hat = Pk1' * Q1 * Pk1;
            [h, lambda] = eig(Vk1_hat \ Vk_hat, 'vector');
            [~, sort_index] = sort(lambda, 'descend');
            h = h(:,sort_index);
            w = Pk * h(:,1) / sqrt(h(:,1)' * Vk_hat * h(:,1));
            w1 = Pk1 * h(:,2) / sqrt(h(:,2)' * Vk1_hat * h(:,2));
            WW(n,:,1) = w'; WW(n,:,2) = w1';
            D(n,1) = 1 / real(w' * Q * w);
            D(n,2) = 1 / real(w1' * Q * w1);
        end        
    else        
        for n = indices
            if RTF
                Q = squeeze(Vk(n,:,:)) + epsilon_ratio*epsilon * eye(M)...
                    + (eye(M)+hf(:,n)*hf(:,n)')/delta_f^2;
                Q1 = squeeze(Vk1(n,:,:)) + epsilon_ratio*epsilon * eye(M)...
                    + (eye(M)+hf1(:,n)*hf1(:,n)')/delta_f^2;
            else
                Q = squeeze(Vk(n,:,:)) + epsilon_ratio*epsilon * eye(M);
                Q1 = squeeze(Vk1(n,:,:)) + epsilon_ratio*epsilon * eye(M);
            end
            Wf = squeeze(W(n,:,:)).';
            WVk = Wf * Q;
            WVk1 = Wf * Q1;
            Pk = WVk \ ek;
            Pk1 = WVk1 \ ek;
            Vk_hat = Pk' * Q * Pk;
            Vk1_hat = Pk1' * Q1 * Pk1;
            [h, lambda] = eig(Vk1_hat \ Vk_hat, 'vector');
            [~, sort_index] = sort(lambda, 'descend');
            h = h(:,sort_index);
            w = Pk * h(:,1) / sqrt(h(:,1)' * Vk_hat * h(:,1));
            w1 = Pk1 * h(:,2) / sqrt(h(:,2)' * Vk1_hat * h(:,2));
            WW(n,:,1) = w'; WW(n,:,2) = w1';
            D(n,1) = 1 / real(w' * Q * w);
            D(n,2) = 1 / real(w1' * Q * w1);
        end        
    end
end
end

function [W_hat,J] = update_What_P(W, W_hat, C, varargin)
% Update W_hat & J (68) of [1]
global diagonal_method;  global DOA_tik_ratio; global DOA_Null_ratio;
global epsilon; global epsilon_ratio1; global diagonal_method; global Ratio_Rxx;
[N, M, K] = size(W);
J = zeros(N, K, M-K);
E1 = [eye(K) zeros(K,M-K)]; % #source * #mic
E2 = [zeros(M-K,K) eye(M-K)]; % #mic-source *  #mic
if nargin > 3
    hf = varargin{1};
    delta_bg = varargin{2};
    fac_a = varargin{3};
    n_bg = size(hf,3);
    RTF = 1;
else
    RTF = 0;
end
if diagonal_method==0
    if RTF
        for n = 1:N
            P_bg = DOA_tik_ratio * eye(M);
            if K == 1
                Wf = squeeze(W(n,:,:)).';% Wf是[1]中wf的转置，所以wf' = conj(Wf)
            else
                Wf = squeeze(W(n,:,:));% Wf是[1]中wf的转置，所以wf' = conj(Wf)
            end
            W_hat(n,:,1:K) = W(n,:,:);
            for i = n_bg
                P_bg = P_bg + DOA_Null_ratio * hf(:,n,i)*hf(:,n,i)'; % (43) of [4]
            end
            Cf = squeeze(C(n,:,:)) + P_bg * fac_a / delta_bg^2; % (87) of [4]
            J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf) ); % (13) of [1]
            W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
        end
    else
        for n = 1:N
            if K == 1
                Wf = squeeze(W(n,:,:)).';% Wf是[1]中wf的转置，所以wf' = conj(Wf)
            else
                Wf = squeeze(W(n,:,:));% Wf是[1]中wf的转置，所以wf' = conj(Wf)
            end
            W_hat(n,:,1:K) = W(n,:,:);
            Cf = squeeze(C(n,:,:));% + epsilon_ratio1*epsilon * eye(M);
            J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf) ); % (13) of [1]
            %         J(n,:,:) = (( Wf.' * Cf * E1.' ) \ ( Wf.' * Cf * E2.' ))'; % (17) of [3]
            W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
        end
    end
end
if diagonal_method==1
    for n = 1:N
        Wf = squeeze(W(n,:,:));
        W_hat(n,:,1:K) = W(n,:,:);
        Cf = squeeze(C(n,:,:)) + eye(M)*mean(diag(squeeze(C(n,:,:))))*(eps+Ratio_Rxx);
        J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf));
        W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
    end
end
if diagonal_method==2
    for n = 1:N
        Wf = squeeze(W(n,:,:));
        W_hat(n,:,1:K) = W(n,:,:);
        Cf = squeeze(C(n,:,:)) + diag(diag(squeeze(C(n,:,:))))*(eps+Ratio_Rxx);
        J(n,:,:) = ( E2 * Cf * conj(Wf) ) / ( E1 * Cf * conj(Wf));
        W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
    end
end
end

function W_hat = update_What_NP(W, W_hat, C, varargin)
% Update W_hat & J Alogrithm 1 of [1]
global diagonal_method;  global DOA_tik_ratio; global DOA_Null_ratio;
global epsilon; global epsilon_ratio1; global diagonal_method; global Ratio_Rxx;
[N, M, K] = size(W);
E = [zeros(K,M-K);eye(M-K)]; % #mic * #src-mic
W_hat(:,:,1:K) = W;
if nargin > 3
    hf = varargin{1};
    delta_bg = varargin{2};
    fac_a = varargin{3};
    n_bg = size(hf,3);
    RTF = 1;
else
    RTF = 0;
end
if diagonal_method==0
    if RTF
        for n = 1:N
            P_bg = DOA_tik_ratio * eye(M);
            Wf = squeeze(W_hat(n,:,:)).';
            for i = n_bg
                P_bg = P_bg + DOA_Null_ratio * hf(:,n,i)*hf(:,n,i)'; % (43) of [4]
            end
            Cf = squeeze(C(n,:,:)) + P_bg * fac_a / delta_bg^2; % (87) of [4]
            Uf = (Wf * Cf) \ E; % BG update from Algorithm 1 of [1]
            for k = 1:M-K
                u_kf = Uf(:,k);
                u_kf = u_kf / sqrt(u_kf' * Cf * u_kf);
                Uf(:,k) = u_kf;
            end
            W_hat(n,:,(K+1):M) = conj(Uf);
        end
    else
        for n = 1:N
            Wf = squeeze(W_hat(n,:,:)).';
            Cf = squeeze(C(n,:,:));% + epsilon_ratio1*epsilon * eye(M);
            Uf = (Wf * Cf) \ E; % BG update from Algorithm 1 of [1]
            for k = 1:M-K
                u_kf = Uf(:,k);
                u_kf = u_kf / sqrt(u_kf' * Cf * u_kf);
                Uf(:,k) = u_kf;
            end
            W_hat(n,:,(K+1):M) = conj(Uf);
        end
    end
end
if diagonal_method==1
    for n = 1:N
        Wf = squeeze(W_hat(n,:,:)).';
        Cf = squeeze(C(n,:,:)) + eye(M)*mean(diag(squeeze(C(n,:,:))))*(eps+Ratio_Rxx);
        Uf = (Wf * Cf) \ E; % BG update from Algorithm 1 of [1]
        for k = 1:M-K
            u_kf = Uf(:,k);
            u_kf = u_kf / sqrt(u_kf' * Cf * u_kf);
            Uf(:,k) = u_kf;
        end
        W_hat(n,:,(K+1):M) = conj(Uf);
    end
end
if diagonal_method==2
    for n = 1:N
        Wf = squeeze(W_hat(n,:,:)).';
        Cf = squeeze(C(n,:,:)) + diag(diag(squeeze(C(n,:,:))))*(eps+Ratio_Rxx);
        Uf = (Wf * Cf) \ E; % BG update from Algorithm 1 of [1]
        for k = 1:M-K
            u_kf = Uf(:,k);
            u_kf = u_kf / sqrt(u_kf' * Cf * u_kf);
            Uf(:,k) = u_kf;
        end
        W_hat(n,:,(K+1):M) = conj(Uf);
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

function Cxx = cal_Cxx(X) % 旧版Cxx计算方法，暂时保留，(9) of [1]
[N, T, M] = size(X);
Cxx = zeros(N, M, M);
for n = 1:N
    Xf = squeeze(X(n,:,:)); % M x 1
    if T == 1
        % 当T=1时，由于squeeze的特性，Xf与[3]中xf相等，直接用原公式
        Cf = Xf * Xf';
    else
        % 本Xf为[1]中xf的转置，因此 xf*xf' = Xf.' * conj(Xf)
        Cf = Xf.' * conj(Xf) / T; 
    end
    Cxx(n,:,:) = Cf;
end
end