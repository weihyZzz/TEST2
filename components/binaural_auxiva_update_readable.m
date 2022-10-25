function [Y, W, D, V, C, W_hat, W_res, obj_vals] =  binaural_auxiva_update_readable(X, W, D, V, C, W_hat, partition, varargin)
%% 仅保留核心算法功能的AuxIVA（OverIVA）代码
% AuxIVA for online and offline BSS
% References:
%   [1] INDEPENDENT VECTOR ANALYSIS WITH MORE MICROPHONES THAN SOURCES
%   [2] AN AUXILIARY-FUNCTION APPROACH TO ONLINE INDEPENDENT VECTOR ANALYSIS FOR REAL-TIME BLIND SOURCE SEPARATION
% Default batch Aux-IVA parameter
%   [3] Overdetermined independent vector analysis
option.iter_num = 10; option.inner_iter_num = 2; option.verbose = true; option.select = 1;
option.parti = 0; option.thFactor = 1/50; option.online = 0;
if nargin > 7
    user_option = varargin{1};
    for fn = fieldnames(user_option)'
        option.(fn{1}) = user_option.(fn{1});
    end
end
%% Initalization

iter_num = option.iter_num; % 外迭代
inner_iter_num = option.inner_iter_num; % 内迭代
verbose = option.verbose; % 是否观察收敛
alpha = 1; % V滑动平均因子；offline = 1，online < 1
gamma = 1; % C滑动平均因子；offline = 1，online < 1
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

% M>2时Xn * Xn'的快速计算方式，size(XX) = N * T * M^2
XXn = reshape(X, [N, T, M, 1]) .* conj(reshape(X, [N, T, 1, M]));    XX = reshape(XXn,[N, T, M^2]);

if M > K  % overdetermined
    Cxx = cal_Cxx(X);
    C = gamma * Cxx + (1 - gamma) * C; % online
    W = W_hat(:,1:M,1:K);
end

[spec_indices,par_select] = selectpar(partition, 0, 0, XX, K, N, 0); % select & partition initalize

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
if M > K % overdetermined update W_hat & J
    [W_hat] = update_What(W, W_hat, C); % (13) of [1]
end
%% Main Iteration
for iter = 1:iter_num
    for k = 1:K
        % faster calculation of Y
        Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
        if D_open
            R = D(:,k) .* abs(Y(:,:,k)).^2;
        else
            R = abs(Y(:,:,k)).^2;
        end
        Vxx(:,:,:,k) = cal_Vxx(R, XX, partition{k}); % select off
    end
    for inner_iter = 1:inner_iter_num
        for k = 1:K % 按照标准公式编写的更新代码 (12) of [1]
            ek = zeros(M,1); ek(k) = 1;
            Vk = alpha * Vxx(:,:,:,k) + (1 - alpha) * V(:,:,:,k); % online
            if M == K
                W_hat = W;
            end
            % 有当前源先验的话就使用DOA
            if option.prior(k) == 1
                hf = cal_RTF(N,16000,option.mic_pos,option.theta(k,:));
                delta_f = option.deltaf;
                [W(:,:,k), D(:,k)] = update_w_std(W_hat, Vk, ek,spec_indices{k}, hf, delta_f);
            else
                [W(:,:,k), D(:,k)] = update_w_std(W_hat, Vk, ek,spec_indices{k});
            end
            if M > K % overdetermined update W_hat & J
                [W_hat] = update_What(W, W_hat, C); % (13) of [1]
            end
        end
    end
    if verbose
        if M == K
            W_hat = W;
        end
        obj_vals(iter+1) = obj_mm(W_hat, D, X, V, partition, alpha);
        fprintf('iter = %d (W), obj = %f\n', iter, obj_vals(iter+1));
    end
end
%% After-Processing
for k = 1:K
    % faster calculation of Y
    Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
    if D_open
        R = D(:,k) .* abs(Y(:,:,k)).^2;
    else
        R = abs(Y(:,:,k)).^2;
    end
    Vxx(:,:,:,k) = cal_Vxx(R, XX, partition{k});
end
for k=1:size(V,4)
    V(:,:,:,k) = alpha * Vxx(:,:,:,k) + (1 - alpha) * V(:,:,:,k); % online;
end


if ~verbose
    if M == K
        W_hat = W;
    end
    obj_vals = obj_mm(W_hat, D, X, V, partition, 1);
end

if M > K
    [W_res,~] = rescale(W_hat,D); % overdetermined
else
    [W_res,~] = rescale(W,D); % determined
end

for k = 1:K
    % faster calculation of Y
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
% 按照标准公式 (12) of [1] 编写的W更新函数
global epsilon; global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;
[N, M, ~] = size(W); WW = zeros(N, M);
D = ones(N, 1);
if nargin > 4
    hf = varargin{1};
    delta_f = varargin{2};
    RTF = 1;
else
    RTF = 0;
end
   
for n = indices
    if RTF
        Q = squeeze(V(n,:,:)) + epsilon_ratio*epsilon * eye(M)...
            + (eye(M)+hf(:,n)*hf(:,n)')/delta_f^2;
    else
        Q = squeeze(V(n,:,:)) + epsilon_ratio*epsilon * eye(M);
    end
    Wf = squeeze(W(n,:,:)).';
    WV = Wf * Q;
    w = WV \ ek;
    w = w / sqrt(w' * Q * w);
    WW(n,:) = w';
    D(n,1) = 1 / real(w' * Q * w);
end

end

function [W_hat,J] = update_What(W, W_hat, C)
% Update W_hat & J (12)(13) of [1]
[N, M, K] = size(W);
J = zeros(N, K, M-K);
E1 = [eye(K) zeros(K,M-K)]; % #source * #mic
E2 = [zeros(M-K,K) eye(M-K)]; % #mic-source *  #mic
for n = 1:N
    if K == 1
        Wf = squeeze(W(n,:,:)).';% Wf是[1]中wf的转置，所以wf' = conj(Wf)
    else
        Wf = squeeze(W(n,:,:));% Wf是[1]中wf的转置，所以wf' = conj(Wf)
    end
    W_hat(n,:,1:K) = W(n,:,:);
    Cf = squeeze(C(n,:,:));% + epsilon_ratio1*epsilon * eye(M);
%     J(n,:,:) = ( E2 * Cf * Wf ) / ( E1 * Cf * Wf ); % (13) of [1]
    J(n,:,:) = (( Wf.' * Cf * E1.' ) \ ( Wf.' * Cf * E2.' ))'; % (17) of [3]
    W_hat(n,1:K,(K+1):M) = permute(J(n,:,:),[1,3,2]);
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