function [Y, W, D, V, C, W_hat, W_res, obj_vals] =  binaural_auxiva_update_multi(X, W, D, V, C, W_hat, partition, varargin)
% AuxIVA for online and offline BSS
% References:
%   [1] INDEPENDENT VECTOR ANALYSIS WITH MORE MICROPHONES THAN SOURCES
%   [2] AN AUXILIARY-FUNCTION APPROACH TO ONLINE INDEPENDENT VECTOR ANALYSIS FOR REAL-TIME BLIND SOURCE SEPARATION
%   [3] Overdetermined independent vector analysis
%   [4] A Unified Bayesian View on Spatially Informed Source Separation and Extraction based on Independent Vector Analysis

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
global Ratio_Rxx; global PowerRatio;  global GammaRatio; global OrthogonalIVA;

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
theta = option.theta; % doa
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
%     % M>2时Xn * Xn'的遍历计算，size(XX) = N * T * M^2, 目前复杂度最高，待优化
%     if diagonal_method == 0
%         % 0: using a tiny portion of the mean diagnal values of Xnn；
%         % 1: using a tiny portion of the diagnal value of Xnn；
%         for n = 1:N
%             for t = 1:T
%                 Xn = squeeze(X(n,t,:));
%                 Xnn = Xn * Xn';
%                 Xnn = Xnn + eye(M)*mean(diag(Xnn))*Ratio_Rxx; % diagonal loading factor;
%                 XX(n,t,:) = Xnn(:);
%             end
%         end
%     end
%
%     if diagonal_method == 1
%         for n = 1:N  % Xn * Xn'的计算，size(XX) = N * T * M^2, 目前复杂度最高，待优化
%             for t = 1:T
%                 Xn = squeeze(X(n,t,:));
%                 Xnn = Xn * Xn';
%                 Xnn = Xnn + eye(M)*diag(diag(Xnn))*Ratio_Rxx; % diagonal loading factor;
%                 XX(n,t,:) = Xnn(:);
%             end
%         end
%     end
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

%% Main Iteration
for iter = 1:iter_num
    for k = 1:K
%                     for n = 1:N
%                         if T == 1
%                             Y(n,:,k) = W(n,:,k) * (squeeze(X(n,:,:)));
%                         else
%                             Y(n,:,k) = W(n,:,k) * (squeeze(X(n,:,:))).';
%                         end
%                     end
        % faster calculation of Y
        Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
        if D_open
            R = D(:,k) .* abs(Y(:,:,k)).^PowerRatio;
        else
            R = abs(Y(:,:,k)).^PowerRatio;
        end
        if select
            Vxx(:,:,:,k) = cal_Vxx(R, XX, par_select{k}); % select on
        else
            Vxx(:,:,:,k) = cal_Vxx(R, XX, partition{k}); % select off
        end
    end
    for inner_iter = 1:inner_iter_num

        for k = 1:K % 程序原始的更新代码（采用零空间方法）,与标准公式结果相同
            ind = 1:M;
            ind(k) = [];
            if M == K % determined
                for n = 1:N
                    WW = squeeze(W(n,:,:));
                    H(n,:) = -null(WW(:,ind).')';
                end
            else % overdetermined
                for n = 1:N
                    WW = squeeze(W_hat(n,:,:));
                    H(n,:) = -null(WW(:,ind).')';
                end
            end
            if option.annealing % 是否使用退火因子
                fac_a = max(0.5-iter/iter_num, 0); % annealing factor
            else
                fac_a = 1; % =1即等效于不退火
            end
            Vk = alpha * GammaRatio(k) * Vxx(:,:,:,k) + (1 - alpha * GammaRatio(k)) * V(:,:,:,k); % online
            % 有当前源先验的话就使用DOA
            if option.prior(k) == 1
                hf = cal_RTF(N,16000,option.mic_pos,theta(:,k));
                delta_f = option.deltaf;
                [W(:,:,k), D(:,k)] = update_w(W(:,:,k), Vk, H, spec_indices{k}, hf, delta_f, fac_a);
            else
                [W(:,:,k), D(:,k)] = update_w(W(:,:,k), Vk, H, spec_indices{k});
            end
            if M > K % overdetermined update W_hat & J
                if option.prior(K+1) == 1
                    hf_bg = cal_RTF(N,16000,option.mic_pos,theta(:,K+1:end));
                    delta_bg = option.deltabg;
                    [W_hat] = update_What(W, W_hat, C, hf_bg, delta_bg, fac_a); % (87) of [4]
                else
                    [W_hat] = update_What(W, W_hat, C); % (13) of [1]
                end
            end
        end

%         for k = 1:K % 按照标准公式编写的更新代码 (12) of [1]
%             ek = zeros(M,1); ek(k) = 1;
%             Vk = alpha * GammaRatio(k) * Vxx(:,:,:,k) + (1 - alpha * GammaRatio(k)) * V(:,:,:,k); % online
%             % 有当前源先验的话就使用DOA
%             if option.prior(k) == 1
%                 hf = cal_RTF(N,16000,option.mic_pos,theta(k,:));
%                 delta_f = option.deltaf;
%                 [W(:,:,k), D(:,k)] = update_w_std(W, Vk, ek,spec_indices{k}, hf, delta_f);
%             else
%                 [W(:,:,k), D(:,k)] = update_w_std(W, Vk, ek,spec_indices{k});
%             end
%             if M > K % overdetermined update W_hat & J
%                 [W_hat] = update_What(W, W_hat, C); % (13) of [1]
%             end
%         end

        if OrthogonalIVA==1
            sub_carr_num = size(W,1);
            for i=1:sub_carr_num
                tmp =squeeze(W(i,:,:));
                W(i,:,:) =(tmp*tmp')^(1/2)*tmp;
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
    if option.DOA_update % 是否在迭代过程中更新DOA
        theta_tmp = doa_update(W,option.esti_mic_dist,16000);
        theta = 0.96 * theta + 0.04 * theta_tmp;
    end
end
%% After-Processing
for k = 1:K
    %         for n = 1:N
    %             if T == 1
    %                 Y(n,:,k) = W(n,:,k) * (squeeze(X(n,:,:)));
    %             else
    %                 Y(n,:,k) = W(n,:,k) * (squeeze(X(n,:,:))).';
    %             end
    %         end
    % faster calculation of Y
    Y(:,:,k) = sum(X .* permute(repmat(W(:,:,k),1,1,T),[1,3,2]), 3);
    if D_open
        R = D(:,k) .* abs(Y(:,:,k)).^PowerRatio;
    else
        R = abs(Y(:,:,k)).^PowerRatio;
    end
    if select
        Vxx(:,:,:,k) = cal_Vxx(R, XX, par_select{k});
    else
        Vxx(:,:,:,k) = cal_Vxx(R, XX, partition{k});
    end
end

for k=1:size(V,4)
    V1(:,:,:,k) = alpha * GammaRatio(k) * Vxx(:,:,:,k) + (1 - alpha*GammaRatio(k)) * V(:,:,:,k); % online
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
    %         for n = 1:N
    %             if T == 1
    %                 Y(n,:,k) = W_res(n,:,k) * (squeeze(X(n,:,:)));
    %             else
    %                 Y(n,:,k) = W_res(n,:,k) * (squeeze(X(n,:,:))).';
    %             end
    %         end
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
                    + (eye(M)+hf(:,n)*hf(:,n)') * fac_a/delta_f^2;
            else
                Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon* eye(M);
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
                    P_t = P_t + DOA_Null_ratio * fac_a * hf(:,n,i)*hf(:,n,i)'; % (43) of [4]
                end
%                 Q = squeeze(V(n,:,:)) + epsilon_ratio*epsilon * eye(M)...
%                     + (eye(M)+hf(:,n)*hf(:,n)') * fac_a/delta_f^2;
                Q = squeeze(V(n,:,:)) + P_t / delta_f^2; % (79) of [4] 
            else
                Q = squeeze(V(n,:,:)) + epsilon_ratio*epsilon * eye(M);
            end
            %  Q = squeeze(V(n,:,:)) + diag([epsilon1 epsilon2  ])
            h = H(n,:)';
            w = Q \ h; % sum(abs(Q \ h-inv(Q)*h))% (78) of [4]
            % w = w / norm(w);
            w = w / sqrt(w' * Q * w); % real w
            W(n,:) = w'; % W = [w]'
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

function [WW, D] = update_w_std(W, V, ek, indices, varargin)
% 按照标准公式 (12) of [1] 编写的W更新函数
global diagonal_method; global frameNum; global frameStart; global epsilon_start_ratio;
global epsilon; global Ratio_Rxx; global epsilon_ratio;    global DOA_tik_ratio; global DOA_Null_ratio;
[N, M, ~] = size(W); WW = zeros(N, M);
D = ones(N, 1);
if nargin > 4
    hf = varargin{1};
    delta_f = varargin{2};
    RTF = 1;
else
    RTF = 0;
end
if diagonal_method == 0
    if  frameNum <= frameStart
        for n = indices
            if RTF
                Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (eye(M)+hf(:,n)*hf(:,n)')/delta_f^2;
            else
                Q = squeeze(V(n,:,:)) + epsilon_start_ratio*epsilon_ratio*epsilon* eye(M);
            end
            Wf = squeeze(W(n,:,:)).';
            WV = Wf * Q;
            w = WV \ ek;
            w = w / sqrt(w' * Q * w);
            WW(n,:) = w';
            D(n,1) = 1 / real(w' * Q * w);
        end        
    else        
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
end
end

function [W_hat,J] = update_What(W, W_hat, C, varargin)
% Update W_hat & J (12)(13) of [1]
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

%   G_func = {@(x) w1(1)*(x + delta).^n_orders(1)+w1(2)*(x+delta).^n_orders(2),@(x) w2(1)*(x+delta).^n_orders(1)+w2(2)*(x+delta).^n_orders(2)};
%     dG_func = {@(x) w1(1)*n_orders(1)*(x+delta+eps).^(n_orders(1) - 1)+w1(2)*n_orders(2)*(x+delta+eps).^(n_orders(2) - 1), ...
%        @(x) w2(1)*n_orders(1)*(x+delta+eps).^(n_orders(1)-1)+w2(2)*n_orders(2)*(x+delta+eps).^(n_orders(2)-1)};
% dd=sum(R(indices,:), 1);
%  OrderParam.w1(1)* OrderParam.n_orders1*(dd).^(OrderParam.n_orders1 - 1)+...
%  +OrderParam.w1(2)*OrderParam.n_orders2*(dd).^(OrderParam.n_orders2 - 1)
%     dG_func = {@(x) w1(1)*n_orders(1)*(x+delta+eps).^(n_orders(1) - 1)+w1(2)*n_orders(2)*(x+delta+eps).^(n_orders(2) - 1), ...
%        @(x) w2(1)*n_orders(1)*(x+delta+eps).^(n_orders(1)-1)+w2(2)*n_orders(2)*(x+delta+eps).^(n_orders(2)-1)};

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