function [Y, W, D1, D2, obj_vals] = binaural_nmfiva_batch_update(X, W, Gr, Fr, varargin)
% NMF IVA
% Augumented IVA with NMF method for TWO sources with demxing matrix rescaling based on the minimum distoriton principle
%%
% [1] Fast AuxIVA for binaural audio separation: N. Ono, "Fast Stereo Independent Vector Analysis and its
% Implementation on Mobile Phone," in Proc. International Workshop on Acoustic % Signal Enhancement (IWAENC), Aachen, Germany,
% 2012, pp. 1-4.
%
% Minimum distortion principle:  K. Matsuoka and S. Nakashima, "Minimal distortion principle for
% blind source separation," in Proc. Int. Conf. Independent Compon. Anal. Blind Source Separation, 2001, pp. 722--727.

option.forgetting_fac = .04;
option.iter_num = 2;
option.verbose = false; % what is verbose?
option.nmf_iter_num = 1;
option.nmf_fac_num = 1;
option.nmf_beta = 0;

if nargin > 4
    user_option = varargin{1};
    for fn = fieldnames(user_option)'
        option.(fn{1}) = user_option.(fn{1});
    end
end
%%
epsilon = eps;
alpha = option.forgetting_fac;
iter_num = option.iter_num;
verbose = option.verbose;
nmf_fac_num = option.nmf_fac_num; % num of bases
nmf_option.iter_num = option.nmf_iter_num;
nmf_option.beta = option.nmf_beta;
nmf_option.p = option.nmf_p;
nmf_option.b = option.nmf_b;
nmf_option.nmfupdate = option.nmfupdate;
%     global seed; seed = rng(20);

[N, T, M] = size(X); % #freq *  #frame * #mic
[~, ~, K] = size(W); % #freq *  #mic * #source

W1_f = rand(N, nmf_fac_num); % Basis matrix
H1_f = rand(nmf_fac_num, T); % Activation matrix
D1 = 1 ./ (W1_f * H1_f + epsilon); % Power spectrogram

W2_f = rand(N, nmf_fac_num);
H2_f = rand(nmf_fac_num, T);
D2 = 1 ./ (W2_f * H2_f + epsilon);

Y(:,:,1) = X(:,:,1) .* W(:,1,1) + X(:,:,2) .* W(:,2,1);
Y(:,:,2) = X(:,:,1) .* W(:,1,2) + X(:,:,2) .* W(:,2,2);

R1 = sqrt(sum(D1 .* abs(Y(:,:,1)).^2))'; % Ref.1 (4)
R2 = sqrt(sum(D2 .* abs(Y(:,:,2)).^2))'; % Ref.1 (4)

if verbose
    obj_vals = zeros(iter_num + 1, 1);
    obj_vals(1) = obj(W, D1, D2, W, D1, D2, X, Gr, Fr); % 计算目标函数
end

for iter = 1:iter_num    
    W_old = W;
    D1_old = D1;
    D2_old = D2;    
    for n = 1:N        
        V1 = [abs(X(n,:,1)).^2 .* D1(n,:); ...
            X(n,:,2) .* conj(X(n,:,1)) .* D1(n,:); ...
            X(n,:,1) .* conj(X(n,:,2)) .* D1(n,:);...
            abs(X(n,:,2)).^2 .* D1(n,:)] * Fr{1}(R1); % Ref.1 (5)
        V1 = reshape(V1, 2, 2);
        
        V2 = [abs(X(n,:,1)).^2 .* D2(n,:); ...
            X(n,:,2) .* conj(X(n,:,1)) .* D2(n,:); ...
            X(n,:,1) .* conj(X(n,:,2)) .* D2(n,:);...
            abs(X(n,:,2)).^2 .* D2(n,:)] * Fr{2}(R2); % Ref.1 (5)
        V2 = reshape(V2, 2, 2);
        E = auxfun_min_sol(V1 / T, V2 / T);
        W(n,:,:) = conj(E);
    end
    
    Y(:,:,1) = X(:,:,1) .* W(:,1,1) + X(:,:,2) .* W(:,2,1);
    Y(:,:,2) = X(:,:,1) .* W(:,1,2) + X(:,:,2) .* W(:,2,2);
    
    switch option.nmfupdate
        case 0 % IS-NMF
            W1_f_old = W1_f;
            H1_f_old = H1_f;
            [W1_f, H1_f] = nmf(abs(Y(:,:,1)).^2 .* Fr{1}(R1)', W1_f, H1_f, nmf_option);
            W2_f_old = W2_f;
            H2_f_old = H2_f;
            [W2_f, H2_f] = nmf(abs(Y(:,:,2)).^2 .* Fr{2}(R2)', W2_f, H2_f, nmf_option);
        case 1 % GGD-NMF
            beta = option.nmf_beta;
            W1_f_old = W1_f;
            H1_f_old = H1_f;
            [W1_f, H1_f] = nmf(abs(Y(:,:,1)).^beta .* Fr{1}(R1)', W1_f, H1_f, nmf_option);
            W2_f_old = W2_f;
            H2_f_old = H2_f;
            [W2_f, H2_f] = nmf(abs(Y(:,:,2)).^beta .* Fr{2}(R2)', ...
                W2_f, H2_f, nmf_option);
    end
    if any(isnan(W1_f))
        warning('Nan');
    end
    D1 = 1 ./ (W1_f * H1_f + epsilon);
    D2 = 1 ./ (W2_f * H2_f + epsilon);
    R1 = sqrt(sum(D1 .* abs(Y(:,:,1)).^2))';
    R2 = sqrt(sum(D2 .* abs(Y(:,:,2)).^2))';
    if verbose
        obj_vals(iter+1) = obj(W, D1, D2,W_old, D1_old, D2_old, X, Gr, Fr);
        fprintf('%d, obj = %.4f\n', iter, obj_vals(iter+1));
    end
end

if ~verbose
    obj_vals = obj(W, D1, D2, W_old, D1_old, D2_old, X, Gr, Fr);
end

end

function E = auxfun_min_sol(V1, V2)
epsilon = eps;
H = (V1 + epsilon * eye(2)) \ V2;  % H = V1^(-1)*V2
lambda1 = (trace(H) + sqrt(trace(H)^2 - 4 * det(H))) / 2; % Ref.1 (16)
lambda2 = (trace(H) - sqrt(trace(H)^2 - 4 * det(H))) / 2; % Ref.1 (17)
E = [H(2,2) - real(lambda1) -H(1,2); ... % Ref.1 (18)
    -H(2,1) H(1,1) - real(lambda2)];
E = E * diag(diag(E \ eye(2))); % E * diag of E^(-1), what does it mean?  inv(E)-(E\eye(2))
% W = E';

end

function val = obj(W, D1, D2, W_hat, D1_hat, D2_hat, X, Gr, Fr)
epsilon = eps;
T = size(X, 2);
Y_hat(:,:,1) = X(:,:,1) .* W_hat(:,1,1) ...
    + X(:,:,2) .* W_hat(:,2,1);
Y_hat(:,:,2) = X(:,:,1) .* W_hat(:,1,2) ...
    + X(:,:,2) .* W_hat(:,2,2);

R1_hat = sqrt(sum(D1_hat .* abs(Y_hat(:,:,1)).^2))';
R2_hat = sqrt(sum(D2_hat .* abs(Y_hat(:,:,2)).^2))';

Y(:,:,1) = X(:,:,1) .* W(:,1,1) + X(:,:,2) .* W(:,2,1);
Y(:,:,2) = X(:,:,1) .* W(:,1,2) + X(:,:,2) .* W(:,2,2);

val = sum(Gr{1}(R1_hat)) + sum(Gr{2}(R2_hat)) ...
    + sum((abs(Y(:,:,1)).^2 .* D1) ...
    * Fr{1}(R1_hat)) / 2 ...
    - sum((abs(Y_hat(:,:,1)).^2 .* D1_hat) ...
    * Fr{2}(R1_hat)) / 2 ...
    + sum((abs(Y(:,:,2)).^2 .* D2) ...
    * Fr{2}(R2_hat)) / 2 ...
    - sum((abs(Y_hat(:,:,2)).^2 .* D2_hat) ...
    * Fr{2}(R2_hat)) / 2 ...
    - sum(sum(log(D1 + epsilon))) / 2 ...
    - sum(sum(log(D2 + epsilon))) / 2 ...
    - T * sum(log(abs(W(:,1,1) .* W(:,2,2) ...
    - W(:,1,2) .* W(:,2,1))));

end
