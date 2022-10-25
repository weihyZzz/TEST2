function [W, H, varargout] = nmf_online(V, W0, H0, varargin)
% NMF IVA
% Augumented IVA with NMF method for TWO sources with
% demxing matrix rescaling based on the minimum distoriton 
% principle
%
% Reference
%
% Nonnegative Matrix Factorization:
% [1] D. Kitamura, "Determined Blind Source Separation Unifying Independent 
%     Vector Analysis and Nonnegative Matrix Factorization", 
%     IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2016
% [2] "Generalized independent low-rank matrix
%     analysis using heavy-tailed distributions for
%     blind source separation"
% [3] "ONLINE ALGORITHMS FOR NONNEGATIVE MATRIX FACTORIZATION
%     WITH THE ITAKURA-SAITO DIVERGENCE"

[M, N] = size(V);
R = size(W0, 2);

option.beta = 1;
option.iter_num = 1;
option.verbose = false;

if nargin > 3
    user_option = varargin{1};
    for fn = fieldnames(user_option)'
        option.(fn{1}) = user_option.(fn{1});
    end
end

beta = option.beta;
p = option.p;
b = option.b;
nmfupdate = option.nmfupdate;
iter_num = option.iter_num;
verbose = option.verbose;

W = W0;
H = H0;
Lambda = W * H + eps;
div_vals = zeros(iter_num, 1);
alpha = 0.04^(1/N);
for iter = 1:iter_num
    % 计算目标函数
    div_vals(iter) = betaDiv(V + eps, Lambda + eps, beta / p);
    
    switch nmfupdate 
        case 0 % IS-NMF
            % update of W
            wf = W .* (((Lambda.^(-2) .* V) * H' + eps) ...
                ./ ((Lambda.^(-1)) * H' + eps)).^b; % Ref.1 (29) & Ref.2 (58)
            W = alpha * W + (1-alpha) * wf;
            Lambda = W * H + eps;
            
            % update of H
            hf = H .* ((W'*(Lambda.^(-2) .* V) + eps) ...
                ./(W' * (Lambda.^(-1)) + eps)).^b; % Ref.1 (30) & Ref.2 (59)
            H = alpha * H + (1-alpha) * hf;
            Lambda = W * H + eps;
            
        case 1 % GGD-NMF
            Z = beta/2 .* V .* Lambda.^(1 - beta/p);
            W = W .* (((Lambda.^(-2) .* Z) * H' + eps) ...
                ./ ((Lambda.^(-1)) * H' + eps)).^(p/(beta+p)); % Ref.1 (29) & Ref.2 (60)(61)
            Lambda = W * H + eps;
            
            % update of H
            Z = beta/2 .* V .* Lambda.^(1 - beta/p);
            H = H .* ((W'*(Lambda.^(-2) .* Z) + eps) ...
                ./(W' * (Lambda.^(-1)) + eps)).^(p/(beta+p)); % Ref.1 (30)            
            Lambda = W * H + eps;
    end

    if verbose
        msg = 'iter = %d/%d, div val = %f\n';    
        fprintf(msg, iter, iter_num, div_vals(iter));
    end

end

if nargout > 2
    varargout{1} = div_vals;
end

function val = betaDiv(V1, V2, beta)
beta = beta - 1;
if beta == 0
    val = sum((V1(:) ./ V2(:)) - log(V1(:) ./ V2(:)) - 1);
elseif beta == 1
    val = sum(V1(:) .* (log(V1(:)) - log(V2(:))) + V2(:) - V1(:));
else
    val = sum(max(1 / (beta * (beta - 1)) ...
                  * (V1(:).^beta + (beta - 1) * V2(:).^beta ...
                    - beta * V1(:) .* V2(:).^(beta-1)), 0));
end




