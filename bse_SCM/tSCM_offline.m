function [rt,rn,Rx,Rn] = tSCM_offline(X,maxIt,Af,rt,rn,Rn_hat,alpha,beta,rou,v)

%% # Original paper
% [1] Blind Speech Extraction Based on Rank-Constrained Spatial Covariance Matrix Estimation With Multivariate Generalized Gaussian Distribution
%
% [syntax]
%   [rt,rn,Rx] =  SCM_offline(X,XX,ns,MNMF_nb,MNMF_it,MNMF_drawConv,Af,rt,rn,Rn_hat);
% [inputs]
%         X: the observed signal (M x F x T )
%        ns: number of sources
%         K: number of bases (default: ceil(T/10))
%     maxIt: number of iteration (default: 50)
%        Af: mixing matrix (M x N x F x T)
%        rt: initial variances of the directional target speech (F x T)
%        rn: variances of the diffuse noise (F x T)
%    Rn_hat: initial the full-rank SCM of diffuse noise (M x M x F)
%     alpha: the shape parameter of the inverse gamma distribution
%      beta: the scale parameter of the inverse gamma distribution
%       rou: the shape parameter ρ of the multivariate GGD
% [outputs]
%        rt: the variances of the directional target speech (F x T)
%        rn: the variances of the diffuse noise (F x T)
%        Rx: the sum of the SCMs of the directional target speech and diffuse noise (M x M x F x T)
% Check errors and set default values
%% initialization parameters
[M,F,T] = size(X);
N = M;
delta = 0.001; % to avoid numerical conputational instability
if (nargin < 2)
    maxIt = 300;
end
% q = min(0.5,2/(rou+2));
lamda = 0.9 * ones(F,1);
b= zeros(M,F);
for ff = 1:F
    b(:,ff) = Af(:,1,ff);% 第一个是源
end
    Rn = local_Rn(F, M, Af, lamda, Rn_hat);% (14) of [1]
    Rx = local_Rx(F, T, M, N, Af, rt, rn, Rn);% (12) of [1]
%% main iterate
for it = 1:maxIt
    fprintf('\b\b\b\b%4d', it);
    alpha = local_alpha(E, T, M, v, X, Rx);% F * T 
    c_FT = local_c_FT(F,T,X,Rx,rou);% under three lines of (68) in [1]    
    %%%%% Update rt %%%%%% F * T      
    [rt_nume,rt_deno] = local_rt(X,F,T,Af,rt,Rx,alpha,c_FT,beta);% (66) of [1]
    rt = rt .* (rt_nume./rt_deno).^1/2;% (66) of [1]
    %%%%% Update rn %%%%%% F * T
    [rn_nume,rn_deno] = local_rn(X,F,T,Rx,Rn,alpha,c_FT);% (67) of [1]
    rn = rn .* (rn_nume./rn_deno).^1/2;% (67) of [1]
    %%%%% Update lamda %%%%%% sclar weight 
    [lamda_nume,lamda_deno] = local_lamda(X,F,T,Rx,c_FT,b,rn);% (68) of [1]
    lamda = lamda .* (lamda_nume./lamda_deno).^1/2;    % (68) of [1]
    %%%%% Update Rn Rx %%%%%% 
    Rn = local_Rn(F,M,Af,lamda,Rn_hat);% (14) of [1]
    Rx = local_Rx(F,T,M,N,Af,rt,rn,Rn);% (12) of [1]
end
fprintf(' SCM done.\n');
end
function [Rn] = local_Rn(F,M,Af,lamda,Rn_hat)
Rn = zeros(M,M,F);lamdabb= zeros(M,M,F);n=1;
for f = 1:F
    lamdabb(:,:,f) = lamda(f)*Af(:,n,f)*Af(:,n,f)';% second term in (14) of [1]
end
Rn = Rn_hat + lamdabb;% (14) of [1]
end
function [Rx] = local_Rx(F,T,M,N,Af,rt,rn,Rn)
AA = zeros(M,M,F);
for f = 1:F
    for n = 1:N
            AA(:,:,f) = Af(:,n,f)*Af(:,n,f)';
    end
end
Rx_part1 = zeros(M,M,F,T);Rx_part2 = zeros(M,M,F,T);
for f = 1:F
    for t = 1:T
        Rx_part1(:,:,f,t) = rt(f,t) * AA(:,:,f);% first term in (12) of [1]
        Rx_part2(:,:,f,t) = rn(f,t) * Rn(:,:,f);% second term in (12) of [1]
    end
end
Rx = Rx_part1 + Rx_part2;% (12) of [1]
end
function [c_FT] = local_c_FT(F,T,X,Rx,rou)
c_FT = zeros(F,T);
for f = 1:F
    for t = 1:T
        c_FT(f,t) = rou/(2*(X(:,f,t)'*inv(Rx(:,:,f))*X(:,f,t)).^(1-rou/2));% under three lines of (68) in [1] 
    end
end
end
function [alpha] = local_alpha(F, T, M, v, X, Rx)
alpha = zeros(F,T);
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        alpha(f,t) = (2 * M + v)./v./(1 + 2/v * X(:,f,t)' * Rx_inv * X(:,f,t));
    end
end
end
function [rt_nume,rt_deno] = local_rt(X,F,T,Af,rt,Rx,alpha,c_FT,beta)
rt_nume = zeros(F,T);rt_deno = zeros(F,T);
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        rt_nume(f,t) = alpha(f,t) * c_FT(f,t) * abs(X(:,f,t)' * Rx_inv * Af(:,1,f)).^2 + beta/rt(f,t).^2;% 分子(66) of [1]
        rt_deno(f,t) = Af(:,1,f)' * Rx_inv * Af(:,1,f) + (alpha+1)/rt(f,t);% 分母(66) of [1]
    end
end
end
function [rn_nume,rn_deno] = local_rn(X,F,T,Rx,Rn,alpha,c_FT)
rn_nume = zeros(F,T);rn_deno = zeros(F,T);
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        rn_nume(f,t) = alpha(f,t) * c_FT(f,t) * X(:,f,t)' * Rx_inv * Rn(:,:,f) * Rx_inv * X(:,f,t);% 分子(67) of [1]
        rn_deno(f,t) = trace(Rx_inv * Rn(:,:,f));% 分母(67) of [1]
    end
end
end
function [lamda_nume,lamda_deno] = local_lamda(X,F,T,Rx,c_FT,b,rn)
lamda_nume_tmp = zeros(F,T);lamda_deno_tmp = zeros(F,T);
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        lamda_nume_tmp(f,t) = c_FT(f,t) * rn(f,t) * abs(b(:,f)' * Rx_inv * X(:,f,t)).^2;% 分子(68) of [1]
        lamda_deno_tmp(f,t) = rn(f,t) * b(:,f)' * Rx_inv * b(:,f);% 分母(68) of [1]
    end
end
lamda_nume = sum(lamda_nume_tmp,2);
lamda_deno = sum(lamda_deno_tmp,2);
end