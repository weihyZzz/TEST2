function [rt_FT,rn_FT,Rx_MMFT,Rn_MMF] = SCM_offline(X,maxIt,Af,rt_FT,rn_FT,Rn_hat,alpha,beta,rou)

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
%       rou: the shape parameter �� of the multivariate GGD
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
q = min(0.5,2/(rou+2));
lamda_FT = zeros(F,1);b= zeros(M,F);
for f = 1:F
    [vec,val] = eig(Rn_hat(:,:,f));vall = diag(val);
    [valu,num]=min(vall);b(:,f)= vec(:num+1);%bȡ��С��0����ֵ��Ӧ������������lamdaȡ0����ֵ
    lamda_FT(f) = vall(num);
%     if vall(1) < 1e-20
%         vec(1,1) = vec(1,1)/sum(abs(vec(:,1)));
%         vec(2,1) = vec(2,1)/sum(abs(vec(:,1)));
%         lamda_FT(f) = vall(2);b(:,f) = vec(:,1);
%     else
%         vec(1,2) = vec(1,2)/sum(abs(vec(:,2)));
%         vec(2,2) = vec(2,2)/sum(abs(vec(:,2)));
%         lamda_FT(f) = vall(1);b(:,f) = vec(:,2);
%     end
end
% b= zeros(M,F);
% for ff = 1:F
%     b(:,ff) = Af(:,1,ff);% ��һ����Դ
% end
    Rn_MMF = local_Rn(F,M,b,lamda_FT,Rn_hat);% (14) of [1]
    Rx_MMFT = local_Rx(F,T,M,N,Af,rt_FT,rn_FT,Rn_MMF);% (12) of [1]
%% main iterate
for it = 1:maxIt
    fprintf('\b\b\b\b%4d', it);
    c_FT = local_c_FT(F,T,X,Rx_MMFT,rou);% under three lines of (68) in [1]    
    %%%%% Update rt %%%%%% F * T      
    [rt_nume,rt_deno] = local_rt(X,F,T,Af,rt_FT,Rx_MMFT,c_FT,alpha,beta);% (66) of [1]
    rt_FT = rt_FT .* max((rt_nume./rt_deno).^q,eps);% (66) of [1]

    Rn_MMF = local_Rn(F,M,Af,lamda_FT,Rn_hat);% (14) of [1]
    Rx_MMFT = local_Rx(F,T,M,N,Af,rt_FT,rn_FT,Rn_MMF);% (12) of [1]
    c_FT = local_c_FT(F,T,X,Rx_MMFT,rou);% under three lines of (68) in [1]    
    %%%%% Update rn %%%%%% F * T
    [rn_nume,rn_deno] = local_rn(X,F,T,Rx_MMFT,Rn_MMF,c_FT);% (67) of [1]
    rn_FT = rn_FT .* max((rn_nume./rn_deno).^q,eps);% (67) of [1]

    Rn_MMF = local_Rn(F,M,Af,lamda_FT,Rn_hat);% (14) of [1]
    Rx_MMFT = local_Rx(F,T,M,N,Af,rt_FT,rn_FT,Rn_MMF);% (12) of [1]
    c_FT = local_c_FT(F,T,X,Rx_MMFT,rou);% under three lines of (68) in [1]    
    %%%%% Update lamda %%%%%% sclar weight 
    [lamda_nume,lamda_deno] = local_lamda(X,F,T,Rx_MMFT,c_FT,b,rn_FT);% (68) of [1]
    lamda_FT = lamda_FT .* max((lamda_nume./lamda_deno).^q,eps);    % (68) of [1]
    %%%%% Update Rn Rx %%%%%% 
    Rn_MMF = local_Rn(F,M,Af,lamda_FT,Rn_hat);% (14) of [1]
    Rx_MMFT = local_Rx(F,T,M,N,Af,rt_FT,rn_FT,Rn_MMF);% (12) of [1]
end
fprintf(' SCM done.\n');
end
function [Rn] = local_Rn(F,M,b,lamda,Rn_hat)
Rn = zeros(M,M,F);lamdabb= zeros(M,M,F);%n=1;
for f = 1:F
    lamdabb(:,:,f) = lamda(f)*b(:,f)*b(:,f)';% second term in (14) of [1]
end
Rn = Rn_hat + lamdabb;% (14) of [1]
end
function [Rx] = local_Rx(F,T,M,N,Af,rt,rn,Rn)
AA_nt = zeros(M,M,F);AA_nn = zeros(M,M,F);n=1;
for f = 1:F
    AA_nt(:,:,f) = Af(:,n,f)*Af(:,n,f)';% 1e-5 * eye(2,2)
     for nn = n+1:N
          AA_nn(:,:,f) = AA_nn(:,:,f) + Af(:,nn,f)*Af(:,nn,f)';
     end
end
Rx_part1 = zeros(M,M,F,T);Rx_part2 = zeros(M,M,F,T);
for f = 1:F
    for t = 1:T
        Rx_part1(:,:,f,t) = rt(f,t) * (AA_nt(:,:,f) + 1e-5 * AA_nn(:,:,f));% first term in (12) of [1]
        Rx_part2(:,:,f,t) = rn(f,t) * Rn(:,:,f);% second term in (12) of [1]
    end
end
Rx = Rx_part1 + Rx_part2;% (12) of [1]
end
function [c_FT] = local_c_FT(F,T,X,Rx,rou)
c_FT = zeros(F,T);
for f = 1:F
    for t = 1:T
        c_FT(f,t) = rou/(2*(X(:,f,t)'* inv(Rx(:,:,f))*X(:,f,t)).^(1-rou/2));% under three lines of (68) in [1] 
    end
end
end
function [rt_nume,rt_deno] = local_rt(X,F,T,Af,rt,Rx,c_FT,alpha,beta)
rt_nume = zeros(F,T);rt_deno = zeros(F,T);nt = 1;
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        rt_nume(f,t) = c_FT(f,t) * abs(X(:,f,t)' * Rx_inv * Af(:,nt,f)).^2 + beta / (rt(f,t).^2 + eps);% ����(66) of [1]
        rt_deno(f,t) = Af(:,1,f)' * Rx_inv * Af(:,nt,f) + (alpha+1) / (rt(f,t) + eps);% ��ĸ(66) of [1]
    end
end
end
function [rn_nume,rn_deno] = local_rn(X,F,T,Rx,Rn,c_FT)
rn_nume = zeros(F,T);rn_deno = zeros(F,T);
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        rn_nume(f,t) = c_FT(f,t) * X(:,f,t)' * Rx_inv * Rn(:,:,f) * Rx_inv * X(:,f,t);% ����(67) of [1]
        rn_deno(f,t) = trace(Rx_inv * Rn(:,:,f));% ��ĸ(67) of [1]
    end
end
end
function [lamda_nume,lamda_deno] = local_lamda(X,F,T,Rx,c_FT,b,rn)
lamda_nume_tmp = zeros(F,T);lamda_deno_tmp = zeros(F,T);
for f = 1:F
    for t = 1:T
        Rx_inv = inv(Rx(:,:,f));
        lamda_nume_tmp(f,t) = c_FT(f,t) * rn(f,t) * abs(b(:,f)' * Rx_inv * X(:,f,t)).^2;% ����(68) of [1]
        lamda_deno_tmp(f,t) = rn(f,t) * b(:,f)' * Rx_inv * b(:,f);% ��ĸ(68) of [1]
    end
end
lamda_nume = sum(lamda_nume_tmp,2);
lamda_deno = sum(lamda_deno_tmp,2);
end