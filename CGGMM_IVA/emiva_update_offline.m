function [Y, W, V, parm,CostFunc] =  emiva_update_offline(Y, X, W, V, k,t, partition,parm,option )
% EMIVA for  offline BSS
% Inputs:
%   Y    : Estimated signal #K * T * F;
%   X    : Mixed signal #K * T * F;
%   W    : Demixing matrix;
%   V    : Auxiliary function;
%   k    : Present signal channel state;
%   t    : Present signal frame state;
%
% Outputs:
%   Y    : Estimated signal #K * T * F;
%   W    : Demixing matrix;
%   V    : Auxiliary function;
%
% Ref.[1] Independent Vector Analysis for Blind Speech Separation Using Complex Generalized Gaussian Mixture Model with Weighted Variance,Xinyu Tang
% 
%%
global epsilon_start_ratio;global epsilon;global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;
%% init
epsi = parm.epsi;
tmp_beta = parm.tmp_beta ;
[K,T,F] = size(Y);
M = K;% determined case
batch_update_num = option.batch_update_num;
beta = option.EMIVA_beta;
parti = option.parti; % 是否对载波进行分块（子块）
select = option.select; % 是否选择子块
thFactor = option.thFactor; % 子块选择阈值
EMIVA_ratio = 1; % batch
if option.online == 1
EMIVA_ratio = option.EMIVA_ratio;
end
V_init = V;
J = zeros(K,1);
tmp_cf = zeros(F,1);
I = parm.I;
% pre_CostFunction = 0;
%% select threshold
Y_p = permute(Y,[3,2,1]);%[K,T,F] = size(X);
YYn = reshape(Y_p(:,1:batch_update_num,:), [F, batch_update_num, M, 1]) .* conj(reshape(Y_p(:,1:batch_update_num,:), [F, batch_update_num, 1, M]));    YY = reshape(YYn,[F, batch_update_num, M^2]);
YE_mean = zeros(1,M);
for m = 1:M
    YE_mean(:,m) = mean(mean(abs(YY(:,:,m^2))));
    YE_TH = YE_mean * thFactor; % select threshold
end
[spec_indices, par_select] = selectpar(partition, select, parti, YY, K, F, YE_TH); % select & partition initalize

%% F processsing F
if select
    par_indice = par_select{k}.index{1}; % select on
else
    par_indice = spec_indices{k}; % select off par.index{n}
end
for f = spec_indices{k}% 1:size(spec_indices{k},2)
    parm.sigema2{k,1}(f,:) = (tmp_beta{k,1} .* sum( parm.q{k,1}(1:t,:) .* abs( repmat(abs(Y(k,1:t,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1}(1:t,:))) .^ repmat(beta{k,1},t,1),1)./ (sum(parm.q{k,1}(1:t,:))+epsi)) .^(2./beta{k,1}) + eps  ;%{Kx1}(FxI) % (11) of [1] 
%     for i = 1 : I(k)        
        parm.sigema_rou2{k,f}(1:t,:) = repmat(parm.sigema2{k,1}(f,:),t,1) .* parm.rou2{k,1}(1:t,:) ;%{KxF}(TxI)               
%     end
    %% Auxiliary Function-based-update W            
    if f<size(par_indice,2)+1 && f == par_indice(f)
    phi_V = sum(parm.q{k,1}(1:t,:) .* repmat(tmp_beta{k,1},t,1) .* abs(1./ parm.sigema_rou2{k,f}(1:t,:)) .^ repmat(beta{k,1}./2 ,t,1) .* ...
            repmat(abs(Y(k,1:t,f)).',1,I(k)) .^ repmat(beta{k,1} - 2 ,t,1) , 2); % (15) of [1] 
    phi_V = min(phi_V , 1000);
%% update output  
    V(:,:,k,f) = repmat(phi_V.',K,1) .* X(:,1:t,f) * X(:,1:t,f)' ./ t; % (15) of [1] 
    else V(:,:,k,f) = V(:,:,k,f-1);
    end
    %% 有当前源先验的话就使用DOA
    [w] = update_w(W, V(:,:,k,f), option, parm, f ,k, M, K);
    W(k,:,f) = w';   
%% update output
    Y(k,1 : batch_update_num,f) = w' * X(:,1 : batch_update_num,f);    
end
    V =EMIVA_ratio.*V + (1-EMIVA_ratio).* V_init;%V_init(:,:,k,f) % (15) of [1] 
%% the cost function
    J(k,1) = logJ(Y(:,1:batch_update_num,:),k,batch_update_num,F,beta,parm);
    for f= 1:F
        tmp_cf(f,1) = - log(abs(det(W(:,:,f))));
    end
    CostFunc = sum(J) / T + sum(tmp_cf);
function J = logJ(Y,k,T,F,beta,parm)
   J= -sum(sum(parm.q{k,1} .* repmat(log(parm.mixCoef{k,1}),T,1))) + F/2 * sum(sum(parm.q{k,1} .* log(parm.rou2{k,1}))) + 1/2 * sum(sum(parm.q{k,1} .* repmat(sum(log(parm.sigema2{k,1})),T,1)))...
          + sum(sum(parm.q{k,1} .* squeeze(permute(sum(abs(repmat(sqrt(gamma(3./beta{k,1}) ./ gamma(1./beta{k,1})),F,1,T) .* repmat(permute(abs(Y(k,:,:)),[3 1 2]),1,I(k),1) ./ repmat(sqrt(parm.sigema2{k,1}),1,1,T) ./ ...
          permute(repmat(sqrt(parm.rou2{k,1}),1,1,F),[3 2 1])).^ repmat(beta{k,1},F,1,T),1),[3 2 1]))));% (9) of [1]基本一致
end
%% update w
function [w] = update_w(W, V , option, parm, f, k, M,K)
I_e = parm.I_e;
% KK = MM;
   if option.prior(k) == 1
%         hf = cal_RTF(FF,16000,option.mic_pos,theta(:,k));
        hf = option.hf;
        fac_a = option.fac_a;
        delta_f = option.deltaf;
        Q = squeeze(V) ...
                    + (DOA_tik_ratio * eye(M) + DOA_Null_ratio *hf(:,f,k)*hf(:,f,k)') * fac_a/delta_f;
        w = (W(:,:,f) * Q + eps.*eye(K)) \ I_e(:,k);   % (16) of [1]
        w = w / sqrt(w' * Q * w);
   else
        w = (W(:,:,f)*V + eps.*eye(K)) \ I_e(:,k);   % (16) of [1]
        w = w/sqrt(w'*V*w); % (17) of [1] 
        
   end
end
end

