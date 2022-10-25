function [Y, W, V, parm,CostFunc] =  emiva_update_online(in_buffer,Y ,X, W, V,t, partition,parm,option )
% EMIVA for online BSS
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
%% init
epsi = parm.epsi;
tmp_beta = parm.tmp_beta ;
I = parm.I;%% online

% [~,in_num,~] = size(Y);
[K,T,F] = size(X);

 in_num = parm.in_num;
% in_num = t;
M = K;
% H = zeros(F, M); % #freq * #mic
batch_update_num = option.batch_update_num;
beta = option.EMIVA_beta;
parti = option.parti; % 是否对载波进行分块（子块）
select = option.select; % 是否选择子块
thFactor = option.thFactor; % 子块选择阈值

% epsilon_start_ratio = option.epsilon_start_ratio;
% epsilon = option.epsilon;
% epsilon_ratio = option.epsilon_ratio;



%% select threshold
in_buffer_p = permute(in_buffer,[3,2,1]);%[K,T,F] = size(X);

YYn = reshape(in_buffer_p, [F,in_num, M, 1]) .* conj(reshape(in_buffer_p, [F, in_num, 1, M]));    YY = reshape(YYn,[F, in_num, M^2]);
YE_mean = zeros(1,M);
for m = 1:M
    YE_mean(:,m) = mean(mean(abs(YY(:,:,m^2))));
    YE_TH = YE_mean * thFactor; % select threshold
end
[spec_indices,par_select] = selectpar(partition, select, parti, YY, K, F, YE_TH); % select & partition initalize 
% EMIVA_ratio = 1;% batch
Vxx = zeros(K, M, M, F);% online
g = zeros(1,K);
J = zeros(K,1);
tmp_cf = zeros(F,1);
pre_CostFunction = 0;

sigema2_temp = cell(K,1);
mixCoef_temp = cell(K,1);
for kk=1:K
    sigema2_temp{kk,1} = 0.0001*ones(F,I(kk));% online
    mixCoef_temp{kk,1} = zeros(1,I(kk));% online
%     parm.mixCoef{k,1} = 1/I(k) .* ones(1,I(k));%cell(Kx1){1xI(K)}
end
%% iter
on_iter_num = option.iter_num ;
for loop = 1 : on_iter_num
for k = 1 : K 
       for i = 1 :I(k)           %Y(k,t,:)
%            g(1,i) = logP(squeeze(in_buffer(k,in_num,:)), cellfun(@(c)c(t,i),parm.sigema_rou2(k,:)).', beta{k,1}(1,i), F);
           c_temp = zeros(F,1);for f=1:size(c_temp,1) c_temp(f)=parm.sigema_rou2{k,f}(t,i);end
           g(1,i) = logP(squeeze(in_buffer(k,in_num,:)), c_temp, beta{k,1}(1,i), F, option);              

       end
       if sum(isnan(g)==1)>0
       else
           parm.q{k,1}(t,:) =  parm.mixCoef{k,1} .* exp(g - max(g)) ./ sum(parm.mixCoef{k,1} .* exp(g - max(g)));%update posterior probability %cell(Kx1){TxI(K)}
       end
       EMIVA_ratio = option.EMIVA_ratio;
           % S-STEP
       parm.rou2{k,1}(t,:) =((tmp_beta{k,1} .* sum(abs( abs(repmat(squeeze(in_buffer(k,in_num,:)),1,I(k))) .* sqrt(1 ./ parm.sigema2{k,1})) .^ repmat(beta{k,1},F,1),1)) ./ F) .^ (2./beta{k,1}) + eps;%{Kx1}(TxI)
           % Y(k,t,:)
       mixCoef_temp{k,1} = mean(parm.q{k,1}(t-in_num+1:t))+epsi;  % (10) of [1] 
       parm.mixCoef{k,1} =  EMIVA_ratio .* mixCoef_temp{k,1}+ (1-EMIVA_ratio).* parm.mixCoef{k,1};% online
%% F processsing
if select
    par_indice = par_select{k}.index{1}; % select on
else
    par_indice = spec_indices{k}; % select off par.index{n}
end
for f = spec_indices{k}% 1:size(spec_indices{k},2)
 if f == spec_indices{k}(1)
    %disp(['The F index = ',num2str(f),'/',num2str(F)]);      
    sigema2_temp{k,1}(f,:) = (tmp_beta{k,1} .* sum( parm.q{k,1}(t-in_num+1:t,:) .* abs( repmat(abs(in_buffer(k,:,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1}(t-in_num+1:t,:))) .^ repmat(beta{k,1},in_num,1),1)./ (sum(parm.q{k,1}(t-in_num+1:t,:))+epsi)) .^(2./beta{k,1}) + eps  ;%{Kx1}(FxI) % (11) of [1] 
%     re_y = abs( repmat(abs(in_buffer(k,:,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1}(t-in_num+1:t,:))) .^ repmat(beta{k,1},in_num,1);
%     sigema2_temp{k,1}(f,:) = (tmp_beta{k,1} .* sum( parm.q{k,1}(t-in_num+1:t,:) .* re_y,1)./ (sum(parm.q{k,1}(t-in_num+1:t,:))+epsi)) .^(2./beta{k,1}) + eps  ;%{Kx1}(FxI) % (11) of [1] 
    parm.sigema2{k,1}(f,:) = EMIVA_ratio .* sigema2_temp{k,1}(f,:) + (1-EMIVA_ratio).* parm.sigema2{k,1}(f,:) ;% online
    parm.sigema_rou2{k,f}(t-in_num+1:t,:) = repmat(parm.sigema2{k,1}(f,:),in_num,1) .* parm.rou2{k,1}(t-in_num+1:t,:) ;%{KxF}(TxI)               
    %% Auxiliary Function-based-update W            
    phi_V = sum(parm.q{k,1}(t-in_num+1:t,:) .* repmat(tmp_beta{k,1},in_num,1) .* abs(1./ parm.sigema_rou2{k,f}(t-in_num+1:t,:)) .^ repmat(beta{k,1}./2 ,in_num,1) .* ...
            repmat(abs(in_buffer(k,:,f)).',1,I(k)) .^ repmat(beta{k,1} - 2 ,in_num,1) , 2); % (15) of [1]  
%     phi_rl = abs(1./ parm.sigema_rou2{k,f}(t-in_num+1:t,:)) .^ repmat(beta{k,1}./2 ,in_num,1);
%     phi_yl = repmat(abs(in_buffer(k,:,f)).',1,I(k)) .^ repmat(beta{k,1} - 2 ,in_num,1);
%     phi_V = sum(parm.q{k,1}(t-in_num+1:t,:) .* repmat(tmp_beta{k,1},in_num,1) .* phi_rl .* phi_yl , 2); % (15) of [1] 
    phi_V = min(phi_V , 1000);
%% update output  
    Vxx(:,:,k,f) =  repmat(phi_V.',K,1) .* X(:,t-in_num+1:t,f) * X(:,t-in_num+1:t,f)' ./ t; % (15) of [1] 
    V(:,:,k,f) = EMIVA_ratio .* Vxx(:,:,k,f) + (1-EMIVA_ratio).* V(:,:,k,f) ;% online
% 有当前源先验的话就使用DOA
    [w] = update_w(W, V(:,:,k,f) ,option,parm,f ,k ,M, K);
    W(k,:,f) = w';   
    Y(k,batch_update_num+1:T,f) = w' * X(:,batch_update_num+1:T,f);% update 之后的Y
 else
    sigema2_temp{k,1}(f,:) = (tmp_beta{k,1} .* sum( parm.q{k,1}(t-in_num+1:t,:) .* abs( repmat(abs(Y(k,t-in_num+1:t,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1}(t-in_num+1:t,:))) .^ repmat(beta{k,1},in_num,1),1)./ (sum(parm.q{k,1}(t-in_num+1:t,:))+epsi)) .^(2./beta{k,1}) + eps  ;%{Kx1}(FxI) % (11) of [1] 
    parm.sigema2{k,1}(f,:) = EMIVA_ratio .* sigema2_temp{k,1}(f,:) + (1-EMIVA_ratio).* parm.sigema2{k,1}(f,:) ;% online  
    parm.sigema_rou2{k,f}(t-in_num+1:t,:) = repmat(parm.sigema2{k,1}(f,:),in_num,1) .* parm.rou2{k,1}(t-in_num+1:t,:) ;%{KxF}(TxI)               
    %% Auxiliary Function-based-update W            
    phi_V = sum(parm.q{k,1}(t-in_num+1:t,:) .* repmat(tmp_beta{k,1},in_num,1) .* abs(1./ parm.sigema_rou2{k,f}(t-in_num+1:t,:)) .^ repmat(beta{k,1}./2 ,in_num,1) .* ...
            repmat(abs(Y(k,t-in_num+1:t,f)).',1,I(k)) .^ repmat(beta{k,1} - 2 ,in_num,1) , 2); % (15) of [1] 
    phi_V = min(phi_V , 1000);
%% update output  
    Vxx(:,:,k,f) =  repmat(phi_V.',K,1) .* X(:,t-in_num+1:t,f) * X(:,t-in_num+1:t,f)' ./ t; % (15) of [1] 
    V(:,:,k,f) = EMIVA_ratio .* Vxx(:,:,k,f) + (1-EMIVA_ratio).* V(:,:,k,f) ;% online
    % 有当前源先验的话就使用DOA
    [w] = update_w(W, V(:,:,k,f), option, parm,k,f,M,K);
    W(k,:,f) = w';   
    Y(k,batch_update_num+1:T,f) = w' * X(:,batch_update_num+1:T,f);% update 之后的Y         
 end
end
    J(k,1) = logJ(Y,k,t,F,beta,parm);  
end%K
for f= 1:F
        tmp_cf(f,1) = - log(abs(det(W(:,:,f))));
end
    CostFunc = sum(J) / t + sum(tmp_cf);
    %if loop == iter_num  disp(['      CostFunction = ',num2str(CostFunc)]); end
    if abs(pre_CostFunction - CostFunc) < 1e-8*F
        break;
    end
    pre_CostFunction = CostFunc ;
end%iter
end           

%% cost func
function J = logJ(Y,k,T,F,beta,parm)
    I = parm.I;%% online
   J= -sum(sum(parm.q{k,1}(T-parm.in_num+1:T,:) .* repmat(log(parm.mixCoef{k,1}),parm.in_num,1))) + F/2 * sum(sum(parm.q{k,1}(T-parm.in_num+1:T,:) .* log(parm.rou2{k,1}(T-parm.in_num+1:T,:)))) + 1/2 * sum(sum(parm.q{k,1}(T-parm.in_num+1:T,:) .* repmat(sum(log(parm.sigema2{k,1})),parm.in_num,1)))...
          + sum(sum(parm.q{k,1}(T-parm.in_num+1:T,:) .* squeeze(permute(sum(abs(repmat(sqrt(gamma(3./beta{k,1}) ./ gamma(1./beta{k,1})),F,1,parm.in_num) .* repmat(permute(abs(Y(k,T-parm.in_num+1:T,:)),[3 1 2]),1,I(k),1) ./ repmat(sqrt(parm.sigema2{k,1}),1,1,parm.in_num) ./ ...
          permute(repmat(sqrt(parm.rou2{k,1}(T-parm.in_num+1:T,:)),1,1,F),[3 2 1])).^ repmat(beta{k,1},F,1,parm.in_num),1),[3 2 1]))));% (9) of [1]基本一致
end
%% pdf
function gc = logP(S,sigema_rou2,beta,F,option)
if option.logp == 1
    gc = F * (log(beta/2) + 0.5 * log(gamma(3/beta)) - 1.5 * log(gamma(1/beta))) -...
     0.5*sum(log(sigema_rou2)) - sum(abs(sqrt(gamma(3/beta) ./ gamma(1/beta) ./ (sigema_rou2 + eps)) .* S) .^ beta);
elseif option.logp == 2
    gc = - F * log(pi) - F * log(gamma(1+ 2/beta)) -  sum(log(sigema_rou2)) ...
    - sum(abs( S ./ sqrt(sigema_rou2 )) .^ beta);
end
end
%% Update W (12)
function [w] = update_w(W, V ,option,parm,k,f,M,K)
% global epsilon_start_ratio;global epsilon;global epsilon_ratio; 
%epsilon_start_ratio = option.epsilon_start_ratio;epsilon = option.epsilon;epsilon_ratio = option.epsilon_ratio;
global epsilon_start_ratio;global epsilon;global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;

%     % Update W (12)
% if option.annealing % 是否使用退火因子
%     fac_a = max(0.5-iter/iter_num, 0); % annealing factor
% else
%     fac_a = 1; % =1即等效于不退火
% end
% ff = varargin{1};
% k1 = varargin{2};
% MM = varargin{3};
% KK = MM;
I_e = parm.I_e;
   if option.prior(k) == 1
%         hf = cal_RTF(FF,16000,option.mic_pos,theta(:,k));
        fac_a = option.fac_a;
        hf = option.hf;
        delta_f = option.deltaf;
        Q = squeeze(V) ...
                    + (DOA_tik_ratio * eye(M) + DOA_Null_ratio *hf(:,f,k)*hf(:,f,k)') * fac_a/delta_f;
% + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)
%         Q = squeeze(V) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(MM)...
%                     + (lamda_e*eye(MM)+hf(:,f)*hf(:,f)') * fac_a/delta_f^2;
%        Q1 = epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
%                     + (eye(M)+hf(:,f)*hf(:,f)') * fac_a/delta_f^2;
%        h = H(f,:)';
%        w = (W(:,:,f)*squeeze(V(:,:,k,f))+Q1 )\ h; %sum(abs(Q \ h-inv(Q)*h))
        w = (W(:,:,f) * Q+ eps.*eye(K)) \ I_e(:,k);   % (16) of [1]
        w = w / sqrt(w' * Q * w);
   else
        w = (W(:,:,f)*V + eps.*eye(K)) \ I_e(:,k);   % (16) of [1]
        w = w/sqrt(w'*V*w); % (17) of [1]         
   end
end



