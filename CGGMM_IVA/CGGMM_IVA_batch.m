function [s_hat, label,W, parm,loop,CostFunction,obj_set] = CGGMM_IVA_batch(x, option)
%input: 
% x: mixed data, K x nn
% nfft: fft point
% I: mixture state
% max_iterations
% beta: the shape paramater matrix, K x I
% 所有声道的状态数可以不同, 这样可以复杂度比较高， 戴晓明，2020/10/15
% Ref.[1] Independent Vector Analysis for Blind Speech Separation Using Complex Generalized Gaussian Mixture Model with Weighted Variance
%%
global epsilon_start_ratio;global epsilon;global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;global SubBlockSize; global SB_ov_Size;
%% FFT
win_size = option.win_size;
fft_size = win_size;
spec_coeff_num = fft_size / 2 + 1;
[nmic, nn] = size(x);
win_size = hanning(fft_size,'periodic');
inc = fix(1*fft_size/2);
for l=1:nmic
    X(l,:,:) = stft(x(l,:).', fft_size, win_size, inc).';
end
%% Initialization
beta = option.EMIVA_beta;
max_iterations = option.EMIVA_max_iter;
parti = option.parti; % 是否对载波进行分块（子块）
select = option.select; % 是否选择子块
thFactor = option.thFactor; % 子块选择阈值
parti = option.parti;
partisize = option.partisize;

[~,I] = cellfun(@size,beta);%% slow
% [X,~,~] = whitening( X , nmic);
[K,T,F] = size(X);
Y = X;
W = zeros(K,K,F); 
M=K;
V = zeros(K, M, M, F);% batch
for f = 1:F
    W(:,:,f) = eye(K);
end
I_e = eye(K);
parm.mixCoef = cell(K,1);
for k = 1:K
    parm.mixCoef{k,1} = rand(1,I(k));
    parm.mixCoef{k,1} = parm.mixCoef{k,1} ./ sum(parm.mixCoef{k,1});
end
parm.q = cell(K,1);
parm.sigema2 = cell(K,1);
parm.rou2 = cell(K,1);
parm.sigema_rou2 = cell(K,F);
for k=1:K
    parm.sigema2{k,1} = 0.0001*ones(F,I(k));
    for f =1:F
        parm.sigema_rou2{k,f} = 0.0001*ones(T,I(k));
    end
end

if option.annealing % 是否使用退火因子
    fac_a = max(0.5-iter/iter_num, 0); % annealing factor
else
    fac_a = 1; % =1即等效于不退火
end
option.fac_a = fac_a;
       
epsi = 1e-8;
J = zeros(K,1);
tmp_cf = zeros(F,1);
g = zeros(1,K);
pre_CostFunction = 0;
obj_set = [];
%% 子块与index
if parti  % 子块算法初始化
    block_size = SubBlockSize;        block_overlap = floor(SubBlockSize*SB_ov_Size);
    block_starts = 1:block_size - block_overlap  :spec_coeff_num - block_size - 1;
    for n = 1:length(block_starts)
        partition_index{n} = block_starts(n):block_starts(n) + block_size - 1;
    end
else
    partition_index = {1:spec_coeff_num * partisize};
end
partition_size = cellfun(@(x) length(x), partition_index);

par1.num = length(partition_index);     par1.size = partition_size;     par1.index = partition_index;    par1.contrast = {'blank'};
par1.contrast_derivative =  {'blank'};

par2.num = length(partition_index);     par2.size = partition_size;    par2.index = partition_index;     par2.contrast =  {'blank'};
par2.contrast_derivative =  {'blank'};

if option.mix_model == 0 % 对于不用混合模型的情况，可以加入源3和源4
    par3.num = length(partition_index);    par3.size = partition_size;    par3.index = partition_index;    par3.contrast = {'blank'};
    par3.contrast_derivative =  {'blank'};
    
    par4.num = length(partition_index);    par4.size = partition_size;    par4.index = partition_index;    par4.contrast = {'blank'};
    par4.contrast_derivative =  {'blank'};
    
    partition = {par1, par2, par3, par4};
else
    partition = {par1, par2};
end
%% select threshold
Y_p = permute(Y,[3,2,1]);%[K,T,F] = size(X);
YYn = reshape(Y_p, [F, T, M, 1]) .* conj(reshape(Y_p, [F, T, 1, M]));    YY = reshape(YYn,[F, T, M^2]);
YE_mean = zeros(1,M);
for m = 1:M
    YE_mean(:,m) = mean(mean(abs(YY(:,:,m^2))));
    YE_TH = YE_mean * thFactor; % select threshold
end
[spec_indices, par_select] = selectpar(partition, select, parti, YY, K, F, YE_TH); % select & partition initalize
    fid=fopen('indice.txt','a');fprintf(fid,'indice= ');fprintf(fid,'%g  \n ',size(spec_indices{k},2));
%% F processsing F
if select
    par_indice = par_select{k}.index{1}; % select on
else
    par_indice = spec_indices{k}; % select off par.index{n}
end
%% DOA
if option.DOA_esti 
    x1=x.';                   %(1:option.batch_update_num * spec_coeff_num,:)
    theta = doa_estimation(x1,option.esti_mic_dist,K,16000); % 仅用prebatch的信息更新DOA
    option.theta = theta*pi/180;
    if M == 2
        option.mic_pos = [0,option.esti_mic_dist];%2mic
    else
        option.mic_pos = [0,option.esti_mic_dist,2*option.esti_mic_dist,3*option.esti_mic_dist];%4mic    
    end %Only support 2 or 4 mic
    for k = 1 : K
        if option.prior(k) == 1
        hf_temp(:,:,k) = cal_RTF(F,16000,option.mic_pos,option.theta(:,k));
        option.hf = hf_temp;
        end
    end
end
%% Iterate
tmp_beta = cellfun(@(x) x .* abs(gamma(3./x) ./ gamma(1./x)) .^ (x./2) , beta ,'UniformOutput',false);
% fprintf('Iteration:    ');
tic
for loop = 1 : max_iterations
%     fprintf('\b\b\b\b%4d\n', loop);
%     disp(['The iter = ',num2str(loop),'/',num2str(max_iterations)]);
    %% E-STEP
    for k = 1 : K
%        clear g;
        for t = 1 : T
            for i = 1 :I(k)
%                g(1,i) = logP(squeeze(Y(k,t,:)),cellfun(@(c)c(t,i),parm.sigema_rou2(k,:)).',beta{k,1}(1,i), F); % 原始算法
                c_temp = zeros(F,1);for f=1:size(c_temp,1) c_temp(f)=parm.sigema_rou2{k,f}(t,i);end % 此种计算方式更快
                g(1,i) = logP(squeeze(Y(k,t,:)), c_temp, beta{k,1}(1,i), F, option);              
            end
            if sum(isnan(g)==1)>0
                
            else
                parm.q{k,1}(t,:) =  parm.mixCoef{k,1} .* exp(g - max(g)) ./ sum(parm.mixCoef{k,1} .* exp(g - max(g)));%update posterior probability %cell(Kx1){TxI(K)}
            end
           %% S-STEP
            parm.rou2{k,1}(t,:) = ((tmp_beta{k,1} .* sum(abs( abs(repmat(squeeze(Y(k,t,:)),1,I(k))) .* sqrt(1 ./ parm.sigema2{k,1})) .^ repmat(beta{k,1},F,1),1)) ./ F) .^ (2./beta{k,1}) + eps;%{Kx1}(TxI)
        end
            %             parm.rou2(k,:,t) = ones(1,I);
        parm.mixCoef{k,1} = mean(parm.q{k,1})+epsi;
        for f = 1 : size(spec_indices{k},2)
%             re_y = abs( repmat(abs(Y(k,:,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1})) .^ repmat(beta{k,1},T,1);
%             parm.sigema2{k,1}(f,:) = (tmp_beta{k,1} .* sum( parm.q{k,1} .* re_y,1)./ (sum(parm.q{k,1})+epsi)) .^(2./beta{k,1}) + eps  ;%{Kx1}(FxI) % (11) of [1] 
            parm.sigema2{k,1}(f,:) =  (tmp_beta{k,1} .* sum( parm.q{k,1} .* abs( repmat(abs(Y(k,:,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1})) .^ repmat(beta{k,1},T,1),1)...
                                   ./ (sum(parm.q{k,1})+epsi)) .^(2./beta{k,1}) + eps ;%{Kx1}(FxI)
%             for i = 1 : I(k)
                parm.sigema_rou2{k,f} = repmat(parm.sigema2{k,1}(f,:),T,1) .* parm.rou2{k,1};%{KxF}(TxI)
      
%                 min(max(repmat(parm.sigema2(k,i,f), 1,T) .* squeeze(parm.rou2(k,i,:)).' , 0.000001));%KxTxFxI
%             end       
           %% Auxiliary Function-based-update W
%            if f<size(par_indice,2)+1 && f == par_indice(f)  
            phi_V = sum(parm.q{k,1} .* repmat(tmp_beta{k,1},T,1) .* abs(1./ parm.sigema_rou2{k,f}) .^ repmat(beta{k,1}./2 ,T,1) .* ...
                repmat(abs(Y(k,:,f)).',1,I(k)) .^ repmat(beta{k,1} - 2 ,T,1) , 2);
            phi_V = min(phi_V , 1000);
            V(:,:,k,f) =  repmat(phi_V.',K,1) .* X(:,:,f) * X(:,:,f)' ./ T;
%            else V(:,:,k,f) = V(:,:,k,f-1);
%            end

            %% DOA
           if option.prior(k) == 1
               fac_a = option.fac_a;
               hf = option.hf;
               delta_f = option.deltaf;
               Q = squeeze(V(:,:,k,f)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (DOA_tik_ratio * eye(M) + DOA_Null_ratio *hf(:,f,k)*hf(:,f,k)') * fac_a/delta_f^2;

               w = (W(:,:,f) * Q) \ I_e(:,k);   % (16) of [1]
               w = w / sqrt(w' * Q * w);
          else
               w = (W(:,:,f)*V(:,:,k,f) + eps.*eye(K)) \ I_e(:,k);   % (16) of [1]
               w = w/sqrt(w'*V(:,:,k,f)*w); % (17) of [1]         
          end            
          W(k,:,f) = w';
           %% update output
          Y(k,:,f) = w' * X(:,:,f);
        end
       %% the cost function
        J(k,1) = -sum(sum(parm.q{k,1} .* repmat(log(parm.mixCoef{k,1}),T,1))) + F/2 * sum(sum(parm.q{k,1} .* log(parm.rou2{k,1}))) + 1/2 * sum(sum(parm.q{k,1} .* repmat(sum(log(parm.sigema2{k,1})),T,1)))...
          + sum(sum(parm.q{k,1} .* squeeze(permute(sum(abs(repmat(sqrt(gamma(3./beta{k,1}) ./ gamma(1./beta{k,1})),F,1,T) .* repmat(permute(abs(Y(k,:,:)),[3 1 2]),1,I(k),1) ./ repmat(sqrt(parm.sigema2{k,1}),1,1,T) ./ ...
          permute(repmat(sqrt(parm.rou2{k,1}),1,1,F),[3 2 1])).^ repmat(beta{k,1},F,1,T),1),[3 2 1]))));
    end
    for f= 1:F
        tmp_cf(f,1) = - log(abs(det(W(:,:,f))));
    end
    CostFunction = sum(J) / T + sum(tmp_cf);
    disp(['The iter = ',num2str(loop),'/',num2str(max_iterations),'   obj = ',num2str(CostFunction)]);    

%     disp(['      CostFunction = ',num2str(CostFunction)]); 
    if abs(pre_CostFunction - CostFunction) < 1e-8*F
        break;
    end
    pre_CostFunction = CostFunction ;
    obj_set = [obj_set,CostFunction];
end
toc
%% Minimal distortion principle and Output
for f = 1:F
    W(:,:,f) = diag(diag(pinv(W(:,:,f))))*W(:,:,f); 
    Y(:,:,f) = W(:,:,f) * X(:,:,f);
end
%% Re-synthesize the obtained source signals

for k=1:K
    s_est(k,:) = istft( squeeze(Y(k,:,:)).' , nn, win_size, inc)';
end
label = cell(k,1);
for k = 1:K
    label{k} = 'target';
end
%% 自动分离单路程序
if option.singleout
    mean_s_est = mean(s_est,2);[data,loc]=max(mean_s_est);
    s_hat = s_est(loc,:);
else
    s_hat = s_est;
end
end
%% CCMM 
function gc = logP(S,sigema_rou2,beta,F,option)
if option.logp == 1
    gc = F * (log(beta/2) + 0.5 * log(gamma(3/beta)) - 1.5 * log(gamma(1/beta))) -...
     0.5*sum(log(sigema_rou2)) - sum(abs(sqrt(gamma(3/beta) ./ gamma(1/beta) ./ (sigema_rou2 + eps)) .* S) .^ beta);
elseif option.logp == 2
    gc = - F * log(pi) - F * log(gamma(1+ 2/beta)) -  sum(log(sigema_rou2)) ...
    - sum(abs( S ./ sqrt(sigema_rou2 )) .^ beta);
end
end