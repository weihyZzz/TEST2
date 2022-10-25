function [s_est, label,W, parm,loop,CostFunction,obj_set] = CGGMM_IVA_batch(x, option)
%input: 
% x: mixed data, K x nn
% nfft: fft point
% I: mixture state
% max_iterations
% beta: the shape paramater matrix, K x I
% 所有声道的状态数可以不同, 这样可以复杂度比较高， 戴晓明，2020/10/15
% Ref.[1] Independent Vector Analysis for Blind Speech Separation Using Complex Generalized Gaussian Mixture Model with Weighted Variance
%%
global epsilon_start_ratio;global epsilon;global epsilon_ratio; global DOA_tik_ratio; global DOA_Null_ratio;
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
[~,I] = cellfun(@size,beta);%% slow
% [X,~,~] = whitening( X , nmic);
[K,T,F] = size(X);
M = K;
W = zeros(K,K,F); 
% XXn = reshape(X, [N, T, M, 1]) .* conj(reshape(X, [N, T, 1, M]));    XX = reshape(XXn,[N, T, M^2]);
% Cxx = cal_Cxx(X); % (10) of [2] || (P1) of [1]
% if option.singleout % FIVE方法需要pre-whitening，令Cxx等于单位矩阵;
%     for f = 1:F
%         % We will need the inverse square root of Cx
%         [e_vec,e_val] = eig(squeeze(Cxx(f,:,:)),'vector');
%         Q_H = fliplr(e_vec) .* sqrt(flipud(e_val)).';
%         X(:,:,f) = X(:,:,f) * (Q_H^-1).';
%     end
% end
Y = X;
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
W_single = zeros(M,F);
Y_single = zeros(T,F);
pre_CostFunction = 0;
obj_set = [];
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
%     if option.singleout
%         K=1;    
%     else
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
%             RR(t) = norm(squeeze(Y(1,t,:)));             
        end
%         ck = sum(RR)/T;
            %             parm.rou2(k,:,t) = ones(1,I);
        parm.mixCoef{k,1} = mean(parm.q{k,1})+epsi;
        for f = 1 : F
%             re_y = abs( repmat(abs(Y(k,:,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1})) .^ repmat(beta{k,1},T,1);
%             parm.sigema2{k,1}(f,:) = (tmp_beta{k,1} .* sum( parm.q{k,1} .* re_y,1)./ (sum(parm.q{k,1})+epsi)) .^(2./beta{k,1}) + eps  ;%{Kx1}(FxI) % (11) of [1] 
            parm.sigema2{k,1}(f,:) =  (tmp_beta{k,1} .* sum( parm.q{k,1} .* abs( repmat(abs(Y(k,:,f)).',1,I(k)) .* sqrt(1 ./ parm.rou2{k,1})) .^ repmat(beta{k,1},T,1),1)...
                                   ./ (sum(parm.q{k,1})+epsi)) .^(2./beta{k,1}) + eps ;%{Kx1}(FxI)
%             for i = 1 : I(k)
                parm.sigema_rou2{k,f} = repmat(parm.sigema2{k,1}(f,:),T,1) .* parm.rou2{k,1};%{KxF}(TxI)
      
%                 min(max(repmat(parm.sigema2(k,i,f), 1,T) .* squeeze(parm.rou2(k,i,:)).' , 0.000001));%KxTxFxI
%             end       
           %% Auxiliary Function-based-update W
            phi_V = sum(parm.q{k,1} .* repmat(tmp_beta{k,1},T,1) .* abs(1./ parm.sigema_rou2{k,f}) .^ repmat(beta{k,1}./2 ,T,1) .* ...
                repmat(abs(Y(k,:,f)).',1,I(k)) .^ repmat(beta{k,1} - 2 ,T,1) , 2);
            phi_V = min(phi_V , 1000);
            V(:,:,k,f) =  repmat(phi_V.',K,1) .* X(:,:,f) * X(:,:,f)' ./ T;
            if option.singleout
                if k==1
                    Vf = V(:,:,k,f);
                    [rm,lambda] = eig(Vf,'vector'); % Algorithm 3.1 of [1]
                    [lambda_m, min_eig_index] = min(lambda);
                    w = lambda_m^-1/2 * rm(:,min_eig_index);
%                     w = w * ck^-1/2;
                    W_single(:,f) = w;
                  %  IVE_NORMOLIZATION~
                    Y(1,:,f) = w' * X(:,:,f);
                else
                    break;
                end           
            %% DOA
            else
                if option.prior(k) == 1
                    %         hf = cal_RTF(FF,16000,option.mic_pos,theta(:,k));
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
        end%F
       %% the cost function
       if ~option.singleout
           J(k,1) = -sum(sum(parm.q{k,1} .* repmat(log(parm.mixCoef{k,1}),T,1))) + F/2 * sum(sum(parm.q{k,1} .* log(parm.rou2{k,1}))) + 1/2 * sum(sum(parm.q{k,1} .* repmat(sum(log(parm.sigema2{k,1})),T,1)))...
          + sum(sum(parm.q{k,1} .* squeeze(permute(sum(abs(repmat(sqrt(gamma(3./beta{k,1}) ./ gamma(1./beta{k,1})),F,1,T) .* repmat(permute(abs(Y(k,:,:)),[3 1 2]),1,I(k),1) ./ repmat(sqrt(parm.sigema2{k,1}),1,1,T) ./ ...
          permute(repmat(sqrt(parm.rou2{k,1}),1,1,F),[3 2 1])).^ repmat(beta{k,1},F,1,T),1),[3 2 1]))));
       end
       end
    
% end
if ~option.singleout

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
else
    disp(['The iter = ',num2str(loop),'/',num2str(max_iterations)]);
    obj_set = [0];
    CostFunction = 0;
end
end
toc
%% Minimal distortion principle and Output
if ~option.singleout
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
else
    for f = 1:F
    Y_single(:,f) = (W_single(:,f)' * X(:,:,f)).';
    end 
%     Y_single = squeeze(sum(permute(repmat(W_single,1,1,T),[1,3,2]).*X,1));
    %% Re-synthesize the obtained source signals
    s_est = istft( Y_single.' , nn, win_size, inc)';% 1 * nsamples
    label = cell(1);
    for k = 1:K
        label{k} = 'single';
    end
end
% X = permute( X,[3,2,1]);
% Y = backProjection(permute(Y,[3,2,1]) ,X(:,:,2));
% Y = permute(Y,[3,2,1]);
% %% Re-synthesize the obtained source signals
% 
% for k=1:K
%     s_est(k,:) = istft( squeeze(Y(k,:,:)).' , nn, win_size, inc)';
% end
% label = cell(k,1);
% for k = 1:K
%     label{k} = 'target';
% end
% %% 自动分离单路程序
% if option.singleout
%     mean_s_est = mean(abs(s_est),2);[data,loc]=max(mean_s_est);
%     s_hat = s_est(loc,:);
% else
%     s_hat = s_est;
% end
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
%% Cxx
function Cxx = cal_Cxx(X) % 旧版Cxx计算方法，暂时保留，(9) of [3]
[N, T, M] = size(X);
Cxx = zeros(N, M, M);
for n = 1:N
    Xf = squeeze(X(n,:,:));
    if T == 1
        % 当T=1时，由于squeeze的特性，Xf与[3]中xf相等，直接用原公式
        Cf = Xf * Xf';
    else
        % 本Xf为[3]中xf的转置，因此(xf*xf') = Xf.' * conj(Xf)
        Cf = Xf.' * conj(Xf) / T; 
    end
    Cxx(n,:,:) = Cf;
end
end