function [s_est, label,W, parm,loop] = CGGMM_IVA_batch_fast(x, option)
%input: 
% x: mixed data, K x nn
% nfft: fft point
% I: mixture state
% max_iterations
% beta: the shape paramater matrix, K x I
% fast: 所有声道的状态数相同, 这样可以实现快速算法， 戴晓明，2020/10/15
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

% [X,~,~] = whitening( X , nmic);
%% initialization
beta = option.EMIVA_beta;
max_iterations = option.EMIVA_max_iter;
I = size(beta{1,1},2);%% fast

[K,T,F] = size(X);
% % de-mean
% for f= 1:F
%     X(:,:,f) = bsxfun(@minus,X_init(:,:,f), mean(X_init(:,:,f),2));
% end
Y = X;
W = zeros(K,K,F); 
M=K;
V = zeros(K, M, M, F);% batch
for f = 1:F
    W(:,:,f) = eye(K);
end
I_e = eye(K);
parm.mixCoef = zeros(K,I);
for k = 1:K
    parm.mixCoef(k,:) = rand(1,I);
    parm.mixCoef(k,:) = parm.mixCoef(k,:) ./ sum(parm.mixCoef(k,:));
%     parm.mixCoef{k,1} = 1/I .* ones(1,I);%cell(Kx1){1xI(K)}
end

parm.q = zeros(T,I,K);
parm.sigema2 = 0.0001*ones(F,I,K);
parm.rou2 = zeros(T,I,K);
parm.sigema_rou2 =0.0001*ones(T,I,F,K);

if option.annealing % 是否使用退火因子
    fac_a = max(0.5-iter/iter_num, 0); % annealing factor
else
    fac_a = 1; % =1即等效于不退火
end
option.fac_a = fac_a;       
% for f= 1:F
%     parm.sigema2{k,f}() = repmat(diag(Y(:,:,f) * Y(:,:,f)'),1,I) ./ T .* parm.mixCoef.';
% end
epsi = 1e-8;
J = zeros(K,1);
tmp_cf = zeros(F,1);
g = zeros(1,K);
pre_CostFunction = 0;
%% DOA
if option.DOA_esti 
    x1=x.';
    theta = doa_estimation(x1(1:option.batch_update_num * spec_coeff_num,:),option.esti_mic_dist,K,16000); % 仅用prebatch的信息更新DOA
    option.theta = theta*pi/180;
    if M == 2
        option.mic_pos = [0,option.esti_mic_dist];%2mic
    else
        option.mic_pos = [0,option.esti_mic_dist,2*option.esti_mic_dist,3*option.esti_mic_dist];%4mic    
    end %Only support 2 or 4 mic
end
%% iterate
tmp_beta = cellfun(@(x) x .* abs(gamma(3./x) ./ gamma(1./x)) .^ (x./2) , beta ,'UniformOutput',false);
fprintf('Iteration:    ');
for loop = 1 : max_iterations
    fprintf('\b\b\b\b%4d', loop);
%     disp(['The iter = ',num2str(loop),'/',num2str(max_iterations)]);
    %% E-STEP
    for k = 1 : K
        if option.prior(k) == 1
        hf_temp(:,:,k) = cal_RTF(F,16000,option.mic_pos,option.theta(:,k));
        option.hf = hf_temp;
        end
        clear g;
        for t = 1 : T
            for i = 1 :I
                g(1,i) = logP(squeeze(Y(k,t,:)), squeeze(parm.sigema_rou2(t,i,:,k)), beta{k,1}(1,i), F);              
            end
            if sum(isnan(g)==1)>0
                
            else
                parm.q(t,:,k) =  parm.mixCoef(k,:) .* exp(g - max(g)) ./ sum(parm.mixCoef(k,:) .* exp(g - max(g)));%update posterior probability %\{TxIxK}
            end
           %% S-STEP
            parm.rou2(t,:,k) = ((beta{k,1} .* sum( abs(  repmat(squeeze(Y(k,t,:)),1,I) .* sqrt(1 ./ parm.sigema2(:,:,k))  ) .^ repmat(beta{k,1},F,1),   1)) ./ (1 * F)) .^ (2./beta{k,1}) + eps;%(TxIxK)
%             parm.rou2(t,:,k) = ((beta{k,1} .* sum(abs( abs(repmat(squeeze(Y(k,t,:)),1,I)) .* sqrt(1 ./ (parm.sigema2(:,:,k).* repmat(gamma(1./beta{k,1}) ./ gamma(3./beta{k,1}),F,1)))) .^ repmat(beta{k,1},F,1),1)) ./ (2* F)) .^ (2./beta{k,1}) + eps;%(TxIxK)
%             parm.tmp_rou2(t,:,k) = parm.rou2(t,:,k) .* gamma(1./beta{k,1}) ./ gamma(3./beta{k,1});
        end
            %             parm.rou2(k,:,t) = ones(1,I);
        parm.mixCoef(k,:) = mean(parm.q(:,:,k))+epsi;
        for f = 1 : F
            parm.sigema2(f,:,k) =  (beta{k,1} .* sum( parm.q(:,:,k) .* abs( repmat(Y(k,:,f).',1,I) .* sqrt(1 ./ parm.rou2(:,:,k))) .^ repmat(beta{k,1},T,1)  ,1)...
                                   ./ ( 1 * sum(parm.q(:,:,k))+eps)) .^(2./beta{k,1}) + eps ;%(FxIxK)
%             parm.sigema2(f,:,k) =  (beta{k,1} .* sum( parm.q(:,:,k) .* abs( repmat(abs(Y(k,:,f)).',1,I) .* sqrt(1 ./ (parm.rou2(:,:,k).* repmat(gamma(1./beta{k,1}) ./ gamma(3./beta{k,1}),T,1)))) .^ repmat(beta{k,1},T,1),1)...
%                                    ./ ( 2 * sum(parm.q(:,:,k))+epsi)) .^(2./beta{k,1}) + eps ;%(FxIxK)  
%             parm.tmp_sigema2(f,:,k) = parm.sigema2(f,:,k) .* gamma(1./beta{k,1}) ./ gamma(3./beta{k,1}) ;                   
            for i = 1 : I
                parm.sigema_rou2(:,i,f,k) = parm.sigema2(f,i,k) .* parm.rou2(:,i,k);%(TxIxFxK)
%                 parm.sigema_rou2(:,:,f,k) = parm.sigema_rou2(:,:,f,k) .* repmat(gamma(1./beta{k,1}) ./ gamma(3./beta{k,1}) , T,1);
%                 min(max(repmat(parm.sigema2(k,i,f), 1,T) .* squeeze(parm.rou2(k,i,:)).' , 0.000001));%KxTxFxI
            end       
           %% Auxiliary Function-based-update W
            phi_V = sum(parm.q(:,:,k) .* repmat(beta{k,1},T,1) .* abs( repmat(Y(k,:,f).',1,I) ./  sqrt(parm.sigema_rou2(:,:,f,k)) ) .^ repmat(beta{k,1} ,T,1) .* ...
                repmat((abs(Y(k,:,f)).').^(-2),1,I) , 2);
            phi_V = min(phi_V , 1000);
            V(:,:,k,f) =  repmat(phi_V.',K,1) .* X(:,:,f) * X(:,:,f)' ./ T;
%% DOA
     if option.prior(k) == 1
         %         hf = cal_RTF(FF,16000,option.mic_pos,theta(:,k));
         fac_a = option.fac_a;
         hf = option.hf;
         delta_f = option.deltaf;
         Q = squeeze(V(:,:,k,f)) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
                    + (DOA_tik_ratio * eye(M) + DOA_Null_ratio *hf(:,f,k)*hf(:,f,k)') * fac_a/delta_f^2;

%         Q = squeeze(V) + epsilon_start_ratio*epsilon_ratio*epsilon * eye(MM)...
%                     + (lamda_e*eye(MM)+hf(:,f)*hf(:,f)') * fac_a/delta_f^2;
%        Q1 = epsilon_start_ratio*epsilon_ratio*epsilon * eye(M)...
%                     + (eye(M)+hf(:,f)*hf(:,f)') * fac_a/delta_f^2;
%        h = H(f,:)';
%        w = (W(:,:,f)*squeeze(V(:,:,k,f))+Q1 )\ h; %sum(abs(Q \ h-inv(Q)*h))
         w = (W(:,:,f) * Q) \ I_e(:,k);   % (16) of [1]
         w = w / sqrt(w' * Q * w);
     else
         w = (W(:,:,f)*V(:,:,k,f) + eps.*eye(K)) \ I_e(:,k);   % (16) of [1]
         w = w/sqrt(w'*V(:,:,k,f)*w); % (17) of [1]         
     end
%             w = (W(:,:,f)*V(:,:,k,f) + eps.*eye(K)) \ I_e(:,k);  
%             w = w/sqrt(w'*V(:,:,k,f)*w);
            W(k,:,f) = w';
           %% update output
            Y(k,:,f) = w' * X(:,:,f);
        end
       %% the cost function
        J(k,1) = -sum(sum(parm.q(:,:,k) .* repmat(log(parm.mixCoef(k,:)),T,1))) + F/2 * sum(sum(parm.q(:,:,k) .* log(parm.rou2(:,:,k)))) +  1/2 * sum(sum(parm.q(:,:,k) .* repmat(sum(log(parm.sigema2(:,:,k))),T,1)))...
          + sum(sum(parm.q(:,:,k) .* squeeze(permute(sum(abs( repmat(permute(abs(Y(k,:,:)),[3 1 2]),1,I,1) ./ repmat(sqrt(parm.sigema2(:,:,k)),1,1,T) ./ ...
          permute(repmat(sqrt(parm.rou2(:,:,k)),1,1,F),[3 2 1])).^ repmat(beta{k,1},F,1,T),1),[3 2 1])))) + F * sum(sum(parm.q(:,:,k) .* repmat(log(gamma(2./ beta{k,1} +1 )),T,1))) + sum(sum(parm.q(:,:,k) .* log(parm.q(:,:,k)+eps)));
    end
    for f= 1:F
        tmp_cf(f,1) = - log(abs(det(W(:,:,f))));
    end
    CostFunction = sum(J) / T + sum(tmp_cf) + K*F*log(pi);
%     disp(['      CostFunction = ',num2str(CostFunction)]); 
%     if abs(pre_CostFunction - CostFunction) < 1e-6*F
%         break;
%     end
    CostF(loop) = CostFunction ;
    pre_CostFunction = CostFunction ;
end
    fprintf('\n');
%% minimal distortion principle and Output
for f = 1:F
    W(:,:,f) = diag(diag(pinv(W(:,:,f))))*W(:,:,f); 
    Y(:,:,f) = W(:,:,f) * X(:,:,f);
end
% Y = backProjection(Y , X_init);
%% Re-synthesize the obtained source signals
for k=1:K
    s_est(k,:) = istft( squeeze(Y(k,:,:)).' , nn, win_size, inc)';
end
label = cell(k,1);
for k = 1:K
    label{k} = 'target';
end
end
%% CCMM 
function gc = logP(S,sigema_rou2,beta,F)
gc = - F * log(pi) - F * log(gamma(1+ 2/beta)) -  sum(log(sigema_rou2)) ...
    - sum(abs( S ./ sqrt(sigema_rou2 )) .^ beta);
end