function [s_est, label,W, parm,loop] = CGGMM_IVA_online(x, option)
%input: 
% x: mixed data, K x nn
% nfft: fft point
% I: mixture state
% max_iterations
% beta: the shape paramater matrix, K x I
% 所有声道的状态数可以不同, 这样可以复杂度比较高， 戴晓明，2020/10/15
% Ref.[1] Independent Vector Analysis for Blind Speech Separation Using Complex Generalized Gaussian Mixture Model with Weighted Variance
%% FFT
win_size = option.win_size;
fft_size = win_size;
[nmic, nn] = size(x);
win_size = hanning(fft_size,'periodic');
inc = fix(1*fft_size/2);
for l=1:nmic
    X(l,:,:) = stft(x(l,:).', fft_size, win_size, inc).';
end

%% Initialization
[K,T,F] = size(X);
buffer_size = option.Lb;
in_buffer = zeros(K, buffer_size, F);
in_buffer_perm = zeros(K, T, F);
in_buffer_batch = zeros(K, T, F);
in_buffer_batch = X;
if option.perm
    perm = randperm(size(in_buffer_batch,2));
    in_buffer_perm = in_buffer_batch(:,perm,:);
else
    in_buffer_perm = in_buffer_batch;
end

beta = option.EMIVA_beta;
max_iterations = option.EMIVA_max_iter;
online_iter_num = option.iter_num ;
[~,I] = cellfun(@size,beta);%%
parm.I = I;
% [X,~,~] = whitening( X , nmic);
% % de-mean
% for f= 1:F
%     X(:,:,f) = bsxfun(@minus,X_init(:,:,f), mean(X_init(:,:,f),2));
% end
Y = X;
W = zeros(K,K,F); 
for f = 1:F
    W(:,:,f) = eye(K);
end
M = K;
V = zeros(K, M, M, F);% batch
I_e = eye(K);
parm.I_e = I_e;
parm.mixCoef = cell(K,1);
for k = 1:K
    parm.mixCoef{k,1} = rand(1,I(k));
    parm.mixCoef{k,1} = parm.mixCoef{k,1} ./ sum(parm.mixCoef{k,1});
%     parm.mixCoef{k,1} = 1/I(k) .* ones(1,I(k));%cell(Kx1){1xI(K)}
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
epsi = 1e-8;
parm.epsi = epsi;
J = zeros(K,1);
tmp_cf = zeros(F,1);
pre_CostFunction = 0;
%% batch_size init
batch_update_num = option.batch_update_num; % option

%% Iterate
tmp_beta = cellfun(@(x) x .* abs(gamma(3./x) ./ gamma(1./x)) .^ (x./2) , beta ,'UniformOutput',false);
parm.tmp_beta = tmp_beta ;

for loop = 1 : max_iterations
    disp(['The iter = ',num2str(loop),'/',num2str(max_iterations)]);
    %% E-STEP
    for k = 1 : K
        clear g;
        for t = 1 : batch_update_num
            for i = 1 :I(k)
                g(1,i) = logP(squeeze(Y(k,t,:)), cellfun(@(c)c(t,i),parm.sigema_rou2(k,:)).', beta{k,1}(1,i), F);              
            end
            if sum(isnan(g)==1)>0
                
            else
                parm.q{k,1}(t,:) =  parm.mixCoef{k,1} .* exp(g - max(g)) ./ sum(parm.mixCoef{k,1} .* exp(g - max(g)));%update posterior probability %cell(Kx1){TxI(K)}
            end
           %% S-STEP
            parm.rou2{k,1}(t,:) = ((tmp_beta{k,1} .* sum(abs( abs(repmat(squeeze(Y(k,t,:)),1,I(k))) .* sqrt(1 ./ parm.sigema2{k,1})) .^ repmat(beta{k,1},F,1),1)) ./ F) .^ (2./beta{k,1}) + eps;%{Kx1}(TxI) % (12) of [1] 
        end
        parm.mixCoef{k,1} = mean(parm.q{k,1}(1 : batch_update_num,:))+epsi;
        option.online = 0; in_num = t ; parm.in_num = in_num;
        [Y, W, V, parm] =  binaural_emiva_update_multi(Y, X, W, V, k, t, parm,option);       
        %% the cost function
        J(k,1) = logJ(Y(:,1:batch_update_num,:),k,batch_update_num,F,beta,parm);
    end
    for f= 1:F
        tmp_cf(f,1) = - log(abs(det(W(:,:,f))));
    end
    CostFunction = sum(J) / T + sum(tmp_cf);
    disp(['      CostFunction = ',num2str(CostFunction)]); 
    if abs(pre_CostFunction - CostFunction) < 1e-8*F
        break;
    end
    pre_CostFunction = CostFunction ;
end
    disp(['Prebatch process done!']);
    option.online = 1;
%% After processing
tic
    
for loop = 1 : online_iter_num
%    disp(['The iter = ',num2str(loop),'/',num2str(online_iter_num),'*********************']);
    %% E-STEP
for k = 1 : K 
    in_buffer(k,:,:) = Y(k,t - buffer_size+ 1 : t,:);
    for t = batch_update_num + 1 : T
            disp(['Iter = ',num2str(loop),'/',num2str(online_iter_num),' The frame index = ',num2str(t),'/',num2str(T)]);
            for i = 1 :I(k)
                g(1,i) = logP(squeeze(Y(k,t,:)), cellfun(@(c)c(t,i),parm.sigema_rou2(k,:)).', beta{k,1}(1,i), F);              
            end
            if sum(isnan(g)==1)>0
                
            else
                parm.q{k,1}(t,:) =  parm.mixCoef{k,1} .* exp(g - max(g)) ./ sum(parm.mixCoef{k,1} .* exp(g - max(g)));%update posterior probability %cell(Kx1){TxI(K)}
            end
%             t_hat = min(t,buffer_size); t = 
            [Y, W, V, parm] =  binaural_emiva_update_multi(Y, X, W, V, k, t, parm,option);       
            
%             in_buffer(:,2:buffer_size,:) = in_buffer(:,1:buffer_size-1,:);
%             in_buffer(:,1,:) = in_buffer_perm(:,t,:);     in_num = min(t,buffer_size);  parm.in_num = in_num;
% %             if t > 1
% %             detect_range = round(0.3*F):round(0.5*F);
% %             pastf_energy = sum(sum(abs(in_buffer_perm(detect_range,t-1,:)).^2));
% %             presentf_energy = sum(sum(abs(in_buffer_perm(detect_range,t,:)).^2));
% %                if presentf_energy > 50*pastf_energy 
% %                    fprintf('%%%%%%sound detected%%%%%%\n');
% %                end
% %             end
%              [Y, W, V, parm] =  binaural_emiva_update_multi_Lb(flip(in_buffer(:,1:in_num,:),2),Y, X, W, V, k, t, parm,option);                       

    end
    J(k,1) = logJ(Y,k,T,F,beta,parm);     
end
    for f= 1:F
        tmp_cf(f,1) = - log(abs(det(W(:,:,f))));
    end
    CostFunction = sum(J) / T + sum(tmp_cf);
    disp(['      CostFunction = ',num2str(CostFunction)]); 
    if abs(pre_CostFunction - CostFunction) < 1e-8*F
        break;
    end
    pre_CostFunction = CostFunction ;
end
toc
%% Minimal distortion principle and Output
for f = 1:F
    W(:,:,f) = diag(diag(pinv(W(:,:,f))))*W(:,:,f); 
    Y(:,:,f) = W(:,:,f) * X(:,:,f); % (2) of [1] 
end
%% BackProjection
   X = permute(X,[3,2,1]);
   Y = backProjection(permute(Y,[3,2,1]) ,X(:,:,1));
   Y = permute(Y,[3,2,1]);
%% Re-synthesize the obtained source signals

for k=1:K
    s_est(k,:) = istft( squeeze(Y(k,:,:)).' , nn, win_size, inc)';
end
for k = 1:K
    label{k} = 'target';
end
end
%% CCMM 
function gc = logP(S,sigema_rou2,beta,F)
gc = F * (log(beta/2) + 0.5 * log(gamma(3/beta)) - 1.5 * log(gamma(1/beta))) -...
     0.5*sum(log(sigema_rou2)) - sum(abs(sqrt(gamma(3/beta) ./ gamma(1/beta) ./ (sigema_rou2 + eps)) .* S) .^ beta);% CGGMM PDF可更换？与[1]中PDF不一致
end
function J = logJ(Y,k,T,F,beta,parm)
   I = parm.I;
   J= -sum(sum(parm.q{k,1} .* repmat(log(parm.mixCoef{k,1}),T,1))) + F/2 * sum(sum(parm.q{k,1} .* log(parm.rou2{k,1}))) + 1/2 * sum(sum(parm.q{k,1} .* repmat(sum(log(parm.sigema2{k,1})),T,1)))...
          + sum(sum(parm.q{k,1} .* squeeze(permute(sum(abs(repmat(sqrt(gamma(3./beta{k,1}) ./ gamma(1./beta{k,1})),F,1,T) .* repmat(permute(abs(Y(k,:,:)),[3 1 2]),1,I(k),1) ./ repmat(sqrt(parm.sigema2{k,1}),1,1,T) ./ ...
          permute(repmat(sqrt(parm.rou2{k,1}),1,1,F),[3 2 1])).^ repmat(beta{k,1},F,1,T),1),[3 2 1]))));% (9) of [1]基本一致
end