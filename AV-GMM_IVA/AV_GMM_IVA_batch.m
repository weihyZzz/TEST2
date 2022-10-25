function [s_est, label,W, parm,loop,CostFunction,obj_set] = AV_GMM_IVA_batch(x, option)
%input: 
% x: mixed data, K x nn
% nfft: fft point
% I: mixture state
% max_iterations
% Ref.[1] Speech Separation Using Independent Vector Analysis with an Amplitude Variable Gaussian Mixture Model
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
% X = whitening(permute(X,[3,2,1]), nmic);
% X = permute(X,[3,2,1]);
%% Initialization
max_iterations = option.AVGMMIVA_max_iter;
% [X,~,~] = whitening( X , nmic);
[K,T,F] = size(X);
% % de-mean
% for f= 1:F
%     X(:,:,f) = bsxfun(@minus,X_init(:,:,f), mean(X_init(:,:,f),2));
% end
Y = X;
W = zeros(K,K,F); 
M=K;
for f = 1:F
    W(:,:,f) = eye(K);
end
I = 2;
parm.mixCoef = zeros(K,I);
for k = 1:K
    parm.mixCoef(k,:) = rand(1,I);
    parm.mixCoef(k,:) = parm.mixCoef(k,:) ./ sum(parm.mixCoef(k,:));
end
parm.q = zeros(T,I,K);
parm.vfd = 0.0001*ones(F,I,K);
parm.ht = 0.0001*ones(T,K);
parm.vh = [0.0001*ones(T,1,F,K) 0.0002*ones(T,1,F,K)];
parm.Mf = zeros(I,K);
%% prewhite
for f = 1:F
    Cf(f,:) = mean(X(:,:,f) * X(:,:,f)',1);
    for k = 1:K
        X(k,:,f) = Cf(f,k).^(-0.5).* X(k,:,f);
    end
end

%% 未修改：
epsi = 1e-8;
J = zeros(K,1);
tmp_cf = zeros(F,1);
g = zeros(1,K);
pre_CostFunction = 0;
obj_set = [];

%% Iterate
tic
for loop = 1 : max_iterations
    %% E-STEP
    for k = 1 : K

        clear g;
        for t = 1 : T
            for i = 1 :I
                g(1,i) = logP(squeeze(Y(k,t,:)), squeeze(parm.vh(t,i,:,k)), F);              
            end
            if sum(isnan(g)==1)>0                
            else
                parm.q(t,:,k) =  parm.mixCoef(k,:) .* exp(g - max(g)) ./ sum(parm.mixCoef(k,:) .* exp(g - max(g)));% (11) of [1] update posterior probability %\{TxIxK}
            end
            parm.ht(t,k) = F ./ sum(parm.q(t,:,k) * parm.vfd(:,:,k)' * squeeze((abs(Y(k,t,:)).^2)));% (17) of [1]
        end
        parm.mixCoef(k,:) = mean(parm.q(:,:,k))+epsi;
        for f = 1 : F
%             for i = 1 : I
%                 parm.vfd(f,i,k) = sum(parm.q(:,i,k),1) ./ sum(sum(parm.q(:,i,k) .* parm.ht(:,k) * (abs(Y(k,:,f)).^2)));
%                 parm.vh(:,i,f,k) = parm.vfd(f,i,k) .* parm.ht(:,k);%(TxIxFxK)
%                 temp_vh = permute(repmat(squeeze(parm.vh(:,1,f,:)-parm.vh(:,2,f,:)),1,1,I),[1,3,2]);         
%                 parm.Mf(i,:) = sum(squeeze(parm.q(:,i,:)) .* squeeze(temp_vh(:,i,:)) * X(:,:,f) * X(:,:,f)',1);
%             end
            parm.vfd(f,:,k) = sum(parm.q(:,:,k),1) ./ sum(sum(parm.q(:,:,k)' * parm.ht(:,k) * (abs(Y(k,:,f)).^2)));% (16) of [1]
            parm.vh(:,:,f,k) =  parm.ht(:,k) * parm.vfd(f,:,k);%(TxIxFxK)
            temp_vh = squeeze(parm.vh(:,1,f,:)-parm.vh(:,2,f,:)); % first element in the matrix in (13) of [1]         
            for i = 1:I parm.Mf(i,:) = sum(squeeze(parm.q(:,i,:)) .* temp_vh * X(:,:,f) * X(:,:,f)',1);end
            Mf = parm.Mf;
            lamada = (Mf(1,1)+Mf(2,2))/2-sqrt((Mf(1,1)-Mf(2,2))^2/4+abs(Mf(1,2))^2);% 求特征值 under (15) of [1]
            [eigvec ,eigval] = eig(parm.Mf);% 求特征值 under (15) of [1]
            [~ ,num] = min(diag(eigval));
            w = eigvec(:,num); 
            W(k,:,f) = w';                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               %% update output
            Y(k,:,f) = w' * X(:,:,f);
%             Y(:,:,f) = diag(diag(Cf(f,k) .^(0.5) .* pinv(W(:,:,f)))) * Y(:,:,f) ;% post-whitening
            cost_temp1(f) = sum(sum(squeeze(parm.q(:,1,k)) .* temp_vh(:,k) * Y(k,:,f) * Y(k,:,f)'))+sum(sum(squeeze(parm.q(:,2,k)) .* temp_vh(:,k) * Y(k,:,f) * Y(k,:,f)'));
            cost_temp2(f) = lamada .* w' * w;
        end
       
       %% the cost function
       J(k) = -sum(cost_temp1) + sum(cost_temp2); 
    end
    CostFunction = abs(sum(J)) / T / F;
    disp(['The iter = ',num2str(loop),'/',num2str(max_iterations),'   obj = ',num2str(CostFunction)]);    
    obj_set = [obj_set,CostFunction];
end
toc
%% Minimal distortion principle and Output

for f = 1:F
    Cf_temp = diag(diag(repmat(Cf(f,:),2,1)));
    W(:,:,f) = diag(diag( Cf_temp .^(0.5) * pinv(W(:,:,f))));%*W(:,:,f); 
    Y(:,:,f) = W(:,:,f) * X(:,:,f);
end

% X = permute( X,[3,2,1]);
% Y = backProjection(permute(Y,[3,2,1]) ,X(:,:,2));
% Y = permute(Y,[3,2,1]);
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
function gc = logP(S,vh,F,option) 
    gc = - F * log(2*pi) - sum(log(vh)) - 0.5 * sum(abs( S ./ sqrt(vh )) .^ 2);
end