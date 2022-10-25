function [s_hat, label,W, parm,loop] = CGGMM_IVA_online_revise(x, option)
%input: 
% x: mixed data, K x nn
% nfft: fft point
% I: mixture state
% max_iterations
% beta: the shape paramater matrix, K x I
% ����������״̬�����Բ�ͬ, �������Ը��ӶȱȽϸߣ� ��������2020/10/15
% Ref.[1] Independent Vector Analysis for Blind Speech Separation Using Complex Generalized Gaussian Mixture Model with Weighted Variance
%% 
global epsilon; global SubBlockSize; global SB_ov_Size;
epsilon = 1e-7; % ��ֵ�Խ��Ӱ��ϴ�Ŀǰ���Է���online�����1e-7�ȽϺ���
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
% f = (0:size(X,2)*size(X,3)-1)*16000/size(X,2)*size(X,3);
% figure;
% plot(f,abs(reshape(X(2,:,:),[1,size(X,2)*size(X,3)])));

if option.annealing % �Ƿ�ʹ���˻�����
    fac_a = max(0.5-iter/iter_num, 0); % annealing factor
else
    fac_a = 1; % =1����Ч�ڲ��˻�
end
option.fac_a = fac_a;
%% Initialization
pmulti = option.pmulti;
detect_low = option.detect_low;
detect_up = option.detect_up;
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
parti = option.parti;
partisize = option.partisize;

beta = option.EMIVA_beta;
max_iterations = option.EMIVA_max_iter;
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

g = zeros(1,K);
%% �ӿ���index
if parti  % �ӿ��㷨��ʼ��
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

if option.mix_model == 0 % ���ڲ��û��ģ�͵���������Լ���Դ3��Դ4
    par3.num = length(partition_index);    par3.size = partition_size;    par3.index = partition_index;    par3.contrast = {'blank'};
    par3.contrast_derivative =  {'blank'};
    
    par4.num = length(partition_index);    par4.size = partition_size;    par4.index = partition_index;    par4.contrast = {'blank'};
    par4.contrast_derivative =  {'blank'};
    
    partition = {par1, par2, par3, par4};
else
    partition = {par1, par2};
end
%% batch_size init
batch_update_num = option.batch_update_num; % option
%% DOA
if option.DOA_esti 
    x1=x.';
    theta = doa_estimation(x1(1:batch_update_num * spec_coeff_num,:),option.esti_mic_dist,K,16000); % ����prebatch����Ϣ����DOA
    option.theta = theta * pi/180;
    if M == 2 option.mic_pos = [0,option.esti_mic_dist];%2mic
    else option.mic_pos = [0,option.esti_mic_dist,2*option.esti_mic_dist,3*option.esti_mic_dist];end %Only support 2 or 4 mic
    for k = 1 : K
        if option.prior(k) == 1
            hf_temp(:,:,k) = cal_RTF(F,16000,option.mic_pos,option.theta(:,k));
            option.hf = hf_temp;
        end
    end
end
%% V init
if option.initial_rand == 0 
    V = zeros(K, M, M, F);% batch
elseif option.initial_rand == 1
    diag_dom = epsilon * rand(M);
    %dom_num = 100*epsilon;
    for i = 1:M
        diag_dom(i,i) = sum(abs(diag_dom(i,:)));% + dom_num*rand(1);
    end
    V = repmat(diag_dom, 1, 1, spec_coeff_num, K);
    if option.initial_rand == 2
        V(:,:,1/3*spec_coeff_num:end,:) = repmat(eps * eye(M), 1, 1, 2/3*spec_coeff_num+1, K);
    end
    V = permute(V, [3 1 2 4]);
    V = permute(V, [4 2 3 1]);
elseif option.initial_rand == 2
    V = zeros(spec_coeff_num, M, M, K);
    for k = 1:K
        if option.prior(k) == 1 && option.DOA_init == 1
            hf = option.hf;
            delta_f = option.deltaf;
            for n = 1:spec_coeff_num
                V(n,:,:,k) = (eye(M)+hf(:,n)*hf(:,n)')/delta_f^2;
            end
        else
            Vss = repmat(eps * eye(M), 1, 1, spec_coeff_num);
            V(:,:,:,k) = permute(Vss, [3 1 2]);            
        end
    end
    V = permute(V, [4 2 3 1]);
end

%% Iterate
tmp_beta = cellfun(@(x) x .* abs(gamma(3./x) ./ gamma(1./x)) .^ (x./2) , beta ,'UniformOutput',false);
parm.tmp_beta = tmp_beta ;
%% batch����W��V����ø��õ������ٶ�
if option.pre_batch == 1 || option.pre_batch == 2
    % �����س�ʼ��V���ڶ��ַ�ʽδ��ɣ�Ϊ��auxivaͳһ���ó������ʽ
    % ǰbatch_update_num֡prebatch( EMIVA_ratio = 1)
    option.online = 0;
    for loop = 1 : max_iterations   
    %% E-STEP
    for k = 1 : K
        clear g;
        for t = 1 : batch_update_num
            for i = 1 :I(k)
                c_temp = zeros(F,1);for f=1:size(c_temp,1) c_temp(f)=parm.sigema_rou2{k,f}(t,i);end % ���ּ��㷽ʽ����
                g(1,i) = logP(squeeze(Y(k,t,:)), c_temp, beta{k,1}(1,i), F, option);              
%                 g(1,i) = logP(squeeze(Y(k,t,:)), cellfun(@(c)c(t,i),parm.sigema_rou2(k,:)).', beta{k,1}(1,i), F);              
            end
            if sum(isnan(g)==1)>0
                
            else
                parm.q{k,1}(t,:) =  parm.mixCoef{k,1} .* exp(g - max(g)) ./ sum(parm.mixCoef{k,1} .* exp(g - max(g)));%update posterior probability %cell(Kx1){TxI(K)}
            end
           %% S-STEP
            parm.rou2{k,1}(t,:) = ((tmp_beta{k,1} .* sum(abs( abs(repmat(squeeze(Y(k,t,:)),1,I(k))) .* sqrt(1 ./ parm.sigema2{k,1})) .^ repmat(beta{k,1},F,1),1)) ./ F) .^ (2./beta{k,1}) + eps;%{Kx1}(TxI) % (12) of [1] 
        end
        parm.mixCoef{k,1} = mean(parm.q{k,1}(1 : batch_update_num,:))+epsi;
        [Y, W, V, parm,CostFunc] =  emiva_update_offline(Y, X, W, V, k, t, partition, parm,option);       
    end
%     fprintf('\b\b\b\b%4d', loop);
    disp(['The iter = ',num2str(loop),'/',num2str(max_iterations),'   obj = ',num2str(CostFunc)]);    
    end
    option.online = 1;
end
if option.pre_batch == 3 %% ���ز�ͬ��ʼ��V
% ǰbatch_update_num֡( EMIVA_ratio < 1����ӳ�ʼ��V�ĳ�ʼ��ƽ�� V =EMIVA_ratio.*V + (1-EMIVA_ratio).* V_init)
     for loop = 1 : max_iterations   
    %% E-STEP
    for k = 1 : K
        clear g;
        for t = 1 : batch_update_num
            for i = 1 :I(k)
                c_temp = zeros(F,1);for f=1:size(c_temp,1) c_temp(f)=parm.sigema_rou2{k,f}(t,i);end % ���ּ��㷽ʽ����
                g(1,i) = logP(squeeze(Y(k,t,:)), c_temp, beta{k,1}(1,i), F, option);              
%                 g(1,i) = logP(squeeze(Y(k,t,:)), cellfun(@(c)c(t,i),parm.sigema_rou2(k,:)).', beta{k,1}(1,i), F);              
            end
            if sum(isnan(g)==1)>0
                
            else
                parm.q{k,1}(t,:) =  parm.mixCoef{k,1} .* exp(g - max(g)) ./ sum(parm.mixCoef{k,1} .* exp(g - max(g)));%update posterior probability %cell(Kx1){TxI(K)}
            end
           %% S-STEP
            parm.rou2{k,1}(t,:) = ((tmp_beta{k,1} .* sum(abs( abs(repmat(squeeze(Y(k,t,:)),1,I(k))) .* sqrt(1 ./ parm.sigema2{k,1})) .^ repmat(beta{k,1},F,1),1)) ./ F) .^ (2./beta{k,1}) + eps;%{Kx1}(TxI) % (12) of [1] 
        end
        parm.mixCoef{k,1} = mean(parm.q{k,1}(1 : batch_update_num,:))+epsi;
        [Y, W, V, parm,CostFunc] =  emiva_update_offline(Y, X, W, V, k, t, partition, parm,option);       
    end
%     fprintf('\b\b\b\b%4d', loop);
    disp(['The iter = ',num2str(loop),'/',num2str(max_iterations),'   obj = ',num2str(CostFunc)]);    
     end
end
    disp(['\n','Prebatch process done!']); 
%% %%%%%%%%%%%%Online processing%%%%%%%%%%%%%%%%%%%%%
% fprintf('Frame no:    ');
if option.DOA_esti_online % ���߸���theta 
    theta_set = [];N_interval = 8;% DOA���������֡����16K����������£�Լ8֡Ϊһ�룻
    for doa_frame = batch_update_num+1:N_interval:fix(size(x1,1)/2048) % ÿ��N_interval֡��һ��DOA���ƣ�У���Ƕȣ�
    theta_temp = doa_estimation(x1((doa_frame-N_interval) * spec_coeff_num+1:doa_frame * spec_coeff_num,:),option.esti_mic_dist,K,16000); % ����prebatch����Ϣ����DOA
    theta = 0.6 * theta + 0.4 * theta_temp; % ����һ�εĹ��ƽǶ���һ������ƽ����
    option.theta = theta * pi/180; theta_set = [theta_set;theta];end % ��������theta�����
end

    for t = batch_update_num + 1 : T 
    if option.DOA_esti_online % ���߸���hf��Ȩ
        if mod(t-batch_update_num-1,N_interval) == 0 && t-batch_update_num-1 ~=0 && t<T% ÿ��N_interval֡��һ��DOA���Ƽ�Ȩ��
        for k = 1 : K
            if option.prior(k) == 1
                hf_temp(:,:,k) = cal_RTF(F,16000,option.mic_pos,theta_set((t-batch_update_num-1)/N_interval,k));
            end
        end
        option.hf = 0.4 * option.hf + 0.6 * hf_temp;end % ����һ�����Ľ���󻬶�ƽ����
    end 

            in_num = min(t,buffer_size);  parm.in_num = in_num; %in_buffer(:,2:buffer_size,:) = in_buffer(:,1:buffer_size-1,:);
%             in_buffer(:,1,:) = Y(:,t,:);% ��Ƶ��֡����Y,�����ø��º��Y����
%% Detection
           detection=1;average_prenergy = 500;%% �趨����֡��������ֵ
        if detection && t>batch_update_num + 1 && buffer_size>3% ÿ������������ǵ�ǰ֡+��buffer_size-2��������֡
            in_buffer(:,1,:)=in_buffer_perm(:,t,:);%���뵱ǰ֡��Ϊinbuffer�ĵ�һ֡
            detect_range1 = 1:round(0.25*spec_coeff_num);detect_range2=round(0.75*spec_coeff_num):spec_coeff_num;
            front_energy = sum(sum(abs(in_buffer_perm(:,t-1,detect_range1)).^2));
            back_energy = sum(sum(abs(in_buffer_perm(:,t-1,detect_range2)).^2));% ��һ֡֡�ڵ�Ƶ���Ƶ������
            currentframe_energy = sum(sum(abs(in_buffer_perm(:,t-1,:)).^2));% ��һ֡��������
            if front_energy > 50*back_energy  && currentframe_energy > average_prenergy
               fprintf('%%%%%%previous sound detected%%%%%%\n');               
            else 
                in_buffer(:,2:end-1,:)= in_buffer(:,3:end,:);%�����һ֡��������֡�ͽ���һ֡����Ĩ������Ϊ����
                in_buffer(:,end,:)= zeros(M,spec_coeff_num);
            end
        else in_buffer(:,1:in_num,:) = flip(Y(:,t - in_num+ 1 : t,:),2);%in_buffer(:,1,:) = Y(:,t,:);  
        end

            %% online������
            [Y, W, V, parm, CostFunc] = emiva_update_online(flip(in_buffer(:,1:in_num,:),2),Y, X, W, V, t, partition, parm,option);                       
            disp(['frame no = ',num2str(t),'/',num2str(T),'  obj = ',num2str(CostFunc)]);          
% fprintf('\b\b\b\b%4d', t);
    end
%     figure;
% subplot(1,3,1);plot(0:size(pastf_energy_set,2)-1,pastf_energy_set);grid on 
% title('pastf energy');
% subplot(1,3,2);plot(0:size(presentf_energy_set,2)-1,presentf_energy_set);grid on 
% title('presentf energy');
% subplot(1,3,3);plot(0:size(presentf_energy_set,2)-1,pastf_energy_set./presentf_energy_set);grid on 
% title('pastf/presentf');
% disp(['\nOnline process done!']);
%% Minimal distortion principle and Output
for f = 1:F
    W(:,:,f) = diag(diag(pinv(W(:,:,f))))*W(:,:,f); 
    Y(:,:,f) = W(:,:,f) * X(:,:,f); % (2) of [1] 
end
%% BackProjection
%    X = permute(X,[3,2,1]);
%    Y = backProjection(permute(Y,[3,2,1]) ,X(:,:,1));
%    Y = permute(Y,[3,2,1]);
%% Re-synthesize the obtained source signals

for k=1:K
    s_est(k,:) = istft( squeeze(Y(k,:,:)).' , nn, win_size, inc)';
end
label = cell(k,1);
for k = 1:K
    label{k} = 'target';
end
%% �Զ����뵥·����
if option.singleout
    mean_s_est = mean(s_est,2);[data,loc]=max(mean_s_est);
    s_hat = s_est(loc,:);
else
    s_hat = s_est;
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
end
