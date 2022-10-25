function [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option)
global epsilon; global SubBlockSize; global SB_ov_Size;
epsilon = 1e-7; % 该值对结果影响较大，目前测试发现online情况下1e-7比较合理
parti = option.parti;
partisize = option.partisize;

%%
win_size = option.win_size;inc = win_size / 2;
fft_size = win_size;spec_coeff_num = fft_size / 2 + 1;
win_ana = option.win_ana;win_syn = option.win_syn;

%%
[mic_num, sample_num] = size(x);
in_frame_num = fix((sample_num - win_size) / inc) + 1;

buffer_size = option.Lb;
startFrame = 1;

out_delay = inc * buffer_size + 1;
out_frame_num = fix((sample_num + out_delay - win_size) / inc) + 1;
batch_size = out_frame_num;

s_est = zeros(source_num, sample_num + out_delay);

in_buffer = zeros(spec_coeff_num, buffer_size, mic_num);
in_buffer_batch = zeros(spec_coeff_num, batch_size, mic_num);
in_buffer_perm = zeros(spec_coeff_num, batch_size, mic_num);
out_buffer = zeros(spec_coeff_num, buffer_size, source_num);
out_buffer_batch = zeros(spec_coeff_num, batch_size, source_num);

%% 初始化
Ws = zeros(spec_coeff_num, mic_num, source_num);
for n = 1:spec_coeff_num
    Ws(n,1:source_num,:) = eye(source_num);
end
W_hat = Ws;
if mic_num > source_num
    W_hat = zeros(spec_coeff_num, mic_num, mic_num); % #freq * #mic * #mic
    W_hat(:, (source_num+1):mic_num, (source_num+1):mic_num) =...
        repmat(reshape(-eye(mic_num - source_num),[1,mic_num-source_num,mic_num-source_num]),[spec_coeff_num,1,1]);
    W_hat(:, :, 1:source_num) = Ws;
end
Ds = ones(spec_coeff_num, source_num);
if option.initial_rand == 0
    Vs = zeros(spec_coeff_num, mic_num, mic_num, source_num);
    for k = 1:source_num
        if option.prior(k) == 1 && option.DOA_init == 1
            hf = cal_RTF(spec_coeff_num,16000,option.mic_pos,option.theta(k,:));
            delta_f = option.deltaf;
            for n = 1:spec_coeff_num
                Vs(n,:,:,k) = (eye(mic_num)+hf(:,n)*hf(:,n)')/delta_f^2;
            end
        else
            Vss = repmat(eps * eye(mic_num), 1, 1, spec_coeff_num);
            Vs(:,:,:,k) = permute(Vss, [3 1 2]);
        end
    end

%     Vs = repmat(eps * eye(mic_num), 1, 1, spec_coeff_num, source_num);
%     Vs = permute(Vs, [3 1 2 4]);
    Cs = repmat(epsilon * eye(mic_num), 1, 1, spec_coeff_num); 
else
    diag_dom = epsilon * rand(mic_num);
    %dom_num = 100*epsilon;
    for i = 1:mic_num
        diag_dom(i,i) = sum(abs(diag_dom(i,:)));% + dom_num*rand(1);
    end
    Vs = repmat(diag_dom, 1, 1, spec_coeff_num, source_num);
    Cs = repmat(diag_dom, 1, 1, spec_coeff_num); 
    if option.initial_rand == 2
        Vs(:,:,1/3*spec_coeff_num:end,:) = repmat(eps * eye(mic_num), 1, 1, 2/3*spec_coeff_num+1, source_num);
    end
    Vs = permute(Vs, [3 1 2 4]);
end
Cs = permute(Cs, [3 1 2]);

if option.verbose
    fprintf('\n');
end

%%
n_orders = option.n_orders_online; delta = option.delta; mix_model = option.mix_model;
if mix_model == 2
    % 设定混合概率初值
    n_orders_1 = n_orders; n_orders_2 = n_orders;
    Param_1 = make_initial_guess_EM(n_orders_1); Param_2 = make_initial_guess_EM(n_orders_2); w1 = [Param_1.w1 Param_1.w2]; w2 = [Param_2.w1 Param_2.w2];
    G_func = {@(x) w1(1)*(x + delta).^n_orders_1(1)+w1(2)*(x+delta).^n_orders_1(2),@(x) w2(1)*(x+delta).^n_orders_2(1)+w2(2)*(x+delta).^n_orders_2(2)};
    dG_func = {@(x) w1(1)*n_orders_1(1)*(x+delta+eps).^(n_orders_1(1) - 1)+w1(2)*n_orders_1(2)*(x+delta+eps).^(n_orders_1(2) - 1), ...
        @(x) w2(1)*n_orders_2(1)*(x+delta+eps).^(n_orders_2(1)-1)+w2(2)*n_orders_2(2)*(x+delta+eps).^(n_orders_2(2)-1)};
else if mix_model == 1
        Param = make_initial_guess(n_orders); w1 = Param.w1; w2 = Param.w2;
        G_func = {@(x) w1(1)*(x + delta).^n_orders(1)+w1(2)*(x+delta).^n_orders(2),@(x) w2(1)*(x+delta).^n_orders(1)+w2(2)*(x+delta).^n_orders(2)};
        dG_func = {@(x) w1(1)*n_orders(1)*(x+delta+eps).^(n_orders(1) - 1)+w1(2)*n_orders(2)*(x+delta+eps).^(n_orders(2) - 1), ...
            @(x) w2(1)*n_orders(1)*(x+delta+eps).^(n_orders(1)-1)+w2(2)*n_orders(2)*(x+delta+eps).^(n_orders(2)-1)};
    else
        G_func = {@(x) (x + delta).^n_orders(1), @(x) (x + delta).^n_orders(2),@(x) (x + delta).^n_orders(3), @(x) (x + delta).^n_orders(4)};
        dG_func = {@(x) n_orders(1) * (x + delta + eps).^(n_orders(1) - 1), @(x) n_orders(2) * (x + delta + eps).^(n_orders(2) - 1)...
            ,@(x) n_orders(3) * (x + delta + eps).^(n_orders(3) - 1), @(x) n_orders(4) * (x + delta + eps).^(n_orders(4) - 1)};
    end
end
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

par1.num = length(partition_index);     par1.size = partition_size;     par1.index = partition_index;    par1.contrast = G_func{1};
par1.contrast_derivative = dG_func{1};

par2.num = length(partition_index);     par2.size = partition_size;    par2.index = partition_index;    par2.contrast = G_func{2};
par2.contrast_derivative = dG_func{2};

if option.mix_model == 0 % 对于不用混合模型的情况，可以加入源3和源4
    par3.num = length(partition_index);    par3.size = partition_size;    par3.index = partition_index;    par3.contrast = G_func{3};
    par3.contrast_derivative = dG_func{3};
    
    par4.num = length(partition_index);    par4.size = partition_size;    par4.index = partition_index;    par4.contrast = G_func{4};
    par4.contrast_derivative = dG_func{4};
    
    partition = {par1, par2, par3, par4};
else
    partition = {par1, par2};
end

global  frameNum;
%%
for n_frame = 1:batch_size
    win_sample_range = inc*(n_frame-1) + 1:min(inc*(n_frame-1) + win_size, sample_num);
    zero_padding_num = max(0, win_size - length(win_sample_range));
    
    xw = [x(:,win_sample_range) zeros(mic_num, zero_padding_num)] .* win_ana;
    Xw = fft(xw.', fft_size, 1);
    
    in_buffer_batch(:,n_frame,:) = Xw(1:spec_coeff_num,:);
end

if option.DOA_esti || ~option.mix_sim
    theta = doa_estimation(x(:,1:option.batch_update_num * spec_coeff_num).',option.esti_mic_dist,source_num,16000); % 仅用prebatch的信息更新DOA
    option.theta = theta*pi/180;
    if mic_num == 2
        option.mic_pos = [0,option.esti_mic_dist];%2mic
    else
        option.mic_pos = [0,option.esti_mic_dist,2*option.esti_mic_dist,3*option.esti_mic_dist];%4mic    
    end %Only support 2 or 4 mic
end

% batch更新Ws、Vs、Cs，获得更好的收敛速度
if option.pre_batch == 1 || option.pre_batch == 2
    % pre_batch = 1前batch_update_num帧prebatch(alpha = 1),第八帧开始online(alpha < 1);
    % pre_batch = 2前batch_update_num帧prebatch(alpha = 1),第1帧开始online(alpha < 1);
    batch_update_num = option.batch_update_num;
    Vs = zeros(spec_coeff_num, mic_num, mic_num, source_num); epsilon = 1e-32; 
    alpha = option.forgetting_fac; option.forgetting_fac = 1; option.online = 0;
    online_iter_num = option.iter_num; option.iter_num = option.prebatch_iter_num; 
    [out_buffer_batch,Ws,Ds,Vs,Cs,~,~,~] = binaural_auxiva_update_multi(in_buffer_batch(:,1:batch_update_num,:), ...
        Ws, Ds, Vs, Cs, W_hat, partition, option);
    option.forgetting_fac = alpha; option.online = 1;
    option.iter_num = online_iter_num;
    startFrame = batch_update_num + 1; epsilon = 1e-7;
    for n_frame = 1:batch_update_num
        Sw = squeeze(out_buffer_batch(:,n_frame,:)); Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
        s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) = s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
            + real(ifft(Sw, fft_size)).' .* win_syn;
    end
    if option.pre_batch == 2
        startFrame = 1;
    end
end

if option.pre_batch == 3
    % 前batch_update_num帧preonline(alpha < 1), 从头开始online(alpha < 1)
    batch_update_num = option.batch_update_num;
%     Vs = zeros(spec_coeff_num, mic_num, mic_num, source_num); epsilon = 1e-32;
    [~,Ws,Ds,Vs,Cs,~,~,~] = binaural_auxiva_update_multi(in_buffer_batch(:,1:batch_update_num,:), ...
        Ws, Ds, Vs, Cs, W_hat, partition, option);
    startFrame = 1; % epsilon = 1e-7;
end
%rng(0);
if option.perm
    perm = randperm(size(in_buffer_batch,2));
    in_buffer_perm = in_buffer_batch(:,perm,:);
else
    in_buffer_perm = in_buffer_batch;
end

for k = 1:source_num
    label{k} = 'target';
end

global OrderEst; global OutIter_Num; global GammaRatioThr; global GammaRatioSet; global GammaRatio;
detection=1;average_prenergy = 500;%% 设定语音帧的能量阈值
% if detection
% signalpower = squeeze(sum(abs(in_buffer_batch).^2,1));%frameflag=startFrame-1;
% % in_buffer_speech(:,1:frameflag,:) = in_buffer_batch(:,1:frameflag,:);
% [maxpo,maxloc]= max(signalpower,[],1);
% [minpo,minloc]= min(signalpower,[],1);
% if maxpo > 1*10^3 average_prenergy = sum((maxpo+minpo))/800;
% else average_prenergy = sum((maxpo+minpo))/4;end
% end
frameflag = 0;
if  OrderEst ==0
    for n_frame = startFrame:out_frame_num
        frameNum =n_frame;    in_buffer(:,2:buffer_size,:) = in_buffer(:,1:buffer_size-1,:);
%         in_buffer(:,1,:) = in_buffer_perm(:,n_frame,:);  
        in_num = min(n_frame,buffer_size);
        if n_frame>1
%             detect_range = round(0.3*spec_coeff_num):round(0.5*spec_coeff_num);
%             pastf_energy = sum(sum(abs(in_buffer_perm(detect_range,n_frame-1,:)).^2));
%             presentf_energy = sum(sum(abs(in_buffer_perm(detect_range,n_frame,:)).^2));
%             if presentf_energy > 50*pastf_energy 
%                fprintf('%%%%%%sound detected%%%%%%\n');
%             end
        if detection && n_frame>startFrame% 每次输入的数据是当前帧+（buffer_size-2）个语音帧
            in_buffer(:,1,:)=in_buffer_perm(:,n_frame,:);%输入当前帧作为inbuffer的第一帧
            detect_range1 = 1:round(0.25*spec_coeff_num);detect_range2=round(0.75*spec_coeff_num):spec_coeff_num;
            front_energy = sum(sum(abs(in_buffer_perm(detect_range1,n_frame-1,:)).^2));
            back_energy = sum(sum(abs(in_buffer_perm(detect_range2,n_frame-1,:)).^2));% 上一帧帧内低频与高频的能量
            currentframe_energy = sum(sum(abs(in_buffer_perm(:,n_frame-1,:)).^2));% 上一帧的总能量
            if front_energy > 50*back_energy  && currentframe_energy > average_prenergy
               fprintf('%%%%%%previous sound detected%%%%%%\n');               
            else 
                in_buffer(:,2:end-1,:)= in_buffer(:,3:end,:);%如果上一帧不是语音帧就将上一帧数据抹除不作为先验
                in_buffer(:,end,:)= zeros(spec_coeff_num,mic_num);
            end
        else in_buffer(:,1,:) = in_buffer_perm(:,n_frame,:);  
        end

        end
       
        %     if n_frame == 1
        %         alpha = option.forgetting_fac; option.forgetting_fac = 1;
        %     end
        [out_buffer, Ws, Ds, Vs, Cs, W_hat, W_res, obj_vals] = binaural_auxiva_update_multi(flip(in_buffer(:,1:in_num,:),2),Ws, Ds, Vs, Cs, W_hat, partition ,option);        
        %     if n_frame == 1
        %         option.forgetting_fac = alpha;
        %     end
        fprintf('frame no = %d/%d, obj = %.4e\n', ...
            n_frame, in_frame_num, obj_vals(end));
%         fprintf('frame no = %d/%d\n',  n_frame, in_frame_num);
        dd =squeeze(out_buffer(:,in_num,:));
        if sum(abs(dd(:,1)).^2)/sum(abs(dd(:,2)).^2)<1/GammaRatioThr
             GammaRatio(1) = GammaRatioSet;
        end
        if sum(abs(dd(:,2)).^2)/sum(abs(dd(:,1)).^2)<1/GammaRatioThr
             GammaRatio(2) = GammaRatioSet;
        end
        if option.perm
            for s = 1:source_num
                for n = 1:spec_coeff_num
                    out_buffer(n,:,s) = W_res(n,:,s) * (squeeze(in_buffer_batch(n,n_frame+1-in_num:n_frame,:)));
                end
            end
        end
        
        % 混合CGDD模型EM估计，目前是无EM硬判+递归平均更新方式
        if mix_model == 1
            ita = option.ita;
            % 硬判决混合概率
            w1_c = calc_mix_prob(dd(:,1)',n_orders,w1); % 代表source1的一对混合概率
            w2_c = calc_mix_prob(dd(:,2)',n_orders,w2); % 代表source2的一对混合概率
            % 递归更新混合概率
            w1(1) = (1-ita)*w1(1) + ita*w1_c(1); w1(2) = (1-ita)*w1(2) + ita*w1_c(2);
            w2(1) = (1-ita)*w2(1) + ita*w2_c(1); w2(2) = (1-ita)*w2(2) + ita*w2_c(2);
            % 更新contrast function
            G_func = {@(x) w1(1)*(x + delta).^n_orders(1)+w1(2)*(x+delta).^n_orders(2),@(x) w2(1)*(x+delta).^n_orders(1)+w2(2)*(x+delta).^n_orders(2)};
            dG_func = {@(x) w1(1)*n_orders(1)*(x+delta+eps).^(n_orders(1) - 1)+w1(2)*n_orders(2)*(x+delta+eps).^(n_orders(2) - 1), ...
                @(x) w2(1)*n_orders(1)*(x+delta+eps).^(n_orders(1)-1)+w2(2)*n_orders(2)*(x+delta+eps).^(n_orders(2)-1)};
            partition{1}.contrast = G_func{1}; partition{1}.contrast_derivative = dG_func{1};
            partition{2}.contrast = G_func{2}; partition{2}.contrast_derivative = dG_func{2};
            if n_frame == out_frame_num
                % 在最后一帧查看最终的contrast function
                string1 = [num2str(w1(1)) '*(x)^' num2str(n_orders(1)) '+' num2str(w1(2)) '*(x)^' num2str(n_orders(2))];
                string2 = [num2str(w2(1)) '*(x)^' num2str(n_orders(1)) '+' num2str(w2(2)) '*(x)^' num2str(n_orders(2))];
                fprintf('G_func(1)=%s\nG_func(2)=%s\n',string1,string2);
            end
        else if mix_model == 2
                ita = 0.1;
                % EM算法，每个频点假设独立分布
                Data_1 = [dd(:,1) zeros(size(dd,1),1)];
                [~, Param_1] = EM(Data_1, Param_1);
                w1_c = [Param_1.w1 Param_1.w2];
                Data_2 = [dd(:,2) zeros(size(dd,1),1)];
                [~, Param_2] = EM(Data_2, Param_2);
                w2_c = [Param_1.w1 Param_2.w2];
                % 递归更新混合概率
                w1 = (1-ita)*w1 + ita*w1_c; w2 = (1-ita)*w2 + ita*w2_c;
                n_orders_1 = (1-ita)*n_orders_1 + ita*Param_1.n_orders;
                n_orders_2 = (1-ita)*n_orders_2 + ita*Param_2.n_orders;
                % 更新contrast function
                G_func = {@(x) w1(1)*(x + delta).^n_orders_1(1)+w1(2)*(x+delta).^n_orders_1(2),@(x) w2(1)*(x+delta).^n_orders_2(1)+w2(2)*(x+delta).^n_orders_2(2)};
                dG_func = {@(x) w1(1)*n_orders_1(1)*(x+delta+eps).^(n_orders_1(1) - 1)+w1(2)*n_orders_1(2)*(x+delta+eps).^(n_orders_1(2) - 1), ...
                    @(x) w2(1)*n_orders_2(1)*(x+delta+eps).^(n_orders_2(1)-1)+w2(2)*n_orders_2(2)*(x+delta+eps).^(n_orders_2(2)-1)};
                partition{1}.contrast = G_func{1}; partition{1}.contrast_derivative = dG_func{1};
                partiton{2}.contrast = G_func{2}; partition{2}.contrast_derivative = dG_func{2};
                if n_frame == out_frame_num
                    % 在最后一帧查看最终的contrast function
                    string1 = [num2str(w1(1)) '*(x)^' num2str(n_orders_1(1)) '+' num2str(w1(2)) '*(x)^' num2str(n_orders_1(2))];
                    string2 = [num2str(w2(1)) '*(x)^' num2str(n_orders_2(1)) '+' num2str(w2(2)) '*(x)^' num2str(n_orders_2(2))];
                    fprintf('G_func(1)=%s\nG_func(2)=%s\n',string1,string2);
                end
            end
        end
%       
        Sw = squeeze(out_buffer(:,in_num,:));   Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
        s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) =  s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
            + real(ifft(Sw, fft_size)).' .* win_syn;
        dd= real(ifft(Sw, fft_size)).';
    end
end

if  OrderEst == 1
    global order_gamma;  global OrderEstUppThr; global OrderEstLowThr;
    %order_gamma =1;
    for n_frame = startFrame:out_frame_num
        frameNum = n_frame;    in_buffer(:,2:buffer_size,:) = in_buffer(:,1:buffer_size-1,:);
        in_buffer(:,1,:) = in_buffer_perm(:,n_frame,:);     in_num = min(n_frame,buffer_size);
        for OutIter = 1: OutIter_Num
            %     if n_frame == 1
            %         alpha = option.forgetting_fac; option.forgetting_fac = 1;
            %     end
            [out_buffer, Ws, Ds, Vs, Cs, W_hat, W_res,~] = binaural_auxiva_update_multi(flip(in_buffer(:,1:in_num,:),2),Ws, Ds, Vs, Cs, W_hat, partition ,option);
            %     if n_frame == 1
            %         option.forgetting_fac = alpha;
            %     end
            %     fprintf('frame no = %d/%d, obj = %.4e\n', ...
            % 	    n_frame, in_frame_num, obj_vals(end));
            if option.perm
                for s = 1:source_num
                    for n = 1:spec_coeff_num
                        out_buffer(n,:,s) = W_res(n,:,s) * (squeeze(in_buffer_batch(n,n_frame+1-in_num:n_frame,:)));
                    end
                end
            end
            dd =squeeze(out_buffer(:,in_num,:));  %[c1(OutIter),besterr] = estimateCGGDNewton(dd(:,1)); [c2(OutIter),besterr] = estimateCGGDNewton(dd(:,2));
            [c1(OutIter)] = estimateCGGDNewton_1(dd(:,1)); [c2(OutIter)] = estimateCGGDNewton_1(dd(:,2));
            %n_orders = option.n_orders_online;
            %         c1(OutIter)
            %         c2(OutIter) OrderEstUppThr; global OrderEstLowThr;
%             figure;  plot(abs(dd(:,1))); figure;  plot(abs(dd(:,2)));
            if c1(OutIter)>2*OrderEstUppThr  c1(OutIter) =2*OrderEstUppThr;  end
            if c2(OutIter)>2*OrderEstUppThr  c2(OutIter) =2*OrderEstUppThr;  end
            if c1(OutIter)<2*OrderEstLowThr  c1(OutIter) =2*OrderEstLowThr ;  end
            if c2(OutIter)<2*OrderEstLowThr  c2(OutIter) =2*OrderEstLowThr;  end           
            n_orderstmp = [c1(OutIter)/2 c2(OutIter)/2 ];
            n_orders = n_orderstmp * order_gamma + n_orders * (1-order_gamma);
            
            G_func = {@(x) x.^n_orders(1), @(x) x.^n_orders(2),@(x) x.^n_orders(3), @(x) x.^n_orders(4)};
            dG_func = {@(x) n_orders(1) * (x + delta + eps).^(n_orders(1) - 1), @(x) n_orders(2) * (x + delta + eps).^(n_orders(2) - 1)...
                ,@(x) n_orders(3) * (x + delta + eps).^(n_orders(3) - 1), @(x) n_orders(4) * (x + delta + eps).^(n_orders(4) - 1)};
        end
        fprintf('frame no = %d/%d\n',  n_frame, in_frame_num);
        %      [alpha_hat1, sigma1]=ggd_GCM(real(dd(:,1)),5)
        %       [alpha_hat2, sigma2]=ggd_GCM(real(dd(:,2)),5)
        %     [alpha_hat2, sigma1]=ggd_GCM(imag(dd(:,1)),5)
        Sw = squeeze(out_buffer(:,in_num,:));   Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
        s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) =  s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
            + real(ifft(Sw, fft_size)).' .* win_syn;
        dd= real(ifft(Sw, fft_size)).';
    end
    if n_frame == out_frame_num
       fprintf('orders1 = %f, orders2 = %f',n_orders(1),n_orders(2)); 
    end
end

    
% for s = 1:source_num
%     for n = 1:spec_coeff_num
%         out_buffer_batch(n,:,s) = Ws(n,:,s) * (squeeze(in_buffer_batch(n,:,:))).';
%     end
% end


% for n_frame = 1:batch_size
%     Sw = squeeze(out_buffer_batch(:,n_frame,:));
%     Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
%     
%     s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) = ...
%         s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
%         + real(ifft(Sw, fft_size)).' .* win_syn;
% end
