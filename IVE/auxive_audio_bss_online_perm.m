function [s_est,label] = auxive_audio_bss_online_perm(x,option)
global epsilon; global SubBlockSize; global SB_ov_Size;
epsilon = 1e-7; % 该值对结果影响较大，目前测试发现online情况下1e-7比较合理
parti = option.parti;
partisize = option.partisize;
source_num = 1;
%%
win_size = option.win_size;
inc = win_size / 2;
fft_size = win_size;
spec_coeff_num = fft_size / 2 + 1;

win_ana = option.win_ana;
win_syn = option.win_syn;

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
out_buffer = zeros(spec_coeff_num, buffer_size);
out_buffer_batch = zeros(spec_coeff_num, batch_size);

%% 初始化
Ws = zeros(spec_coeff_num, mic_num);
for n = 1:spec_coeff_num
    Ws(n,1,:) = 1;
end
W_hat = zeros(spec_coeff_num, mic_num, mic_num); % #freq * #mic * #mic
W_hat(:, 2:mic_num, 2:mic_num) =...
    repmat(reshape(-eye(mic_num - 1),[1, mic_num-1, mic_num-1]),[spec_coeff_num,1,1]);
W_hat(:, :, 1) = Ws;

Ds = ones(spec_coeff_num, 1);
if option.initial_rand == 0
    Vs = zeros(spec_coeff_num, mic_num, mic_num);
    if option.prior(1) == 1 && option.DOA_init == 1
        hf = cal_RTF(spec_coeff_num,16000,option.mic_pos,option.theta(1,:));
        delta_f = option.deltaf;
        for n = 1:spec_coeff_num
            Vs(n,:,:) = (eye(mic_num)+hf(:,n)*hf(:,n)')/delta_f^2;
        end
    else
        Vss = repmat(eps * eye(mic_num), 1, 1, spec_coeff_num);
        Vs(:,:,:) = permute(Vss, [3 1 2]);
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
    Vs = repmat(diag_dom, 1, 1, spec_coeff_num);
    Cs = repmat(diag_dom, 1, 1, spec_coeff_num); 
    if option.initial_rand == 2
        Vs(:,:,1/3*spec_coeff_num:end,:) = repmat(eps * eye(mic_num), 1, 1, 2/3*spec_coeff_num+1);
    end
    Vs = permute(Vs, [3 1 2]);
end
Cs = permute(Cs, [3 1 2]);

if option.verbose
    fprintf('\n');
end

%%
n_orders = option.n_orders_online; delta = option.delta;
G_func = {@(x) (x + delta).^n_orders(1)};
dG_func = {@(x) n_orders(1) * (x + delta + eps).^(n_orders(1) - 1)};

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
partition = {par1};

global  frameNum;
%%
for n_frame = 1:batch_size
    win_sample_range = inc*(n_frame-1) + 1:min(inc*(n_frame-1) + win_size, sample_num);
    zero_padding_num = max(0, win_size - length(win_sample_range));
    
    xw = [x(:,win_sample_range) zeros(mic_num, zero_padding_num)] .* win_ana;
    Xw = fft(xw.', fft_size, 1);
    
    in_buffer_batch(:,n_frame,:) = Xw(1:spec_coeff_num,:);
end

% batch更新Ws、Vs、Cs，获得更好的收敛速度
if option.pre_batch == 1 || option.pre_batch == 2
    % 前七帧prebatch(alpha = 1),第八帧开始online(alpha < 1)
    batch_update_num = option.batch_update_num;
    Vs = zeros(spec_coeff_num, mic_num, mic_num, source_num); epsilon = 1e-32; 
    alpha = option.forgetting_fac; option.forgetting_fac = 1; option.online = 0;
    [out_buffer_batch,Ws,Ds,Vs,Cs,~,~,~] = binaural_auxive_update_multi(in_buffer_batch(:,1:batch_update_num,:), ...
        Ws, Ds, Vs, Cs, W_hat, partition, option);
    option.forgetting_fac = alpha; option.online = 1;
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
    % 前七帧preonline(alpha < 1), 从头开始online(alpha < 1)
    batch_update_num = option.batch_update_num;
%     Vs = zeros(spec_coeff_num, mic_num, mic_num, source_num); epsilon = 1e-32;
    [~,Ws,Ds,Vs,Cs,~,~,~] = binaural_auxive_update_multi(in_buffer_batch(:,1:batch_update_num,:), ...
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

label = {'single'};

for n_frame = startFrame:out_frame_num
    frameNum =n_frame;    in_buffer(:,2:buffer_size,:) = in_buffer(:,1:buffer_size-1,:);
    in_buffer(:,1,:) = in_buffer_perm(:,n_frame,:);     in_num = min(n_frame,buffer_size);
    if n_frame>1
        detect_range = round(0.3*spec_coeff_num):round(0.5*spec_coeff_num);
        pastf_energy = sum(sum(abs(in_buffer_perm(detect_range,n_frame-1,:)).^2));
        presentf_energy = sum(sum(abs(in_buffer_perm(detect_range,n_frame,:)).^2));
        if presentf_energy > 50*pastf_energy
            fprintf('%%%%%%sound detected%%%%%%\n');
        end
    end
    
    [out_buffer, Ws, Ds, Vs, Cs, W_hat, W_res, obj_vals] = binaural_auxive_update_multi(flip(in_buffer(:,1:in_num,:),2),Ws, Ds, Vs, Cs, W_hat, partition ,option);
    
    fprintf('frame no = %d/%d, obj = %.4e\n', ...
        n_frame, in_frame_num, obj_vals(end));
    
    if option.perm
        for s = 1:source_num
            for n = 1:spec_coeff_num
                out_buffer(n,:,s) = W_res(n,:,s) * (squeeze(in_buffer_batch(n,n_frame+1-in_num:n_frame,:)));
            end
        end
    end
    
    Sw = squeeze(out_buffer(:,in_num,:));   Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
    s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) =  s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
        + real(ifft(Sw, fft_size)).' .* win_syn;
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
