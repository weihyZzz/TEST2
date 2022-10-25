function [s_est] = auxiva_audio_bss_online_single(x,source_num,option)

global epsilon;
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

out_delay = inc * buffer_size + 1;
out_frame_num = fix((sample_num + out_delay - win_size) / inc) + 1;
batch_size = out_frame_num;

s_est = zeros(source_num, sample_num + out_delay);

in_buffer = zeros(spec_coeff_num, buffer_size, mic_num);
out_buffer = zeros(spec_coeff_num, buffer_size, source_num);

%%
Ws = zeros(spec_coeff_num, mic_num, source_num);
for n = 1:spec_coeff_num
    Ws(n,1:source_num,:) = eye(source_num);
end
Ds = ones(spec_coeff_num, source_num);
Vs = repmat(eps * eye(mic_num), 1, 1, spec_coeff_num, source_num);
Vs = permute(Vs, [3 1 2 4]);

if option.verbose
    fprintf('\n');
end

%%
n_orders = option.n_orders_online;
G_func = {@(x) x.^n_orders(1), @(x) x.^n_orders(2),@(x) x.^n_orders(3), @(x) x.^n_orders(4)};
dG_func = {@(x) n_orders(1) * (x + eps).^(n_orders(1) - 1), @(x) n_orders(2) * (x + eps).^(n_orders(2) - 1)...
       ,@(x) n_orders(3) * (x + eps).^(n_orders(3) - 1), @(x) n_orders(4) * (x + eps).^(n_orders(4) - 1)};
if parti  % 子块算法初始化
    block_size = 100;        block_overlap = 50;
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
    
par3.num = length(partition_index);    par3.size = partition_size;    par3.index = partition_index;    par3.contrast = G_func{3};
par3.contrast_derivative = dG_func{3};
    
par4.num = length(partition_index);    par4.size = partition_size;    par4.index = partition_index;    par4.contrast = G_func{4};
par4.contrast_derivative = dG_func{4};
    
partition = {par1, par2, par3, par4};

%%
for n_frame = 1:out_frame_num
    
    win_sample_range = inc*(n_frame-1) + 1:...
        min(inc*(n_frame-1) + win_size, sample_num);
    zero_padding_num = max(0, win_size - length(win_sample_range));
    
    xw = [x(:,win_sample_range) ...
        zeros(mic_num, zero_padding_num)] .* win_ana;
    Xw = fft(xw.', fft_size, 1);
    
    in_buffer(:,2:buffer_size,:) = in_buffer(:,1:buffer_size-1,:);
    in_buffer(:,1,:) = Xw(1:spec_coeff_num, :);
    
    in_num = min(n_frame,buffer_size);
%     in_num = buffer_size;
    
    [out_buffer, Ws, Ds, Vs, ~,~] = ...
        binaural_auxiva_update_multi(flip(in_buffer(:,1:in_num,:),2), Ws, Ds, Vs,...
			       partition, option);
%     fprintf('frame no = %d/%d, obj = %.4e\n', ...
% 	    n_frame, in_frame_num, obj_vals(end));
    fprintf('frame no = %d/%d\n', ...
	    n_frame, in_frame_num);

    Sw = squeeze(out_buffer(:,in_num,:));
    Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];

    s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) = ...
        s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
        + real(ifft(Sw, fft_size)).' .* win_syn;

end

