function [s_est,label] = auxive_audio_bss_batch(x,option)

global epsilon;
epsilon = 1e-32; % offline 用1e-32
parti = option.parti;
whitening_open = option.whitening_open;
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
in_frame_num = fix((sample_num - win_size) / inc) + 1; % time frame
batch_overlap = 0;    batch_size = in_frame_num; % time frame
batch_num = fix((in_frame_num - batch_size)  / (batch_size - batch_overlap)) + 1;
sample_max = inc * (in_frame_num-1) + win_size;
s_est = zeros(1, sample_max);
in_buffer = zeros(spec_coeff_num, batch_size, mic_num);
out_buffer = zeros(spec_coeff_num, batch_size);
if whitening_open 
    mic_num = source_num; 
    in_buffer_white = zeros(spec_coeff_num, batch_size, mic_num);
    out_buffer_white = zeros(spec_coeff_num, batch_size);
end

%%
Ws = zeros(spec_coeff_num, mic_num, source_num);
for n = 1:spec_coeff_num
    Ws(n,1,:) = 1;
end
W_hat = zeros(spec_coeff_num, mic_num, mic_num); % #freq * #mic * #mic
W_hat(:, 2:mic_num, 2:mic_num) =...
    repmat(reshape(-eye(mic_num - 1),[1, mic_num-1, mic_num-1]),[spec_coeff_num,1,1]);
W_hat(:, :, 1) = Ws;
Ds = ones(spec_coeff_num, 1);
Vs = zeros(spec_coeff_num, mic_num, mic_num);
Cs = zeros(spec_coeff_num, mic_num, mic_num);

if option.verbose
    fprintf('\n');
end

%%
n_orders = option.n_orders_batch; delta = option.delta;
G_func = {@(x) (x + delta).^n_orders(1)};
dG_func = {@(x) n_orders(1) * (x + delta + eps).^(n_orders(1) - 1)};
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

partition = {par1};

%% 离线算法(batch)
% 观测信号FFT
for n_frame = 1:batch_size
    win_sample_range = inc*(n_frame-1) + 1: min(inc*(n_frame-1) + win_size, sample_num);
    zero_padding_num = max(0, win_size - length(win_sample_range));
    xw = [x(:,win_sample_range) zeros(size(x,1), zero_padding_num)] .* win_ana;
    Xw = fft(xw.', fft_size, 1);
    in_buffer(:,n_frame,:) = Xw(1:spec_coeff_num,:);
end


% AuxIVE
if whitening_open == 1
    in_buffer_white = whitening(in_buffer, 1);
    % tic;
    [out_buffer_white, Ws, Ds, Vs, Cs, ~, ~, obj_vals] = binaural_auxive_update_multi(in_buffer_white, Ws, Ds, Vs, Cs, W_hat, partition, option);
    %toc;
    out_buffer = backProjection(out_buffer_white, in_buffer(:,:,1));
else
    tic;
    [out_buffer, Ws, Ds, Vs, Cs, ~, ~, obj_vals] = ...
        binaural_auxive_update_multi(in_buffer, Ws, Ds, Vs, Cs, W_hat, partition, option);
    toc;
end
project_back = option.project_back;
% fixing the frequency-wise scales of the signals 
if project_back == 1 
    out_buffer = backProjection(out_buffer, in_buffer(:,:,1));
end

label = {'single'};

% 估计信号做IFFT
for n_frame = 1:batch_size
    Sw = squeeze(out_buffer(:,n_frame,:));
    Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
    
    s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) = ...
        s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
        + real(ifft(Sw, fft_size)).' .* win_syn;
end

