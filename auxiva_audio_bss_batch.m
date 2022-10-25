function [s_est,label] = auxiva_audio_bss_batch(x,source_num,option)

global epsilon;
epsilon = 1e-32; % offline 用1e-32
parti = option.parti;
whitening_open = option.whitening_open;
partisize = option.partisize;

%%
win_size = option.win_size;inc = win_size / 2;
fft_size = win_size;spec_coeff_num = fft_size / 2 + 1;
win_ana = option.win_ana;win_syn = option.win_syn;

%% Buffer initialization
[mic_num, sample_num] = size(x);
in_frame_num = fix((sample_num - win_size) / inc) + 1; % time frame
batch_overlap = 0;    batch_size = in_frame_num+1; % time frame
batch_num = fix((in_frame_num - batch_size)  / (batch_size - batch_overlap)) + 1;
sample_max = inc * in_frame_num + win_size;
s_est = zeros(source_num, sample_max);
in_buffer = zeros(spec_coeff_num, batch_size, mic_num);
out_buffer = zeros(spec_coeff_num, batch_size, source_num);
if whitening_open 
    mic_num = source_num; 
    in_buffer_white = zeros(spec_coeff_num, batch_size, mic_num);
    out_buffer_white = zeros(spec_coeff_num, batch_size, source_num);
end

%% Matrix initialization
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
Vs = zeros(spec_coeff_num, mic_num, mic_num, source_num);
Cs = zeros(spec_coeff_num, mic_num, mic_num);

if option.verbose
    fprintf('\n');
end

%% Aux Function & partition initialization
n_orders = option.n_orders_batch; delta = option.delta;
G_func = {@(x) (x + delta).^n_orders(1), @(x) (x + delta).^n_orders(2),@(x) (x + delta).^n_orders(3), @(x) (x + delta).^n_orders(4)};
dG_func = {@(x) n_orders(1) * (x + delta + eps).^(n_orders(1) - 1), @(x) n_orders(2) * (x + delta + eps).^(n_orders(2) - 1)...
       ,@(x) n_orders(3) * (x + delta + eps).^(n_orders(3) - 1), @(x) n_orders(4) * (x + delta + eps).^(n_orders(4) - 1)};
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

%% Batch update
% 观测信号FFT
for n_frame = 1:batch_size
    win_sample_range = inc*(n_frame-1) + 1: min(inc*(n_frame-1) + win_size, sample_num);
    zero_padding_num = max(0, win_size - length(win_sample_range));
    xw = [x(:,win_sample_range) zeros(size(x,1), zero_padding_num)] .* win_ana;
    Xw = fft(xw.', fft_size, 1);
    in_buffer(:,n_frame,:) = Xw(1:spec_coeff_num,:);
end

if option.DOA_esti || ~option.mix_sim
    theta = doa_estimation(x.',option.esti_mic_dist,source_num,16000);
    option.theta = theta*pi/180;
    if mic_num == 2
        option.mic_pos = [0,option.esti_mic_dist];%2mic
    else 
        option.mic_pos = [0,option.esti_mic_dist,2*option.esti_mic_dist,3*option.esti_mic_dist];%4mic
    end
end

switch option.batch_type
    case 1
        % AuxIVA
        if whitening_open == 1
            in_buffer_white = whitening(in_buffer, source_num);
           % tic;
            [out_buffer_white, Ws, Ds, Vs, Cs, ~, ~, obj_vals] = binaural_auxiva_update_multi(in_buffer_white, Ws, Ds, Vs, Cs, W_hat, partition, option);
            %toc;
            out_buffer = backProjection(out_buffer_white, in_buffer(:,:,1));
        else
           % tic;
            [out_buffer, Ws, Ds, Vs, Cs, ~, ~, obj_vals] = ...
                binaural_auxiva_update_multi(in_buffer, Ws, Ds, Vs, Cs, W_hat, partition, option);
%             [out_buffer, Ws, Ds, Vs, Cs, ~, ~, obj_vals] = ...
%                 binaural_overiva_ip2_update(in_buffer, Ws, Ds, Vs, Cs, W_hat, partition, option);
            if option.project_back
                out_buffer = backProjection(out_buffer, in_buffer(:,:,1));
            end
            %     [out_buffer,Ws] = overiva_py(in_buffer,Ws,source_num,0,'laplace',0,0);
            %     out_buffer = backProjection(out_buffer, in_buffer(:,:,1));
           % toc;
        end
    case 2
        % NMF IVA
        if whitening_open == 1
            in_buffer_white = whitening(in_buffer, source_num);
            tic;
            [out_buffer_white, Ws, D1, D2, obj_vals] = binaural_nmfiva_batch_update(in_buffer_white, Ws, G_func, dG_func, option);
            toc;
            out_buffer = backProjection(out_buffer_white, in_buffer(:,:,1));
        else
            tic;
            [out_buffer, Ws, D1, D2, obj_vals] = binaural_nmfiva_batch_update(in_buffer, Ws, G_func, dG_func, option);
            toc;
            if option.project_back
                out_buffer = backProjection(out_buffer, in_buffer(:,:,1));
            end
        end
    case 3
        % AuxIVA-ISS
        [out_buffer,~] =auxiva_iss(in_buffer,option);

    case 4% SMM_IVA
            tic;
%            in_buffer_white = whitening(in_buffer, source_num);
%            in_buffer = in_buffer_white;
            upS1=1;upS2=1;
            nEM = 10;%pi1 = [0.4;0.6];pi2 = [0.1;0.9];
           [out_buffer ,nu1 ,pi1 ,nu2, pi2 ,W ,F] =  EM_IVA_noiseFree(in_buffer,upS1,upS2,nEM);
            toc;
            if option.project_back
                out_buffer = backProjection(out_buffer, in_buffer(:,:,1));
            end
end

% label = sort_est_sig(out_buffer(:,1:3,:));
label = cell(1,source_num);
for k = 1:source_num
    label{k} = 'target';
end

% 估计信号做IFFT
for n_frame = 1:batch_size
    Sw = squeeze(out_buffer(:,n_frame,:));
    Sw = [Sw; conj(flipud(Sw(2:spec_coeff_num-1,:)))];
    
    s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) = ...
        s_est(:,inc*(n_frame-1) + 1:inc*(n_frame-1) + win_size) ...
        + real(ifft(Sw, fft_size)).' .* win_syn;
end

