%% 实录信号处理 %%
target_source = real_src_num;
file_name = num2str(file_tag);
if file_tag < 10
    file_name = strcat(num2str(file_tag));%'0',
else if file_tag < 100 && file_tag >= 10
        file_name = strcat(num2str(file_tag));
    end
end
% folder_name = './data/mix/'; % 实录文件夹名称
% wav_name = 'xiaomi_record.wav'; % 实录音频名称
% mix_file = strcat(folder_name,wav_name);
[xR_t, Fssr] = audioread(mix_file); real_mic = size(xR_t,2);
%         clip_start = 70*Fssr; clip_end = size(xR_t,1);
%% 写入两段音频的名称格式
        real_result = 1;
        method='method1';
%% 读取未混合的实录信号
SDR_improvement = 0;
     if SDR_improvement == 1
        target_index = cell2mat(target_idx); intf_index = cell2mat(intf_idx);
        num_points = 320000; % 原始源信号最长限定在10s
        fsResample = 16000;
        n_tgt = length(target_index); n_intf = length(intf_index);
        [target_tag, intf_tag] = source_select_new(n_tgt,n_intf,target_index,intf_index);
        n_src = n_tgt+n_intf;
        dataLenRe = num_points * DebugRato;
        src_sig_resample = zeros(dataLenRe, n_src); % time x source
        fs = zeros(1,n_src);
        for k = 1:n_tgt
            [src, fs(k)] = audioread(target_tag{k});
%             src_sig(:,k) = src(1:dataLenRe,:);
            src_sig_resample(:,k) = src;
        end
        for l = n_tgt+1 : n_src
            [src, fs(l)] = audioread(intf_tag{l-n_tgt});
%             src_sig(:,l) = src(1:dataLenRe,:);
            src_sig_resample(:,l) = resample(src_sig(:,l), fsResample, fs(1), 100);
        end
        s = src_sig_resample.'; % fs_ref = fs(1);
     else
        s=xR_t.';
     end
%         xR_t = preprocess(xR_t, 2, deavg);
        if sim_mic == real_mic
            xR = xR_t.';
        else if sim_mic < real_mic
                xR = xR_t(:,[3,4]).';
            end
        end
        fs_ref = Fssr;