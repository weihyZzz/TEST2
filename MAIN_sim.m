%% Main simulation (origin version)
%% Simulation Initialize
clear all; close all;
addpath('room');addpath('components');addpath('bss_eval');addpath('plot');addpath('EM');addpath('nmfiva');addpath('IVE');addpath('audioSignalProcessTools');
filenameTmp ='Single_source';%AGC_method 可将仿真参数作为文件名 test_online_offline   test_AWGN_Lb test_AWGN_Lb
mkdir_str=strcat('./Simulation_Single/',filenameTmp);
mkdir(mkdir_str);%一运行就  会在当前文件夹下创建simulation文件夹
mkdir_str1 =strcat(mkdir_str,'/'); sound_dir= strcat(mkdir_str1,'/sound');
mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);
mkdir_str =strcat(mkdir_str,'.m');Save_link1=Save_link_file('MAIN_sim.m',mkdir_str);%将当前函数另存为nk文件
if isOctave
    graphics_toolkit('gnuplot');    warning('off', 'Octave:broadcast');
end
% mix_file1 = 'data/dev1_female3_male3_mix_1.wav';
% mix_file2 = 'data/dev1_female3_male3_mix_2.wav';
mix_file = 'data/mix/raw_cut.wav';

%% case setting
%%% Room simulation setting %%%
DebugRato = 10/10;     % 测试用；2是用2倍长度的测试数据；0.5是用1/2的数据；
mix_type = [1];      % 0-实录，1-仿真
room_case = [1:1]; 
    % Scenario : 1-the oldest one with angle ; 2-the oldest one
    %            3-mic>num Scenario; 4-Block-online Scenario 1
    %            5-Block-online Scenario 2; 6-online Scenario(200ms)
    % 仅加入了第一个房间的DOA信息，要使用的话注意
src_comb_case = [8];
% src_comb代表不同的信号源组合，s1代表target，s2代表interference
% Source combination: 1-female1+female2 ; 2-female1+male ; 3-female1+washer
%                     4-male+washer ; 5-wake up words+female1 ;
%                     6-wake up words+male ; 7-wake up words+washer
%                     8-wake up words+GWN; 9-wake up words+music
muteOn_case = [0]; % 选择第二个source，将前几秒进行mute（第二个source一般为target source)
normal_case = [3]; % 混合信号归一化类型，0-不归一化，1-功率归一化，2-幅值归一化，3-SNR混合（附带归一化）
deavg_case = [1]; % 混合信号去均值，仿真和实录均适用
mix_SNR_case = [0]; % 目标信号和噪声（或者干扰信号）的混合信噪比，若mix_SNR=0，则为1:1混合，仅在normal_case=3时使用
sim_mic_num = [2]; % 仿真麦克风数目。若实录要用4mic方法也需调成4，否则只会用两个通道
desire_source_case = [1]; % 期望最后得到的源数目，该值<=2，=1时为单源提取
determined_case = [0]; % 0:overdetermined(mic > src); 1:determined(mic = src)
target_SIRSDR = 1; % 1:只显示target source的SIRSDR，0:全部显示 (开关形式，未写入批处理）
%%% Pre-batch & Initialize %%%
initial_case = [0]; % C & V initialization, 0:normal, 1:use first-frame-batch
pre_batch_case = [0]; % using pre-batch to initialize Vs, 0: no pre-batch; 1:prebatch(continue); 2:prebatch(restart); 3:preonline(restart);
prebatch_num_case = [10]; % pre_batch_num
initial_rand_case = [0]; % V initialization using random diagonal domiance matrix; 1:full, 2:only low freq(1/3), 0:off;
%%% Essential setting %%%
online_case = [0];  %是否是在线。1 是在线。0 是离线；
batch_type_case = [1]; % batch IVA的类型，1代表auxiva，2代表nmfiva，默认为1。2的在线模式不可用
Lb_case = [1]; % length of blocks [1,2,4,6,8,16], default using 1
SNR_range =[200]; %200 dB,表明没有noise；
tao_case =[0]/10; % default value is 0, default hanning windowing operation;
                        % >1,  designed based special criterion;
                        % windowing factor default value is 1.2; 10^5, indicates no windowing;
win_exp_ratio_range =[20:1:20]/20; % default value is 1,effective for tao>0;                       
taoMeanAdd_case = [1]/10; % effective for tao>0; add some value, such as 0.1 to robustify the window desgin;
taoMean_case = [1]; % default value is 1,effective for tao>0;
D_open_case = [1]; % using D
perm_on_case = [0]; % 混合信号帧乱序输入
forgetting_fac_case = [0.04]; % 仅在线使用，相当于论文中的1-alpha [0.3 0.2 0.1 0.08 0.06 0.04]
gamma_case = [4:4]/100; % 仅在线mic>src使用，为C的forgetting factor
delta_case = [0]/10^0; % using in contrast function: (||x||^2 + delta)^n_orders
whitening_case = [0]; % 是否使用白化，未调试完成，暂时不要用
n_orders_case = {[0.5 0.5 1/2 1/2]}; % 离线的orders {[1/6 1/2],[1/2 1],[1/2 1/2],[1/8 1/2]}
n_orders_casenum = size(n_orders_case,2);
AWGN_on_case = [0]; % 是否加入白噪声（25dB）
%%% Partition %%%
parti_case = [0]; % 是否使用子块BSS
SubBlockSize_case = [100]; %  使用子块的大小。
SB_ov_Size_case = [1]/2; %  overlap 的比例，大小为 round(SubBlockSize*SB_ov_Size)；
partisize_case = [1];
select_case = [0]; % 是否选择子块（若parti=1，此处为选子块；若parti=0，此处为选子载波）
thFactor_case = [0]/200; % 选择子块阈值因子
%%% Epsilon test %%%
diagonal_method_case =[0]; %  0: using a tiny portion of the mean diagnal values of Xnn；          
                             % 1: using a tiny portion of the diagnal value of Xnn；       
Ratio_Rxx_case =[1:1]/10^3; %1:3:10 default is 0; 2 is a robust value;
epsilon_ratio_range =[10:10]/10; % 1:4 10:10:60 robustify inversion of update_w ;
epsilon_ratio1_range =[10]/10; % 1:4 10:10:60 robustify inversion of update_w ;
frameStart_range =[1]; % 1 表明初始用一样的 epsilon；  表明前frameStart前用 epsilon_start_ratio;
epsilon_start_ratio_range =[1:1:1]/1; %  表明前frameStart前用 ;
%%% Iteration num %%%
iter_num_case = [20]; % 算法迭代次数，online 固定为2次，offline的值为case的值
inner_iter_num_case = [1]; % 算法内迭代次数，online固定为1次，offline的值为case的值
total_iter_num_case = [1]; % 重复仿真次数，混合算法测试时为了算随机初始化时的平均SDR和SIR用的。
%%% Order estimation %%%
OrderEst_range =[0]; % 是否进行order estimation；
OutIter_Num_range =[3]; % order 估计的次数；
OrderEstUppThr_range =[10]/10; % order 估计的上限；
OrderEstLowThr_range =[1]/10; %  order 估计的下限；
order_gamma_range =[8]/10; %  order 估计的滑动系数；
n_orders1_range =[0.5]; % pdf 的估计指数（在线orders）
n_orders2_range =[0.5]; % pdf 的估计指数（在线orders）
verbose_range =[1]; % 0 不保留中间结果，1 保留中间结果。
%%% Gamma ratio %%%
GammaRatioThr_range =[10^2]; % 是否弱化在没有发声估计的门限，非常大等价不弱化；
GammaRatioSet_range =[10]/10;  % 是否弱化在没有发声估计的gamma 值。
%%% Mix Model Estimation %%%
mix_model_case = [0]; % 是否使用混合CGG模型,1：硬判，2：EM，0：不使用
ita_case = [0.9]; % 混合CGG模型中混合概率和beta值的递归平均系数
%%% NMF Setting %%%
nmf_iter_num_case = [1]; % nmf内迭代次数，1即可，过大出现奇异值
nmf_fac_num_case = [9]; % nmf基数目, 9 是典型值
nmf_b_case = [1/2]; % IS-NMF指数值；p=beta=2时，GGD-NMF等效为b=1/2的IS-NMF
nmf_beta_case = [2]; % beta值，用于GGD-NMF
nmf_p_case = [0.5]; % p值，用于GGD-NMF
nmf_update_case = [0]; % nmf update的模型，0是用IS-NMF，1用GGD-NMF
%%% Prior AuxIVA %%%
prior_case = {[0,0]}; % 输入为(1,k)维向量，第k项为1代表有第k个源的先验信息，
                            % 第k项为0代表没有第k个源的先验，则不用DOA方法更新
%deltaf_case = sqrt([1e9]); % 归一化系数
DOA_init_case = [0]; %是否使用DOA初始化；
DOA_tik_ratio_range =[0:2:0 ]/100; % DOA Tik 加权值， 
DOA_Null_ratio_range =[0:1:0]/100; % DOA  加权值， 
deltaf_case = ([10^20]); % 归一化系数
%%% IVE %%%
IVE_method_case = [1]; % IVE的方法，1为IP-1（等效overiva）, 2为IP-2（目前归一化错误），3具体看auxive_update说明
%%% Parameters initialization %%%
global DOA_tik_ratio; global DOA_Null_ratio; global Ratio_Rxx;  global win_exp_ratio; global PowerRatio;global diagonal_method;global epsilon_ratio1; 
global frameNum; global frameStart; global epsilon_start_ratio;global seed; global epsilon; epsilon = 1e-32; global epsilon_ratio; global SubBlockSize; global SB_ov_Size;
global OrderEst; global OutIter_Num; global order_gamma;  global OrderEstUppThr; global OrderEstLowThr;
global GammaRatioSet; global GammaRatio; global GammaRatioThr;
SNR = 45; % 加入白噪声的信噪比（in1 dB）
SIR_case = []; SDR_case = []; SAR_case = []; SNR_case = [];
SIR = []; SDR = []; SAR = []; SIR_time_all = [];
SIR_time_all = [];
timeblock_Length = 1; % online SIR仿真分块长度（in second）
case_num = 0;
room_imp_on = 1; seed = rng(10); PowerRatio = 2;
if mix_file == 1 file_num = 1; else file_num = 110; end
%% 批处理
tic
for mix_sim = mix_type for room_type = room_case for src_comb = src_comb_case for muteOn = muteOn_case for normal_type = normal_case for deavg = deavg_case for mix_SNR = mix_SNR_case for sim_mic = sim_mic_num for file_tag = file_num for prior = prior_case
%% 房间信道仿真
    if mix_sim % 使用房间信道进行仿真
        [xR, s, Fss1, mic_pos, theta] = generate_sim_mix(src_comb, sim_mic, room_type, room_imp_on, DebugRato, muteOn, normal_type, deavg, mix_SNR);
    else % 使用实录信号
        file_name = num2str(file_tag);
        if file_tag < 10
            file_name = strcat('00',num2str(file_tag));
        else if file_tag < 100 && file_tag >= 10
                file_name = strcat('0',num2str(file_tag));
            end
        end
        [x_pca, Fssp] = audioread('data/music.wav');
        mix_file = strcat('data/N/N1_2/',file_name,'.wav');
        [xR_t, Fssr] = audioread(mix_file); real_mic = size(xR_t,2);
%         clip_start = 70*Fssr; clip_end = size(xR_t,1);
        xR_t = preprocess(xR_t, normal_type, deavg);
        if sim_mic == real_mic
            xR = xR_t.';
        else if sim_mic < real_mic
                xR = xR_t(:,[3,4]).';
            end
        end
        Fss1 = Fssr;
    end        
for SNR = SNR_range    
    xR_w = awgn(xR,SNR(1),'measured',0); noise = xR_w - xR; % 加高斯白噪声
    audiowrite([sound_dir,'/mix.wav'], xR', Fss1);  audiowrite([sound_dir,'/mix_w.wav'], xR_w', Fss1);
    xR_t = audioread([sound_dir,'/mix.wav']);       xR_tw = audioread([sound_dir,'/mix_w.wav']);
    xR_1 = xR_t';    xR_1w = xR_tw'; % 存成wav后再读取的混合信号，一般用xR即可，性能特别不好再用xR_1试试
    win_type = 'hann';    win_size = 4096;    inc = win_size / 2;    fft_size = win_size;
    spec_coeff_num = fft_size / 2 + 1; % frequency bin
%% 批处理
for determined = determined_case for online = online_case for batch_type = batch_type_case for tao = tao_case for win_exp_ratio = win_exp_ratio_range for taoMeanAdd = taoMeanAdd_case for taoMean = taoMean_case
    if online
        win_type = 'hamming'; % 建议online使用汉明窗
    end
    if tao >0 
        [win_ana, ~]= WinGen(win_size,tao,taoMeanAdd,taoMean,inc,win_exp_ratio); 
        % [win_ana1, ~]= WinGen(win_size,1.2,taoMeanAdd,taoMean,inc,win_exp_ratio); plot(win_ana,'b'); hold on;   plot(win_ana1,'r'); 
    else
        [win_ana, ~] = generate_win(win_type, win_size, inc);
    end
    win_syn = ones(1, win_size);    
for desire_source = desire_source_case for D_open = D_open_case for perm_on = perm_on_case for iter_num = iter_num_case  for inner_iter_num = inner_iter_num_case  for forgetting_fac = forgetting_fac_case for gamma = gamma_case for delta = delta_case for whitening_open = whitening_case
for initial = initial_case for initial_rand = initial_rand_case for pre_batch = pre_batch_case for batch_update_num = prebatch_num_case for Lb = Lb_case for n_orders_num = 1:n_orders_casenum for AWGN_on = AWGN_on_case for parti = parti_case  for SubBlockSize = SubBlockSize_case 
for SB_ov_Size = SB_ov_Size_case for select = select_case for partisize = partisize_case for thFactor = thFactor_case for diagonal_method = diagonal_method_case for Ratio_Rxx = Ratio_Rxx_case   for epsilon_ratio = epsilon_ratio_range  for epsilon_ratio1 =epsilon_ratio1_range
for OrderEst = OrderEst_range for OutIter_Num = OutIter_Num_range for order_gamma = order_gamma_range for OrderEstUppThr = OrderEstUppThr_range for OrderEstLowThr = OrderEstLowThr_range for frameStart =frameStart_range for epsilon_start_ratio =epsilon_start_ratio_range            
for GammaRatioThr = GammaRatioThr_range for  GammaRatioSet = GammaRatioSet_range for mix_model = mix_model_case for ita = ita_case for n_orders1 =n_orders1_range for n_orders2 =n_orders2_range for verbose = verbose_range for nmfupdate = nmf_update_case
for nmf_iter_num = nmf_iter_num_case for nmf_fac_num = nmf_fac_num_case for nmf_b = nmf_b_case for nmf_beta = nmf_beta_case for nmf_p = nmf_p_case for deltaf = deltaf_case for DOA_Null_ratio = DOA_Null_ratio_range  for DOA_tik_ratio = DOA_tik_ratio_range   
for DOA_init = DOA_init_case for IVE_method = IVE_method_case for total_iter_num = total_iter_num_case 

    seed = rng(10); % 固定nmf仿真随机种子
    if batch_type == 1 
        total_iter = 1; % auxIVA采用固定初始化，不需要重复仿真
    else if batch_type == 2 
          total_iter = total_iter_num; % nmfIVA采用随机初始化，需要重复仿真
        end
    end
for ITER = 1:total_iter
    %% 混合信号初始化
    if AWGN_on
        x = xR_w;
    else
        x = xR; % 一般用xR即可，性能特别不好再用xR_1试试。实录信号需要改成xR_1
    end    
    [mic_num, sample_num] = size(x); source_num = desire_source;
    if determined source_num = mic_num; end
   
    %% 仿真参数设置
    option.D_open = D_open; option.iter_num = iter_num; option.inner_iter_num = inner_iter_num;
    if verbose ==0     option.verbose = false; end
    if verbose ==1     option.verbose = true; end
    option.forgetting_fac = forgetting_fac; option.select = select; option.ita = ita; option.IVE_method = IVE_method;
    option.parti = parti; option.thFactor = thFactor; option.online = online; option.win_size = win_size;
    option.win_ana = win_ana; option.win_syn = win_syn; option.Lb = Lb; option.batch_type = batch_type;
    option.perm = perm_on; option.whitening_open = whitening_open; option.partisize = partisize;
    option.n_orders_online = [n_orders1 n_orders2]; % online n_orders 
    option.n_orders_batch = cell2mat(n_orders_case(n_orders_num)); option.pre_batch = pre_batch;
    option.diagonal_method = diagonal_method; option.gamma = gamma; option.initial = initial;
    option.batch_update_num = batch_update_num; option.delta = delta; option.mix_model = mix_model;
    % 对4mic的情况，需要有四个orders，这时候用orders1和orders2重复一次
    if sim_mic == 4 && determined option.n_orders_online = [n_orders1 n_orders2 n_orders1 n_orders2]; end
    option.nmf_iter_num = nmf_iter_num; option.nmf_fac_num = nmf_fac_num; option.nmf_beta = nmf_beta;
    option.prior = cell2mat(prior); option.nmf_p = nmf_p; option.nmfupdate = nmfupdate; option.nmf_b = nmf_b;
    if mix_type option.mic_pos = mic_pos; option.theta = theta; option.deltaf = deltaf;end
    GammaRatio = ones(1,8); option.initial_rand = initial_rand;  option.DOA_init = DOA_init;
    %% 信号处理和盲分离
    if online
        % 在线算法(online)
        if source_num ~= 1
            [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option); % 带乱序输入的online版本
%         [s_est] = auxiva_audio_bss_online_single(x,source_num,option); 
%         [s_est,label] = nmfiva_audio_bss_online_perm(x,source_num,option); 
        else
            [s_est,label] = auxive_audio_bss_online_perm(x,option); % IVE
        end
        out_type = 'online';
    else
        % 离线算法(batch)
        if source_num ~= 1
            [s_est,label] = auxiva_audio_bss_batch(x,source_num,option); % IVA batch(including nmfiva)
        else
            [s_est,label] = auxive_audio_bss_batch(x,option); % IVE batch
        end
        out_type = 'batch';
    end    
  
    %% SIR SDR计算
    if mix_sim
        L = min(size(s,2), size(s_est,2));
        if source_num == 1
            [SDR,~,SAR,perm] = bss_eval_sources(s_est(1,1:L), s(1,1:L));
            SIR = 0;
        end
        if AWGN_on % 自己修改的带noise的SDR计算，不完善，不建议使用
           % [SDR,SIR,SNR,SAR,perm] = bss_eval_noise(s_est(:,1:L), s(1:2,1:L),noise(:,1:L));
            [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(:,1:L));
            SNR = zeros(source_num,1);
        else % bss_eval 3.0 版本SDR计算，建议使用
            if source_num == 2 % 对于4*2和2*2
                [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(:,1:L)); max_comb = 1:2;
                s_est = s_est(perm',:);
            else if source_num > 2 % 对于4*4，需要选两路SIR之和最大的组合
                    perm_comb = nchoosek(1:source_num,2); [comb_num,~] = size(perm_comb);
                    SDR_c = zeros(2,comb_num); SIR_c = zeros(2,comb_num); SAR_c = zeros(2,comb_num); perm_c = zeros(2,comb_num);
                    for cn = 1:comb_num
                        [SDR_c(:,cn),SIR_c(:,cn),SAR_c(:,cn),perm_c(:,cn)] = bss_eval_sources(s_est(perm_comb(cn,:),1:L), s(:,1:L));
                    end
                    [~, max_perm_index] = max(sum(SDR_c,1)); max_comb = perm_comb(max_perm_index,:);
                    SDR = SDR_c(:,max_perm_index); SIR = SIR_c(:,max_perm_index); SAR = SAR_c(:,max_perm_index); perm = perm_c(:,max_perm_index);
                    s_est = s_est(max_comb(perm'),:); label = label{max_comb(perm')};
                end
            end
            SNR = zeros(2,1);
        end
        if target_SIRSDR
            SIR = SIR(1); SDR = SDR(1); SAR = SAR(1); SNR = SNR(1);
        end
        fprintf('%s\nSDR = %s\nSIR = %s\n',out_type,num2str(SDR'),num2str(SIR'));
        
        if online % 计算 online SIR improvment
            tap_Length = timeblock_Length * Fss1;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length);
            T = [0 : ceil(L / tap_Length)] * timeblock_Length;            SIR_time_all = [SIR_time_all;SIR_time];
        end 
        if total_iter ~= total_iter_num
            % 固定初始化和随机初始化一起仿真时，需要将固定初始化的数据进行复制来减少仿真时间。
            SIR_case = [SIR_case repmat(SIR,[1,total_iter_num])];  SDR_case = [SDR_case repmat(SDR,[1,total_iter_num])];  SAR_case = [SAR_case repmat(SAR,[1,total_iter_num])];  SNR_case = [SNR_case repmat(SNR,[1,total_iter_num])];
        else
            SIR_case = [SIR_case SIR];  SDR_case = [SDR_case SDR];  SAR_case = [SAR_case SAR];  SNR_case = [SNR_case SNR];
        end
    end
    strsave= strcat(mkdir_str1,filenameTmp,'.mat'); sav=['save ' strsave]; eval(sav);
    %% 保存数据(case)和音频
    case_num = case_num + 1;    case_str = num2str(case_num);
%     filenameTmp1 = strcat('case_',case_str,'.mat'); strsave= strcat(mkdir_str1,filenameTmp1);
%     sav=['save ' strsave]; eval(sav);
    % Save separated wave files
    sep1_str = label{1};
    audiowrite([sound_dir,'/sep1_case',case_str,'_',sep1_str,'_',out_type,'.wav'], s_est(1,:)', Fss1);
    if size(s_est,1) == 2
        sep2_str = label{2};
        audiowrite([sound_dir,'/sep2_case',case_str,'_',sep2_str,'_',out_type,'.wav'], s_est(2,:)', Fss1);
    end
    if size(s_est,1) == 4
        audiowrite([sound_dir,'/sep3_case',case_str,'_',out_type,'.wav'], s_est(3,:)', Fss1);   audiowrite([sound_dir,'/sep4_case',case_str,'_',out_type,'.wav'], s_est(4,:)', Fss1);   
    end
    % 波形&语谱图绘制，仅测试实录信号时使用 -1.8152   -1.8152   -3.3037   -3.3037
    if mix_sim == 0
        plot_sound(s_est,xR,Fss1,label);
%         print(gcf,'-djpeg',['.\plot\save\',case_str+1,'.jpeg']);
    end
    toc
  end
end
end
%if ~select  %        break;     end
end
%if parti || select        break;    end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
%% 保存数据(all)&画图
% 需要单独画图可用这里读取mat文件
% load_case = 1;  load_case_str = num2str(load_case);
% load(strcat(mkdir_str1,'case_',load_case_str,'.mat')); %
%   load(strcat(mkdir_str1,'.mat'));
if mix_sim
    loadMat= 0;
    if loadMat == 1
        
        clear all; close all;
        filenameTmp ='DOA_ratio';%可将仿真参数作为文件名 test_online_offline   test_AWGN_Lb test_AWGN_Lb
        mkdir_str=strcat('./Simulation_Single/',filenameTmp);
        mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
        mkdir_str1 =strcat(mkdir_str,'/'); sound_dir = strcat(mkdir_str1,'/sound');
        mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);  loadstrcase = strcat(mkdir_str1,filenameTmp,'.mat')
        loadsr=['load  ' loadstrcase];       eval(loadsr);        SIR_case
        SDR_case
        Ratio_Rxx_case
        epsilon_ratio_range
        
        if target_SIRSDR R_num = 1; else R_num = 2; end
        case_num = size(SIR_case,2) / total_iter_num;
        SIR_total = reshape(SIR_case,R_num, total_iter_num, case_num);  SDR_total = reshape(SDR_case,R_num, total_iter_num, case_num);
        SAR_total = reshape(SAR_case,R_num, total_iter_num, case_num);  SNR_total = reshape(SNR_case,R_num, total_iter_num, case_num);
        
        %online =1;
        if online
         %   plotSIR_time(SIR_time_all,T,max(size(epsilon_ratio_range,2),size(Ratio_Rxx_case,2)),3); % 绘制时变SIR         
sub_case_num = case_num;
packFigNum =   1;  % 一个fig 分成几个(=packFigNum)subfugure，进行情况比较
SortedPlotThr = 5; % >1,表明显示排序。
SortedPlotNum = 5; % 表明排序显示case的数目,在 sub_case_num 比较大的时候可以清晰显示
plotRatio =1;      %一次多少比例的图； default=1； 2 表明画50% 的图；   
close all;  
% plotSIR_time(SIR_time_all,T,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % 绘制时变SIR
plotSIR_time1(SIR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % 绘制时变SIR 多source
%function plotSIR_time(SIR,T,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
%sub_case_num      同一大类比较在一个Figure 画的数目, 比如一共 10 =size(SIR_time_all,1)/sourceNum (=2),一个图画2*5比较。
%packFigNum       一次把几种subcase 都放到一起来plot
%SortedPlotThr;   如果case_num太多显示不了，判决是否显示 SortedPlotNum case，
%如果是1，缺省显示不排序的情况；如果>1, 表明排序显示。
%SortedPlotNum;   如果case_num太多显示不了，显示 SortedPlotNum case；
%plotRatio       一次多少比例的图； default=1； 2 表明画50% 的图；
        else            
                        %plotSDR(case_num,SDR_total); % 绘制普通SIR、SDR
            %plotSDR(case_num,SDR_total,SIR_total); % 绘制普通SIR、SDR  
sub_case_num = 2;
packFigNum =  1;  %一个fig 分成几个(=packFigNum)subfugure，进行情况比较 
SortedPlotThr = 1;%sub_case_num * packFigNum < SortedPlotThr不排序
SortedPlotNum = 2;%表明排序显示case的数目,在 sub_case_num 比较大的时候可以清晰显示
plotRatio =1;     %一次多少比例的图, default=1, 2 表明画50% 的图；
close all;
%function plotSDR3 plotSDR4
%sub_case_num  同一大类比较在一个Figure 画的数目
%packFigNum    一个fig 分成几个(=packFigNum)subfugure，进行情况比较
%SortedPlotThr 如果case_num太多显示不了，判决是否显示 SortedPlotNum case 
%SortedPlotNum 如果case_num太多显示不了，显示 SortedPlotNum case；
%plotRatio     一次多少比例的图, default=1, 2 表明画50% 的图
plotSDR3(SDR_total,SIR_total,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio);%排序errorbar  
plotSDR4(SDR_total,SIR_total,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio);%均值排序boxplot
        end   
           
        
        
        
    end
end