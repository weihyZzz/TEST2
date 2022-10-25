%% Main simulation (origin version)
%% Simulation Initialize
clear all; close all;
addpath('room');addpath('components');addpath('bss_eval');addpath('plot');addpath('EM');addpath('nmfiva');addpath('IVE');addpath('audioSignalProcessTools');
filenameTmp ='Single_source';%AGC_method �ɽ����������Ϊ�ļ��� test_online_offline   test_AWGN_Lb test_AWGN_Lb
mkdir_str=strcat('./Simulation_Single/',filenameTmp);
mkdir(mkdir_str);%һ���о�  ���ڵ�ǰ�ļ����´���simulation�ļ���
mkdir_str1 =strcat(mkdir_str,'/'); sound_dir= strcat(mkdir_str1,'/sound');
mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);
mkdir_str =strcat(mkdir_str,'.m');Save_link1=Save_link_file('MAIN_sim.m',mkdir_str);%����ǰ�������Ϊnk�ļ�
if isOctave
    graphics_toolkit('gnuplot');    warning('off', 'Octave:broadcast');
end
% mix_file1 = 'data/dev1_female3_male3_mix_1.wav';
% mix_file2 = 'data/dev1_female3_male3_mix_2.wav';
mix_file = 'data/mix/raw_cut.wav';

%% case setting
%%% Room simulation setting %%%
DebugRato = 10/10;     % �����ã�2����2�����ȵĲ������ݣ�0.5����1/2�����ݣ�
mix_type = [1];      % 0-ʵ¼��1-����
room_case = [1:1]; 
    % Scenario : 1-the oldest one with angle ; 2-the oldest one
    %            3-mic>num Scenario; 4-Block-online Scenario 1
    %            5-Block-online Scenario 2; 6-online Scenario(200ms)
    % �������˵�һ�������DOA��Ϣ��Ҫʹ�õĻ�ע��
src_comb_case = [8];
% src_comb����ͬ���ź�Դ��ϣ�s1����target��s2����interference
% Source combination: 1-female1+female2 ; 2-female1+male ; 3-female1+washer
%                     4-male+washer ; 5-wake up words+female1 ;
%                     6-wake up words+male ; 7-wake up words+washer
%                     8-wake up words+GWN; 9-wake up words+music
muteOn_case = [0]; % ѡ��ڶ���source����ǰ�������mute���ڶ���sourceһ��Ϊtarget source)
normal_case = [3]; % ����źŹ�һ�����ͣ�0-����һ����1-���ʹ�һ����2-��ֵ��һ����3-SNR��ϣ�������һ����
deavg_case = [1]; % ����ź�ȥ��ֵ�������ʵ¼������
mix_SNR_case = [0]; % Ŀ���źź����������߸����źţ��Ļ������ȣ���mix_SNR=0����Ϊ1:1��ϣ�����normal_case=3ʱʹ��
sim_mic_num = [2]; % ������˷���Ŀ����ʵ¼Ҫ��4mic����Ҳ�����4������ֻ��������ͨ��
desire_source_case = [1]; % �������õ���Դ��Ŀ����ֵ<=2��=1ʱΪ��Դ��ȡ
determined_case = [0]; % 0:overdetermined(mic > src); 1:determined(mic = src)
target_SIRSDR = 1; % 1:ֻ��ʾtarget source��SIRSDR��0:ȫ����ʾ (������ʽ��δд��������
%%% Pre-batch & Initialize %%%
initial_case = [0]; % C & V initialization, 0:normal, 1:use first-frame-batch
pre_batch_case = [0]; % using pre-batch to initialize Vs, 0: no pre-batch; 1:prebatch(continue); 2:prebatch(restart); 3:preonline(restart);
prebatch_num_case = [10]; % pre_batch_num
initial_rand_case = [0]; % V initialization using random diagonal domiance matrix; 1:full, 2:only low freq(1/3), 0:off;
%%% Essential setting %%%
online_case = [0];  %�Ƿ������ߡ�1 �����ߡ�0 �����ߣ�
batch_type_case = [1]; % batch IVA�����ͣ�1����auxiva��2����nmfiva��Ĭ��Ϊ1��2������ģʽ������
Lb_case = [1]; % length of blocks [1,2,4,6,8,16], default using 1
SNR_range =[200]; %200 dB,����û��noise��
tao_case =[0]/10; % default value is 0, default hanning windowing operation;
                        % >1,  designed based special criterion;
                        % windowing factor default value is 1.2; 10^5, indicates no windowing;
win_exp_ratio_range =[20:1:20]/20; % default value is 1,effective for tao>0;                       
taoMeanAdd_case = [1]/10; % effective for tao>0; add some value, such as 0.1 to robustify the window desgin;
taoMean_case = [1]; % default value is 1,effective for tao>0;
D_open_case = [1]; % using D
perm_on_case = [0]; % ����ź�֡��������
forgetting_fac_case = [0.04]; % ������ʹ�ã��൱�������е�1-alpha [0.3 0.2 0.1 0.08 0.06 0.04]
gamma_case = [4:4]/100; % ������mic>srcʹ�ã�ΪC��forgetting factor
delta_case = [0]/10^0; % using in contrast function: (||x||^2 + delta)^n_orders
whitening_case = [0]; % �Ƿ�ʹ�ð׻���δ������ɣ���ʱ��Ҫ��
n_orders_case = {[0.5 0.5 1/2 1/2]}; % ���ߵ�orders {[1/6 1/2],[1/2 1],[1/2 1/2],[1/8 1/2]}
n_orders_casenum = size(n_orders_case,2);
AWGN_on_case = [0]; % �Ƿ�����������25dB��
%%% Partition %%%
parti_case = [0]; % �Ƿ�ʹ���ӿ�BSS
SubBlockSize_case = [100]; %  ʹ���ӿ�Ĵ�С��
SB_ov_Size_case = [1]/2; %  overlap �ı�������СΪ round(SubBlockSize*SB_ov_Size)��
partisize_case = [1];
select_case = [0]; % �Ƿ�ѡ���ӿ飨��parti=1���˴�Ϊѡ�ӿ飻��parti=0���˴�Ϊѡ���ز���
thFactor_case = [0]/200; % ѡ���ӿ���ֵ����
%%% Epsilon test %%%
diagonal_method_case =[0]; %  0: using a tiny portion of the mean diagnal values of Xnn��          
                             % 1: using a tiny portion of the diagnal value of Xnn��       
Ratio_Rxx_case =[1:1]/10^3; %1:3:10 default is 0; 2 is a robust value;
epsilon_ratio_range =[10:10]/10; % 1:4 10:10:60 robustify inversion of update_w ;
epsilon_ratio1_range =[10]/10; % 1:4 10:10:60 robustify inversion of update_w ;
frameStart_range =[1]; % 1 ������ʼ��һ���� epsilon��  ����ǰframeStartǰ�� epsilon_start_ratio;
epsilon_start_ratio_range =[1:1:1]/1; %  ����ǰframeStartǰ�� ;
%%% Iteration num %%%
iter_num_case = [20]; % �㷨����������online �̶�Ϊ2�Σ�offline��ֵΪcase��ֵ
inner_iter_num_case = [1]; % �㷨�ڵ���������online�̶�Ϊ1�Σ�offline��ֵΪcase��ֵ
total_iter_num_case = [1]; % �ظ��������������㷨����ʱΪ���������ʼ��ʱ��ƽ��SDR��SIR�õġ�
%%% Order estimation %%%
OrderEst_range =[0]; % �Ƿ����order estimation��
OutIter_Num_range =[3]; % order ���ƵĴ�����
OrderEstUppThr_range =[10]/10; % order ���Ƶ����ޣ�
OrderEstLowThr_range =[1]/10; %  order ���Ƶ����ޣ�
order_gamma_range =[8]/10; %  order ���ƵĻ���ϵ����
n_orders1_range =[0.5]; % pdf �Ĺ���ָ��������orders��
n_orders2_range =[0.5]; % pdf �Ĺ���ָ��������orders��
verbose_range =[1]; % 0 �������м�����1 �����м�����
%%% Gamma ratio %%%
GammaRatioThr_range =[10^2]; % �Ƿ�������û�з������Ƶ����ޣ��ǳ���ȼ۲�������
GammaRatioSet_range =[10]/10;  % �Ƿ�������û�з������Ƶ�gamma ֵ��
%%% Mix Model Estimation %%%
mix_model_case = [0]; % �Ƿ�ʹ�û��CGGģ��,1��Ӳ�У�2��EM��0����ʹ��
ita_case = [0.9]; % ���CGGģ���л�ϸ��ʺ�betaֵ�ĵݹ�ƽ��ϵ��
%%% NMF Setting %%%
nmf_iter_num_case = [1]; % nmf�ڵ���������1���ɣ������������ֵ
nmf_fac_num_case = [9]; % nmf����Ŀ, 9 �ǵ���ֵ
nmf_b_case = [1/2]; % IS-NMFָ��ֵ��p=beta=2ʱ��GGD-NMF��ЧΪb=1/2��IS-NMF
nmf_beta_case = [2]; % betaֵ������GGD-NMF
nmf_p_case = [0.5]; % pֵ������GGD-NMF
nmf_update_case = [0]; % nmf update��ģ�ͣ�0����IS-NMF��1��GGD-NMF
%%% Prior AuxIVA %%%
prior_case = {[0,0]}; % ����Ϊ(1,k)ά��������k��Ϊ1�����е�k��Դ��������Ϣ��
                            % ��k��Ϊ0����û�е�k��Դ�����飬����DOA��������
%deltaf_case = sqrt([1e9]); % ��һ��ϵ��
DOA_init_case = [0]; %�Ƿ�ʹ��DOA��ʼ����
DOA_tik_ratio_range =[0:2:0 ]/100; % DOA Tik ��Ȩֵ�� 
DOA_Null_ratio_range =[0:1:0]/100; % DOA  ��Ȩֵ�� 
deltaf_case = ([10^20]); % ��һ��ϵ��
%%% IVE %%%
IVE_method_case = [1]; % IVE�ķ�����1ΪIP-1����Чoveriva��, 2ΪIP-2��Ŀǰ��һ�����󣩣�3���忴auxive_update˵��
%%% Parameters initialization %%%
global DOA_tik_ratio; global DOA_Null_ratio; global Ratio_Rxx;  global win_exp_ratio; global PowerRatio;global diagonal_method;global epsilon_ratio1; 
global frameNum; global frameStart; global epsilon_start_ratio;global seed; global epsilon; epsilon = 1e-32; global epsilon_ratio; global SubBlockSize; global SB_ov_Size;
global OrderEst; global OutIter_Num; global order_gamma;  global OrderEstUppThr; global OrderEstLowThr;
global GammaRatioSet; global GammaRatio; global GammaRatioThr;
SNR = 45; % ���������������ȣ�in1 dB��
SIR_case = []; SDR_case = []; SAR_case = []; SNR_case = [];
SIR = []; SDR = []; SAR = []; SIR_time_all = [];
SIR_time_all = [];
timeblock_Length = 1; % online SIR����ֿ鳤�ȣ�in second��
case_num = 0;
room_imp_on = 1; seed = rng(10); PowerRatio = 2;
if mix_file == 1 file_num = 1; else file_num = 110; end
%% ������
tic
for mix_sim = mix_type for room_type = room_case for src_comb = src_comb_case for muteOn = muteOn_case for normal_type = normal_case for deavg = deavg_case for mix_SNR = mix_SNR_case for sim_mic = sim_mic_num for file_tag = file_num for prior = prior_case
%% �����ŵ�����
    if mix_sim % ʹ�÷����ŵ����з���
        [xR, s, Fss1, mic_pos, theta] = generate_sim_mix(src_comb, sim_mic, room_type, room_imp_on, DebugRato, muteOn, normal_type, deavg, mix_SNR);
    else % ʹ��ʵ¼�ź�
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
    xR_w = awgn(xR,SNR(1),'measured',0); noise = xR_w - xR; % �Ӹ�˹������
    audiowrite([sound_dir,'/mix.wav'], xR', Fss1);  audiowrite([sound_dir,'/mix_w.wav'], xR_w', Fss1);
    xR_t = audioread([sound_dir,'/mix.wav']);       xR_tw = audioread([sound_dir,'/mix_w.wav']);
    xR_1 = xR_t';    xR_1w = xR_tw'; % ���wav���ٶ�ȡ�Ļ���źţ�һ����xR���ɣ������ر𲻺�����xR_1����
    win_type = 'hann';    win_size = 4096;    inc = win_size / 2;    fft_size = win_size;
    spec_coeff_num = fft_size / 2 + 1; % frequency bin
%% ������
for determined = determined_case for online = online_case for batch_type = batch_type_case for tao = tao_case for win_exp_ratio = win_exp_ratio_range for taoMeanAdd = taoMeanAdd_case for taoMean = taoMean_case
    if online
        win_type = 'hamming'; % ����onlineʹ�ú�����
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

    seed = rng(10); % �̶�nmf�����������
    if batch_type == 1 
        total_iter = 1; % auxIVA���ù̶���ʼ��������Ҫ�ظ�����
    else if batch_type == 2 
          total_iter = total_iter_num; % nmfIVA���������ʼ������Ҫ�ظ�����
        end
    end
for ITER = 1:total_iter
    %% ����źų�ʼ��
    if AWGN_on
        x = xR_w;
    else
        x = xR; % һ����xR���ɣ������ر𲻺�����xR_1���ԡ�ʵ¼�ź���Ҫ�ĳ�xR_1
    end    
    [mic_num, sample_num] = size(x); source_num = desire_source;
    if determined source_num = mic_num; end
   
    %% �����������
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
    % ��4mic���������Ҫ���ĸ�orders����ʱ����orders1��orders2�ظ�һ��
    if sim_mic == 4 && determined option.n_orders_online = [n_orders1 n_orders2 n_orders1 n_orders2]; end
    option.nmf_iter_num = nmf_iter_num; option.nmf_fac_num = nmf_fac_num; option.nmf_beta = nmf_beta;
    option.prior = cell2mat(prior); option.nmf_p = nmf_p; option.nmfupdate = nmfupdate; option.nmf_b = nmf_b;
    if mix_type option.mic_pos = mic_pos; option.theta = theta; option.deltaf = deltaf;end
    GammaRatio = ones(1,8); option.initial_rand = initial_rand;  option.DOA_init = DOA_init;
    %% �źŴ����ä����
    if online
        % �����㷨(online)
        if source_num ~= 1
            [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option); % �����������online�汾
%         [s_est] = auxiva_audio_bss_online_single(x,source_num,option); 
%         [s_est,label] = nmfiva_audio_bss_online_perm(x,source_num,option); 
        else
            [s_est,label] = auxive_audio_bss_online_perm(x,option); % IVE
        end
        out_type = 'online';
    else
        % �����㷨(batch)
        if source_num ~= 1
            [s_est,label] = auxiva_audio_bss_batch(x,source_num,option); % IVA batch(including nmfiva)
        else
            [s_est,label] = auxive_audio_bss_batch(x,option); % IVE batch
        end
        out_type = 'batch';
    end    
  
    %% SIR SDR����
    if mix_sim
        L = min(size(s,2), size(s_est,2));
        if source_num == 1
            [SDR,~,SAR,perm] = bss_eval_sources(s_est(1,1:L), s(1,1:L));
            SIR = 0;
        end
        if AWGN_on % �Լ��޸ĵĴ�noise��SDR���㣬�����ƣ�������ʹ��
           % [SDR,SIR,SNR,SAR,perm] = bss_eval_noise(s_est(:,1:L), s(1:2,1:L),noise(:,1:L));
            [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(:,1:L));
            SNR = zeros(source_num,1);
        else % bss_eval 3.0 �汾SDR���㣬����ʹ��
            if source_num == 2 % ����4*2��2*2
                [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(:,1:L)); max_comb = 1:2;
                s_est = s_est(perm',:);
            else if source_num > 2 % ����4*4����Ҫѡ��·SIR֮���������
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
        
        if online % ���� online SIR improvment
            tap_Length = timeblock_Length * Fss1;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length);
            T = [0 : ceil(L / tap_Length)] * timeblock_Length;            SIR_time_all = [SIR_time_all;SIR_time];
        end 
        if total_iter ~= total_iter_num
            % �̶���ʼ���������ʼ��һ�����ʱ����Ҫ���̶���ʼ�������ݽ��и��������ٷ���ʱ�䡣
            SIR_case = [SIR_case repmat(SIR,[1,total_iter_num])];  SDR_case = [SDR_case repmat(SDR,[1,total_iter_num])];  SAR_case = [SAR_case repmat(SAR,[1,total_iter_num])];  SNR_case = [SNR_case repmat(SNR,[1,total_iter_num])];
        else
            SIR_case = [SIR_case SIR];  SDR_case = [SDR_case SDR];  SAR_case = [SAR_case SAR];  SNR_case = [SNR_case SNR];
        end
    end
    strsave= strcat(mkdir_str1,filenameTmp,'.mat'); sav=['save ' strsave]; eval(sav);
    %% ��������(case)����Ƶ
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
    % ����&����ͼ���ƣ�������ʵ¼�ź�ʱʹ�� -1.8152   -1.8152   -3.3037   -3.3037
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
%% ��������(all)&��ͼ
% ��Ҫ������ͼ���������ȡmat�ļ�
% load_case = 1;  load_case_str = num2str(load_case);
% load(strcat(mkdir_str1,'case_',load_case_str,'.mat')); %
%   load(strcat(mkdir_str1,'.mat'));
if mix_sim
    loadMat= 0;
    if loadMat == 1
        
        clear all; close all;
        filenameTmp ='DOA_ratio';%�ɽ����������Ϊ�ļ��� test_online_offline   test_AWGN_Lb test_AWGN_Lb
        mkdir_str=strcat('./Simulation_Single/',filenameTmp);
        mkdir(mkdir_str);%һ���оͻ��ڵ�ǰ�ļ����´���simulation�ļ���
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
         %   plotSIR_time(SIR_time_all,T,max(size(epsilon_ratio_range,2),size(Ratio_Rxx_case,2)),3); % ����ʱ��SIR         
sub_case_num = case_num;
packFigNum =   1;  % һ��fig �ֳɼ���(=packFigNum)subfugure����������Ƚ�
SortedPlotThr = 5; % >1,������ʾ����
SortedPlotNum = 5; % ����������ʾcase����Ŀ,�� sub_case_num �Ƚϴ��ʱ�����������ʾ
plotRatio =1;      %һ�ζ��ٱ�����ͼ�� default=1�� 2 ������50% ��ͼ��   
close all;  
% plotSIR_time(SIR_time_all,T,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % ����ʱ��SIR
plotSIR_time1(SIR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % ����ʱ��SIR ��source
%function plotSIR_time(SIR,T,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
%sub_case_num      ͬһ����Ƚ���һ��Figure ������Ŀ, ����һ�� 10 =size(SIR_time_all,1)/sourceNum (=2),һ��ͼ��2*5�Ƚϡ�
%packFigNum       һ�ΰѼ���subcase ���ŵ�һ����plot
%SortedPlotThr;   ���case_num̫����ʾ���ˣ��о��Ƿ���ʾ SortedPlotNum case��
%�����1��ȱʡ��ʾ���������������>1, ����������ʾ��
%SortedPlotNum;   ���case_num̫����ʾ���ˣ���ʾ SortedPlotNum case��
%plotRatio       һ�ζ��ٱ�����ͼ�� default=1�� 2 ������50% ��ͼ��
        else            
                        %plotSDR(case_num,SDR_total); % ������ͨSIR��SDR
            %plotSDR(case_num,SDR_total,SIR_total); % ������ͨSIR��SDR  
sub_case_num = 2;
packFigNum =  1;  %һ��fig �ֳɼ���(=packFigNum)subfugure����������Ƚ� 
SortedPlotThr = 1;%sub_case_num * packFigNum < SortedPlotThr������
SortedPlotNum = 2;%����������ʾcase����Ŀ,�� sub_case_num �Ƚϴ��ʱ�����������ʾ
plotRatio =1;     %һ�ζ��ٱ�����ͼ, default=1, 2 ������50% ��ͼ��
close all;
%function plotSDR3 plotSDR4
%sub_case_num  ͬһ����Ƚ���һ��Figure ������Ŀ
%packFigNum    һ��fig �ֳɼ���(=packFigNum)subfugure����������Ƚ�
%SortedPlotThr ���case_num̫����ʾ���ˣ��о��Ƿ���ʾ SortedPlotNum case 
%SortedPlotNum ���case_num̫����ʾ���ˣ���ʾ SortedPlotNum case��
%plotRatio     һ�ζ��ٱ�����ͼ, default=1, 2 ������50% ��ͼ
plotSDR3(SDR_total,SIR_total,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio);%����errorbar  
plotSDR4(SDR_total,SIR_total,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio);%��ֵ����boxplot
        end   
           
        
        
        
    end
end