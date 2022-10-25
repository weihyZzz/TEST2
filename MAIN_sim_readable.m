%% Main simulation with new RIR simulate function% Simulation Initialize
clear all; close all; addpath('room');addpath('components');addpath('bss_eval');addpath('plot');addpath('EM');addpath('nmfiva');addpath('IVE');addpath('audioSignalProcessTools');addpath('DOA');addpath('MNMF');addpath('dataoppo');
filenameTmp ='test';% AGC_method �ɽ����������Ϊ�ļ��� test_online_offline   test_AWGN_Lb test_AWGN_Lb
%%for files=1:100 filenameTmp =strcat('0dB_f01_80_0_chat_200_60-',num2str(files)); 
%filenameTmp =strcat('0dB_m01_30_0_chat_200_60-',num2str(files)); 
%mkdir_str=strcat('./Simulation_Single/chat-m01_online_auxiva/',filenameTmp);
mkdir_str=strcat('./Simulation_Single/',filenameTmp);
mkdir(mkdir_str);%һ���о�  ���ڵ�ǰ�ļ����´���simulation�ļ���
mkdir_str1 =strcat(mkdir_str,'/'); sound_dir= strcat(mkdir_str1,'sound'); mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);
mkdir_str =strcat(mkdir_str,'.m');Save_link1=Save_link_file('MAIN_sim_readable.m',mkdir_str);%����ǰ�������Ϊnk�ļ�
if isOctave ;    graphics_toolkit('gnuplot');    warning('off', 'Octave:broadcast'); end
% mix_file = 'data/mix/BSS_VECTOR/2020_07_06_14_54_49.wav';% ʵ¼�ź�λ��
mix_file = 0;
%% case setting, Room simulation setting %%%
DebugRato =10/10;                % �����ã�2����2�����ȵĲ������ݣ�0.5����1/2�����ݣ����ܵ���0.1
mix_type = [ 1];                 % 0��ʵ¼��1������
real_src_num = 2;                % ʵ¼�ź�����Ҫ�����Դ����Ŀ
customize_room = 1;              % �Ƿ�ʹ���Զ��巿���С
room_case = [2 ];                % Scenario : 1��the oldest one with angle ; 2��the oldest one   3��mic>num Scenario; 4��Block-online Scenario 1
%                                  5��Block-online Scenario 2; 6��online Scenario(200ms)  7��large mic array room(up to 8 mic)
% ��customize_roomΪ1����caseѡ����Ч���������涨�巿���С
rand_select = 0; % 1:���ѡ��Դ�ź���� 0:�̶�ѡ��Դ�ź����,��Ϊ�̶����
% �̶�ģʽ�¸��������indexѡ��Դ�źź͸����ź�  ��Ϊ�̶�ģʽ�����Բ��õ��������Ŀ��Դ�͸���Դ��Ŀ��������Զ���ȡindex��С  [1 4] [1 5] [1 6] [1 7] [1 8] [2 3] [2 4] [2 5] [2 6] [2 7] [2 8]
target_index_case = {[1 7] }; intf_index_case = {[] };
% target_index_case = {[2 ]}; intf_index_case = {[3]}; ������ȼۡ�
% �ɹ�ѡ���Դ��� = {1��Ů��1, 2��Ů��3, 3������1, 4�����Ļ��Ѵ�, 5��Ӣ�Ļ��Ѵ�,  6��ϴ�»�, 7������, 8����˹������}
sim_mic_num = [2];             % ������˷���Ŀ����ʵ¼Ҫ��4mic����Ҳ�����4������ֻ��������ͨ�� 
target_source_case = [2];      % Ŀ��Դ��Ŀ����ֵ<=2��=1ʱΪ��Դ��ȡ
intf_source_case = [0];        % ����Դ��Ŀ��0=<L<6����Ϊfix select�����ó�index�ĳ���
muteOn_case = [0];             % ѡ��ڶ���source����ǰ�������mute���ڶ���sourceһ��Ϊtarget source)
deavg_case = [1];              % ����ź�ȥ��ֵ�������ʵ¼������
mix_SINR_case = [0];           % Ŀ���źź����������߸����źţ��Ļ���Ÿ���ȣ���mix_SINR=0����Ϊ1:1��ϣ�����normal_case=3ʱʹ��
SINR_diffuse_ratio_case = [1]; % ����ظ������ܸ���֮�ȣ�1-diffuse_ratioΪAWGN�ɷ֣���=1��Ϊ����AWGN
determined_case = [0];         % 0:overdetermined(mic > src); 1:determined(mic = src)
target_SIRSDR = 0;             % 1:ֻ��ʾtarget source��SIRSDR��0:ȫ����ʾ (������ʽ��δд��������
%%  Room environment setting %%
reverbTime_case = [0.08];      % �������ʱ�䣬��λΪ��
angle_case =       [40];       % ��Դ��С�н�,angle from 0 to 360,��λΪ�ȡ�0�Ƚ�ΪX���ҷ���
angle_start_case = [55];       % �Ӹ�λ�ÿ�ʼ����Դ���Ų�,�Ƕ�Ҫ��ͬ�ϡ�>180ʱ����0�ȴ��棬eg.180,225��Ӧ��-45,0����
R_ratio_case =     [0.8];      % R = Դ����˷�����������max_R * R_ratio, ȡֵfrom 0 to 1
tR_ratio_case =   [0.66];      % Ŀ��Դ�����Դ֮��ľ����ֵ
src_permute_type_case = [0];   % 1�����Դ�Ų���0��˳��Դ�Ų�
room_plot_on = 2;              % �Ƿ���Ʒ���ƽ��ͼ��0�������ƣ�1��ÿ�ζ����ƣ�2�������Ƶ�һ��
room_size_case = {[10 8 3]};   % �����С �����
mic_center_case = {[5 4 1.2]}; % ��˷������������� ����Ĭ�ϸ߶���ͬ
mic_distx_case = [0.158];      % ������˾����ˮƽ����ͶӰ���� ��ֵ��x�������� ��ֵ��x�Ḻ����
mic_disty_case = [0];          % ������˾���Ĵ�ֱ����ͶӰ���� ��ֵ��y�������� ��ֵ��y�Ḻ����
%% Pre-batch & Initialize %%
initial_case = [0 ];           % C & V initialization, 0:normal, 1:use first-frame-batch
pre_batch_case = [0 ];         % using pre-batch to initialize Vs, 0: no pre-batch; 1:prebatch(continue); 2:prebatch(restart); 3:preonline(restart);
batch_update_num_case = [ 20]; % pre_batch_num
prebatch_iter_num = 20;
initial_rand_case = [0];       % V initialization using random diagonal domiance matrix; 1:full, 2:only low freq(1/3), 0:off;
%%  Essential setting %% 
online_case = [ 1 ];           %�Ƿ������ߡ�0 �����ߣ�1 �����ߣ�
batch_type_case = [ 1];        % batch IVA�����ͣ�1����auxiva��2����nmfiva��3Ϊauxiva-iss��ע��2������ģʽ������
Lb_case = [1 ];                % length of blocks [1,2,4,6,8,16], default using 1
tao_case =[0]/10;              %>1  default value is 0, default hanning windowing operation; >1,  designed based special criterion;
                               % windowing factor default value is 1.2; 10^5, indicates no windowing;
win_exp_ratio_range =[20:1:20]/20; % default value is 1,effective for tao>0;
win_size_case = [4096];        % window length
taoMeanAdd_case = [1]/10;      % effective for tao>0; add some value, such as 0.1 to robustify the window desgin;
taoMean_case = [1];            % default value is 1,effective for tao>0;
D_open_case = [ 1];            % using D ����Ȩ
perm_on_case = [0];            % ����ź�֡��������
forgetting_fac_case = [1]/1000;% ������ʹ�ã��൱�������е�1-alpha [0.3 0.2 0.1 0.08 0.06 0.04]
gamma_case = [4:4]/100;        % ������mic>srcʹ�ã�ΪC��forgetting factor
delta_case = [0]/10^0;         % using in contrast function: (||x||^2 + delta)^n_orders
whitening_case = [0];          % �Ƿ�ʹ�ð׻���δ������ɣ���ʱ��Ҫ��
project_back = 1;              % �Ƿ���Ҫ������źŷ������������ź���ͬ�ĳ߶�
n_orders_case = {[1/2 1/2 1/2 1/2]}; % ���ߵ�orders {[1/6 1/2],[1/2 1],[1/2 1/2],[1/8 1/2]}
n_orders_casenum = size(n_orders_case,2);
%%  Partition %%%
parti_case = [0 ];            % �Ƿ�ʹ���ӿ�BSS
SubBlockSize_case = [100];    %  ʹ���ӿ�Ĵ�С��
SB_ov_Size_case = [1]/2;      %  overlap �ı�������СΪ round(SubBlockSize*SB_ov_Size)��
partisize_case = [1];
select_case = [0];            % �Ƿ�ѡ���ӿ飨��parti=1���˴�Ϊѡ�ӿ飻��parti=0���˴�Ϊѡ���ز���
thFactor_case = [0]/200;      % ѡ���ӿ���ֵ����
%% Epsilon test %%
diagonal_method_case =[ 0 ];  %  0: using a tiny portion of the mean diagnal values of Xnn��          
                              % 1: using a tiny portion of the diagnal value of Xnn��       
Ratio_Rxx_case =[1:1]/10^3;   %1:3:10 default is 0; 2 is a robust value;
epsilon_ratio_range =[10:10]/10; % 1:4 10:10:60 robustify inversion of update_w ;
epsilon_ratio1_range =[10]/10;% 1:4 10:10:60 robustify inversion of update_w ;
frameStart_range =[1];        % 1 ������ʼ��һ���� epsilon��  ����ǰframeStartǰ�� epsilon_start_ratio;
epsilon_start_ratio_range =[1:1:1]/1; %  ����ǰframeStartǰ�� ;
%% Iteration num %%
iter_num_case = [20 ];         % DOA+20�Σ��㷨����������online �̶�Ϊ2�Σ�offline��ֵΪcase��ֵ
inner_iter_num_case = [1];     % �㷨�ڵ���������online�̶�Ϊ1�Σ�offline��ֵΪcase��ֵ
total_iter_num_case = [1];     % �ظ��������������㷨����ʱΪ���������ʼ��ʱ��ƽ��SDR��SIR�õġ�
%% Order estimation %% 
OrderEst_range =[0];           % �Ƿ����order estimation��
OutIter_Num_range =[1];        % order ���ƵĴ�����
OrderEstUppThr_range =[11]/10; % order ���Ƶ����ޣ�
OrderEstLowThr_range =[4]/10;  % order ���Ƶ����ޣ�
order_gamma_range =[8]/10;     %  order ���ƵĻ���ϵ����
n_orders1_range =[20:20]/40;   % pdf �Ĺ���ָ��������orders��
n_orders2_range =[20:20]/40;   % pdf �Ĺ���ָ��������orders��
verbose_range =[1];            % 0 �������м�����1 �����м�����
%% Gamma ratio %%
GammaRatioThr_range =[10^2];   % �Ƿ�������û�з������Ƶ����ޣ��ǳ���ȼ۲�������
GammaRatioSet_range =[10]/10;  % �Ƿ�������û�з������Ƶ�gamma ֵ��
%% Mix Model Estimation %% 
mix_model_case = [0];          % �Ƿ�ʹ�û��CGGģ��,1��Ӳ�У�2��EM��0����ʹ��
ita_case = [0.9];              % ���CGGģ���л�ϸ��ʺ�betaֵ�ĵݹ�ƽ��ϵ��
%% NMF Setting %%
nmf_iter_num_case = [1];       % nmf�ڵ���������1���ɣ������������ֵ
nmf_fac_num_case = [9];        % nmf����Ŀ, 9 �ǵ���ֵ
nmf_b_case = [1/2];            % IS-NMFָ��ֵ��p=beta=2ʱ��GGD-NMF��ЧΪb=1/2��IS-NMF
nmf_beta_case = [2];           % betaֵ������GGD-NMF
nmf_p_case = [0.5];            % pֵ������GGD-NMF
nmf_update_case = [0];         % nmf update��ģ�ͣ�0����IS-NMF��1��GGD-NMF
%% MNMF Setting %%
MNMF_case = [0];               % 1-ʹ��MNMF 0-��ʹ��MNMF
MNMF_refMic_case = [1];        % reference mic for performance evaluation using bss_eval_sources 1 or 2
MNMF_nb_case = [20];           % number of NMF bases for all sources (total bases)
MNMF_it_case = [300];          % number of MNMF iterations
MNMF_first_batch_case = [20];  % first mini-batch size
MNMF_batch_size_case = [4];    % mini-batch size
MNMF_rho_case = [0.9];         % the weight of last batch;  ȱʡ0.9
MNMF_fftSize_case = [4096];    % window length in STFT [points]
MNMF_shiftSize_case = [2048];  % shift length in STFT [points]
MNMF_delta_case = 10.^[-12];   % to avoid numerical conputational instability
MNMF_p_norm_range =[5:5]/10;   % default is 0.5; sqrt(x);
MNMF_drawConv = false;         % true or false(�����߿���)(true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
%% ILRMA Initialization Setting %%
ILRMA_init_case = [0];         % 1:ʹ��ILRMA��ʼ�� 0:��ʹ��ILRMA��ʼ��
ILRMA_type_case = [2];         % 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
ILRMA_nb_case = [20];          % number of bases (for type=1, nb is # of bases for "each" source. for type=2, nb is # of bases for "all" sources)
ILRMA_it_case = [20];          % iteration of ILRMA
ILRMA_normalize_case = true;   % true or false (true: apply normalization in each iteration of ILRMA to improve numerical stability, but the monotonic decrease of the cost function may be lost. false: do not apply normalization)
ILRMA_dlratio_case = 10.^[-2]; % diagonal loading ratio of ILRMA
ILRMA_drawConv = false;        % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
%% Prior AuxIVA %%
DOA_switch_case=[  1];        % ����DOA,0-[0,0,0,0]&doa_0; 1-[1,1,1,0]&doa_1
% prior_case = {[1,1,1,0]};   % ����Ϊ(1,k)ά��������k��Ϊ1�����е�k��Դ��������Ϣ��
%                             % ��k��Ϊ0����û�е�k��Դ�����飬����DOA��������
%                             % ��ΪOverIVA��IVEʱ����K+1��prior����BG��prior��KΪtarget��Ŀ��
DOA_esti_case = [ 1 ];        % �Ƿ�ʹ��DOA���ƣ�Ϊ0��ʹ�÷�����洫�ݵ�DOA��Ϣ 
DOA_update_case = [ 0];       % �Ƿ��ڵ����и���DOA
DOA_init_case = [ 0];         % �Ƿ�ʹ��DOA��ʼ����
esti_mic_dist_case = 0.158*[10]/10; % DOA����ʱʹ�õ���˷���
% �˴���ʽ���£� P = (DOA_tik_ratio * eye(M) + sum(DOA_Null_ratio * hf * hf') /deltaf^2
DOA_Null_ratio_range = [0.006]/10; % DOA  ��Ȩֵ��[0.1]/10-4mic, 
DOA_tik_ratio_range = [0.5]/1000; % DOA Tik ��Ȩֵ��[0.5]/1000-4mic
deltaf_case =   [30];        % Ŀ��Դ��һ��ϵ��10-4mic
deltabg_case = [0.5];        % ����������һ��ϵ��
annealing_case = [0];        % �Ƿ�ģ���˻�fac_a = max(0.5-iter/iter_num, 0);
PowerRatio_case = [20:20]/10;% ���չ��ʵ����ӿ��ǣ�ȱʡPowerRatio = 2;
%% IVE �ķ�����1ΪIP-1����Чoveriva��, 2ΪIP-2��3���忴auxive_update˵����4ΪFIVE����
IVE_method_case = [2]; 
%% Parameters initialization %%
Initiallize; 
timeblock_Length = 1; % online SIR����ֿ鳤�ȣ�in second��
plot_time_mode = 1; % ����SIR_time/SDR_time��ģʽ  1:0-1,0-2,0-3...  2:0-1,1-2,2-3...
case_num = 0; room_imp_on = 1;   RandomSeed =0; % 0 ÿ��������ͬ�����1 ÿ�������ͬ���
if RandomSeed==0 randn('state',98765); rand('state',12345); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
if RandomSeed==1 randn('state',cputime); rand('state',cputime+1); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
if mix_file == 1 file_num = 1; else file_num = 1; end
% �̶�ѡ��sourceʱ��target��intf����Ŀͳһ��index��Ŀ��case����Ϊ��һcase��ֹ�ظ����档
if ~rand_select target_source_case = [0]; intf_source_case = [0]; end; tic   %% ������
for DOA_switch = DOA_switch_case  %% DOA     
    if DOA_switch    prior_case = {[1,1,1,0]};    DOA_esti_case  = [1];
    else     prior_case = {[0,0,0,0]};    DOA_esti_case = [0];     end
for mix_sim = mix_type 
    if mix_sim == 0 % ʵ¼�ź�������
        filetest = '/test'; % data�ļ����´����Ƶ����Ŀ¼
        fileFolder = fullfile('./data',filetest); dirOutput = dir(fullfile(fileFolder,'*.wav')); fileNames = {dirOutput.name}; fileNum = length(fileNames);
    else   fileNum = 1;    end
for fn = 1:fileNum case_num = 0;
for room_type = room_case for target_idx = target_index_case for intf_idx = intf_index_case for muteOn = muteOn_case for SINR_diffuse_ratio = SINR_diffuse_ratio_case
for deavg = deavg_case for mix_SINR = mix_SINR_case for sim_mic = sim_mic_num for target_source_num = target_source_case  for intf_source_num = intf_source_case for file_tag = file_num for prior = prior_case
for angle = angle_case for angle_start = angle_start_case for R_ratio = R_ratio_case for tR_ratio = tR_ratio_case for src_permute_type = src_permute_type_case for reverbTime = reverbTime_case
for room_size = room_size_case for mic_center = mic_center_case for mic_distx = mic_distx_case for mic_disty = mic_disty_case
    if mix_sim  %% �����ŵ����� ʹ�÷����ŵ����з���        % room��������
        room.room_type = room_type; roon.muteOn = muteOn; room.deavg = deavg; room.room_imp_on = room_imp_on; room.muteOn = muteOn; room.reverbTime = reverbTime;
        room.SINR = mix_SINR; room.sim_mic = sim_mic; room.target_source_num = target_source_num; room.intf_source_num = intf_source_num; room.tR_ratio = tR_ratio;
        room.rand_select = rand_select; target_index = cell2mat(target_idx); intf_index = cell2mat(intf_idx); room.SINR_diffuse_ratio = SINR_diffuse_ratio;
        room.angle = angle; room.angle_start = angle_start; room.R_ratio = R_ratio; room.src_permute_type = src_permute_type; room.DebugRato = DebugRato;
        room.customize_room = customize_room; room.size = cell2mat(room_size); room.mic_center = cell2mat(mic_center); room.mic_distx = mic_distx; room.mic_disty = mic_disty;  
        % RIR���溯��
        [xR, s, fs_ref, mic_pos, theta, target_source, intf_source, layout] ...
            = generate_sim_mix_new(room,target_index,intf_index);
    else % ʹ��ʵ¼�ź�
        target_source = real_src_num;        file_name = num2str(file_tag);
        if file_tag < 10
            file_name = strcat('0',num2str(file_tag));
        else if file_tag < 100 && file_tag >= 10
                file_name = strcat(num2str(file_tag));
            end
        end
%        [x_pca, Fssp] = audioread('data/music.wav');        %  mix_file = strcat(,file_name,'.wav');
        %   mix_file = strcat('O/0dB_f01_80_0_chat_200_60-',file_name,'.wav');
        soundfile = fileNames{fn}; mix_file = strcat(fileFolder,'/',soundfile); foldname = soundfile(1:end-4);
        sound_dir= strcat(mkdir_str1,'sound'); sound_dir = strcat(sound_dir,'/',foldname); mkdir(sound_dir);
        [xR_t, Fssr] = audioread(mix_file); real_mic = size(xR_t,2);  s=xR_t.';
%         clip_start = 70*Fssr; clip_end = size(xR_t,1);      
        xR_t = preprocess(xR_t, 2, deavg);
        if sim_mic == real_mic
            xR = xR_t.';
        else if sim_mic < real_mic
                xR = xR_t(:,[3,4]).';
            end
        end
        fs_ref = Fssr;
    end 
    % �������ź�
    audiowrite([sound_dir,'/mix.wav'], xR.', fs_ref); 
    xR_t = audioread([sound_dir,'/mix.wav']); xR_1 = xR_t.';  % ���wav���ٶ�ȡ�Ļ���źţ�һ����xR���ɣ������ر𲻺�����xR_1����
    % FFT�����ʹ���������
    for win_size = win_size_case
    win_type = 'hann';     inc = win_size / 2;
%% ������
for determined = determined_case for online = online_case for batch_type = batch_type_case for tao = tao_case for win_exp_ratio = win_exp_ratio_range for taoMeanAdd = taoMeanAdd_case for taoMean = taoMean_case    
    if online         win_type = 'hamming'; end % ����onlineʹ�ú�����     
    if tao >0 
        [win_ana, ~]= WinGen(win_size,tao,taoMeanAdd,taoMean,inc,win_exp_ratio); 
        % [win_ana1, ~]= WinGen(win_size,1.2,taoMeanAdd,taoMean,inc,win_exp_ratio); plot(win_ana,'b'); hold on;   plot(win_ana1,'r'); 
    else
        [win_ana, ~] = generate_win(win_type, win_size, inc);
    end
    win_syn =win_ana; %ones(1, win_size);   
for D_open = D_open_case for perm_on = perm_on_case for iter_num = iter_num_case  for inner_iter_num = inner_iter_num_case  for forgetting_fac = forgetting_fac_case for gamma = gamma_case for delta = delta_case for whitening_open = whitening_case
for initial = initial_case for initial_rand = initial_rand_case for pre_batch = pre_batch_case for batch_update_num = batch_update_num_case for Lb = Lb_case for n_orders_num = 1:n_orders_casenum for parti = parti_case  for SubBlockSize = SubBlockSize_case 
for SB_ov_Size = SB_ov_Size_case for select = select_case for partisize = partisize_case for thFactor = thFactor_case for diagonal_method = diagonal_method_case for Ratio_Rxx = Ratio_Rxx_case   for epsilon_ratio = epsilon_ratio_range  for epsilon_ratio1 =epsilon_ratio1_range
for OrderEst = OrderEst_range for OutIter_Num = OutIter_Num_range for order_gamma = order_gamma_range for OrderEstUppThr = OrderEstUppThr_range for OrderEstLowThr = OrderEstLowThr_range for frameStart =frameStart_range for epsilon_start_ratio =epsilon_start_ratio_range            
for GammaRatioThr = GammaRatioThr_range for  GammaRatioSet = GammaRatioSet_range for mix_model = mix_model_case for ita = ita_case for n_orders1 =n_orders1_range for n_orders2 =n_orders2_range for verbose = verbose_range for nmfupdate = nmf_update_case
for nmf_iter_num = nmf_iter_num_case for nmf_fac_num = nmf_fac_num_case for nmf_b = nmf_b_case for nmf_beta = nmf_beta_case for nmf_p = nmf_p_case for deltaf = deltaf_case  for deltabg = deltabg_case for DOA_Null_ratio = DOA_Null_ratio_range  for DOA_tik_ratio = DOA_tik_ratio_range   
for DOA_esti = DOA_esti_case for DOA_update = DOA_update_case for esti_mic_dist = esti_mic_dist_case for DOA_init = DOA_init_case for annealing = annealing_case for  PowerRatio =  PowerRatio_case for IVE_method = IVE_method_case for total_iter_num = total_iter_num_case 
for MNMF = MNMF_case for MNMF_refMic = MNMF_refMic_case for MNMF_nb = MNMF_nb_case for MNMF_it = MNMF_it_case for MNMF_first_batch = MNMF_first_batch_case for MNMF_batch_size = MNMF_batch_size_case for MNMF_rho = MNMF_rho_case for MNMF_fftSize = MNMF_fftSize_case for MNMF_shiftSize = MNMF_shiftSize_case
for MNMF_delta = MNMF_delta_case for MNMF_p_norm = MNMF_p_norm_range for ILRMA_init = ILRMA_init_case for ILRMA_type = ILRMA_type_case for ILRMA_nb = ILRMA_nb_case for ILRMA_it = ILRMA_it_case for ILRMA_normalize = ILRMA_normalize_case for ILRMA_dlratio = ILRMA_dlratio_case
if RandomSeed==0 randn('state',98765); rand('state',12345); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
if RandomSeed==1 randn('state',cputime); rand('state',cputime+1); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
    if batch_type ~= 2 || MNMF == 1
        total_iter = 1; % auxIVA���ù̶���ʼ��������Ҫ�ظ�����
    else if batch_type == 2 
          total_iter = total_iter_num; % nmfIVA���������ʼ������Ҫ�ظ�����
        end
    end    
for ITER = 1:total_iter    %% ����źų�ʼ��
    x = xR; % һ����xR���ɣ������ر𲻺�����xR_1���ԡ�   
    [mic_num, sample_num] = size(x); source_num = target_source;
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
    option.MNMF_nb = MNMF_nb; option.MNMF_it = MNMF_it; option.MNMF_first_batch = MNMF_first_batch; option.MNMF_batch_size = MNMF_batch_size; option.MNMF_rho = MNMF_rho;
    option.MNMF_fftSize = MNMF_fftSize; option.MNMF_shiftSize = MNMF_shiftSize; option.ILRMA_init = ILRMA_init; option.ILRMA_type = ILRMA_type; option.ILRMA_nb = ILRMA_nb;
    option.MNMF_delta = MNMF_delta; option.ILRMA_it = ILRMA_it; option.ILRMA_normalize = ILRMA_normalize; option.ILRMA_dlratio = ILRMA_dlratio;
    option.MNMF_drawConv = MNMF_drawConv; option.ILRMA_drawConv = ILRMA_drawConv;
    % ��4mic���������Ҫ���ĸ�orders����ʱ����orders1��orders2�ظ�һ��
    if sim_mic == 4 && determined option.n_orders_online = [n_orders1 n_orders2 n_orders1 n_orders2]; end
    option.nmf_iter_num = nmf_iter_num; option.nmf_fac_num = nmf_fac_num; option.nmf_beta = nmf_beta;
    option.prior = cell2mat(prior); option.nmf_p = nmf_p; option.nmfupdate = nmfupdate; option.nmf_b = nmf_b; 
    if mix_type option.mic_pos = mic_pos; option.theta = theta; end 
    GammaRatio = ones(1,8); option.initial_rand = initial_rand;  option.DOA_init = DOA_init; 
    option.deltabg = deltabg; option.project_back = project_back; option.annealing = annealing;
    option.prebatch_iter_num = prebatch_iter_num; option.DOA_esti = DOA_esti; option.DOA_update = DOA_update;
    option.esti_mic_dist = esti_mic_dist; option.mix_sim = mix_sim; option.deltaf = deltaf;
    %% �źŴ����ä����
    if online       % �����㷨(online)
        if MNMF == 1
            [s_hat,label] = bss_multichannelNMF_online(x.',source_num,option);
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif source_num ~= 1
            [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option); % �����������online�汾
%         [s_est] = auxiva_audio_bss_online_single(x,source_num,option); 
%         [s_est,label] = nmfiva_audio_bss_online_perm(x,source_num,option); 
        else
            [s_est,label] = auxive_audio_bss_online_perm(x,option); % IVE
        end
        out_type = 'online';
    else
        % �����㷨(batch)
        if MNMF == 1
            [s_hat,label] = bss_multichannelNMF_offline(x.',source_num,option);
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif source_num ~= 1
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
            [SDR_in,~,SAR_in,~] = bss_eval_sources(x(1,1:L), s(1,1:L));
            [SDR_out,~,SAR_out,perm] = bss_eval_sources(s_est(1,1:L), s(1,1:L));
            SIR = 0; SDR=SDR_out-SDR_in; SAR=SAR_out-SAR_in;
        end
%         if AWGN_on % �Լ��޸ĵĴ�noise��SDR���㣬�����ƣ�������ʹ��
%            % [SDR,SIR,SNR,SAR,perm] = bss_eval_noise(s_est(:,1:L), s(1:2,1:L),noise(:,1:L));
%             [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(:,1:L));
%             SNR = zeros(source_num,1);
%         else % bss_eval 3.0 �汾SDR���㣬����ʹ��
            if source_num == 2 || source_num == 4 % ����4*2��2*2
                [SDR_in,SIR_in,SAR_in,~] = bss_eval_sources(x(:,1:L), s(1:source_num,1:L));
                [SDR_out,SIR_out,SAR_out,perm] = bss_eval_sources(s_est(:,1:L), s(1:source_num,1:L)); max_comb = 1:source_num;
                SDR=SDR_out-SDR_in; SIR=SIR_out-SIR_in; SAR=SAR_out-SAR_in;
                s_est = s_est(perm',:);
            elseif source_num > 2 % ����4*4����Ҫѡ��·SIR֮���������
                perm_comb = nchoosek(1:source_num,2); [comb_num,~] = size(perm_comb);
                SDR_c = zeros(2,comb_num); SIR_c = zeros(2,comb_num); SAR_c = zeros(2,comb_num); perm_c = zeros(2,comb_num);
                for cn = 1:comb_num
                    [SDR_c(:,cn),SIR_c(:,cn),SAR_c(:,cn),perm_c(:,cn)] = bss_eval_sources(s_est(perm_comb(cn,:),1:L), s(:,1:L));
                end
                [~, max_perm_index] = max(sum(SDR_c,1)); max_comb = perm_comb(max_perm_index,:);
                SDR = SDR_c(:,max_perm_index); SIR = SIR_c(:,max_perm_index); SAR = SAR_c(:,max_perm_index); perm = perm_c(:,max_perm_index);
                s_est = s_est(max_comb(perm'),:); label = label{max_comb(perm')};
            end
            SNR = zeros(2,1);
%         end
        if target_SIRSDR;      SIR = SIR(1); SDR = SDR(1); SAR = SAR(1); SNR = SNR(1);        end
        fprintf('%s\nSDR = %s\nSIR = %s\n',out_type,num2str(SDR'),num2str(SIR'));        
        if online % ���� online SIR improvment
            tap_Length = timeblock_Length * fs_ref;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length,plot_time_mode); SDR_time = cal_SDR_time(x,s,s_est,tap_Length,plot_time_mode);
            T = [0 : ceil(L / tap_Length)] * timeblock_Length;            SIR_time_all = [SIR_time_all;SIR_time]; SDR_time_all = [SDR_time_all;SDR_time];
        end         
        if total_iter ~= total_iter_num
            % �̶���ʼ���������ʼ��һ�����ʱ����Ҫ���̶���ʼ�������ݽ��и��������ٷ���ʱ�䡣
            SIR_case = [SIR_case repmat(SIR,[1,total_iter_num])];  SDR_case = [SDR_case repmat(SDR,[1,total_iter_num])];  SAR_case = [SAR_case repmat(SAR,[1,total_iter_num])];  SNR_case = [SNR_case repmat(SNR,[1,total_iter_num])];
        else
            SIR_case = [SIR_case SIR];  SDR_case = [SDR_case SDR];  SAR_case = [SAR_case SAR];  SNR_case = [SNR_case SNR];
        end
    end    
    %% ��������(case)����Ƶ
    strsave= strcat(mkdir_str1,filenameTmp,'.mat'); sav=['save ' strsave]; eval(sav);
    case_num = case_num + 1;    case_str = num2str(case_num);
%     filenameTmp1 = strcat('case_',case_str,'.mat'); strsave= strcat(mkdir_str1,filenameTmp1);
%     sav=['save ' strsave]; eval(sav);
    % Save separated wave files
    sep1_str = label{1};
    audiowrite([sound_dir,'/sep1_case',case_str,'_',sep1_str,'_',out_type,'.wav'], s_est(1,:)', fs_ref);
    if size(s_est,1) == 2
        sep2_str = label{2};
        audiowrite([sound_dir,'/sep2_case',case_str,'_',sep2_str,'_',out_type,'.wav'], s_est(2,:)', fs_ref);
    end
    if size(s_est,1) == 4
        audiowrite([sound_dir,'/sep3_case',case_str,'_',out_type,'.wav'], s_est(3,:)', fs_ref);   audiowrite([sound_dir,'/sep4_case',case_str,'_',out_type,'.wav'], s_est(4,:)', fs_ref);   
    end
    % ����&����ͼ���ƣ�������ʵ¼�ź�ʱʹ�� -1.8152   -1.8152   -3.3037   -3.3037
%     if mix_sim == 0
%         plot_sound(s_est,xR,fs_ref,label);
% %         print(gcf,'-djpeg',['.\plot\save\',case_str+1,'.jpeg']);
%     end
    % ���Ʒ��䲼��ͼ
    if room_plot_on == 1 && mix_sim == 1
        plot_room_layout(layout);
    elseif room_plot_on == 2 && mix_sim == 1
        plot_room_layout(layout);     room_plot_on = 0;
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
    loadMat=0;
    if loadMat == 1
        
        clear all; close all;
        filenameTmp ='test';%�ɽ����������Ϊ�ļ��� test_online_offline   test_AWGN_Lb test_AWGN_Lb
        mkdir_str=strcat('./Simulation_Single/',filenameTmp);
        mkdir(mkdir_str);%һ���оͻ��ڵ�ǰ�ļ����´���simulation�ļ���
        mkdir_str1 =strcat(mkdir_str,'/'); sound_dir = strcat(mkdir_str1,'/sound');
        mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);  
        loadstrcase = strcat(mkdir_str1,filenameTmp,'.mat')
        loadsr=['load  ' loadstrcase];       eval(loadsr);       

        
        if target_SIRSDR R_num = 1; else R_num = 2; end
        case_num = size(SDR_case,2) / total_iter_num;
        SIR_total = reshape(SIR_case,R_num, total_iter_num, case_num);  SDR_total = reshape(SDR_case,R_num, total_iter_num, case_num);
        SAR_total = reshape(SAR_case,R_num, total_iter_num, case_num);  SNR_total = reshape(SNR_case,R_num, total_iter_num, case_num);
       case_name = {'non-DOA','DOA'};%�����㷨���� ���������������case1,case2,case3
       case_name={};
        %online =1;
        if online
         %   plotSIR_time(SIR_time_all,T,max(size(epsilon_ratio_range,2),size(Ratio_Rxx_case,2)),3); % ����ʱ��SIR         
sub_case_num = case_num;
packFigNum =   1;  % һ��fig �ֳɼ���(=packFigNum)subfugure����������Ƚ�
SortedPlotThr = 1; % >1,������ʾ����
SortedPlotNum = 1; % ����������ʾcase����Ŀ,�� sub_case_num �Ƚϴ��ʱ�����������ʾ
plotRatio =1;      %һ�ζ��ٱ�����ͼ�� default=1�� 2 ������50% ��ͼ��   
DOA_switch_case=[];
 close all;  plotSIR_time1(SIR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,DOA_switch_case); %DOA_Null_ratio_range DOA_switch_case deltaf_case deltabg_case  deltabg_case deltaf_case  DOA_Null_ratio_range DOA_esti_case angle_start_case prior_casen_orders1_range,n_orders2_range ����ʱ��SIR
 
 close all;  plotSIR_time1(SDR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,DOA_switch_case); %DOA_Null_ratio_range DOA_switch_case deltaf_case deltabg_case  deltabg_case deltaf_case  DOA_Null_ratio_range DOA_esti_case angle_start_case prior_casen_orders1_range,n_orders2_range ����ʱ��SDR

% plotSIR_time1(SDR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,n_orders1_range,n_orders2_range); % ����ʱ��SIR
%function plotSIR_time(SIR,T,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
%sub_case_num      ͬһ����Ƚ���һ��Figure ������Ŀ, ����һ�� 10 =size(SIR_time_all,1)/sourceNum (=2),һ��ͼ��2*5�Ƚϡ�
%packFigNum       һ�ΰѼ���subcase ���ŵ�һ����plot
%SortedPlotThr;   ���case_num̫����ʾ���ˣ��о��Ƿ���ʾ SortedPlotNum case��
%�����1��ȱʡ��ʾ���������������>1, ����������ʾ��
%SortedPlotNum;   ���case_num̫����ʾ���ˣ���ʾ SortedPlotNum case��
%plotRatio       һ�ζ��ٱ�����ͼ�� default=1�� 2 ������50% ��ͼ��
        else
            R_num = size(SDR_case,1) ;
            case_num = size(SDR_case,2) / total_iter_num;
            SIR_total = reshape(SIR_case,R_num, total_iter_num, case_num);  
            SDR_total = reshape(SDR_case,R_num, total_iter_num, case_num);
            SAR_total = reshape(SAR_case,R_num, total_iter_num, case_num);
sub_case_num = 4;
packFigNum =  1;  %һ��fig �ֳɼ���(=packFigNum)subfigure����������Ƚ� 
SortedPlotThr = 1;%sub_case_num * packFigNum < SortedPlotThr������
SortedPlotNum = 4;%����������ʾcase����Ŀ,�� sub_case_num �Ƚϴ��ʱ�����������ʾ
plotRatio =1;     %һ�ζ��ٱ�����ͼ, default=1, 2 ������50% ��ͼ��
close all;
            if ~isempty(case_name)
                plotSDR_name(case_num,SDR_total,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,case_name); % ���ƴ����㷨���Ƶ�SDR
            else
                plotSDR_name(case_num,SDR_total,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % ������ͨSDR
            end
            %plotSDR(case_num,SDR_total); % ������ͨSIR��SDR
            %plotSDR(case_num,SDR_total,SIR_total); % ������ͨSIR��SDR  
        end              
    end
end