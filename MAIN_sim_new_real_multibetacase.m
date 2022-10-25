%% Main simulation with new RIR simulate function
%% Simulation Initialize
clear all; close all;
addpath('room');addpath('components');addpath('bss_eval');addpath('plot');addpath('EM');addpath('nmfiva');addpath('IVE');addpath('audioSignalProcessTools');addpath('DOA');addpath('MNMF');addpath('CGGMM_IVA');addpath('MAIN_sim_components');addpath('oppo');addpath('samples');
mixfile = strcat('samples');%almethod = strcat('_em_batch_7���ļ�');
% path1 = strcat('D:\github-code\IVATEST\',mixfile);t=0;% ��¼�����������ļ�����t
source_path = 'samples';t=0;% ��¼�����������ļ�����t
filestr1 = dir([source_path,'.']); % ���ź��ļ���
for fnum1 = 3 :size(filestr1) % ������Դ��ѡ����ϵ��������,jovi&man&woman&xvxv
    source_path2nd = strcat(source_path,'\',filestr1(fnum1).name,'\');SDR_case_full=[];SDR_time_all_full=[];% Ϊ��һ�𻭳���������Դ��ϵ��������SDR
    filestr2 = dir([source_path2nd,'*.wav']); % ���ź��ļ�
for fnum2 = 1 :size(filestr2,1) % ��������Դ��ϵ��������(ʵ¼����źţ�mix>jovi&man��...)
    filenameTmp = filestr2(fnum2).name; 
%     filenameTmp =strcat('jovi_man_',num2str(files));%test_em_ʵ¼_beta AGC_method �ɽ����������Ϊ�ļ��� test_online_offline   test_AWGN_Lb test_AWGN_Lb
mkdir_str=strcat('./Simulation_Single_em_batch_restshilu_multibeta_1dan/',filestr1(fnum1).name,'/',filenameTmp(1:end-4));
file_state=mkdir_str;
mkdir(mkdir_str);%һ���о�  ���ڵ�ǰ�ļ����´���simulation�ļ���
sound_dir=mkdir_str;
% mkdir_str1 =strcat(mkdir_str,'/'); sound_dir= strcat(mkdir_str1,'/method1_1');
% mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);
% mkdir_str =strcat(mkdir_str,'.m');Save_link1=Save_link_file('MAIN_sim.m',mkdir_str);%����ǰ�������Ϊnk�ļ�
if isOctave
    graphics_toolkit('gnuplot');    warning('off', 'Octave:broadcast');
end
% mix_file = 'data/mix/BSS_VECTOR/2020_07_06_14_54_49.wav';% ʵ¼�ź�λ��
mix_file = 0;
%% case setting
%%% Room simulation setting %%
DebugRato = 10/10;                      % �����ã�2����2�����ȵĲ������ݣ�0.5����1/2�����ݣ�
mix_sim_case = [0];                     % 0��ʵ¼��1������
real_src_num = 2;                       % ʵ¼�ź�����Ҫ�����Դ����Ŀ
customize_room = 1;                     % �Ƿ�ʹ���Զ��巿���С
room_type_case = [2];                   % Scenario : 1��the oldest one with angle ; 2��the oldest one
                                        %            3��mic>num Scenario; 4��Block-online Scenario 1
                                        %            5��Block-online Scenario 2; 6��online Scenario(200ms)
                                        %            7��large mic array room(up to 8 mic)
                                        % ��customize_roomΪ1����caseѡ����Ч���������涨�巿���С
rand_select = 0;                        % 1:���ѡ��Դ�ź���� 0:�̶�ѡ��Դ�ź����,��Ϊ�̶����
% �̶�ģʽ�¸��������indexѡ��Դ�źź͸����ź�
% ��Ϊ�̶�ģʽ�����Բ��õ��������Ŀ��Դ�͸���Դ��Ŀ��������Զ���ȡindex��С
if     fnum1 == 3 targnum=[9 10]  ;elseif fnum1 == 4 targnum=[9 11] ;
elseif fnum1 == 5 targnum=[9 12]  ;elseif fnum1 == 6 targnum=[10 11] ;
elseif fnum1 == 7 targnum=[10 12] ;elseif fnum1 == 8 targnum=[11 12] ;end% Ϊ����SDR_improvementѡ��target
target_index_case = {targnum}; intf_index_case = {[]};
% target_index_case = {[9 10]}; 
% �ɹ�ѡ���Դ��� = {1��Ů��1, 2��Ů��3, 3������1, 4�����Ļ��Ѵ�, 5��Ӣ�Ļ��Ѵ�,
%                    6��ϴ�»�, 7������, 8����˹������, 9-jovi, 10-man, 11-woman, 12-xvxv}
sim_mic_case = [2];                     % ������˷���Ŀ����ʵ¼Ҫ��4mic����Ҳ�����4������ֻ��������ͨ�� 
target_source_num_case = [2];           % Ŀ��Դ��Ŀ����ֵ<=2��=1ʱΪ��Դ��ȡ
intf_source_num_case = [0];             % ����Դ��Ŀ��0=<L<6����Ϊfix select�����ó�index�ĳ���
muteOn_case = [0];                      % ѡ��ڶ���source����ǰ�������mute���ڶ���sourceһ��Ϊtarget source)
deavg_case = [1];                       % ����ź�ȥ��ֵ�������ʵ¼������
mix_SINR_case = [0];                    % Ŀ���źź����������߸����źţ��Ļ���Ÿ���ȣ���mix_SINR=0����Ϊ1:1��ϣ�����normal_case=3ʱ;ʹ����dBΪ��λ����ʾ target/(interference+diffuse_noise)
SINR_diffuse_ratio_case = [1];          % ����ظ������ܸ���֮�ȣ�1-diffuse_ratioΪAWGN�ɷ֣���=1��Ϊ����AWGN
determined_case = [0];                  % 0:overdetermined(mic > src); 1:determined(mic = src)
target_SIRSDR = 0;                      % 1:ֻ��ʾtarget source��SIRSDR��0:ȫ����ʾ (������ʽ��δд��������
%% Room environment setting %%
reverbTime_case = [0.08];
                                        % �������ʱ�䣬��λΪ��
angle_case = [40];                      % ��Դ��С�н�,angle from 0 to 360,��λΪ�ȡ�0�Ƚ�ΪX���ҷ���
angle_start_case = [55];                % �Ӹ�λ�ÿ�ʼ����Դ���Ų�,�Ƕ�Ҫ��ͬ�ϡ�>180ʱ����0�ȴ��棬eg.180,225��Ӧ��-45,0����
R_ratio_case = [0.8];                   % R = Դ����˷�����������max_R * R_ratio, ȡֵfrom 0 to 1
tR_ratio_case = [0.66];                 % Ŀ��Դ�����Դ֮��ľ����ֵ
src_permute_type_case = [0];            % 1�����Դ�Ų���0��˳��Դ�Ų�
room_plot_on = 0;                       % �Ƿ���Ʒ���ƽ��ͼ��0�������ƣ�1��ÿ�ζ����ƣ�2�������Ƶ�һ��
room_size_case = {[10 8 3]};            % �����С �����
mic_center_case = {[5 4 1.2]};          % ��˷������������� ����Ĭ�ϸ߶���ͬ
mic_distx_case = [0.158];               % ������˾����ˮƽ����ͶӰ���� ��ֵ��x�������� ��ֵ��x�Ḻ����
mic_disty_case = [0];                   % ������˾���Ĵ�ֱ����ͶӰ���� ��ֵ��y�������� ��ֵ��y�Ḻ����
%% Pre-batch & Initialize %%
initial_case = [1];                     % C & V initialization, 0:normal, 1:use first-frame-batch
pre_batch_case = [1];                   % using pre-batch to initialize Vs, 0: no pre-batch; 1:prebatch(continue); 2:prebatch(restart); 3:preonline(restart);
batch_update_num_case = [17];           % pre_batch_num
prebatch_iter_num = 20;
initial_rand_case = [0];                % V initialization using random diagonal domiance matrix; 1:full, 2:only low freq(1/3), 0:off;
%% Iteration num %%
iter_num_case = [20];                    % DOA+20�Σ����㷨����������online �̶�Ϊ2�Σ�offline��ֵΪcase��ֵ default=20
inner_iter_num_case = [1];              % �㷨�ڵ���������online�̶�Ϊ1�Σ�offline��ֵΪcase��ֵ
total_iter_num_case = [1];              % �ظ��������������㷨����ʱΪ���������ʼ��ʱ��ƽ��SDR��SIR�õġ�
%% Essential setting %%
online_case = [0];                      %�Ƿ������ߡ�1 �����ߡ�0 �����ߣ�
batch_type_case = [1];                  % batch IVA�����ͣ�1����auxiva��2����nmfiva��3Ϊauxiva-iss��ע��2������ģʽ������ �������㷨�ڿ���
batch_algs_case = [3];                  % batch IVA������,1-auxiva,2-MNMF��3-EMIVA��4-IVE �����Ʊ��������㷨,5-EMIVA_fast
online_algs_case = [3];                 % online IVA������,1-auxiva,2-MNMF��3-EMIVA��4-IVE
Lb_case =  [2];                     % length of blocks [1,2,4,6,8,16], default using 1
tao_case =[0]/10;                       % default value is 0, default hanning windowing operation;
                                        % >1,  designed based special criterion;
                                        % windowing factor default value is 1.2; 10^5, indicates no windowing;
win_exp_ratio_range =[20:1:20]/20;      % default value is 1,effective for tao>0;                       
win_size_case = [4096];
taoMeanAdd_case = [1]/10;               % effective for tao>0; add some value, such as 0.1 to robustify the window desgin;
taoMean_case = [1];                     % default value is 1,effective for tao>0;
D_open_case = [1];                      % using D
perm_on_case = [0];                     % ����ź�֡��������
forgetting_fac_case = [0.04];           % ������ʹ�ã��൱�������е�1-alpha [0.3 0.2 0.1 0.08 0.06 0.04]
gamma_case = [4:4]/100;                 % ������mic>srcʹ�ã�ΪC��forgetting factor
delta_case = [0]/10^0;                  % using in contrast function: (||x||^2 + delta)^n_orders
whitening_case = [0];                   % �Ƿ�ʹ�ð׻���δ������ɣ���ʱ��Ҫ��
project_back = 1;                       % �Ƿ���Ҫ������źŷ������������ź���ͬ�ĳ߶�
n_orders_case = {[1/2 1/2 1/2 1/2]};    % ���ߵ�orders {[1/6 1/2],[1/2 1],[1/2 1/2],[1/8 1/2]}
n_orders_casenum = size(n_orders_case,2);
%% Partition %%
parti_case = [0];                       % �Ƿ�ʹ���ӿ�BSS
SubBlockSize_case = [100];              %  ʹ���ӿ�Ĵ�С��
SB_ov_Size_case = [1]/2;                %  overlap �ı�������СΪ round(SubBlockSize*SB_ov_Size)��
partisize_case = [1];
select_case = [0];                      % �Ƿ�ѡ���ӿ飨��parti=1���˴�Ϊѡ�ӿ飻��parti=0���˴�Ϊѡ���ز���
thFactor_case = [2]/200;                % ѡ���ӿ���ֵ����
%% Epsilon test %%
diagonal_method_case =[0];              %  0: using a tiny portion of the mean diagnal values of Xnn��          
                                        % 1: using a tiny portion of the diagnal value of Xnn��       
Ratio_Rxx_case =[1:1]/10^3;             % 1:3:10 default is 0; 2 is a robust value;
epsilon_ratio_range =[10:10]/10;        % 1:4 10:10:60 robustify inversion of update_w ;
epsilon_ratio1_range =[10]/10;          % 1:4 10:10:60 robustify inversion of update_w ;
frameStart_range =[1];                  % 1 ������ʼ��һ���� epsilon��  ����ǰframeStartǰ�� epsilon_start_ratio;
epsilon_start_ratio_range =[1:1:1]/1;   %  ����ǰframeStartǰ�� ;
%% Order estimation %%
OrderEst_range =[0];                    % �Ƿ����order estimation��
OutIter_Num_range =[1];                 % order ���ƵĴ�����
OrderEstUppThr_range =[11]/10;          % order ���Ƶ����ޣ�
OrderEstLowThr_range =[4]/10;           %  order ���Ƶ����ޣ�
order_gamma_range =[8]/10;              %  order ���ƵĻ���ϵ����
n_orders1_range =[10]/20;                % pdf �Ĺ���ָ��������orders��
n_orders2_range =[10]/20;               % pdf �Ĺ���ָ��������orders��
verbose_range =[1];                     % 0 �������м�����1 �����м�����
%% Gamma ratio %%
GammaRatioThr_range =[10^2];            % �Ƿ�������û�з������Ƶ����ޣ��ǳ���ȼ۲�������
GammaRatioSet_range =[10]/10;           % �Ƿ�������û�з������Ƶ�gamma ֵ��
%% Mix Model Estimation %%
mix_model_case = [0];                   % �Ƿ�ʹ�û��CGGģ��,1��Ӳ�У�2��EM��0����ʹ��
ita_case = [0.9];                       % ���CGGģ���л�ϸ��ʺ�betaֵ�ĵݹ�ƽ��ϵ��
%% NMF Setting %%
nmf_iter_num_case = [1];                % nmf�ڵ���������1���ɣ������������ֵ
nmf_fac_num_case = [9];                 % nmf����Ŀ, 9 �ǵ���ֵ
nmf_b_case = [1/2];                     % IS-NMFָ��ֵ��p=beta=2ʱ��GGD-NMF��ЧΪb=1/2��IS-NMF
nmf_beta_case = [2];                    % betaֵ������GGD-NMF
nmf_p_case = [0.5];                     % pֵ������GGD-NMF
nmf_update_case = [0];                  % nmf update��ģ�ͣ�0����IS-NMF��1��GGD-NMF
%% MNMF Setting %%
%MNMF_case = [0]; % 1-ʹ��MNMF 0-��ʹ��MNMF
MNMF_refMic_case = [1];                 % reference mic for performance evaluation using bss_eval_sources 1 or 2
MNMF_nb_case = [20];                    % number of NMF bases for all sources (total bases)
MNMF_it_case = [300];                   % number of MNMF iterations
MNMF_first_batch_case = [20];           % first mini-batch size
MNMF_batch_size_case = [4];             % mini-batch size
MNMF_rho_case = [0.9];                  % the weight of last batch
MNMF_fftSize_case = [4096];             % window length in STFT [points]
MNMF_shiftSize_case = [2048];           % shift length in STFT [points]
MNMF_delta_case = 10.^[-12];            % to avoid numerical conputational instability
MNMF_p_norm_range =[5:5]/10;            % default is 0.5; sqrt(x);
MNMF_drawConv = false;                  % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
%% ILRMA Initialization Setting
ILRMA_init_case = [1];                  % 1:ʹ��ILRMA��ʼ�� 0:��ʹ��ILRMA��ʼ��
ILRMA_type_case = [2];                  % 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
ILRMA_nb_case = [20];                   % number of bases (for type=1, nb is # of bases for "each" source. for type=2, nb is # of bases for "all" sources)
ILRMA_it_case = [20];                   % iteration of ILRMA
ILRMA_normalize_case = true;            % true or false (true: apply normalization in each iteration of ILRMA to improve numerical stability, but the monotonic decrease of the cost function may be lost. false: do not apply normalization)
ILRMA_dlratio_case = 10.^[-2];          % diagonal loading ratio of ILRMA
ILRMA_drawConv = false;                 % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
%% EM-IVA Setting %%%
%EMIVA_case = [0]; %1-ʹ��EMIVA 0-��ʹ��EMIVA
beta1_case = [20]/10;           
beta2_case = [20]/10;                    
beta1_offset_case = [-2]/10^1;   %0����      
beta2_offset_case = [0]/10^1;            
EMIVA_max_iter_case =[20]; 
EMIVA_ratio_case = [0.4];               %0.2 0.4 0.6 0.8 weight of present frame 
detect_low_case = [0.3];
detect_up_case = [0.7];
pmulti_case = [40];
logp_case = [1];                        % ����logp_case
AVGMMIVA_max_iter_case = [20];          % AV-GMM_IVA����������
singleout_case = [1];                           % �Ƿ���е�·����
%% Prior AuxIVA %%%
DOA_switch_case=[0];                    % ����DOA,0-[0,0,0,0]&doa_0; 1-[1,1,1,0]&doa_1
% prior_case = {[1,1,1,0]};    SIR_time_all_full         % ����Ϊ(1,k)ά��������k��Ϊ1�����е�k��Դ��������Ϣ��
%                                       % ��k��Ϊ0����û�е�k��Դ�����飬����DOA��������
%                                       % ��ΪOverIVA��IVEʱ����K+1��prior����BG��prior��KΪtarget��Ŀ��
% DOA_esti_case = [1];                  % �Ƿ�ʹ��DOA���ƣ�Ϊ0��ʹ�÷�����洫�ݵ�DOA��Ϣ 
DOA_update_case = [0];                  % �Ƿ��ڵ����и���DOA
DOA_init_case = [0];                    % �Ƿ�ʹ��DOA��ʼ����
esti_mic_dist_case = [0.158];           % DOA����ʱʹ�õ���˷���
% �˴���ʽ���£�
%     P = (DOA_tik_ratio * eye(M) + sum(DOA_Null_ratio * hf * hf') /deltaf^2
DOA_Null_ratio_range = [1000]/1000;      % DOA  ��Ȩֵ��[0.1]/10-4mic, 
DOA_tik_ratio_range = [0.5]/1000;       % DOA Tik ��Ȩֵ��[0.5]/1000-4mic
deltaf_case = [40];                     % Ŀ��Դ��һ��ϵ��10-4mic
deltabg_case = [0.5];                   % ����������һ��ϵ��
annealing_case = [0];                   % �Ƿ�ģ���˻�fac_a = max(0.5-iter/iter_num, 0);
%% IVE %%%
% IVE�ķ�����1ΪIP-1����Чoveriva��, 2ΪIP-2��3���忴auxive_update˵����4ΪFIVE����
IVE_method_case = [1 ]; 
%% Parameters initialization %%%
Initiallize;
timeblock_Length = 1; % online SIR����ֿ鳤�ȣ�in second��
plot_time_mode = 1; % ????SIR_time/SDR_time????  1:0-1,0-2,0-3...  2:0-1,1-2,2-3...
case_num = 0;
room_imp_on = 1;   
RandomSeed =0; % 0 ÿ��������ͬ�����1 ÿ�������ͬ���
if RandomSeed==0 randn('state',98765); rand('state',12345); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
if RandomSeed==1 randn('state',cputime); rand('state',cputime+1); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
PowerRatio = 2;
if mix_file == 1 file_tag_case = 1; else file_tag_case = 1; end
% �̶�ѡ��sourceʱ��target��intf����Ŀͳһ��index��Ŀ��case����Ϊ��һcase��ֹ�ظ����档
if ~rand_select target_source_num_case = [0]; intf_source_num_case = [0]; end 
%% ������
tic
%% DOA    
for DOA_switch = DOA_switch_case
    if DOA_switch
    prior_case = {[1,1,0,0]};DOA_esti_case = [1];DOA_esti_online_case = [0];
    else
    prior_case = {[0,0,0,0]};DOA_esti_case = [0];DOA_esti_online_case = [0];end
for mix_sim = mix_sim_case for room_type = room_type_case for target_idx = target_index_case for intf_idx = intf_index_case for muteOn = muteOn_case for SINR_diffuse_ratio = SINR_diffuse_ratio_case
for deavg = deavg_case for mix_SINR = mix_SINR_case for sim_mic = sim_mic_case for target_source_num = target_source_num_case  for intf_source_num = intf_source_num_case for file_tag = file_tag_case for prior = prior_case
for angle = angle_case for angle_start = angle_start_case for R_ratio = R_ratio_case for tR_ratio = tR_ratio_case for src_permute_type = src_permute_type_case for reverbTime = reverbTime_case
for room_size = room_size_case for mic_center = mic_center_case for mic_distx = mic_distx_case for mic_disty = mic_disty_case

%% �����ŵ�����
    if mix_sim % ʹ�÷����ŵ����з���
        % room��������
        room_setup;
        % RIR���溯��
        [xR, s, fs_ref, mic_pos, theta, target_source, intf_source, layout] ...
            = generate_sim_mix_new(room,target_index,intf_index);
    else % ʹ��ʵ¼�ź�
        mix_file_deal;
    end
    
    % �������ź�
%     audiowrite([sound_dir,'/mix.wav'], xR.', fs_ref);  % ʵ¼����
%     xR_t = audioread([sound_dir,'/mix.wav']); 
    xR_1 = xR_t.';  % ���wav���ٶ�ȡ�Ļ���źţ�һ����xR���ɣ������ر𲻺�����xR_1����
    % FFT�����ʹ���������
    for win_size = win_size_case
    win_type = 'hann';     inc = win_size / 2;       
%% ������
for determined = determined_case for online = online_case for batch_type = batch_type_case for tao = tao_case for win_exp_ratio = win_exp_ratio_range for taoMeanAdd = taoMeanAdd_case for taoMean = taoMean_case
    
    if online      win_type = 'hamming'; end% ����onlineʹ�ú�����
    if tao >0 
        [win_ana, ~]= WinGen(win_size,tao,taoMeanAdd,taoMean,inc,win_exp_ratio); 
        % [win_ana1, ~]= WinGen(win_size,1.2,taoMeanAdd,taoMean,inc,win_exp_ratio); plot(win_ana,'b'); hold on;   plot(win_ana1,'r'); 
    else
        [win_ana, ~] = generate_win(win_type, win_size, inc);
    end
%   win_syn = ones(1, win_size); 
    win_syn =win_ana; %ones(1, win_size);
for D_open = D_open_case for perm_on = perm_on_case for iter_num = iter_num_case  for inner_iter_num = inner_iter_num_case  for forgetting_fac = forgetting_fac_case for gamma = gamma_case for delta = delta_case for whitening_open = whitening_case
for initial = initial_case for initial_rand = initial_rand_case for pre_batch = pre_batch_case for batch_update_num = batch_update_num_case for Lb = Lb_case for n_orders_num = 1:n_orders_casenum for parti = parti_case  for SubBlockSize = SubBlockSize_case 
for SB_ov_Size = SB_ov_Size_case for select = select_case for partisize = partisize_case for thFactor = thFactor_case for diagonal_method = diagonal_method_case for Ratio_Rxx = Ratio_Rxx_case   for epsilon_ratio = epsilon_ratio_range  for epsilon_ratio1 =epsilon_ratio1_range
for OrderEst = OrderEst_range for OutIter_Num = OutIter_Num_range for order_gamma = order_gamma_range for OrderEstUppThr = OrderEstUppThr_range for OrderEstLowThr = OrderEstLowThr_range for frameStart =frameStart_range for epsilon_start_ratio =epsilon_start_ratio_range            
for GammaRatioThr = GammaRatioThr_range for  GammaRatioSet = GammaRatioSet_range for mix_model = mix_model_case for ita = ita_case for n_orders1 =n_orders1_range for n_orders2 =n_orders2_range for verbose = verbose_range for nmfupdate = nmf_update_case
for nmf_iter_num = nmf_iter_num_case for nmf_fac_num = nmf_fac_num_case for nmf_b = nmf_b_case for nmf_beta = nmf_beta_case for nmf_p = nmf_p_case for deltaf = deltaf_case for deltabg = deltabg_case for DOA_Null_ratio = DOA_Null_ratio_range  for DOA_tik_ratio = DOA_tik_ratio_range   
for DOA_esti = DOA_esti_case for DOA_esti_online = DOA_esti_online_case for DOA_update = DOA_update_case for esti_mic_dist = esti_mic_dist_case for DOA_init = DOA_init_case for annealing = annealing_case for IVE_method = IVE_method_case for total_iter_num = total_iter_num_case 
for MNMF_refMic = MNMF_refMic_case for MNMF_nb = MNMF_nb_case for MNMF_it = MNMF_it_case for MNMF_first_batch = MNMF_first_batch_case for MNMF_batch_size = MNMF_batch_size_case for MNMF_rho = MNMF_rho_case for MNMF_fftSize = MNMF_fftSize_case for MNMF_shiftSize = MNMF_shiftSize_case
for MNMF_delta = MNMF_delta_case for MNMF_p_norm = MNMF_p_norm_range for ILRMA_init = ILRMA_init_case for ILRMA_type = ILRMA_type_case for ILRMA_nb = ILRMA_nb_case for ILRMA_it = ILRMA_it_case for ILRMA_normalize = ILRMA_normalize_case for ILRMA_dlratio = ILRMA_dlratio_case
for online_algs = online_algs_case for batch_algs = batch_algs_case
for beta1 = beta1_case for beta2 = beta2_case for beta1_offset = beta1_offset_case for beta2_offset = beta2_offset_case 
for EMIVA_max_iter = EMIVA_max_iter_case for EMIVA_ratio = EMIVA_ratio_case  for logp = logp_case for detect_low = detect_low_case for detect_up = detect_up_case for pmulti = pmulti_case for AVGMMIVA_max_iter = AVGMMIVA_max_iter_case
    EMIVA_beta = { [beta1+beta1_offset beta2+beta2_offset];[beta1+beta1_offset beta2+beta2_offset]};                                
if RandomSeed==0 randn('state',98765); rand('state',12345); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
if RandomSeed==1 randn('state',cputime); rand('state',cputime+1); end % ��֤ÿ��SNRѭ������ʼ����һ�� seed = rng(10);        
    if batch_type ~= 2 || batch_algs == 2 || online_algs == 2
        total_iter = 1; % auxIVA���ù̶���ʼ��������Ҫ�ظ�����
    else if batch_type == 2
            seed = rng(10); % �̶�nmf�����������
            total_iter = total_iter_num; % nmfIVA���������ʼ������Ҫ�ظ�����
         end
    end
    
for ITER = 1:total_iter
    %% ����źų�ʼ��
    x = xR; % һ����xR���ɣ������ر𲻺�����xR_1���ԡ�   
    [mic_num, sample_num] = size(x); source_num = target_source;
    if determined source_num = mic_num; end  
    %% �����������
    option_setup;
    %% preprocess for select beta
%     pre_select_beta;
   option.EMIVA_beta = EMIVA_beta;
    %% �źŴ����ä����
    if online
        % �����㷨(online)
        if online_algs == 1 % AuxIVA
            [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option); % �����������online�汾
%           [s_est] = auxiva_audio_bss_online_single(x,source_num,option); 
%           [s_est,label] = nmfiva_audio_bss_online_perm(x,source_num,option);              
        elseif online_algs == 2 % MNMF
            [s_hat,label] = bss_multichannelNMF_online(x.',source_num,option);
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif online_algs == 3 % EMIVA
            tic
            [s_est,label,~, ~,~] = CGGMM_IVA_online_revise(x, option);
            toc
        elseif online_algs == 4 % IVE
            [s_est,label] = auxive_audio_bss_online_perm(x,option); % IVE
        end
        out_type = 'online';
    else
        % �����㷨(batch)
         if batch_algs == 1 % AuxIVA
            [s_est,label] = auxiva_audio_bss_batch(x,source_num,option); % IVA batch(including nmfiva)
        elseif batch_algs == 2 % MNMF
            [s_hat,label] = bss_multichannelNMF_offline(x.',source_num,option);
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 3 % EMIVA
%             [pcaData,projectionVectors,eigVal] = whitening_pre(x',2);
            [s_est,label,~, ~,~,~,obj_set] = CGGMM_IVA_batch(x, option);           
        elseif batch_algs == 4 % IVE
            [s_est,label] = auxive_audio_bss_batch(x,option); % IVE batch
        elseif batch_algs == 5 % EMIVA_fast
            [s_est,label,~, ~,~] = CGGMM_IVA_batch_fast(x, option); 
        elseif batch_algs == 6 % AV-GMM-IVA
            [s_est,label,~, ~,~] = AV_GMM_IVA_batch_single(x, option); 
        elseif batch_algs == 7 % fastmnmf
            [s_hat,label] = subguas_FastMNMF_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        end
        out_type = 'batch';
    end 
    %% SIR SDR����
    if 0 %~mix_sim
        L = min(size(s,2), size(s_est,2));
        if source_num == 1
%            [SDR,~,SAR,perm] = bss_eval_sources(s_est(1,1:L), s(1,1:L));
%             SIR = 0;
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
%               [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(1:source_num,1:L)); max_comb = 1:source_num;
                [SDR_in,SIR_in,SAR_in,~] = bss_eval_sources(x(:,1:L), s(1:source_num,1:L));
                [SDR_out,SIR_out,SAR_out,perm] = bss_eval_sources(s_est(:,1:L), s(1:source_num,1:L)); max_comb = 1:source_num;
                SDR=SDR_out-SDR_in; SIR=SIR_out-SIR_in ; SAR=SAR_out-SAR_in;% ;
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
        if target_SIRSDR
            SIR = SIR(1); SDR = SDR(1); SAR = SAR(1); SNR = SNR(1);
        end
        fprintf('%s\n%s\nSDR = %s\nSIR = %s\n',filenameTmp,out_type,num2str(SDR'),num2str(SIR'));
%         profile viewer
%         if online % ���� online SIR improvment
% %             tap_Length = timeblock_Length * fs_ref;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length); SDR_time = cal_SDR_time(x,s,s_est,tap_Length);
%             tap_Length = timeblock_Length * fs_ref;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length,plot_time_mode); SDR_time = cal_SDR_time(x,s,s_est,tap_Length,plot_time_mode);            
%               T = [0 : ceil(L / tap_Length)] * timeblock_Length;            SIR_time_all = [SIR_time_all;SIR_time]; SDR_time_all = [SDR_time_all;SDR_time];                                                                                                                    SDR_time_all_full = [SDR_time_all_full;SDR_time_all];
%         end 
        
        if total_iter ~= total_iter_num
            % �̶���ʼ���������ʼ��һ�����ʱ����Ҫ���̶���ʼ�������ݽ��и��������ٷ���ʱ�䡣
            SIR_case = [SIR_case repmat(SIR,[1,total_iter_num])];  SDR_case = [SDR_case repmat(SDR,[1,total_iter_num])];  SAR_case = [SAR_case repmat(SAR,[1,total_iter_num])];  SNR_case = [SNR_case repmat(SNR,[1,total_iter_num])];
        else
            SIR_case = [SIR_case SIR];  SDR_case = [SDR_case SDR];  SAR_case = [SAR_case SAR];  SNR_case = [SNR_case SNR];
            SDR_case_full=[SDR_case_full SDR_case];
        end
    end
%     fid=fopen('SDR_online_result_Lb.txt','a');fprintf(fid,'\n%s\n',filenameTmp);fprintf(fid,'SDR = ');fprintf(fid,'%g   ',SDR);
%     fprintf(fid,'SIR = ');fprintf(fid,'%g   ',SIR);fclose(fid);
%     %% �Ŵ��ź�
%     s_est = preprocess(s_est, 2, deavg); 
    %% �ж�SDR�Ƿ����Ҫ��
%     if (max(SDR)>9&&max(SDR)<16)&&(min(SDR)>-4)  
    t=t+1;% ��¼���ϵ��ļ���
    %% ����txt�ļ���¼SDR
%     fid=fopen('SDR_online_result.txt','a');fprintf(fid,'\n%s\n',filenameTmp);fprintf(fid,'SDR = ');fprintf(fid,'%g   ',SDR);
%     fprintf(fid,'SIR = ');fprintf(fid,'%g   ',SIR);fclose(fid);
      %% ��������(case)����Ƶ
%     gcf_temp = strcat('D:\github-code\IVATEST\gcf_save\',filenameTmp(:,1:end-4),'.jpg');
%     saveas(gcf, gcf_temp); %���浱ǰ���ڵ�ͼ��
%     close(figure(gcf));
%     strsave= strcat(mkdir_str1,filenameTmp,'.mat'); sav=['save ' strsave]; eval(sav);
    case_num = case_num + 1;    case_str = num2str(case_num);
%     filenameTmp1 = strcat('case_',case_str,'.mat'); strsave= strcat(mkdir_str1,filenameTmp1);
%     sav=['save ' strsave]; eval(sav);
    %% Save separated wave files
    if real_result == 1 %% ����sep1_method1��ʽ���� 'case_',case_str, 'case_',case_str,
      audiowrite([sound_dir,'/',filenameTmp(:,1:end-4),'_out',case_str,'.wav'], s_est(1,:)', fs_ref);
      if size(s_est,1) == 2
        audiowrite([sound_dir,'/',filenameTmp(:,1:end-4),'_out2ch_method_1.1.wav'], s_est(2,:)', fs_ref);
      end
%        audiowrite([sound_dir,'/',method,'.1_sep1','.wav'], s_est(1,:)', fs_ref);
%       if size(s_est,1) == 2
%         audiowrite([sound_dir,'/',method,'.1_sep2','.wav'], s_est(2,:)', fs_ref);
%       end
      if size(s_est,1) == 4
        audiowrite([sound_dir,'/sep3_',method,'.wav'], s_est(3,:)', fs_ref);   audiowrite([sound_dir,'/sep4_',method,'.wav'], s_est(4,:)', fs_ref);   
      end
    else
      sep1_str = label{1}; %% ����sep1_case1_target_batch��ʽ����
      audiowrite([sound_dir,'/sep1_case',case_str,'_',sep1_str,'_',out_type,'.wav'], s_est(1,:)', fs_ref);
      if size(s_est,1) == 2
          sep2_str = label{2};
          audiowrite([sound_dir,'/sep2_case',case_str,'_',sep2_str,'_',out_type,'.wav'], s_est(2,:)', fs_ref);
      end
      if size(s_est,1) == 4
          audiowrite([sound_dir,'/sep3_case',case_str,'_',out_type,'.wav'], s_est(3,:)', fs_ref);   audiowrite([sound_dir,'/sep4_case',case_str,'_',out_type,'.wav'], s_est(4,:)', fs_ref);   
      end
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
        plot_room_layout(layout);
        room_plot_on = 0;
    end
    toc
%     else
%         status = rmdir(file_state,'s');% ɾ��������SDR�������ļ��м�������
%     end
end
end
end
end
end
end
%if ~select  %        break;     end
end
%if parti || select        break;    
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
end
end
end
end
end
end
end
end
% end
%% ��������(all)&��ͼ
% ��Ҫ������ͼ���������ȡmat�ļ�
% load_case = 1;  load_case_str = num2str(load_case);
% load(strcat(mkdir_str1,'case_',load_case_str,'.mat')); %
%   load(strcat(mkdir_str1,'.mat'));
if mix_sim
    loadMat=1;
    if loadMat == 1
        
%         clear all; close all;
%          for files=1 :3  
%     %filenameTmp =strcat('jovi_man_',num2str(files)); 
%         filenameTmp =strcat('man_jovi_3.wav');%,num2str(files)
%      %   filenameTmp ='test_em_ʵ¼_beta';%�ɽ����������Ϊ�ļ��� test_online_offline   test_AWGN_Lb test_AWGN_Lb
%         mkdir_str=strcat('./Simulation_Single/mix_new_em/jovi_man/',filenameTmp);
%         mkdir(mkdir_str);%һ���оͻ��ڵ�ǰ�ļ����´���simulation�ļ���
%         mkdir_str1 =strcat(mkdir_str,'/'); sound_dir = strcat(mkdir_str1,'/sound');
%         mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);  
%         loadstrcase = strcat(mkdir_str1,filenameTmp,'.mat');
%         loadsr=['load  ' loadstrcase];       eval(loadsr);       

        
        if target_SIRSDR R_num = 1; else R_num = 2; end
         case_num = size(SDR_case,2) / total_iter_num;
%         case_num1 = size(SDR_case_full,2) / total_iter_num;
      
        SIR_total = reshape(SIR_case,R_num, total_iter_num, case_num);  SDR_total = reshape(SDR_case,R_num, total_iter_num, case_num);
        SAR_total = reshape(SAR_case,R_num, total_iter_num, case_num);  SNR_total = reshape(SNR_case,R_num, total_iter_num, case_num);
        case_name={};
%         SDR_total = reshape(SDR_case_full,R_num, total_iter_num, case_num1);
       % case_name = {'non-DOA','DOA'};%�����㷨���� ���������������case1,case2,case3
        
        %plot_case = online_algs_case;% ��Բ�ͬ��Ҫ�����case��ͼ
                                    % DOA_Null_ratio_range DOA_switch_case deltaf_case deltabg_case  deltabg_case deltaf_case  
                                    % DOA_Null_ratio_range DOA_esti_case angle_start_case prior_casen_orders1_range,n_orders2_range
                                    % online_algs_case reverbTime_case  batch_update_num_case
         % online =1;
        if online
         %   plotSIR_time(SIR_time_all,T,max(size(epsilon_ratio_range,2),size(Ratio_Rxx_case,2)),3); % ����ʱ��SIR         
         case_num2 = size(SDR_time_all_full,1) / 2;
         sub_case_num = case_num;
         packFigNum =   1;  % һ��fig �ֳɼ���(=packFigNum)subfugure����������Ƚ�
         SortedPlotThr = 1; % >1,������ʾ����
         SortedPlotNum = 8; % ����������ʾcase����Ŀ,�� sub_case_num �Ƚϴ��ʱ�����������ʾ
         plotRatio =1;      %һ�ζ��ٱ�����ͼ�� default=1�� 2 ������50% ��ͼ��   
         %close all;  
%          plotSIR_time1(SDR_time_all_full,T,case_num2,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); %,plot_case ����ʱ��SIR
         plotSIR_time1(SDR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,Lb_case); %,plot_case ����ʱ��SIR
% plotSIR_time1(SDR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,n_orders1_range,n_orders2_range); % ����ʱ��SIR
%function plotSIR_time(SIR,T,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
%sub_case_num      ͬһ����Ƚ���һ��Figure ������Ŀ, ����һ�� 10 =size(SIR_time_all,1)/sourceNum (=2),һ��ͼ��2*5�Ƚϡ�
%packFigNum       һ�ΰѼ���subcase ���ŵ�һ����plot
%SortedPlotThr;   ���case_num̫����ʾ���ˣ��о��Ƿ���ʾ SortedPlotNum case��
%�����1��ȱʡ��ʾ���������������>1, ����������ʾ��
%SortedPlotNum;   ���case_num̫����ʾ���ˣ���ʾ SortedPlotNum case��
%plotRatio       һ�ζ��ٱ�����ͼ�� default=1�� 2 ������50% ��ͼ��
        else
%             R_num = size(SDR_case_full,1) ;
%             case_num = size(SDR_case_full,1) / total_iter_num;
            R_num = size(SDR_case,1) ;
            case_num = size(SDR_case,2) / total_iter_num;
            sub_case_num = case_num;
            packFigNum =   1;  % һ��fig �ֳɼ���(=packFigNum)subfugure����������Ƚ�
            SortedPlotThr = 1; % >1,������ʾ����
            SortedPlotNum = 8; % ����������ʾcase����Ŀ,�� sub_case_num �Ƚϴ��ʱ�����������ʾ
            plotRatio =1;
            if ~isempty(case_name)
                plotSDR_name(case_num,SDR_total,case_name); % ���ƴ����㷨���Ƶ�SDR
            else
                plotSDR_name(case_num,SDR_total,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % ������ͨSDR
            end
            %plotSDR(case_num,SDR_total); % ������ͨSIR��SDR
            %plotSDR(case_num,SDR_total,SIR_total); % ������ͨSIR��SDR  
        end 
%          end
    end
    close all;
end
end
end

disp(['�����������ļ���',num2str(t),'��']);
% %% �����ļ�
% outputname = almethod(5:end);%'oppo��������11-6';% ����ı�
% path1 = strcat('D:\github-code\IVATEST\Simulation_Single\',mixfile,almethod);% ����ı�
% filestr1 = dir([path1,'.']);
% for fnum1 = 1+2 :size(filestr1)
%     path2 = strcat(path1,'\',filestr1(fnum1).name,'\');% һ��Ŀ¼�ļ���
%     filestr2 = dir([path2,'.']);% һ��Ŀ¼�ļ������ݶ�ȡ
% for fnum2 = 1+2 :size(filestr2)
%     path3 = strcat(path2,'\',filestr2(fnum2).name,'\');
%     filestr3 = dir([path3,'*.wav']);
% for fnum3 = 1:size(filestr3)
%     filenametemp = filestr3(fnum3).name;
%     mkdir_str=strcat('.\',outputname,'\',filestr2(fnum2).name);%filestr3(1:end-4)ȥ����׺.wav
%     mkdir(mkdir_str);% һ���оͻ��ڵ�ǰ�ļ����´�����Ӧ�ļ������ļ���       
%     copyfile([path3,'\',filenametemp],mkdir_str)
% end
% end
% end

