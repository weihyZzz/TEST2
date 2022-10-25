<<<<<<< HEAD
%% Main simulation with new RIR simulate function  Simulation Initialize
clear all; close all; addpath('room');addpath('components');addpath('bss_eval');addpath('plot');addpath('EM');addpath('nmfiva');addpath('IVE');addpath('audioSignalProcessTools');addpath('DOA');addpath('MNMF');addpath('CGGMM_IVA');addpath('MAIN_sim_components');addpath('AV-GMM_IVA');addpath('MNMF_subguas');addpath('PEASSevaluate');addpath('MLDR');addpath('bse_SCM');addpath('AR');addpath('GWPE');%addpath('PEASS');
filenameTmp =strcat('test6');%test_em_实录_beta AGC_method 可将仿真参数作为文件名 test_online_offline   test_AWGN_Lb test_AWGN_Lb
mkdir_str=strcat('./Simulation_Single/',filenameTmp);  mkdir(mkdir_str);%一运行就  会在当前文件夹下创建simulation文件夹
mkdir_str1 =strcat(mkdir_str,'/'); sound_dir= strcat(mkdir_str1);%,'/sound'
mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);
% mkdir_str =strcat(mkdir_str,'.m');Save_link1=Save_link_file('MAIN_sim.m',mkdir_str);%将当前函数另存为nk文件
if isOctave     graphics_toolkit('gnuplot');    warning('off', 'Octave:broadcast');  end
folder_name = './data/mix/';             % 实录文件夹名称 
wav_name = 'xiaomi_record.wav';          % 实录音频名称
mix_file = strcat(folder_name,wav_name); % 要读取的实录音频
%% case setting Room simulation setting %%
=======
%% Main simulation with new RIR simulate function
%% Simulation Initialize
clear all; close all;
addpath('room');addpath('components');addpath('bss_eval');addpath('plot');addpath('EM');addpath('nmfiva');addpath('IVE');addpath('audioSignalProcessTools');addpath('DOA');addpath('MNMF');addpath('CGGMM_IVA');addpath('MAIN_sim_components');addpath('AV-GMM_IVA');addpath('MNMF_subguas');addpath('PEASSevaluate');addpath('MLDR');addpath('bse_SCM');%addpath('PEASS');
filenameTmp =strcat('DOA');%test_em_实录_beta AGC_method 可将仿真参数作为文件名 test_online_offline   test_AWGN_Lb test_AWGN_Lb
mkdir_str=strcat('./Simulation_Single/',filenameTmp);
mkdir(mkdir_str);%一运行就  会在当前文件夹下创建simulation文件夹
mkdir_str1 =strcat(mkdir_str,'/'); sound_dir= strcat(mkdir_str1);%,'/sound'
mkdir(sound_dir); mkdir_str =strcat(mkdir_str1,filenameTmp);
% mkdir_str =strcat(mkdir_str,'.m');Save_link1=Save_link_file('MAIN_sim.m',mkdir_str);%将当前函数另存为nk文件
if isOctave
    graphics_toolkit('gnuplot');    warning('off', 'Octave:broadcast');
end
% mix_file = 'data/mix/BSS_VECTOR/2020_07_06_14_54_49.wav';% 实录信号位置
mix_file = 'data/mix_2020_07_06_14_54_25.wav';
%% case setting
%%% Room simulation setting %%
>>>>>>> my_addnew-ver
DebugRato = 10/10;                      % 测试用；2是用2倍长度的测试数据；0.5是用1/2的数据；
mix_sim_case = [0];                     % 0—实录，1—仿真
real_src_num = 2;                       % 实录信号所需要分离的源的数目
customize_room = 1;                     % 是否使用自定义房间大小
room_type_case = [2];                   % Scenario : 1—the oldest one with angle ; 2—the oldest one
                                        %            3—mic>num Scenario; 4—Block-online Scenario 1
                                        %            5—Block-online Scenario 2; 6—online Scenario(200ms)
                                        %            7—large mic array room(up to 8 mic)
                                        %            8—car four mic
                                        % 若customize_room为1，则case选择无效，需在下面定义房间大小
rand_select = 0;                        % 1:随机选择源信号组合 0:固定选择源信号组合,若为固定组合
% 固定模式下根据输入的index选择源信号和干扰信号
% 若为固定模式，可以不用调整下面的目标源和干扰源数目，程序会自动读取index大小
% if     fnum1 == 3 targnum=[9 10]  ;elseif fnum1 == 4 targnum=[9 11] ;
% elseif fnum1 == 5 targnum=[9 12]  ;elseif fnum1 == 6 targnum=[10 11] ;
% elseif fnum1 == 7 targnum=[10 12] ;elseif fnum1 == 8 targnum=[11 12] ;end% 为计算SDR_improvement选择target
% target_index_case = {targnum}; intf_index_case = {[]};[2,3],[2,4],[2,7],[3,4],[3,7]
target_index_case = {[3 4]}; intf_index_case = {[]};
% 可供选择的源编号 = {1—女声1, 2—女声3, 3—男声1, 4—中文唤醒词, 5—英文唤醒词,
%  23,34,37                  6—洗衣机, 7—音乐吉他, 8—高斯白噪声, 9-jovi, 10-man, 11-woman, 12-xvxv}
sim_mic_case = [2];                     % 仿真麦克风数目。若实录要用4mic方法也需调成4，否则只会用两个通道 
target_source_num_case = [1];           % 目标源数目，该值<=2，=1时为单源提取
intf_source_num_case = [0];             % 干扰源数目，0=<L<6，若为fix select则设置成index的长度
muteOn_case = [0];                      % 选择第二个source，将前几秒进行mute（第二个source一般为target source)
deavg_case = [1];                       % 混合信号去均值，仿真和实录均适用
mix_SINR_case = [0];                    % 目标信号和噪声（或者干扰信号）的混合信干噪比，若mix_SINR=0，则为1:1混合，仅在normal_case=3时;使用以dB为单位，表示 target/(interference+diffuse_noise)，10-20dB
SINR_diffuse_ratio_case = [1];          % 非相关干扰与总干扰之比，1-diffuse_ratio为AWGN成分，若=1即为不加AWGN,该值[0,1]区间,越接近一加的AWGN越少。
determined_case = [0];                  % 0:overdetermined(mic > src); 1:determined(mic = src)
target_SIRSDR = 0;                      % 1:只显示target source的SIRSDR，0:全部显示 (开关形式，未写入批处理）
PEASS_flag = 0;                         %  1:使用PEASS评估指数，
%% Room environment setting %%
reverbTime_case = [50]/1000;            % 房间混响时间，单位为秒;>0.3 s 影响信道仿真时间
angle_case = [40];                      % 两源最小夹角,angle from 0 to 360,单位为度。0度角为X轴右方向
angle_start_case = [55];                % 从该位置开始进行源的排布,角度要求同上。>180时，用0度代替，eg.180,225，应用-45,0代替
R_ratio_case = [0.8];                   % R = 源与麦克风阵列最大距离max_R * R_ratio, 取值from 0 to 1
tR_ratio_case = [0.66];                 % 目标源与干扰源之间的距离比值
src_permute_type_case = [0];            % 1—随机源排布，0—顺序源排布
room_plot_on = 0;                       % 是否绘制房间平面图，0—不绘制，1—每次都绘制，2—仅绘制第一次，3—绘制车载环境
room_size_case = {[10 8 3]};            % 房间大小 长宽高
mic_center_case = {[5 4 1.2]};          % 麦克风线阵中心坐标 阵列默认高度相同
mic_distx_case = [0.158];               % 相邻麦克距离的水平方向投影长度 正值：x轴正方向 负值：x轴负方向
mic_disty_case = [0];                   % 相邻麦克距离的垂直方向投影长度 正值：y轴正方向 负值：y轴负方向
% moving_source_sound
move_sound_case = [0];                  % 选择是否开启移动声源仿真
anglenum_case = [10];                   % 需要改变的角度数目
angle1_start_case = [30];               % 第一个源的开始分布角度
angle1_interval_case = [5];             % 第一个源的分布角度间隔
angle2_start_case = [70];               % 第二个源的开始分布角度
angle2_interval_case = [10];            % 第二个源的分布角度间隔
%% Pre-batch & Initialize %%
initial_case = [1];                     % C & V initialization, 0:normal, 1:use first-frame-batch
pre_batch_case = [1];                   % using pre-batch to initialize Vs, 0: no pre-batch; 1:prebatch(continue); 2:prebatch(restart); 3:preonline(restart);
batch_update_num_case = [17];           % pre_batch_num
prebatch_iter_num = 20;
initial_rand_case = [0];                % V initialization using random diagonal domiance matrix; 1:full, 2:only low freq(1/3), 0:off;
%% Iteration num %%
iter_num_case = [20];                    % DOA+20次，算法迭代次数，online 固定为2次，offline的值为case的值 default=20
inner_iter_num_case = [1];              % 算法内迭代次数，online固定为1次，offline的值为case的值
total_iter_num_case = [1];              % 重复仿真次数，混合算法测试时为了算随机初始化时的平均SDR和SIR用的。
%% Essential setting %%
online_case = [0];                      %是否是在线。1 是在线。0 是离线；
batch_type_case = [1];                  % batch IVA的类型，1代表auxiva，2代表nmfiva，3为auxiva-iss。注意2的在线模式不可用 ，离线算法内控制
<<<<<<< HEAD
batch_algs_case = [9];                 % batch IVA的类型,1-auxiva,2-MNMF，3-EMIVA，4-IVE ,5-EMIVA_fast，6-AV-GMM-IVA,
                                        % 7-subguassmnmf,8-tmnmf.9-fastmnmf,10-t_fastmnmf,11-MLDR;12-fastmnmf2;13-tfastdifferent v ;
                                        % 14-tfast mpdf混合pdf控制本函数内算法; 16-AR_FastMNMF2; 17-AR_FastMNMF1; %18-AR_t_FastMNMF1; %19-AR_t_FastMNMF2;
                                        % 20-FastMNMF_nopart; %21-FastMNMF2_nopart; 22-t_FastMNMF_nopart; 23-ILRMA_GWPE
online_algs_case = [12];                % online IVA的类型,1-auxiva,2-MNMF，3-EMIVA，4-IVE, 5-tmnmf, 6-fastmnmf, 7-tfastmnmf, 8-fastmnmf2, 9-MLDR，10-mpdf,11-t-differv,12-subguass;
=======
batch_algs_case = [1];                  % batch IVA的类型,1-auxiva,2-MNMF，3-EMIVA，4-IVE ,5-EMIVA_fast，6-AV-GMM-IVA,7-subguassmnmf,8-tmnmf.9-fastmnmf,10-t_fastmnmf,11-MLDR;12-fastmnmf2;13-tfastdifferent v ;14-tfast mpdf混合pdf控制本函数内算法
online_algs_case = [12];                 % online IVA的类型,1-auxiva,2-MNMF，3-EMIVA，4-IVE, 5-tmnmf, 6-fastmnmf, 7-tfastmnmf, 8-fastmnmf2, 9-MLDR，10-mpdf,11-t-differv,12-subguass;
>>>>>>> my_addnew-ver
Lb_case =  [20];                        % length of blocks [1,2,4,6,8,16], default using 1
tao_case =[0]/10;                       % default value is 0, default hanning windowing operation;
                                        % >1,  designed based special criterion;
                                        % windowing factor default value is 1.2; 10^5, indicates no windowing;
win_exp_ratio_range =[20:1:20]/20;      % default value is 1,effective for tao>0;                       
win_size_case = [4096];                 % STFT size 取决于平稳随机长度
taoMeanAdd_case = [1]/10;               % effective for tao>0; add some value, such as 0.1 to robustify the window desgin;
taoMean_case = [1];                     % default value is 1,effective for tao>0;
D_open_case = [1];                      % using D
perm_on_case = [0];                     % 混合信号帧乱序输入
forgetting_fac_case = [0.04];           % 仅在线使用，相当于论文中的1-alpha [0.3 0.2 0.1 0.08 0.06 0.04]
gamma_case = [4:4]/100;                 % 仅在线mic>src使用，为C的forgetting factor
delta_case = [0]/10^0;                  % using in contrast function: (||x||^2 + delta)^n_orders
whitening_case = [0];                   % 是否使用白化，未调试完成，暂时不要用
project_back = 1;                       % 是否需要将输出信号放缩到与输入信号相同的尺度
n_orders_case = {[1/2 1/2 1/2 1/2]};    % 离线的orders {[1/6 1/2],[1/2 1],[1/2 1/2],[1/8 1/2]}
n_orders_casenum = size(n_orders_case,2);
%% Partition %%
parti_case = [0];                       % 是否使用子块BSS
SubBlockSize_case = [100];              %  使用子块的大小。
SB_ov_Size_case = [1]/2;                %  overlap 的比例，大小为 round(SubBlockSize*SB_ov_Size)；
partisize_case = [1];
select_case = [0];                      % 是否选择子块（若parti=1，此处为选子块；若parti=0，此处为选子载波）
thFactor_case = [0]/200;                % 选择子块阈值因子
%% Epsilon test %%
diagonal_method_case =[0];              %  0: using a tiny portion of the mean diagnal values of Xnn；          
                                        % 1: using a tiny portion of the diagnal value of Xnn；       
Ratio_Rxx_case =[1:1]/10^3;             % 1:3:10 default is 0; 2 is a robust value;
epsilon_ratio_range =[10:10]/10;        % 1:4 10:10:60 robustify inversion of update_w ;
epsilon_ratio1_range =[10]/10;          % 1:4 10:10:60 robustify inversion of update_w ;
frameStart_range =[1];                  % 1 表明初始用一样的 epsilon；  表明前frameStart前用 epsilon_start_ratio;
epsilon_start_ratio_range =[1:1:1]/1;   %  表明前frameStart前用 ;
%% Order estimation %%
OrderEst_range =[0];                    % 是否进行order estimation；
OutIter_Num_range =[1];                 % order 估计的次数；
OrderEstUppThr_range =[11]/10;          % order 估计的上限；
OrderEstLowThr_range =[4]/10;           % order 估计的下限；
order_gamma_range =[8]/10;              % order 估计的滑动系数；
n_orders1_range =[10]/20;               % pdf 的估计指数（在线orders）
n_orders2_range =[10]/20;               % pdf 的估计指数（在线orders）
verbose_range =[1];                     % 0 不保留中间结果，1 保留中间结果。
%% Gamma ratio %%
GammaRatioThr_range =[10^2];            % 是否弱化在没有发声估计的门限，非常大等价不弱化；
GammaRatioSet_range =[10]/10;           % 是否弱化在没有发声估计的gamma 值。
%% Mix Model Estimation %%
mix_model_case = [0];                   % 是否使用混合CGG模型,1：硬判，2：EM，0：不使用
ita_case = [0.9];                       % 混合CGG模型中混合概率和beta值的递归平均系数
%% NMF Setting %%
nmf_iter_num_case = [1];                % nmf内迭代次数，1即可，过大出现奇异值
nmf_fac_num_case = [10];                % nmf基数目, 9 是典型值
nmf_b_case = [1/2];                     % IS-NMF指数值；p=beta=2时，GGD-NMF等效为b=1/2的IS-NMF
nmf_beta_case = [2];                    % beta值，用于GGD-NMF
nmf_p_case = [0.5];                     % p值，用于GGD-NMF
nmf_update_case = [0];                  % nmf update的模型，0是用IS-NMF，1用GGD-NMF
%% MNMF Setting %%
%MNMF_case = [0]; % 1-使用MNMF 0-不使用MNMF
MNMF_refMic_case = [1];                 % reference mic for performance evaluation using bss_eval_sources 1 or 2
<<<<<<< HEAD
MNMF_nb_case = [40];                    % number of MNMF bases for all sources (total bases)
MNMF_it_case = [100];                   % number of MNMF iterations
=======
MNMF_nb_case = [20];                    % number of MNMF bases for all sources (total bases)
MNMF_it_case = [50];                    % number of MNMF iterations
>>>>>>> my_addnew-ver
MNMF_first_batch_case = [40];           % first mini-batch size
MNMF_batch_size_case = [20];            % mini-batch size
MNMF_rho_case = [0.9];                  % the weight of last batch
MNMF_fftSize_case = [1024];             % window length in STFT [points]
MNMF_shiftSize_case = [256];            % shift length in STFT [points]
MNMF_delta_case = 10.^[-12];            % to avoid numerical conputational instability
MNMF_p_norm_range =[5:5]/10;            % default is 0.5; sqrt(x);
MNMF_drawConv = false;                  % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
sub_beta_case = [4];                    % sub-guass_MNMF BETA:[2.3.4]
tMNMF_v_case = [8];                     %[8] {[1,8]}是普通t,diffv是用cell;t-distribution 's freedom [0-20]
tMNMF_trial_case = [20];                % t-distribution 's   update T&V iternums for initialization previously
mpdf_probability_case = {[1 2 1 1]};    % mpdf四个源可选的不同分布，1-t分布；2-高斯分布
AR_init_it_case = [30];                 % AR init iterations
AR_tap_case = [4];                      % AR tap length
AR_delay_case = [3];                    % AR delay size
AR_init_SCM_case = {'circular'};        % AR init SCM ways {'circular','obs','ILRMA'}
AR_internorm_case = [1];                % AR interval norm
%% ILRMA Initialization Setting
ILRMA_init_case = [0];                  % 1:使用ILRMA初始化 0:不使用ILRMA初始化
ILRMA_type_case = [2];                  % 1 or 2 (1: ILRMA w/o partitioning function, 2: ILRMA with partitioning function)
ILRMA_nb_case = [40];                   % number of bases (for type=1, nb is # of bases for "each" source. for type=2, nb is # of bases for "all" sources)
ILRMA_it_case = [30];                   % iteration of ILRMA
ILRMA_normalize_case = true;            % true or false (true: apply normalization in each iteration of ILRMA to improve numerical stability, but the monotonic decrease of the cost function may be lost. false: do not apply normalization)
ILRMA_dlratio_case = 10.^[-2];          % diagonal loading ratio of ILRMA
ILRMA_drawConv = false;                 % true or false (true: plot cost function values in each iteration and show convergence behavior, false: faster and do not plot cost function values)
%% EM-IVA Setting %%%
%EMIVA_case = [0];                      %1-使用EMIVA 0-不使用EMIVA
beta1_case = [20]/10;                   % 源1的beta拟合
beta2_case = [20]/10;                   % 源2的beta拟合                    
beta1_offset_case = [2]/10^1;           % 源1的beta偏移量
beta2_offset_case = [0]/10^1;           % 源2的beta偏移量
EMIVA_max_iter_case =[20];              % batch最大迭代次数
EMIVA_ratio_case = [0.4];               % 0.2 0.4 0.6 0.8 weight of present frame 
detect_low_case = [0.3];                % 发声检测下界
detect_up_case = [0.7];                 % 发声检测上界
pmulti_case = [40];                     % 发声检测倍数
logp_case = [1];                        % 两种logp_case
AVGMMIVA_max_iter_case = [20];          % AV-GMM_IVA最大迭代次数
singleout_case = [0];                   % 是否进行单路分离
%% MLDR Setting %%
MLDR_iteration_case = [4]; % MLDR循环次数
MLDR_fft_size_case = [1024]; % FFT长度
MLDR_shift_size_case = [256]; % hop长度
% MLDR batch robust setting
MLDR_delta_case = 10.^[-6]; % diagonal loading
MLDR_epsilon_case = 10.^[-6]; % flooring parameter of Rx_tilde
MLDR_epsilon_hat_case = 10.^[-2]; % robust estimation of lambda
MLDR_moving_lambda_case= [0]; % 是否用相邻帧取平均的方式更新lambda 适用于离线
MLDR_moving_average_case = [1]; % moving average of the powers at adjacent frames
% MLDR online robust setting
MLDR_epsilon1_case = 10.^[-3]; % robust estimation of lambda_tilde
MLDR_zeta_case = 10.^[0]; % diagonal loading
MLDR_beta_case = [0.2]; % moving average weight of lambda_tilde
MLDR_rho_case = [0.995]; % scaling factor to compensate for overestimation
MLDR_epsilon1_hat_case = 10.^[0]; % robust estimation of Y
MLDR_gamma_first5_case = [0.3]; % gamma for first five frames
MLDR_gamma_others_case = [0.995]; % gamma for other frames
% SCM setting
SCM_alpha_case = [0.5];% priori distribution's shape parameter 
%% Prior AuxIVA %%%
DOA_switch_case=[1];                    % 控制DOA,0-[0,0,0,0]&doa_0; 1-[1,1,1,0]&doa_1
% prior_case = {[1,1,1,0]};             % 输入为(1,k)维向量，第k项为1代表有第k个源的先验信息，
%                                       % 第k项为0代表没有第k个源的先验，则不用DOA方法更新
%                                       % 当为OverIVA或IVE时，第K+1个prior代表BG的prior（K为target数目）
% DOA_esti_case = [1];                  % 是否使用DOA估计，为0则使用房间仿真传递的DOA信息 
DOA_update_case = [0];                  % 是否在迭代中更新DOA
DOA_init_case = [0];                    % 是否使用DOA初始化；
esti_mic_dist_case = [0.158];           % DOA估计时使用的麦克风间距
% 此处公式如下：
%     P = (DOA_tik_ratio * eye(M) + sum(DOA_Null_ratio * hf * hf') /deltaf^2
DOA_Null_ratio_range = [0.5]/1;      % DOA  加权值，[0.1]/10-4mic, 
DOA_tik_ratio_range = [0.7]/1000;       % DOA Tik 加权值，[0.5]/1000-4mic
deltaf_case = [40];                     % 目标源归一化系数10-4mic
deltabg_case = [0.5];                   % 背景噪声归一化系数
annealing_case = [0];                   % 是否模拟退火，fac_a = max(0.5-iter/iter_num, 0);
%% IVE %%%
% IVE的方法，1为IP-1（等效overiva）, 2为IP-2，3具体看auxive_update说明，4为FIVE方法
IVE_method_case = [4];
%% GWPE %%
run_gwpe_case = [0];                    % 是否在FastMNMF相关算法中使用GWPE（去混响+盲分离）
%% Parameters initialization %%%
Initiallize;
timeblock_Length = 1; % online SIR仿真分块长度（in second）
plot_time_mode = 1;   % mode1: 按照0-1,0-2,0-3...的形式计算SIR_time/SDR_time % mode2: 按照0-1,1-2,2-3...的形式计算SIR_time/SDR_time
case_num = 0;
room_imp_on = 1;   
RandomSeed =0; % 0 每次再现相同结果；1 每次随机不同结果
if RandomSeed==0 randn('state',98765); rand('state',12345); end % 保证每次SNR循环，初始种子一样 seed = rng(10);        
if RandomSeed==1 randn('state',cputime); rand('state',cputime+1); end % 保证每次SNR循环，初始种子一样 seed = rng(10);        
PowerRatio = 2;
if mix_file == 1 file_tag_case = 1; else file_tag_case = 1; end
% 固定选择source时，target和intf的数目统一成index数目，case设置为单一case防止重复仿真。
if ~rand_select target_source_num_case = [0]; intf_source_num_case = [0]; end 
%% 批处理
tic
for DOA_switch = DOA_switch_case %% DOA   
    if DOA_switch
    prior_case = {[1,1,0,0]};DOA_esti_case = [1];DOA_esti_online_case = [0];
    else
    prior_case = {[0,0,0,0]};DOA_esti_case = [0];DOA_esti_online_case = [0];end
for mix_sim = mix_sim_case for room_type = room_type_case for target_idx = target_index_case for intf_idx = intf_index_case for muteOn = muteOn_case for SINR_diffuse_ratio = SINR_diffuse_ratio_case
for deavg = deavg_case for mix_SINR = mix_SINR_case for sim_mic = sim_mic_case for target_source_num = target_source_num_case  for intf_source_num = intf_source_num_case for file_tag = file_tag_case for prior = prior_case
for angle = angle_case for angle_start = angle_start_case for R_ratio = R_ratio_case for tR_ratio = tR_ratio_case for src_permute_type = src_permute_type_case for reverbTime = reverbTime_case
for room_size = room_size_case for mic_center = mic_center_case for mic_distx = mic_distx_case for mic_disty = mic_disty_case
for angle1_start = angle1_start_case  for angle2_start = angle2_start_case for angle1_interval = angle1_interval_case for angle2_interval = angle2_interval_case for move_sound = move_sound_case for anglenum = anglenum_case
    %% 房间信道仿真
    if mix_sim % 使用房间信道进行仿真       
        room_setup;  % room参数设置        
        [xR, s, fs_ref, mic_pos, theta, target_source, intf_source, layout] ...
            = generate_sim_mix_new(room,target_index,intf_index);% RIR仿真函数    
    audiowrite([sound_dir,'/mix.wav'], xR.', fs_ref);  % 保存混合信号， 实录不必
    [xR_t, fss] = audioread([sound_dir,'/mix.wav']); 
    else % 使用实录信号
        mix_file_deal;
    end    
     xR_1 = xR_t.';  % 存成wav后再读取的混合信号，一般用xR即可，性能特别不好再用xR_1试试
    % FFT点数和窗函数设置
    for win_size = win_size_case
    win_type = 'hann';     inc = win_size / 2;       
%% 批处理
for determined = determined_case for online = online_case for batch_type = batch_type_case for tao = tao_case for win_exp_ratio = win_exp_ratio_range for taoMeanAdd = taoMeanAdd_case for taoMean = taoMean_case
for batch_algs = batch_algs_case for online_algs = online_algs_case
    if online      win_type = 'hamming'; end% 建议online使用汉明窗
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
for MNMF_delta = MNMF_delta_case for MNMF_p_norm = MNMF_p_norm_range for sub_beta = sub_beta_case for tMNMF_v = tMNMF_v_case for tMNMF_trial = tMNMF_trial_case for mpdf_pro = mpdf_probability_case 
for AR_init_it = AR_init_it_case for AR_tap = AR_tap_case for AR_delay = AR_delay_case for AR_init_SCM = AR_init_SCM_case for AR_internorm = AR_internorm_case
for ILRMA_init = ILRMA_init_case for ILRMA_type = ILRMA_type_case for ILRMA_nb = ILRMA_nb_case for ILRMA_it = ILRMA_it_case for ILRMA_normalize = ILRMA_normalize_case for ILRMA_dlratio = ILRMA_dlratio_case

for beta1 = beta1_case for beta2 = beta2_case for beta1_offset = beta1_offset_case for beta2_offset = beta2_offset_case 
for EMIVA_max_iter = EMIVA_max_iter_case for EMIVA_ratio = EMIVA_ratio_case  for detect_low = detect_low_case for detect_up = detect_up_case for pmulti = pmulti_case  for logp = logp_case for AVGMMIVA_max_iter = AVGMMIVA_max_iter_case
    EMIVA_beta = { [beta1+beta1_offset beta2+beta2_offset];[beta1+beta1_offset beta2+beta2_offset]};
for MLDR_iteration = MLDR_iteration_case for MLDR_fft_size = MLDR_fft_size_case for MLDR_shift_size = MLDR_shift_size_case for MLDR_delta = MLDR_delta_case for MLDR_epsilon = MLDR_epsilon_case  for MLDR_epsilon_hat = MLDR_epsilon_hat_case for MLDR_moving_lambda = MLDR_moving_lambda_case 
for MLDR_moving_average = MLDR_moving_average_case for MLDR_epsilon1 = MLDR_epsilon1_case for MLDR_zeta = MLDR_zeta_case for MLDR_beta = MLDR_beta_case for MLDR_rho = MLDR_rho_case for MLDR_epsilon1_hat = MLDR_epsilon1_hat_case for MLDR_gamma_first5 = MLDR_gamma_first5_case for MLDR_gamma_others = MLDR_gamma_others_case
<<<<<<< HEAD
for run_gwpe = run_gwpe_case
=======
for SCM_alpha = SCM_alpha_case
>>>>>>> my_addnew-ver
if RandomSeed==0 randn('state',98765); rand('state',12345); end % 保证每次SNR循环，初始种子一样 seed = rng(10);        
if RandomSeed==1 randn('state',cputime); rand('state',cputime+1); end % 保证每次SNR循环，初始种子一样 seed = rng(10);        
    if batch_type ~= 2 || batch_algs == 2 || online_algs == 2
        total_iter = 1; % auxIVA采用固定初始化，不需要重复仿真
    else if batch_type == 2
            seed = rng(10); % 固定nmf仿真随机种子
            total_iter = total_iter_num; % nmfIVA采用随机初始化，需要重复仿真
         end
    end    
for ITER = 1:total_iter
    %% 混合信号初始化
    x = xR; % 一般用xR即可，性能特别不好再用xR_1试试。   
    [mic_num, sample_num] = size(x); source_num = target_source;
    if determined source_num = mic_num; end  
    %% 仿真参数设置
    option_setup;
    %% preprocess for select beta     pre_select_beta;
option.EMIVA_beta = EMIVA_beta;
    %% 信号处理和盲分离
<<<<<<< HEAD
ExecAlgorithm;
=======
    if online
        % 在线算法(online)
        if online_algs == 1 % AuxIVA
            [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option); % 带乱序输入的online版本
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
        elseif online_algs == 5 % t-mnmf
            [s_hat,label] = t_MNMF_bss_online(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif online_algs == 6 % fastmnmf 
            [s_hat,label] = FastMNMF_bss_online(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif online_algs == 7 % t_fastmnmf 
            [s_hat,label] = t_FastMNMF_bss_online(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
         elseif online_algs == 8 % fastmnmf2 
            [s_hat,label] = FastMNMF2_bss_online(x.',source_num,option); 
             s_est = squeeze(s_hat(:,2,:)).';
        elseif online_algs == 9 % MLDR
            [s_hat,label] = MLDR_online(x.',option);
            s_est = s_hat.'; source_num = 1;
        elseif online_algs == 10 % tfastmnmf differnt pdf;                                % s1:t-distribution;s2:guass-distribution 
            [s_hat,label] = mpdf_FastMNMF_bss_online(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).'; 
        elseif online_algs == 11 % tfastmnmf differnt v 
            [s_hat,label] = t_FastMNMFdiffv_bss_online(x.',source_num,option); 
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif online_algs == 12 % subguass_mnmf
            [s_hat,label] = subguas_FastMNMF_bss_online(x.',source_num,option); 
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        end
        out_type = 'online';
    else
        % 离线算法(batch)
%         %% preprocess for select beta
%         pre_select_beta;
        if batch_algs == 1 % AuxIVA
            [s_est,label] = auxiva_audio_bss_batch(x,source_num,option); % IVA batch(including nmfiva)
        elseif batch_algs == 2 % MNMF
tic
            [s_hat,label] = bss_multichannelNMF_offline(x.',source_num,option);
            s_est = squeeze(s_hat(:,MNMF_refMic,:)).';toc
        elseif batch_algs == 3 % EMIVA
%             [pcaData,projectionVectors,eigVal] = whitening_pre(x',2);
            [s_est,label,~, ~,~,~,obj_set] = CGGMM_IVA_batch(x, option);           
        elseif batch_algs == 4 % IVE
            [s_est,label] = auxive_audio_bss_batch(x,option); % IVE batch
        elseif batch_algs == 5 % EMIVA_fast
            [s_est,label,~, ~,~] = CGGMM_IVA_batch_fast(x, option); 
        elseif batch_algs == 6 % AV-GMM-IVA
            [s_est,label,~, ~,~] = AV_GMM_IVA_batch(x, option); 
        elseif batch_algs == 7 % subguassmnmf 
            [s_hat,label] = subguas_FastMNMF_bss_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 8 % t-mnmf
            [s_hat,label,cost] = t_MNMF_bss_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 9 % fastmnmf 
            [s_hat,label] = FastMNMF_bss_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 10 % t_fastmnmf 
            [s_hat,label] = t_FastMNMF_bss_offline(x.',source_num,option); 
              s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 11 % MLDR
            [s_hat,label] = MLDR_batch(x.',option);
            s_est = s_hat.'; source_num = 1;
        elseif batch_algs == 12 % fastmnmf2 
            [s_hat,label] = FastMNMF2_bss_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 13 % tfastmnmf differnt v 
            [s_hat,label] = t_FastMNMFdiffv_bss_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 14 % tfastmnmf differnt pdf; 
                                % s1:t-distribution;s2:guass-distribution 
            [s_hat,label] = mpdf_FastMNMF_bss_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
        elseif batch_algs == 15 % SCM BSE 
                                % s1:t-distribution;s2:guass-distribution 
            [s_hat,label] = bse_SCM_offline(x.',source_num,option); 
             s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
             source_num = 1;
         end
        out_type = 'batch';
    end 
>>>>>>> my_addnew-ver
    %% PEASS
    if PEASS_flag
        PEASS_process;
    end
    %% SIR SDR计算
    if mix_sim
        L = min(size(s,2), size(s_est,2));
        if source_num == 1
%            [SDR,~,SAR,perm] = bss_eval_sources(s_est(1,1:L), s(1,1:L));           SIR = 0;
            [SDR_in,~,SAR_in,~] = bss_eval_sources(x(1,1:L), s(1,1:L));
            [SDR_out,~,SAR_out,perm] = bss_eval_sources(s_est(1,1:L), s(1,1:L));
            SIR = 0; SDR=SDR_out-SDR_in; SAR=SAR_out-SAR_in;
        end
            if source_num == 2 || source_num == 4 % 对于4*2和2*2
%               [SDR,SIR,SAR,perm] = bss_eval_sources(s_est(:,1:L), s(1:source_num,1:L)); max_comb = 1:source_num;               
%               if PEASS_flag
%                 [SDR_in,SIR_in,SAR_in,~] = bss_eval_sources(x(:,1:L), s(1:source_num,1:L));
%                 [SDR_out,SIR_out,SAR_out,perm,option] = bss_eval_sources_withPEASS(s_est(:,1:L), s(1:source_num,1:L),option); max_comb = 1:source_num;
%                 SDR=SDR_out-SDR_in; SIR=SIR_out-SIR_in ; SAR=SAR_out-SAR_in;% ;
%                 PEASS;   
%               else
%                 [SDR_in,SIR_in,SAR_in,~] = bss_eval_sources(x(:,1:L), s(1:source_num,1:L));
                [SDR_out,SIR_out,SAR_out,perm] = bss_eval_sources(s_est(:,1:L), s(1:source_num,1:L)); max_comb = 1:source_num;
                SDR=SDR_out;%-SDR_in; 
                SIR=SIR_out;%-SIR_in ; 
                SAR=SAR_out;%-SAR_in;% ;
%               end
                s_est = s_est(perm',:);
            elseif source_num > 2 % 对于4*4，需要选两路SIR之和最大的组合
                perm_comb = nchoosek(1:source_num,2); [comb_num,~] = size(perm_comb);
                SDR_c = zeros(2,comb_num); SIR_c = zeros(2,comb_num); SAR_c = zeros(2,comb_num); perm_c = zeros(2,comb_num);
%                 for cn = 1:comb_num
%                     [SDR_c(:,cn),SIR_c(:,cn),SAR_c(:,cn),perm_c(:,cn)] = bss_eval_sources(s_est(perm_comb(cn,:),1:L), s(:,1:L));
%                 end
                [SDR_c,SIR_c,SAR_c,perm_c] = bss_eval_sources(s_est(:,1:L), s(:,1:L));
                [~, max_perm_index] = max(sum(SDR_c,1)); max_comb = perm_comb(max_perm_index,:);
                SDR = SDR_c(:,max_perm_index); SIR = SIR_c(:,max_perm_index); SAR = SAR_c(:,max_perm_index); perm = perm_c(:,max_perm_index);
%                 s_est = s_est(max_comb(perm'),:); label = label{max_comb(perm')};
%                 s_est = s_est(max_comb(perm'),:); label = label{max_comb};
            end
            SNR = zeros(2,1);
%         end
        if target_SIRSDR
            SIR = SIR(1); SDR = SDR(1); SAR = SAR(1); SNR = SNR(1);
        end
        fprintf('%s\n%s\nSDR = %s\nSIR = %s\n',filenameTmp,out_type,num2str(SDR'),num2str(SIR'));
%         profile viewer
        if online % 计算 online SIR improvment
%             tap_Length = timeblock_Length * fs_ref;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length); SDR_time = cal_SDR_time(x,s,s_est,tap_Length);
            tap_Length = timeblock_Length * fs_ref;            SIR_time = cal_SIR_time(x,s,s_est,tap_Length,plot_time_mode); SDR_time = cal_SDR_time(x,s,s_est,tap_Length,plot_time_mode);            
              T = [0 : ceil(L / tap_Length)] * timeblock_Length;            %SIR_time_all = [SIR_time_all;SIR_time]; SDR_time_all = [SDR_time_all;SDR_time];                                                                                                                    SDR_time_all_full = [SDR_time_all_full;SDR_time_all];
        end         
        if total_iter ~= total_iter_num
            % 固定初始化和随机初始化一起仿真时，需要将固定初始化的数据进行复制来减少仿真时间。
            SIR_case = [SIR_case repmat(SIR,[1,total_iter_num])];  SDR_case = [SDR_case repmat(SDR,[1,total_iter_num])];  SAR_case = [SAR_case repmat(SAR,[1,total_iter_num])];  SNR_case = [SNR_case repmat(SNR,[1,total_iter_num])];
        else
            SIR_case = [SIR_case SIR];  SDR_case = [SDR_case SDR];  SAR_case = [SAR_case SAR];  SNR_case = [SNR_case SNR];
%             SDR_case_full=[SDR_case_full SDR_case];
        end
    end
    fid=fopen('AR_record.txt','a');fprintf(fid,'\n%s\n',filenameTmp);fprintf(fid,'SDR = ');fprintf(fid,'%g   ',SDR);
    fprintf(fid,'SIR = ');fprintf(fid,'%g   ',SIR);fclose(fid);
      %% 保存数据(case)和音频
%     strsave= strcat(mkdir_str1,filenameTmp,'.mat'); sav=['save ' strsave]; eval(sav);
    case_num = case_num + 1;    case_str = num2str(case_num);
%     filenameTmp1 = strcat('case_',case_str,'.mat'); strsave= strcat(mkdir_str1,filenameTmp1);
%     sav=['save ' strsave]; eval(sav);
    %% Save separated wave files                         (:,1:end-4)(:,1:end-4)'_out1ch_origin.wav'   '_out2ch_case',case_str,'.wav'
%     figure;
%     plot( (0:option.MNMF_it), cost );grid on;%hold on;semilogy( (0:maxIt), cost );
%     set(gca,'FontName','Times','FontSize',16);
%     xlabel('Number of iterations','FontName','Arial','FontSize',16);
%     ylabel('Value of cost function','FontName','Arial','FontSize',16);
%     gcf_temp = strcat('D:\github-code\VAD\BSS_TEST\gcf_save\',filenameTmp,'_case',case_str,'.jpg');
%     saveas(gcf, gcf_temp); %保存当前窗口的图像
%     close(figure(gcf)); close  all;    
    audiowrite([sound_dir,'/',filenameTmp,'_out1ch_case',case_str,'.wav'], s_est(1,:)', fs_ref);
      if size(s_est,1) == 2
        audiowrite([sound_dir,'/',filenameTmp,'_out2ch_case',case_str,'.wav'], s_est(2,:)', fs_ref);
      end
%        audiowrite([sound_dir,'/',method,'.1_sep1','.wav'], s_est(1,:)', fs_ref);
%       if size(s_est,1) == 2
%         audiowrite([sound_dir,'/',method,'.1_sep2','.wav'], s_est(2,:)', fs_ref);
%       end
      if size(s_est,1) == 3
          audiowrite([sound_dir,'/',filenameTmp,'_out2ch_case',case_str,'.wav'], s_est(2,:)', fs_ref);
          audiowrite([sound_dir,'/',filenameTmp,'_out3ch_case',case_str,'.wav'], s_est(3,:)', fs_ref);
%         audiowrite([sound_dir,'/sep3_',method,'.wav'], s_est(3,:)', fs_ref);   audiowrite([sound_dir,'/sep4_',method,'.wav'], s_est(4,:)', fs_ref);   
      end
      if size(s_est,1) == 4
          audiowrite([sound_dir,'/',filenameTmp,'_out2ch_case',case_str,'.wav'], s_est(2,:)', fs_ref);
          audiowrite([sound_dir,'/',filenameTmp,'_out3ch_case',case_str,'.wav'], s_est(3,:)', fs_ref);
          audiowrite([sound_dir,'/',filenameTmp,'_out4ch_case',case_str,'.wav'], s_est(4,:)', fs_ref);
      end
%       sep1_str = label{1}; %% 按照sep1_case1_target_batch方式命名
%       audiowrite([sound_dir,'/sep1_case',case_str,'_',sep1_str,'_',out_type,'.wav'], s_est(1,:)', fs_ref);
%       if size(s_est,1) == 2
%           sep2_str = label{2};
%           audiowrite([sound_dir,'/sep2_case',case_str,'_',sep2_str,'_',out_type,'.wav'], s_est(2,:)', fs_ref);
%       end
%       if size(s_est,1) == 4
%           audiowrite([sound_dir,'/sep3_case',case_str,'_',out_type,'.wav'], s_est(3,:)', fs_ref);   audiowrite([sound_dir,'/sep4_case',case_str,'_',out_type,'.wav'], s_est(4,:)', fs_ref);   
%       end
    % 波形&语谱图绘制，仅测试实录信号时使用 -1.8152   -1.8152   -3.3037   -3.3037
%     if mix_sim == 0
%         plot_sound(s_est,xR,fs_ref,label);
% %         print(gcf,'-djpeg',['.\plot\save\',case_str+1,'.jpeg']);
%     end
    % 绘制房间布局图
    if room_plot_on == 1 && mix_sim == 1
        plot_room_layout(layout);
    elseif room_plot_on == 2 && mix_sim == 1
        plot_room_layout(layout);
        room_plot_on = 0;
    elseif room_plot_on == 3 && mix_sim == 1
        plot_car_layout(layout);
    end
    toc
%     else
%         status = rmdir(file_state,'s');% 删除不符合SDR条件的文件夹及其内容
end
end
end
end
end
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
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
end
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
if ~mix_sim
    loadMat=1;
    if loadMat == 1        
%         clear all; close all;
%          for files=1 :3  
%     %filenameTmp =strcat('jovi_man_',num2str(files)); 
%         filenameTmp =strcat('man_jovi_3.wav');%,num2str(files)
%      %   filenameTmp ='test_em_实录_beta';%可将仿真参数作为文件名 test_online_offline   test_AWGN_Lb test_AWGN_Lb
%         mkdir_str=strcat('./Simulation_Single/mix_new_em/jovi_man/',filenameTmp);
%         mkdir(mkdir_str);%一运行就会在当前文件夹下创建simulation文件夹
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
       % case_name = {'non-DOA','DOA'};%输入算法名称 不输入名称则输出case1,case2,case3        
        %plot_case = online_algs_case;% 针对不同需要仿真的case画图
                                    % DOA_Null_ratio_range DOA_switch_case deltaf_case deltabg_case  deltabg_case deltaf_case  
                                    % DOA_Null_ratio_range DOA_esti_case angle_start_case prior_casen_orders1_range,n_orders2_range
                                    % online_algs_case reverbTime_case  batch_update_num_case
         % online =1;
        if online
         %   plotSIR_time(SIR_time_all,T,max(size(epsilon_ratio_range,2),size(Ratio_Rxx_case,2)),3); % 绘制时变SIR         
%          case_num2 = size(SDR_time_all_full,1) / 2;
         sub_case_num = case_num;
         packFigNum =   1;  % 一个fig 分成几个(=packFigNum)subfugure，进行情况比较
         SortedPlotThr = 1; % >1,表明显示排序。
         SortedPlotNum = 8; % 表明排序显示case的数目,在 sub_case_num 比较大的时候可以清晰显示
         plotRatio =1;      %一次多少比例的图； default=1； 2 表明画50% 的图；   
         %close all;  
%          plotSIR_time1(SDR_time_all_full,T,case_num2,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); %,plot_case 绘制时变SIR
         plotSIR_time1(SDR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,Lb_case); %,plot_case 绘制时变SIR
% plotSIR_time1(SDR_time_all,T,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,n_orders1_range,n_orders2_range); % 绘制时变SIR
%function plotSIR_time(SIR,T,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
%sub_case_num      同一大类比较在一个Figure 画的数目, 比如一共 10 =size(SIR_time_all,1)/sourceNum (=2),一个图画2*5比较。
%packFigNum       一次把几种subcase 都放到一起来plot
%SortedPlotThr;   如果case_num太多显示不了，判决是否显示 SortedPlotNum case，
%如果是1，缺省显示不排序的情况；如果>1, 表明排序显示。
%SortedPlotNum;   如果case_num太多显示不了，显示 SortedPlotNum case；
%plotRatio       一次多少比例的图； default=1； 2 表明画50% 的图；
        else
%             R_num = size(SDR_case_full,1) ;
%             case_num = size(SDR_case_full,1) / total_iter_num;
            R_num = size(SDR_case,1) ;
            case_num = size(SDR_case,2) / total_iter_num;
            sub_case_num = case_num;
            packFigNum =   1;  % 一个fig 分成几个(=packFigNum)subfugure，进行情况比较
            SortedPlotThr = 1; % >1,表明显示排序。
            SortedPlotNum = 8; % 表明排序显示case的数目,在 sub_case_num 比较大的时候可以清晰显示
            plotRatio =1;
            if ~isempty(case_name)
                plotSDR_name(case_num,SDR_total,case_name); % 绘制带有算法名称的SDR
            else
                plotSDR_name(case_num,SDR_total,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio); % 绘制普通SDR
            end
            %plotSDR(case_num,SDR_total); % 绘制普通SIR、SDR
            %plotSDR(case_num,SDR_total,SIR_total); % 绘制普通SIR、SDR  
        end 
%          end
    end
    close all;
end
% end
% end