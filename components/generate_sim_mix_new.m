function [mix,s,fs_ref,mic_pos,theta,n_tgt,n_intf,layout] = generate_sim_mix_new(room,target_index,intf_index)
%% 生成房间仿真信号 20.5.12更新，现在支持更多参数进行房间仿真
% 可以进行SINR混合，并且支持AWGN的添加: 其方差为总干扰方差的(1-diffuse_ratio)倍
% 备注：若不开启随机选择源信号开关，则干扰源和目标源数目会设定为两个index的size，并且返回到主函数里

%% Parameter Setup;
n_mic = room.sim_mic;           % 仿真麦克风数目
n_intf = room.intf_source_num;  % 干扰源数目
n_tgt = room.target_source_num; % 目标源数目
room_type = room.room_type;     % 房间类型，在room_imp_setup函数中使用, 取值为1-6
DebugRato = room.DebugRato;     % 调试用，决定使用多长的信号进行仿真
rand_select = room.rand_select; % 是否随机选择源信号，1:随机，0:按输入索引选择
deavg = room.deavg;             % 混合信号去均值，仿真和实录均适用，1开启0关闭
SINR = room.SINR;               % 目标信号和噪声及干扰信号的混合信干噪比，若SINR=0，则为1:1混合
src_permute_type = room.src_permute_type; % 房间内源摆放位置是否随机，1:随机，0:顺序摆放
R_ratio = room.R_ratio;         % 源摆放半径与最大摆放距离的比例
tR_ratio = room.tR_ratio;       % 目标源与干扰摆放距离的比例
angle = room.angle;             % 两个源之间的最小夹角，单位为度，0度为x轴正向
angle_start = room.angle_start; % 摆放源的起点角度
reverbTime = room.reverbTime;   % 混响时间长度，单位为秒
diffuse_ratio = room.SINR_diffuse_ratio; % 非相关干扰与总干扰之比，1-diffuse_ratio为AWGN成分，若=1即为不加AWGN
customize_room = room.customize_room; % 是否使用自定义房间大小（线阵麦克风）
room_size = room.size;          % 房间大小
mic_center = room.mic_center;   % 麦克风阵列中心坐标
mic_dist = [room.mic_distx, room.mic_disty]; % 相邻麦克距离的水平和垂直方向投影长度
fsResample = 16000;             % 原始源信号采样频率缺省16K
num_points = 160000;            % 原始源信号最长限定在10s
angle1_start = room.angle1_start; % 源1的开始分布角度
angle2_start = room.angle2_start; % 源2的开始分布角度
angle1_interval = room.angle1_interval; % 源1的分布角度间隔
angle2_interval = room.angle2_interval; % 源2的分布角度间隔
move_sound = room.move_sound;   % 是否开启移动声源仿真
anglenum = room.anglenum;       % 需要改变的角度数目
%% Source Select
if rand_select                  % 随机选择源，把源地址存在tag中
    [target_tag, intf_tag] = source_select_new(n_tgt,n_intf);
else                            % 指定源，把源地址存在tag中
    n_tgt = length(target_index); n_intf = length(intf_index);
    [target_tag, intf_tag] = source_select_new(n_tgt,n_intf,target_index,intf_index);
end
% set the source powers
src_std = ones(1,n_tgt);
% src_std(1) = src_std(1) / sqrt(2);
%% Load Source Signal
n_src = n_tgt + n_intf;
dataLenRe = num_points * DebugRato;
src_sig = zeros(dataLenRe, n_src); % time x source
src_sig_resample = zeros(dataLenRe, n_src); % time x source
fs = zeros(1,n_src);
dataLenRe = num_points * DebugRato;
for k = 1:n_tgt
    [src, fs(k)] = audioread(target_tag{k});
    src_sig(:,k) = src(1:dataLenRe,:);
    src_sig_resample(:,k) = resample(src_sig(:,k), fsResample, fs(1), 100);
end
for l = n_tgt+1 : n_src
    [src, fs(l)] = audioread(intf_tag{l-n_tgt});
    src_sig(:,l) = src(1:dataLenRe,:);
    src_sig_resample(:,l) = resample(src_sig(:,l), fsResample, fs(1), 100);
end
s = src_sig_resample.'; fs_ref = fs(1);
%% 过信道，SINR混合和归一化
src_rir = zeros(dataLenRe,n_mic,n_src);
if room_type == 8
    [H,mic_pos,theta,layout] = car_env_setup(n_mic,n_src,room_type,reverbTime);
else
    [H,mic_pos,theta,layout] = room_imp_setup_new(n_mic,n_src,n_tgt,room_type,reverbTime,angle,angle_start,...
        R_ratio,tR_ratio,src_permute_type,customize_room,...
        angle1_start,angle2_start,angle1_interval,angle2_interval,move_sound,anglenum,room_size,mic_center,mic_dist);
end
% Scenario : 1—the oldest one with angle ; 2—the oldest one
%            3—mic>num Scenario; 4—Block-online Scenario 1
%            5—Block-online Scenario 2; 6—online Scenario(200ms)
%            7—large mic array room(up to 8 mic)
%            8—car four mic
if room.move_sound == 1     
    partfreq = fix(size(src_sig_resample,1)/anglenum); % 切分每段的频点数
    for partflag = 1:size(H,4) % 根据角度要把语音切分的总段数
        for k = 1:n_src
            src_rir((partflag-1)*partfreq+1:partflag*partfreq,:,k) = fftfilt(H(:,:,k,partflag),src_sig_resample((partflag-1)*partfreq+1:partflag*partfreq,k)); % time x mic x source
        end
        if mod(size(src_sig_resample,1),partfreq) ~= 0
            src_rir((size(H,4)-1)*partfreq+1:end,:,k) = fftfilt(H(:,:,k,partflag),src_sig_resample((size(H,4)-1)*partfreq+1:end,k)); % 最后一帧的余数数据
        end
    end
else
    for k = 1:n_src
    src_rir(:,:,k) = fftfilt(H(:,:,k),src_sig_resample(:,k)); % time x mic x source
    end
end
% first normalize all separate recording to have unit power at microphone one
ref = 1; % reference mic;
P_mic_ref = squeeze(std(src_rir(:,ref,:),0,1));
for k = 1:n_src
    src_rir(:,:,k) = src_rir(:,:,k) / P_mic_ref(k);
    if k <= n_tgt
         src_rir(:,:,k) = src_rir(:,:,k) * src_std(k);
    end
end

% Total variance of noise components
var_noise_tot = 10^(-SINR / 10) * sum(src_std.^2);

% compute noise variance
sigma_n = sqrt((1 - diffuse_ratio) * var_noise_tot);

% now compute the power of interference signal needed to achieve desired SINR
if n_src - n_tgt > 0
    sigma_i = sqrt((diffuse_ratio / (n_src - n_tgt)) * var_noise_tot);
    src_rir(:,:,n_tgt+1:end) = src_rir(:,:,n_tgt+1:end) * sigma_i;
end

% Mix down the recorded signals
mix = sum(src_rir, 3) + sigma_n * randn(dataLenRe, n_mic);

% amplitude normalization (for better sound effect)
normCoef = max(max(abs(mix)));
mix = mix ./ normCoef;
disp('RIR Simulation Done');

if deavg
    mix = (mix- mean(mix)).';
else
    mix = mix.';
end
end

%%  getComplexGGD得到复高斯分布 暂时没用 先放着
function [Z_gen] = getComplexGGD(N,C,c)
Z = zeros(N,1);
cc = c*2;
for i = 1:N %cc = 2 is Gaussian
    Z(i) = gamrnd(2/cc,1)^(1/cc) * exp(j*2*pi*rand);
end
cNorm = (gamma(2/c)/(gamma(1/c))); 
w =Z*(1/sqrt(cNorm)); %Normalize the complex variance||归一化
waug = [w conj(w)]'; 
Z_gen = C^(1/2)*waug;  
end