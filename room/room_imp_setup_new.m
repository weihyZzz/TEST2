function [H,mic_pos,DOA,layout] = room_imp_setup_new(mic_num,source_num,target_num,Scenario,reverbTime,angle,angle_start,...
    R_ratio,tR_ratio,type,customize_room,angle1_start,angle2_start,angle1_interval,angle2_interval,move_sound,anglenum,varargin)
%% 新版房间仿真设置
% 本版本去除了旧版本的源位置限制，拥有了更多仿真参数选择。
% 现在的源是以麦克风阵列中心为圆心形成圆周分布，圆周半径不大于麦克风阵列中心到距离其最近的墙面
% 并且源分布可以选择随机分布（但两源夹角为angle的倍数）或者顺序分布
% 还可以选择是否绘制房间布局图

% [Input]
% ang为声源的朝向 函数最后的一项'omnidirectional'指的是声源指向性，目前设定成全向传播。
% 需要更严格的声源方向这个值可以设定为'cardioid'，也就是心形指向。
% Scenario: 选择几号房间，目前有房间1-7,房间1-6仅支持4mic以下，房间7可支持8mic
% reverbTime: 混响时间，单位为秒
% angle: 两个源之间的最小夹角, 单位为度，0度为x轴正向
% angle_start: 源从哪个角度开始分布，单位为度，0度为x轴正向
% R_ratio: 半径比例，控制源与麦的间距
% type:1—开启随机分布，0—以angle_start为起点顺序分布

% [Output]
% H:房间信道, size: filterLength x mic x src;
% mic_pos:麦克风间距,第一个参考麦间距为0, size: 1 x mic
% angle_src:各个源的到达角,为弧度制, size: 1 x src

%% Reading room config
% 只需要修改一下读取方式，就可以直接读取有最大mic数目的txt，从中选择我们需要的mic数目
config_name = ['room_S',num2str(Scenario),'_4.txt'];

fid = fopen(config_name);
[room_config,~] = textscan(fid,'%9s%[^\n]','commentstyle','%');
fclose(fid);

param = room_config{1};
value = room_config{2};
if customize_room
    room_size = varargin{1};
    mic_center = varargin{2};
    mic_dist = varargin{3};
else
    room_size=eval(['[' value{strcmp(param,'room_size')} ']'])';
    mic_center=eval(['[' value{strcmp(param,'sc')} ']'])'; % txt文件里有麦克风阵列中心坐标时，直接读取
end
%% 读取txt文件
channels = mic_num;
sensor_xyz=zeros(3,channels);
sensor_off=zeros(3,channels);
tm_sensor=zeros(3,3,channels);
for sensor_No=1:channels
    sensor_xyz(:,sensor_No)=eval(['[' value{strcmp(param,['sp' int2str(sensor_No)])} ']'])';
    sensor_off(:,sensor_No)=eval(['[' value{strcmp(param,['so' int2str(sensor_No)])} ']'])';
    c_psi=cos(pi/180*sensor_off(1,sensor_No)); s_psi=sin(pi/180*sensor_off(1,sensor_No));
    c_theta=cos(-pi/180.*sensor_off(2,sensor_No)); s_theta=sin(-pi/180.*sensor_off(2,sensor_No));
    c_phi=cos(pi/180.*sensor_off(3,sensor_No)); s_phi=sin(pi/180.*sensor_off(3,sensor_No));
    tm_sensor(:,:,sensor_No)=[c_theta.*c_psi c_theta.*s_psi -s_theta;...
        s_phi.*s_theta.*c_psi-c_phi.*s_psi s_phi.*s_theta.*s_psi+c_phi.*c_psi s_phi.*c_theta;...
        c_phi.*s_theta.*c_psi+s_phi.*s_psi c_phi.*s_theta.*s_psi-s_phi.*c_psi c_phi.*c_theta];
end
%% 通过麦克风阵列中心坐标 计算 麦克位置三维坐标
if customize_room
    mic_distx = mic_dist(1);
    mic_disty = mic_dist(2);
    mic_half = mic_num/2;
    for sensor_No = 1:mic_half
        sensor_xyz(1,sensor_No)=mic_center(1)-(mic_half+0.5-sensor_No)*mic_distx;
        sensor_xyz(2,sensor_No)=mic_center(2)-(mic_half+0.5-sensor_No)*mic_disty;
        sensor_xyz(3,sensor_No)=mic_center(3);
        sensor_xyz(1,mic_num+1-sensor_No)=mic_center(1)+(mic_half+0.5-sensor_No)*mic_distx;
        sensor_xyz(2,mic_num+1-sensor_No)=mic_center(2)+(mic_half+0.5-sensor_No)*mic_disty;
        sensor_xyz(3,mic_num+1-sensor_No)=mic_center(3);
    end
end
%% 麦克风间距和源分布角度计算
mic_pos = zeros(1,mic_num); % 麦克风间距
ref_mic_xyz = sensor_xyz(:,1); % 参考麦克风（1号麦）坐标
for sensor_No = 2:mic_num
    mic_pos(sensor_No) = norm(sensor_xyz(:,sensor_No)-ref_mic_xyz);
end
% 麦克风阵列中心坐标计算，此处前提是麦克风为线阵
if isempty(mic_center)
    mic_center = ref_mic_xyz + 1/2 * (sensor_xyz(:,mic_num)-ref_mic_xyz);
end
% 麦克风阵列中心到四面墙的距离 [西 东 南 北]
x_center = mic_center(1); y_center = mic_center(2); z_center = mic_center(3);
c2w_distance = [x_center,room_size(1)-x_center,y_center,room_size(2)-y_center];

% 到墙面的最小距离
[min_dist,ind] = sort(c2w_distance); 

% 最小距离若过小,则采用第二小的距离，并且摆放范围限定在半圆
if min_dist(1) < 0.5 % 源到麦克风最小距离不得小于50cm
    start = [-90 90 0 -180]; % c2w_distance中每面墙对应的角度起点
    angle_max = 180; 
    angle_start = start(ind(1));
    max_R = min_dist(2);
else
    angle_max = 360; max_R = min_dist(1);
end

if move_sound == 1 % only support 2 src
    angle1_all = angle1_start + (0:anglenum)* angle1_interval;
    angle2_all = angle2_start + (0:anglenum)* angle2_interval;
    angle1_all = angle1_all * pi / 180 ; angle2_all = angle2_all * pi / 180;
    angle_src_all = [angle1_all' angle2_all'];
%% RIR计算 % 根据角度，中心坐标和半径计算各个源摆放点的坐标
    for flag = 1 : anglenum
    angle_src = angle_src_all(flag,:);    
    src_loc = cal_loc(R_ratio,tR_ratio,target_num,angle_src,max_R,mic_center);

    %计算每个源到各个麦克风之间的精确DOA
    DOA = cal_DOA(angle_src,sensor_xyz); % mic x src
    % % 单数值DOA
    % DOA = angle_src;

    % 获取房间信道（H_ref只是为了得到RIR长度）
    H_ref = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,1),[0,0,0],'omnidirectional',room_size,sensor_xyz);
    length_H = size(H_ref,1);
    if flag == 1
    H = zeros(length_H,mic_num,source_num,flag);end
    H(:,:,1,flag) = H_ref;
    for k = 2:source_num
        H(:,:,k,flag) = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,k),[0,0,0],'omnidirectional',room_size,sensor_xyz);
    end 
    end
else
    % 把angle_start+0:angle_max上所有的夹角为angle的点都找出来
    angle_all = angle_start + (0:angle:angle_max); 
    angle_all(angle_all>360) = angle_all(angle_all>360)-360;
    angle_all(angle_all>180) = -360+angle_all(angle_all>180);
    angle_all = angle_all * pi / 180;
    if type % 随机分布，任意两源夹角为angle的整数倍
        angle_src = angle_all(randperm(length(angle_all),source_num));
    else % 从angle_start开始按顺序分布
        angle_src = angle_all(1:source_num);
    end
%% RIR计算    % 根据角度，中心坐标和半径计算各个源摆放点的坐标
    src_loc = cal_loc(R_ratio,tR_ratio,target_num,angle_src,max_R,mic_center);

    %计算每个源到各个麦克风之间的精确DOA
    DOA = cal_DOA(angle_src,sensor_xyz); % mic x src
    % % 单数值DOA
    % DOA = angle_src;

    % 获取房间信道（H_ref只是为了得到RIR长度）
    H_ref = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,1),[0,0,0],'omnidirectional',room_size,sensor_xyz);
    length_H = size(H_ref,1);
    H = zeros(length_H,mic_num,source_num);
    H(:,:,1) = H_ref;
    for k = 2:source_num
        H(:,:,k) = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,k),[0,0,0],'omnidirectional',room_size,sensor_xyz);
    end    
end
   
% 输出房间布局参数，用于绘图
layout.R = max_R*R_ratio;
layout.tR = max_R*R_ratio*tR_ratio;
layout.mic_center = mic_center;
layout.angle_src = angle_src;
layout.src_loc = src_loc;
layout.sensor_xyz = sensor_xyz;
layout.room_size = room_size;
layout.reverbTime = reverbTime;
end

function loc = cal_loc(R_ratio,tR_ratio,target_num,angle,max_R,mic_center)
%% 坐标计算，角度以X轴正向为0，输入的angle范围为[-pi,pi]
    M = length(angle);
    loc = zeros(3,M); % 3 * src
    x_center = mic_center(1); y_center = mic_center(2); z_center = mic_center(3); 
    R = R_ratio * max_R;
    R_all = repmat(R,1,M);
    R_all(1:target_num) = R_all(1:target_num) * tR_ratio;
    for m = 1:M
        loc(1,m) = x_center + R_all(m) * cos(angle(m)); % x
        loc(2,m) = y_center + R_all(m) * sin(angle(m)); % y
        loc(3,m) = z_center; % z，默认与麦同高
    end
end

function DOA = cal_DOA(angle_src,sensor_xyz)
%% 精确DOA计算，size = mic x src, 每一列对应着同一个src到各个mic的DOA
    n_src = size(angle_src,2); n_mic = size(sensor_xyz,2);
    ref_ang = atan((sensor_xyz(2,2)-sensor_xyz(2,1)) / (sensor_xyz(1,2)-sensor_xyz(1,1)));    
    DOA = zeros(1,n_src);
    for k = 1:n_src
        DOA(:,k) = angle_src(k)-ref_ang;
    end
end


