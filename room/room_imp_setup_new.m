function [H,mic_pos,DOA,layout] = room_imp_setup_new(mic_num,source_num,target_num,Scenario,reverbTime,angle,angle_start,...
    R_ratio,tR_ratio,type,customize_room,angle1_start,angle2_start,angle1_interval,angle2_interval,move_sound,anglenum,varargin)
%% �°淿���������
% ���汾ȥ���˾ɰ汾��Դλ�����ƣ�ӵ���˸���������ѡ��
% ���ڵ�Դ������˷���������ΪԲ���γ�Բ�ֲܷ���Բ�ܰ뾶��������˷��������ĵ������������ǽ��
% ����Դ�ֲ�����ѡ������ֲ�������Դ�н�Ϊangle�ı���������˳��ֲ�
% ������ѡ���Ƿ���Ʒ��䲼��ͼ

% [Input]
% angΪ��Դ�ĳ��� ��������һ��'omnidirectional'ָ������Դָ���ԣ�Ŀǰ�趨��ȫ�򴫲���
% ��Ҫ���ϸ����Դ�������ֵ�����趨Ϊ'cardioid'��Ҳ��������ָ��
% Scenario: ѡ�񼸺ŷ��䣬Ŀǰ�з���1-7,����1-6��֧��4mic���£�����7��֧��8mic
% reverbTime: ����ʱ�䣬��λΪ��
% angle: ����Դ֮�����С�н�, ��λΪ�ȣ�0��Ϊx������
% angle_start: Դ���ĸ��Ƕȿ�ʼ�ֲ�����λΪ�ȣ�0��Ϊx������
% R_ratio: �뾶����������Դ����ļ��
% type:1����������ֲ���0����angle_startΪ���˳��ֲ�

% [Output]
% H:�����ŵ�, size: filterLength x mic x src;
% mic_pos:��˷���,��һ���ο�����Ϊ0, size: 1 x mic
% angle_src:����Դ�ĵ����,Ϊ������, size: 1 x src

%% Reading room config
% ֻ��Ҫ�޸�һ�¶�ȡ��ʽ���Ϳ���ֱ�Ӷ�ȡ�����mic��Ŀ��txt������ѡ��������Ҫ��mic��Ŀ
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
    mic_center=eval(['[' value{strcmp(param,'sc')} ']'])'; % txt�ļ�������˷�������������ʱ��ֱ�Ӷ�ȡ
end
%% ��ȡtxt�ļ�
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
%% ͨ����˷������������� ���� ���λ����ά����
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
%% ��˷����Դ�ֲ��Ƕȼ���
mic_pos = zeros(1,mic_num); % ��˷���
ref_mic_xyz = sensor_xyz(:,1); % �ο���˷磨1��������
for sensor_No = 2:mic_num
    mic_pos(sensor_No) = norm(sensor_xyz(:,sensor_No)-ref_mic_xyz);
end
% ��˷���������������㣬�˴�ǰ������˷�Ϊ����
if isempty(mic_center)
    mic_center = ref_mic_xyz + 1/2 * (sensor_xyz(:,mic_num)-ref_mic_xyz);
end
% ��˷��������ĵ�����ǽ�ľ��� [�� �� �� ��]
x_center = mic_center(1); y_center = mic_center(2); z_center = mic_center(3);
c2w_distance = [x_center,room_size(1)-x_center,y_center,room_size(2)-y_center];

% ��ǽ�����С����
[min_dist,ind] = sort(c2w_distance); 

% ��С��������С,����õڶ�С�ľ��룬���Ұڷŷ�Χ�޶��ڰ�Բ
if min_dist(1) < 0.5 % Դ����˷���С���벻��С��50cm
    start = [-90 90 0 -180]; % c2w_distance��ÿ��ǽ��Ӧ�ĽǶ����
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
%% RIR���� % ���ݽǶȣ���������Ͱ뾶�������Դ�ڷŵ������
    for flag = 1 : anglenum
    angle_src = angle_src_all(flag,:);    
    src_loc = cal_loc(R_ratio,tR_ratio,target_num,angle_src,max_R,mic_center);

    %����ÿ��Դ��������˷�֮��ľ�ȷDOA
    DOA = cal_DOA(angle_src,sensor_xyz); % mic x src
    % % ����ֵDOA
    % DOA = angle_src;

    % ��ȡ�����ŵ���H_refֻ��Ϊ�˵õ�RIR���ȣ�
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
    % ��angle_start+0:angle_max�����еļн�Ϊangle�ĵ㶼�ҳ���
    angle_all = angle_start + (0:angle:angle_max); 
    angle_all(angle_all>360) = angle_all(angle_all>360)-360;
    angle_all(angle_all>180) = -360+angle_all(angle_all>180);
    angle_all = angle_all * pi / 180;
    if type % ����ֲ���������Դ�н�Ϊangle��������
        angle_src = angle_all(randperm(length(angle_all),source_num));
    else % ��angle_start��ʼ��˳��ֲ�
        angle_src = angle_all(1:source_num);
    end
%% RIR����    % ���ݽǶȣ���������Ͱ뾶�������Դ�ڷŵ������
    src_loc = cal_loc(R_ratio,tR_ratio,target_num,angle_src,max_R,mic_center);

    %����ÿ��Դ��������˷�֮��ľ�ȷDOA
    DOA = cal_DOA(angle_src,sensor_xyz); % mic x src
    % % ����ֵDOA
    % DOA = angle_src;

    % ��ȡ�����ŵ���H_refֻ��Ϊ�˵õ�RIR���ȣ�
    H_ref = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,1),[0,0,0],'omnidirectional',room_size,sensor_xyz);
    length_H = size(H_ref,1);
    H = zeros(length_H,mic_num,source_num);
    H(:,:,1) = H_ref;
    for k = 2:source_num
        H(:,:,k) = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,k),[0,0,0],'omnidirectional',room_size,sensor_xyz);
    end    
end
   
% ������䲼�ֲ��������ڻ�ͼ
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
%% ������㣬�Ƕ���X������Ϊ0�������angle��ΧΪ[-pi,pi]
    M = length(angle);
    loc = zeros(3,M); % 3 * src
    x_center = mic_center(1); y_center = mic_center(2); z_center = mic_center(3); 
    R = R_ratio * max_R;
    R_all = repmat(R,1,M);
    R_all(1:target_num) = R_all(1:target_num) * tR_ratio;
    for m = 1:M
        loc(1,m) = x_center + R_all(m) * cos(angle(m)); % x
        loc(2,m) = y_center + R_all(m) * sin(angle(m)); % y
        loc(3,m) = z_center; % z��Ĭ������ͬ��
    end
end

function DOA = cal_DOA(angle_src,sensor_xyz)
%% ��ȷDOA���㣬size = mic x src, ÿһ�ж�Ӧ��ͬһ��src������mic��DOA
    n_src = size(angle_src,2); n_mic = size(sensor_xyz,2);
    ref_ang = atan((sensor_xyz(2,2)-sensor_xyz(2,1)) / (sensor_xyz(1,2)-sensor_xyz(1,1)));    
    DOA = zeros(1,n_src);
    for k = 1:n_src
        DOA(:,k) = angle_src(k)-ref_ang;
    end
end


