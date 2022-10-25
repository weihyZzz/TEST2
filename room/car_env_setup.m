function [H,mic_pos,DOA,layout] = car_env_setup(mic_num,source_num,Scenario,reverbTime)
%% ����4mic��������
% ����4mic����˳��
% | ��2��  ��3�� |
% | ��1��  ��4�� |
%  source��˵���ˣ��̶��ڳ�����λ�ϣ����λ��д��������˳���mic��ͬ
%% seat loc setting
% ����λ1��λ��Ϊreference
x_length = 0.6; % ����y��ľ��룬���복ǰ�����ľ���
y_length = 0.4; % ����x��ľ��룬��������ĳ��ž���
%% Reading room config
config_name = ['room_S',num2str(Scenario),'_4.txt'];

fid = fopen(config_name);
[room_config,~] = textscan(fid,'%9s%[^\n]','commentstyle','%');
fclose(fid);

param = room_config{1};
value = room_config{2};

room_size=eval(['[' value{strcmp(param,'room_size')} ']'])';
mic_center=eval(['[' value{strcmp(param,'sc')} ']'])'; % txt�ļ�������˷�������������ʱ��ֱ�Ӷ�ȡ
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
%% ��˷���
mic_pos = zeros(1,mic_num); % ��˷���
ref_mic_xyz = sensor_xyz(:,1); % �ο���˷磨1��������
for sensor_No = 2:mic_num
    mic_pos(sensor_No) = norm(sensor_xyz(:,sensor_No)-ref_mic_xyz);
end
% ��˷����������������
if isempty(mic_center)
    mic_center = 1/2 * (sensor_xyz(:,1) + sensor_xyz(:,3));
end
%% sourceλ�ú�RIR����
seat_loc = zeros(3,4); % ˵���˹̶�����λ��
seat_loc(:,1) = [x_length, y_length, ref_mic_xyz(3)]; 
seat_loc(:,2) = [x_length, room_size(2)-y_length, ref_mic_xyz(3)]; 
seat_loc(:,3) = [room_size(1)-x_length, room_size(2)-y_length, ref_mic_xyz(3)];
seat_loc(:,4) = [room_size(1)-x_length, y_length, ref_mic_xyz(3)];
src_loc = seat_loc(:,1:source_num);
% ����DOA ��
seat_DOA = zeros(4,4); % mic x src
seat_DOA(:,1) = [pi/2, -pi/2, atan((seat_loc(2,1)-sensor_xyz(2,3))/(seat_loc(1,1)-sensor_xyz(1,3))), ...
                atan((seat_loc(2,1)-sensor_xyz(2,4))/(seat_loc(1,1)-sensor_xyz(1,4)))]; % src1 DOA
seat_DOA(:,2) = [pi/2, -pi/2, atan((seat_loc(2,2)-sensor_xyz(2,3))/(seat_loc(1,2)-sensor_xyz(1,3))), ...
                atan((seat_loc(2,2)-sensor_xyz(2,4))/(seat_loc(1,2)-sensor_xyz(1,4)))]; % src2 DOA
seat_DOA(:,3) = [atan((seat_loc(2,3)-sensor_xyz(2,1))/(seat_loc(1,3)-sensor_xyz(1,1))), ...
                atan((seat_loc(2,3)-sensor_xyz(2,2))/(seat_loc(1,3)-sensor_xyz(1,2))), -pi/2, pi/2]; % src3 DOA
seat_DOA(:,4) = [atan((seat_loc(2,4)-sensor_xyz(2,1))/(seat_loc(1,4)-sensor_xyz(1,1))), ...
                atan((seat_loc(2,4)-sensor_xyz(2,2))/(seat_loc(1,4)-sensor_xyz(1,2))), -pi/2, pi/2]; % src4 DOA
DOA = seat_DOA(:,1:source_num);

H_ref = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,1),[0,0,0],'omnidirectional',room_size,sensor_xyz);
length_H = size(H_ref,1);
H = zeros(length_H,mic_num,source_num);
H(:,:,1) = H_ref;
for k = 2:source_num
    H(:,:,k) = roomsimove_single_ch(config_name,channels,reverbTime,src_loc(:,k),[0,0,0],'omnidirectional',room_size,sensor_xyz);
end   

% ������䲼�ֲ��������ڻ�ͼ
layout.src_loc = src_loc;
layout.sensor_xyz = sensor_xyz;
layout.room_size = room_size;
layout.reverbTime = reverbTime;
end