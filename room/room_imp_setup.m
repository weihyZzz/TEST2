function [H1,H2,mic_pos,theta] = room_imp_setup(mic_num,Scenario)

% mic_pos是麦克风位置（实际上是间距），这个在房间信道的txt文件里面可以得到，这个是已知的
% theta是到达角，通过声源位置和麦克风位置得到（声源位置是loc_s1和loc_s2)。
% 那个ang是声源的朝向 函数最后的一项'omnidirectional'指的是声源指向性，目前设定成全向传播。
%需要更严格的声源方向这个值可以设定为'cardioid'，也就是心形指向。

switch Scenario
    case 1 % the oldest one with angle
        angle = pi/2;
        rx1 = 2 - 2 * sin(angle/2); rx2 = 2 + 2 * sin(angle/2);
        ry1 = 0.55 + 2 * cos(angle/2); ry2 = ry1;
        loc_s1 = [rx1;ry1;1.4]; ang_s1 = 180 - angle/2 * 180/pi;
        loc_s2 = [rx2;ry2;1.4]; ang_s2 = 180 + angle/2 * 180/pi;
        config_name = ['room_S1_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,loc_s1,[ang_s1,0,0],'omnidirectional');
        H2=roomsimove_single(config_name,loc_s2,[ang_s2,0,0],'omnidirectional');
        if mic_num == 2
            mic_pos = [0,0.0566];
        elseif mic_num == 4
            mic_pos = [0,0.0566,0.0566*2,0.0566*3];
        end
        %theta2 = [atan((sqrt(2)-0.0283)/sqrt(2)),atan((sqrt(2)+0.0283)/sqrt(2))];
        theta = [-pi/4;pi/4]; % 
    case 2 % the oldest one 
        config_name = ['room_S1_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,[0.7144; 1.5321; 1.4],[140; 0; 0],'omnidirectional');
        H2=roomsimove_single(config_name,[3.2856; 1.5321; 1.4],[220; 0; 0],'omnidirectional');
        mic_pos = [0,0.0566];
        theta = [320*pi/180,40*pi/180];
    case 3 % mic>num Scenario
        angle = 2*pi/3;
        rx1 = 4.1 - 2 * cos(angle/2); rx2 = rx1;
        ry1 = 3.76 + 2 * sin(angle/2); ry2 = 3.76 - 2 * sin(angle/2);
        loc_s1 = [rx1;ry1;1.2]; ang_s1 = 90 + angle/2 * 180/pi;
        loc_s2 = [rx2;ry2;1.2]; ang_s2 = 90 - angle/2 * 180/pi;
        config_name = ['room_S2_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,loc_s1,[ang_s1,0,0],'omnidirectional');
        H2=roomsimove_single(config_name,loc_s2,[ang_s2,0,0],'omnidirectional');
    case 4 % Block-Online Scenario 1
        config_name = ['room_S3_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,[3.5; 1.5; 1.2],[0; 0; 0],'omnidirectional');
        H2=roomsimove_single(config_name,[2; 1.5; 1.2],[0; 0; 0],'omnidirectional');
        mic_pos = [0,0.1];
        theta = [pi-atan(1/1.05);atan(1/0.45)];
    case 5 % Block-Online Scenario 2
        config_name = ['room_S4_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,[3; 3; 1.2],[0; 0; 0],'omnidirectional');
        H2=roomsimove_single(config_name,[3.5; 3; 1.2],[0; 0; 0],'omnidirectional');
    case 6 % online Scenario 
        config_name = ['room_S5_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,[6.9; 1.6; 1.2],[0; 0; 0],'omnidirectional');
        H2=roomsimove_single(config_name,[6.9670; 1.35; 1.2],[0; 0; 0],'omnidirectional');
    case 7 % huge room
        config_name = ['room_S6_',num2str(mic_num),'.txt'];
        H1=roomsimove_single(config_name,[6.9; 1.6; 1.2],[0; 0; 0],'omnidirectional');
        H2=roomsimove_single(config_name,[6.9670; 1.35; 1.2],[0; 0; 0],'omnidirectional');
end


