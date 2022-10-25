function plot_room_layout(layout)
%% �����䲼��ͼ����
R = layout.R; 
tR = layout.tR;
mic_center = layout.mic_center; % 1 x 3
angle_src = layout.angle_src; % 1 x src
src_loc = layout.src_loc; % 3 x src
sensor_xyz = layout.sensor_xyz; % 3 x mic
room_size = layout.room_size; % 1 x 3
reverbTime = layout.reverbTime;

figure; hold on;
n_src = size(src_loc,2);
angle_src = angle_src * 180/pi;
X_src = src_loc(1,:); Y_src = src_loc(2,:);
X_c = mic_center(1); Y_c = mic_center(2);
X_mic = sensor_xyz(1,:); Y_mic = sensor_xyz(2,:);
% ����Դ�����
str_src = [repmat('src',n_src,1) num2str([1:n_src].') repmat(',',n_src,1) num2str(angle_src')];
scatter(X_src,Y_src,25,'b','filled'); text(X_src,Y_src+0.2,str_src,'FontSize',12);
% ������˷����е�
scatter(X_mic,Y_mic,30,'k*');
scatter(X_c,Y_c,10,'r.'); text(X_c,Y_c+0.2,'mic','FontSize',12);
% ����Դ���ڵ�Բ��
theta=0:2*pi/3600:2*pi;
CircleX=X_c + R*cos(theta);
CircleY=Y_c + R*sin(theta);
sc = plot(CircleX,CircleY,':m','Linewidth',1);
if tR ~= R % ��Ŀ��ԴԲ��Ҳ���Ƴ���
    CircleXt=X_c + tR*cos(theta);
    CircleYt=Y_c + tR*sin(theta);
    sc2 = plot(CircleXt,CircleYt,':g','Linewidth',1);
end
% ���Ʒ����Ե
room_edge = [ 0 0 room_size(1) room_size(1) 0; 0 room_size(2) room_size(2) 0 0];
re = plot(room_edge(1,:),room_edge(2,:),'k');
% �޶������С���ر������ᣬ��ʾ���䳤��Ͱ뾶
axis([0 room_size(1) 0 room_size(2)]); axis equal; axis off;
%     xlabel(num2str(room_size(1))); ylabel(num2str(room_size(2)));
title(['Roomsize = (' num2str(room_size(1)) '*' num2str(room_size(2))...
       ') m, Reverb Time = ', num2str(reverbTime*1000), ' ms']);
if tR ~= R
    legend([sc,sc2],{['R=' num2str(R)],['tR=' num2str(tR)]});
else
    legend(sc,['R=' num2str(R)]);
end
% ������߿�����
h = gcf;
myboldify(h);
hold off;
end