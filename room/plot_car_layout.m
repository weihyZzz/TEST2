function plot_car_layout(layout)
%% �����ڲ���ͼ����
src_loc = layout.src_loc; % 3 x src
sensor_xyz = layout.sensor_xyz; % 3 x mic
room_size = layout.room_size; % 1 x 3
reverbTime = layout.reverbTime;

figure; hold on;
n_src = size(src_loc,2);
n_mic = size(sensor_xyz,2);

X_src = src_loc(1,:); Y_src = src_loc(2,:);
X_mic = sensor_xyz(1,:); Y_mic = sensor_xyz(2,:);
% ����Դ�����
str_src = [repmat('src',n_src,1) num2str([1:n_src].')];
scatter(X_src,Y_src,25,'b','filled'); text(X_src,Y_src-0.05,str_src,'FontSize',12);
% ������˷����е�
mic_src = [repmat('mic',n_mic,1) num2str([1:n_mic].')];
scatter(X_mic,Y_mic,30,'k*'); text(X_mic,Y_mic-0.05,mic_src,'FontSize',12);
% ���Ʒ����Ե
room_edge = [ 0 0 room_size(1) room_size(1) 0; 0 room_size(2) room_size(2) 0 0];
re = plot(room_edge(1,:),room_edge(2,:),'k');
% �޶������С���ر������ᣬ��ʾ���䳤��Ͱ뾶
axis([0 room_size(1) 0 room_size(2)]); axis equal; axis off;
%     xlabel(num2str(room_size(1))); ylabel(num2str(room_size(2)));
title(['Carsize = (' num2str(room_size(1)) '*' num2str(room_size(2))...
       ') m, Reverb Time = ', num2str(reverbTime*1000), ' ms']);
% ������߿�����
h = gcf;
myboldify(h);
hold off;
end