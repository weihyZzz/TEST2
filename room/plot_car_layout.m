function plot_car_layout(layout)
%% 画车内布局图函数
src_loc = layout.src_loc; % 3 x src
sensor_xyz = layout.sensor_xyz; % 3 x mic
room_size = layout.room_size; % 1 x 3
reverbTime = layout.reverbTime;

figure; hold on;
n_src = size(src_loc,2);
n_mic = size(sensor_xyz,2);

X_src = src_loc(1,:); Y_src = src_loc(2,:);
X_mic = sensor_xyz(1,:); Y_mic = sensor_xyz(2,:);
% 绘制源坐标点
str_src = [repmat('src',n_src,1) num2str([1:n_src].')];
scatter(X_src,Y_src,25,'b','filled'); text(X_src,Y_src-0.05,str_src,'FontSize',12);
% 绘制麦克风阵列点
mic_src = [repmat('mic',n_mic,1) num2str([1:n_mic].')];
scatter(X_mic,Y_mic,30,'k*'); text(X_mic,Y_mic-0.05,mic_src,'FontSize',12);
% 绘制房间边缘
room_edge = [ 0 0 room_size(1) room_size(1) 0; 0 room_size(2) room_size(2) 0 0];
re = plot(room_edge(1,:),room_edge(2,:),'k');
% 限定房间大小，关闭坐标轴，显示房间长宽和半径
axis([0 room_size(1) 0 room_size(2)]); axis equal; axis off;
%     xlabel(num2str(room_size(1))); ylabel(num2str(room_size(2)));
title(['Carsize = (' num2str(room_size(1)) '*' num2str(room_size(2))...
       ') m, Reverb Time = ', num2str(reverbTime*1000), ' ms']);
% 字体和线宽设置
h = gcf;
myboldify(h);
hold off;
end