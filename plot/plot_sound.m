function plot_sound(s_est,x,fs,label)
win_type = 'hann';
win_size = 1024;
% win_size = 2048;
inc = win_size / 2;
fft_size = win_size;

[win_ana, win_syn] = generate_win(win_type, win_size, inc);
win_syn = ones(1, win_size);

if size(s_est,1) == 2
    figure
    subplot(2,2,1)
    % grid on
    % plot(s_est(1,:),'DisplayName','estimated source 1');
    % legend;
    % subplot(2,1,2)
    spectrogram(s_est(1,:)',win_ana,inc,fft_size,fs,'yaxis');
    title(strcat('estimated source 1 ',label{1}));
    colormap('hot');
    subplot(2,2,2)
    % grid on
    % plot(s_est(2,:),'DisplayName','estimated source 2');
    % legend;
    % subplot(2,1,2)
    spectrogram(s_est(2,:),win_ana,inc',fft_size,fs,'yaxis');
    title(strcat('estimated source 2 ',label{2}));
    colormap('hot');
    subplot(2,2,3)
    % grid on
    % plot(x(1,:),'DisplayName','mix');
    % legend;
    % subplot(2,1,2)
    spectrogram(x(1,:)',win_ana,inc,fft_size,fs,'yaxis');
    title('mix signal');
    colormap('hot');
elseif size(s_est,1) == 1
    subplot(1,2,1)
    % grid on
    % plot(s_est(1,:),'DisplayName','estimated source 1');
    % legend;
    % subplot(2,1,2)
    spectrogram(s_est(1,:)',win_ana,inc,fft_size,fs,'yaxis');
    title(strcat('estimated source 1 ',label{1}));
    colormap('hot'); 
    subplot(1,2,2)
    % grid on
    % plot(x(1,:),'DisplayName','mix');
    % legend;
    % subplot(2,1,2)
    spectrogram(x(1,:)',win_ana,inc,fft_size,fs,'yaxis');
    title('mix signal');
    colormap('hot');
end
h  =gcf;
myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)