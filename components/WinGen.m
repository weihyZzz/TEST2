function [win_ana, win_syn]= WinGen(FFT_data_Len,tao,taoMeanAdd,taoMean,inc,win_exp_ratio)
%tao_range =[  10^5]/10;%10^5 对应于没有加窗； 比较好的值可以是 30
% tao =[  3*10^2]/10;%10^5 对应于没有加窗； 比较好的值可以是 2
% taoMean is default value 1;
% taoMeanAdd is  为了鲁棒参数；
tao_Msc_RS_2 = round(tao*FFT_data_Len/2);
m = -tao_Msc_RS_2:1:tao_Msc_RS_2;
windowing_test = (taoMeanAdd+taoMean+taoMean*cos(2*pi*m/(tao*FFT_data_Len)));
windowing_t = [windowing_test((tao_Msc_RS_2-FFT_data_Len/2)+1:tao_Msc_RS_2)';windowing_test(tao_Msc_RS_2+2:end-(tao_Msc_RS_2-FFT_data_Len/2))'];
win_ana =windowing_t/mean(windowing_t);
win_ana =win_ana.^win_exp_ratio;
win_ana =win_ana.';
win_syn = generate_win_syn(win_ana, inc);
end
function win_syn = generate_win_syn(win_ana, inc)
win_size = length(win_ana);
win_syn = win_ana;
end


%                 end