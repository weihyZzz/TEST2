function subplotSIR_time(SIR,T,sub_case_num)

% »æÖÆ online SIR
NumofSource =2;
case_num = size(SIR,1) / NumofSource;
SIR_zero = zeros(case_num*2,1);
SIR = [SIR_zero SIR];
NumSubCase = case_num/sub_case_num;
for i=1:NumSubCase

SIR_Sub=SIR((i-1)*NumofSource*sub_case_num+1:i*NumofSource*sub_case_num,:);  
LineStyle = {'-ro','-bv', '-ms' ,'-gd' ,'-kh' ,'--bo', '--gv', '--rd' ,'--c<' ,'--ms' ,'--kh','-bo','-gv', '-rs' ,'-bd' ,'-yh' ,'--yo', '--gv', '--rd' ,'--c<' ,'--ms' ,'--kh'};
figure;
subplot(2,1,1)

for cn = 1:sub_case_num
    plot(T,SIR_Sub(2*cn-1,:),LineStyle{cn},'DisplayName',['case',num2str(cn)]);
    hold on
end
xlabel('Time(s)'); ylabel('SIR impovement(dB)')
title('Source1 SIR improvement')
legend;
subplot(2,1,2)
for cn = 1:sub_case_num
    plot(T,SIR_Sub(2*cn,:),LineStyle{cn},'DisplayName',['case',num2str(cn)]);
    hold on
end
xlabel('Time(s)'); ylabel('SIR impovement(dB)')
title('Source2 SIR improvement')
legend;
h  =gcf;
myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
end