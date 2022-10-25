function plotSDR3(SDR_total,SIR_total,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
% errorbar plot
Source_sort=1;%自动排序的基准source
SIR_std = []; SDR_std = [];
SIR_avg = mean(SIR_total,2); SIR_avg = squeeze(SIR_avg);
SDR_avg = mean(SDR_total,2); SDR_avg = squeeze(SDR_avg);
SIR_std = std(SIR_total,0,2); SIR_std = squeeze(SIR_std);
SDR_std = std(SDR_total,0,2); SDR_std = squeeze(SDR_std);
n_src = size(SDR_total,1);%source数
if n_src == 1  
    SDR_avg = SDR_avg.'; SDR_std = SDR_std.';
    SIR_avg = SIR_avg.'; SIR_std = SIR_std.';
end
color=['g','r','b','c','m','y'];
Legend_str = [repmat('source',n_src,1) num2str((1:n_src).')];%自动设定legend
for i = 1:n_src
    Legend_cell{i} = Legend_str(i,:);
end
NumSubCase = case_num/sub_case_num/packFigNum;
for i=1:floor(NumSubCase/plotRatio)
    figure;
    SDR_Suball=SDR_avg(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SDR_std_Suball=SDR_std(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SIR_Suball=SIR_avg(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SIR_std_Suball=SIR_std(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
for k=1:packFigNum
    SDR_Sub=SDR_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    SDR_std_Sub=SDR_std_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    SIR_Sub=SIR_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    SIR_std_Sub=SIR_std_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    
% 绘制SDR  
subplot(2*packFigNum,1,(2*k)-1);
if sub_case_num*packFigNum<SortedPlotThr
hold on;
x_case = repmat([1:sub_case_num],n_src,1);
X_label_SDR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
for cn = 1:sub_case_num
    X_label_SDR_cell{cn} = X_label_SDR(cn,:);
end
for j = 1:n_src
    errorbar(x_case(j,:)+(2*j-1-n_src)*0.145,SDR_Sub(j,:),SDR_std_Sub(j,:),'s','color',color(j));   
end
set(gca,'XTick',1:sub_case_num,'XTickLabel',X_label_SDR_cell);
else%排序画图
SDR_tmp=SDR_Sub(Source_sort,:);%以哪一个源为排列基准
plot_num=min(sub_case_num,SortedPlotNum);
SDR_sorted=zeros(n_src,plot_num);%行数是source大小
SDR_std_sorted=zeros(n_src,plot_num);
[~,SDR_tmp_sorted_index]=sort(SDR_tmp,'descend');%降序排列
X_label_SDR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
for index=1:plot_num
    SDR_sorted(:,index)=SDR_Sub(:,SDR_tmp_sorted_index(index));
    SDR_std_sorted(:,index)=SDR_std_Sub(:,SDR_tmp_sorted_index(index));
    X_label_SDR_cell{index} = X_label_SDR(SDR_tmp_sorted_index(index),:);
end
hold on;
x_case = repmat([1:plot_num],n_src,1);
for j = 1:n_src
    errorbar(x_case(j,:)+(2*j-1-n_src)*0.145,SDR_sorted(j,:),SDR_std_sorted(j,:),'s','color',color(j));  
end
set(gca,'XTick',1:plot_num,'XTickLabel',X_label_SDR_cell);
end
legend(Legend_cell);
ylabel('SDR(dB)');
set(gca,'Fontname','Monospaced');
XLabelFontSize=2;
YLabelFontSize =2;
ZLabelFontSize=2;
FontSize =2;
LegendFontSize =10;
TitleFontSize =10;
h  =gcf;
myboldify(h);

% 绘制SIR
subplot(2*packFigNum,1,2*k)
if sub_case_num*packFigNum<SortedPlotThr
hold on;
x_case = repmat([1:sub_case_num],n_src,1);
X_label_SIR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
for cn = 1:sub_case_num
    X_label_SIR_cell{cn} = X_label_SIR(cn,:);
end
for j = 1:n_src
    errorbar(x_case(j,:)+(2*j-1-n_src)*0.145,SIR_Sub(j,:),SIR_std_Sub(j,:),'s','color',color(j));   
end
set(gca,'XTick',1:sub_case_num,'XTickLabel',X_label_SIR_cell);
else%排序画图
SIR_tmp=SIR_Sub(Source_sort,:);%以哪一个源为排列基准
plot_num=min(sub_case_num,SortedPlotNum);
SIR_sorted=zeros(n_src,plot_num);%行数是source大小
SIR_std_sorted=zeros(n_src,plot_num);
[~,SIR_tmp_sorted_index]=sort(SIR_tmp,'descend');%降序排列
X_label_SIR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
for index=1:plot_num
    SIR_sorted(:,index)=SIR_Sub(:,SIR_tmp_sorted_index(index));
    SIR_std_sorted(:,index)=SIR_std_Sub(:,SIR_tmp_sorted_index(index));
    X_label_SIR_cell{index} = X_label_SIR(SIR_tmp_sorted_index(index),:);
end
hold on;
x_case = repmat([1:plot_num],n_src,1);
for j = 1:n_src
    errorbar(x_case(j,:)+(2*j-1-n_src)*0.145,SIR_sorted(j,:),SIR_std_sorted(j,:),'s','color',color(j));  
end
set(gca,'XTick',1:plot_num,'XTickLabel',X_label_SIR_cell);
end
legend(Legend_cell);
ylabel('SIR(dB)');
set(gca,'Fontname','Monospaced');
XLabelFontSize=2;
YLabelFontSize =10;
ZLabelFontSize=10;
FontSize =10;
LegendFontSize =10;
TitleFontSize =10;
h  =gcf;
myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

end
end