function plotSDR4(SDR_total,SIR_total,case_num,sub_case_num,packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
% boxplot
Source_sort=1;%均值排序的基准source
SIR_avg = mean(SIR_total,2); SIR_avg = squeeze(SIR_avg);
SDR_avg = mean(SDR_total,2); SDR_avg = squeeze(SDR_avg);
n_src = size(SDR_total,1);%source数
if n_src == 1  
    SDR_avg = SDR_avg.';
    SIR_avg = SIR_avg.';
end
color=['g','r','b','c','m','y'];
Legend_str = [repmat('source',n_src,1) num2str((1:n_src).')];%自动设定legend
for i = 1:n_src
    Legend_cell{i} = Legend_str(i,:);
end
NumSubCase = case_num/sub_case_num/packFigNum;
for i=1:floor(NumSubCase/plotRatio)
    figure;
    SDR_Suball=SDR_total(:,:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SDR_avg_Suball=SDR_avg(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SIR_Suball=SIR_total(:,:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SIR_avg_Suball=SIR_avg(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
for k=1:packFigNum
    SDR_Sub=SDR_Suball(:,:,(k-1)*sub_case_num+1:k*sub_case_num);
    SDR_avg_Sub=SDR_avg_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    SIR_Sub=SIR_Suball(:,:,(k-1)*sub_case_num+1:k*sub_case_num);
    SIR_avg_Sub=SIR_avg_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    
% 绘制SDR   
subplot(2*packFigNum,1,(2*k)-1);
if sub_case_num*packFigNum<SortedPlotThr
    hold on;
    x_case = 1:sub_case_num;
    X_label_SDR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
    for cn = 1:sub_case_num
        X_label_SDR_cell{cn} = X_label_SDR(cn,:);
    end
    for j = 1:n_src
        boxplot(squeeze(SDR_Sub(j,:,:)),'widths',0.1,'Colors','k','positions',x_case-0.1*(n_src-j),'symbol','+');
    end
    set(gca,'xtick',x_case-0.05*(n_src-1));
    set(gca,'XTickLabel',X_label_SDR_cell);
    h = findobj(gca,'Tag','Box');
    for si=1:n_src
        for j=1:sub_case_num
            patch(get(h(j+(si-1)*sub_case_num),'XData'),get(h(j+(si-1)*sub_case_num),'YData'),color(si),'FaceAlpha',.5);
        end
    end
    c = get(gca, 'Children');
    legend(c(1:sub_case_num:(n_src-1)*sub_case_num+1),Legend_cell);
else%排序画图
    SDR_tmp=SDR_avg_Sub(Source_sort,:);%以哪一个源为排列基准
    plot_num=min(sub_case_num,SortedPlotNum);
    X_label_SDR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
    [~,SDR_tmp_sorted_index]=sort(SDR_tmp,'descend');%降序排列
    for index=1:plot_num
        SDR_sorted(:,:,index)=SDR_Sub(:,:,SDR_tmp_sorted_index(index));
        X_label_SDR_cell{index} = X_label_SDR(SDR_tmp_sorted_index(index),:);
    end
    x_case = 1:plot_num;
    hold on;
    for j=1:n_src
        boxplot(squeeze(SDR_sorted(j,:,:)),'widths',0.1,'Colors','k','positions',x_case-0.1*(n_src-j),'symbol','+');
    end
    set(gca,'xtick',x_case-0.05*(n_src-1));
    set(gca,'XTickLabel',X_label_SDR_cell);
    h = findobj(gca,'Tag','Box');
    for si=1:n_src
        for j=1:plot_num
            patch(get(h(j+(si-1)*plot_num),'XData'),get(h(j+(si-1)*plot_num),'YData'),color(si),'FaceAlpha',.5);
        end
    end
    c = get(gca, 'Children');
    legend(c(1:plot_num:(n_src-1)*plot_num+1),Legend_cell);
end
ylabel('SDR(dB)');
set(gca,'Fontname','Monospaced');
h=gcf;
myboldify(h);

% 绘制SIR
subplot(2*packFigNum,1,2*k)
if sub_case_num*packFigNum<SortedPlotThr
    hold on;
    x_case = 1:sub_case_num;
    X_label_SIR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
    for cn = 1:sub_case_num
        X_label_SIR_cell{cn} = X_label_SIR(cn,:);
    end
    for j = 1:n_src
        boxplot(squeeze(SIR_Sub(j,:,:)),'widths',0.1,'Colors','k','positions',x_case-0.1*(n_src-j),'symbol','+');
    end
    set(gca,'xtick',x_case-0.05*(n_src-1));
    set(gca,'XTickLabel',X_label_SIR_cell);
    h = findobj(gca,'Tag','Box');
    for si=1:n_src
        for j=1:sub_case_num
            patch(get(h(j+(si-1)*sub_case_num),'XData'),get(h(j+(si-1)*sub_case_num),'YData'),color(si),'FaceAlpha',.5);
        end
    end
    c = get(gca, 'Children');
    legend(c(1:sub_case_num:(n_src-1)*sub_case_num+1),Legend_cell);
else%排序画图
    SIR_tmp=SIR_avg_Sub(Source_sort,:);%以哪一个源为排列基准
    plot_num=min(sub_case_num,SortedPlotNum);
    X_label_SIR = [repmat('case',sub_case_num,1) num2str((1:sub_case_num).')];%自动设定Xlabel
    [~,SIR_tmp_sorted_index]=sort(SIR_tmp,'descend');%降序排列
    for index=1:plot_num
        SIR_sorted(:,:,index)=SIR_Sub(:,:,SIR_tmp_sorted_index(index));
        X_label_SIR_cell{index} = X_label_SIR(SIR_tmp_sorted_index(index),:);
    end
    x_case = 1:plot_num;
    hold on;
    for j=1:n_src
        boxplot(squeeze(SIR_sorted(j,:,:)),'widths',0.1,'Colors','k','positions',x_case-0.1*(n_src-j),'symbol','+');
    end
    set(gca,'xtick',x_case-0.05*(n_src-1));
    set(gca,'XTickLabel',X_label_SIR_cell);
    h = findobj(gca,'Tag','Box');
    for si=1:n_src
        for j=1:plot_num
            patch(get(h(j+(si-1)*plot_num),'XData'),get(h(j+(si-1)*plot_num),'YData'),color(si),'FaceAlpha',.5);
        end
    end
    c = get(gca, 'Children');
    legend(c(1:plot_num:(n_src-1)*plot_num+1),Legend_cell);
end
ylabel('SIR(dB)');
set(gca,'Fontname','Monospaced');
h=gcf;
myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

end
end