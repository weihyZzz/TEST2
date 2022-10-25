function plotSDR_name(case_num,varargin)
Source_sort = 1; % 排序的基准source

SDR_total = varargin{1}; n_src = size(SDR_total,1);
sub_case_num = varargin{2}; packFigNum = varargin{3}; SortedPlotThr = varargin{4};
SortedPlotNum = varargin{5}; plotRatio = varargin{6};
SDR_avg = mean(SDR_total,2); SDR_avg = squeeze(SDR_avg);
SDR_std = std(SDR_total,0,2); SDR_std = squeeze(SDR_std);
if n_src == 1  
    SDR_avg = SDR_avg.'; SDR_std = SDR_std.';
end
plot_num=min(sub_case_num,SortedPlotNum);

% 自动设定Legend
Legend_str = [repmat('source',n_src,1) num2str((1:n_src).')];
for k = 1:n_src
    Legend_cell{k} = Legend_str(k,:);
end
% 自动设定Xlabel
if nargin == 7 
    X_label = [repmat('case',case_num,1) num2str((1:case_num).')];              
    for cn = 1:case_num                   
        X_label_cell{cn} = X_label(cn,:);       
    end
else
    X_label = varargin{7}; 
    for cn = 1:case_num             
        X_label_cell{cn} = cell2mat(X_label(cn));    
    end
end

NumSubCase = case_num/sub_case_num/packFigNum;
for i=1:floor(NumSubCase/plotRatio)
    figure;
    SDR_Suball=SDR_avg(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    SDR_std_Suball=SDR_std(:,(i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
    X_label_Suball=X_label_cell((i-1)*sub_case_num*packFigNum+1:i*sub_case_num*packFigNum);
for k=1:packFigNum
    SDR_Sub=SDR_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    SDR_std_Sub=SDR_std_Suball(:,(k-1)*sub_case_num+1:k*sub_case_num);
    X_label_Sub=X_label_Suball((k-1)*sub_case_num+1:k*sub_case_num);
% 绘制SDR
subplot(packFigNum,1,k);
if sub_case_num*packFigNum<SortedPlotThr % 不排序画图
    b = bar(SDR_Sub(:,1:plot_num)');
    hold on;
    ch = get(b,'children');
    set(b,'barwidth',1);
    b(1).FaceColor='flat';
    b(1).CData=repmat([0 0.8 0.8],plot_num,1);
    x_case = repmat([1:plot_num],n_src,1);
    for j = 1:n_src
        errorbar(x_case(j,:)+(2*j-1-n_src)*0.145, SDR_Sub(j,1:plot_num), SDR_std_Sub(j,1:plot_num),'LineStyle','none','LineWidth',1,'color','k');  
    end
    set(gca,'XTick',1:plot_num,'XTickLabel',X_label_Sub(1:plot_num));    
else                                    % 排序画图
    SDR_sorted=zeros(n_src,plot_num); SDR_std_sorted=zeros(n_src,plot_num);
    SDR_tmp=SDR_Sub(Source_sort,:); % 以哪一个源为排序基准
    [~,SDR_tmp_sorted_index]=sort(SDR_tmp,'descend'); % 降序排列
    for index=1:plot_num
        SDR_sorted(:,index)=SDR_Sub(:,SDR_tmp_sorted_index(index));
        SDR_std_sorted(:,index)=SDR_std_Sub(:,SDR_tmp_sorted_index(index));
        X_label_cell_sorted{index} = X_label_Sub{SDR_tmp_sorted_index(index)};
    end
    b = bar(SDR_sorted');
    hold on;
    ch = get(b,'children');
    set(b,'barwidth',1);
    b(1).FaceColor='flat';
    b(1).CData=repmat([0 0.8 0.8],plot_num,1);
    x_case = repmat([1:plot_num],n_src,1);
    for j = 1:n_src
        errorbar(x_case(j,:)+(2*j-1-n_src)*0.145, SDR_sorted(j,:), SDR_std_sorted(j,:),'LineStyle','none','LineWidth',1,'color','k');  
    end
    set(gca,'XTick',1:plot_num,'XTickLabel',X_label_cell_sorted); 
end
legend(Legend_cell);
ylabel('SDR(dB)');
% title('washer+female2, win=hann, winsize=4096, iter=10');
% set(gca,'Fontname','Monospaced');
XLabelFontSize=2;
YLabelFontSize =2;
ZLabelFontSize=2;
FontSize =2;
LegendFontSize =10;
TitleFontSize =10;
h  =gcf;
myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)

end
end
end