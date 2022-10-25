function plotSDR(case_num,varargin)
if nargin == 2
    SDR_total = varargin{1}; n_src = size(SDR_total,1);
    SDR_std = [];
    SDR_avg = mean(SDR_total,2); SDR_avg = squeeze(SDR_avg);
    SDR_std = std(SDR_total,0,2); SDR_std = squeeze(SDR_std);
    if n_src == 1
        SDR_avg = SDR_avg.'; SDR_std = SDR_std.';
    end
end
if nargin > 2 
    SIR_total =  varargin{2};
    SIR_std = [];
    SIR_avg = mean(SIR_total,2); SIR_avg = squeeze(SIR_avg);
    SIR_std = std(SIR_total,0,2); SIR_std = squeeze(SIR_std);
end

% 自动设定XLabel和Legend
x_case = repmat([1:case_num],n_src,1);
X_label = [repmat('case',case_num,1) num2str((1:case_num).')];
for cn = 1:case_num
    X_label_cell{cn} = X_label(cn,:);
end
Legend_str = [repmat('source',n_src,1) num2str((1:n_src).')];
for k = 1:n_src
    Legend_cell{k} = Legend_str(k,:);
end

% 绘制SDR
figure
b = bar(SDR_avg');
hold on;
ch = get(b,'children');
set(b,'barwidth',1);
b(1).FaceColor='flat';
b(1).CData=repmat([0 0.8 0.8],case_num,1);
for k = 1:n_src
    errorbar(x_case(k,:)+(2*k-1-n_src)*0.145, SDR_avg(k,:), SDR_std(k,:),'LineStyle','none','LineWidth',1,'color','k');  
end
set(gca,'XTickLabel',X_label_cell);
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

if nargin > 2
    % 绘制SIR
    figure
    b = bar(SIR_avg');
    hold on;
    ch = get(b,'children');
    set(b,'barwidth',1);
    b(1).FaceColor='flat';
    b(1).CData=repmat([0 0.8 0.8],case_num,1);
    for k = 1:n_src
        errorbar(x_case(k,:)+(2*k-1-n_src)*0.145, SIR_avg(k,:), SIR_std(k,:),'LineStyle','none','LineWidth',1,'color','k');
    end
    legend(Legend_cell);
    ylabel('SIR(dB)');
    % title('washer+female2, win=hann, winsize=4096, iter=10');
%     set(gca,'Fontname','Monospaced');
    h  =gcf;
    myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
end
end

% % 绘制SAR
% figure(3)
% b = bar(SAR_avg');
% hold on;
% ch = get(b,'children');
% set(b,'barwidth',1);
% b(1).FaceColor='flat';
% b(1).CData=repmat([0 0.8 0.8],case_num,1);
% errorbar(x_case(1,:)-0.145,SAR_avg(1,:),SAR_std(1,:),'LineStyle','none','LineWidth',1,'color','k');  
% errorbar(x_case(2,:)+0.145,SAR_avg(2,:),SAR_std(2,:),'LineStyle','none','LineWidth',1,'color','k');  
% set(gca,'XTickLabel',{'700','800','900','1000','1100','1200','1300','1400','1500','1600','1700','1800','1900','2000'});
% set(gca,'XTickLabel',{'无噪，全带宽2048点','无噪，全带宽1024点','无噪，子带全选','无噪，子带选择','有噪，全带宽2048点','有噪，全带宽1024点','有噪，子带全选','有噪，子带选择'});
% 
% legend('洗衣机声','女声');
% ylabel('SAR(dB)');
% % title('washer+female2, win=hann, winsize=4096, iter=10');
% set(gca,'Fontname','Monospaced');
% XLabelFontSize=2;
% YLabelFontSize =10;
% ZLabelFontSize=10;
% FontSize =10;
% LegendFontSize =10;
% TitleFontSize =10;
% h  =gcf;
% myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
% 
% end