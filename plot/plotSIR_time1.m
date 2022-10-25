function plotSIR_time1(SIR,T,case_num,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio,varargin)
%sub_case_num      同一大类比较图的数目,
%packFigNum       一次把几种 subcase 都放到一起来plot
%SortedPlotThr;   如果case_num太多显示不了，判决是否显示 SortedPlotNum case，
%如果是1，缺省显示不排序的情况；如果>1, 表明排序显示。
%SortedPlotNum;   如果case_num太多显示不了，显示 SortedPlotNum case；
%plotRatio       一次多少比例的图； default=1； 2 表明画50% 的图；

% 设定 legend
Legend_cell = cell(case_num,1);
if nargin == 8
    Legend_str = [repmat('case',case_num,1) num2str((1:case_num).')];
    for i = 1:case_num
        Legend_cell{i} = Legend_str(i,:);
    end
else
%     Legend_cell{1} = ['case' num2str(1) ' aux ' ];%num2str(param(i)) reverbTime   
%     Legend_cell{2} = ['case' num2str(2) ' em ' ];%num2str(param(i)) reverbTime   
%    
    param=cell2mat(varargin);%{1}
    for i = 1:case_num
 % %       Legend_cell{i} = ['case' num2str(i) 'beta 1,2= ' num2str(param(2*i-1,1)),',',num2str(param(2*i-1,2))];% num2str(i-1) reverbTime   
 %Legend_cell{i} = ['case' num2str(i) ' Lb =' num2str(param(i))];%num2str(param(i)) reverbTime   
 Legend_cell{i} = ['case' num2str(i) ' FFT size =' num2str(param(i))];%num2str(param(i)) reverbTime   
    end
%     n_orders1 = varargin{1};
%     n_orders2 = varargin{2};
%     n_orders1_num = size(n_orders1,2);
%     n_orders2_num = size(n_orders2,2);
%     for i = 1:n_orders1_num
%         for j = 1:n_orders2_num
%             Legend_cell{(i-1)*n_orders2_num+j} = ['case' num2str((i-1)*n_orders2_num+j) '=' num2str(n_orders1(i)) ',' num2str(n_orders2(j))];
%         end
%     end
end

% 绘制 online SIR
NumofSource = size(SIR,1) / case_num;
SIR_zero = zeros(case_num*NumofSource,1); 
SIR = [SIR_zero SIR];
NumSubCase = case_num/sub_case_num/packFigNum;
LineStyle ='-bo -go -ro -co -mo -yo -ko -bd -gd -rd -cd -md -yd -kd -bp -gp -rp -cp -mp -yp -kp -bh -gh -rh -ch -mh -yh -kh -b> -g> -r> -c> -m> -y> -k> -bs -gs -rs -cs -ms -ys -ks -bo -go -ro -co -mo -yo -ko -b< -g< -r< -c< -m< -y< -k< -b* -g* -r* -c* -m* -y* -k* --bo --go --ro --co --mo --yo --ko --bd --gd --rd --cd --md --yd --kd --bp --gp --rp --cp --mp --yp --kp --bh --gh --rh --ch --mh --yh --kh --b> --g> --r> --c> --m> --y> --k> --bs --gs --rs --cs --ms --ys --ks --bo --go --ro --co --mo --yo --ko --b< --g< --r< --c< --m< --y< --k< --b* --g* --r* --c* --m* --y* --k*';
for i=1:floor(NumSubCase/plotRatio)
    figure;     
    SIR_Suball=SIR((i-1)*NumofSource*sub_case_num*packFigNum+1:i*NumofSource*sub_case_num*packFigNum,:);
    for k=1:packFigNum
        SIR_Sub=SIR_Suball((k-1)*NumofSource*sub_case_num+1:k*NumofSource*sub_case_num,:);
        for j=1:NumofSource
        subplot(NumofSource*packFigNum,1,NumofSource*(k-1)+j);
        if sub_case_num*packFigNum<SortedPlotThr
            hold on;
            for cn = 1:sub_case_num
                plot(T,SIR_Sub(NumofSource*(cn-1)+j,:),LineStyle((cn-1)*4+1:cn*4-1));      
            end
            legend(Legend_cell,'location','EastOutside');
        else
            plot_num=min(SortedPlotNum,sub_case_num);
            tmp = SIR_Sub(:,3:end);
            tmp_sort = tmp(j:NumofSource:end,:);
            [~, tmp_sorted_index] = sort(mean(tmp_sort,2),'descend'); 
            Legend_cell_sorted = cell(plot_num,1);
            hold on;
            for cn = 1:plot_num
                plot(T,SIR_Sub(NumofSource*(tmp_sorted_index(cn)-1)+j,:),LineStyle((cn-1)*4+1:cn*4-1));
                Legend_cell_sorted{cn} = Legend_cell{tmp_sorted_index(cn)};
            end
            legend(Legend_cell_sorted,'location','EastOutside');
        end
        xlabel('Time(s)'); 
        ylabel('SDR impovement(dB)'); 
        title_str=['Source' num2str(j) ' SDR improvement'];
        title(title_str);       
        h  =gcf; 
        myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
        end
    end
end