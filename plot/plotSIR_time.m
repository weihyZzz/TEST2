function plotSIR_time(SIR,T,sub_case_num, packFigNum,SortedPlotThr,SortedPlotNum,plotRatio)
%sub_case_num      同一大类比较图的数目,
%packFigNum       一次把几种 subcase 都放到一起来plot
%SortedPlotThr;   如果case_num太多显示不了，判决是否显示 SortedPlotNum case，
%如果是1，缺省显示不排序的情况；如果>1, 表明排序显示。
%SortedPlotNum;   如果case_num太多显示不了，显示 SortedPlotNum case；
%plotRatio       一次多少比例的图； default=1； 2 表明画50% 的图；

% 绘制 online SIR
NumofSource =2; case_num = size(SIR,1) / NumofSource;SIR_zero = zeros(case_num*2,1); SIR = [SIR_zero SIR];
NumSubCase = case_num/sub_case_num/packFigNum; NumSubCase = case_num/sub_case_num/packFigNum;
LineStyle ='-bo -go -ro -co -mo -yo -ko -bd -gd -rd -cd -md -yd -kd -bp -gp -rp -cp -mp -yp -kp -bh -gh -rh -ch -mh -yh -kh -b> -g> -r> -c> -m> -y> -k> -bs -gs -rs -cs -ms -ys -ks -bo -go -ro -co -mo -yo -ko -b< -g< -r< -c< -m< -y< -k< -b* -g* -r* -c* -m* -y* -k* --bo --go --ro --co --mo --yo --ko --bd --gd --rd --cd --md --yd --kd --bp --gp --rp --cp --mp --yp --kp --bh --gh --rh --ch --mh --yh --kh --b> --g> --r> --c> --m> --y> --k> --bs --gs --rs --cs --ms --ys --ks --bo --go --ro --co --mo --yo --ko --b< --g< --r< --c< --m< --y< --k< --b* --g* --r* --c* --m* --y* --k*';

for i=1:floor(NumSubCase/plotRatio)
    figure;     SIR_Suball=SIR((i-1)*NumofSource*sub_case_num*packFigNum+1:i*NumofSource*sub_case_num*packFigNum,:);
    for k=1:packFigNum
        SIR_Sub=SIR_Suball((k-1)*NumofSource*sub_case_num+1:k*NumofSource*sub_case_num,:);
        % LineStyle = {'-ro','-bv', '-ms' ,'-gd' ,'-kh' ,'--bo', '--gv', '--rd' ,'--c<' ,'--ms' ,'--kh','-bo','-gv', '-rs' ,'-bd' ,'-yh' ,'--yo', '--gv', '--rd' ,'--c<' ,'--ms' ,'-ro','-bv', '-ms' ,'-gd' ,'-kh' ,'--bo', '--gv', '--rd' ,'--c<' ,'--ms' ,'--kh','-bo','-gv', '-rs' ,'-bd' ,'-yh' ,'--yo', '--gv', '--rd' ,'--c<' ,'--ms''--kh'};
        % LineStyle ={'-bo', '-go', -ro -co -mo -yo -ko -bd -gd -rd -cd -md -yd -kd -bp -gp -rp -cp -mp -yp -kp -bh -gh -rh -ch -mh -yh -kh -b> -g> -r> -c> -m> -y> -k> -bs -gs -rs -cs -ms -ys -ks -bo -go -ro -co -mo -yo -ko';
        subplot(2*packFigNum,1,(2*k)-1);
        if sub_case_num*packFigNum<SortedPlotThr
            for cn = 1:sub_case_num
                plot(T,SIR_Sub(2*cn-1,:),LineStyle((cn-1)*4+1:cn*4-1),'DisplayName',['case',num2str(cn)]);
                hold on
            end
        else
            %   [SIR_Sub_sorted(i,:), SIR_Sub_sorted_index(i,:)] =sort(mean(SIR_Sub(i,3:end)));
            tmp = SIR_Sub(:,3:end);
            tmp_odd= tmp(1:2:end,:);               tmp_even= tmp(2:2:end,:);
            [SIR_odd_sorted, SIR_odd_sorted_index] = sort(mean(tmp_odd,2)); BorderOddIndex = find(SIR_odd_sorted_index==sub_case_num);
            [SIR_even_sorted, SIR_even_sorted_index] = sort(mean(tmp_even,2)); BorderEvenIndex = find(SIR_even_sorted_index==sub_case_num);
            for cn = 1:min(SortedPlotNum,size(SIR_Sub,1))
                plot(T,SIR_Sub(2*SIR_odd_sorted_index(end-cn+1)-1,:),LineStyle((cn-1)*4+1:cn*4-1),'DisplayName',['case',num2str(SIR_odd_sorted_index(end-cn+1))]);
                hold on
            end
            %plot(T,SIR_Sub(2*sub_case_num-1,:),LineStyle((cn)*4+1:(cn+1)*4-1),'DisplayName',['case',num2str(sub_case_num)]);
            plot(T,SIR_Sub(2*SIR_even_sorted_index(1)-1,:),LineStyle((sub_case_num-1)*4+1:sub_case_num*4-1),'DisplayName',['case',num2str(SIR_odd_sorted_index(1))]);
        end
        xlabel('Time(s)'); ylabel('SIR impovement(dB)'); title('Source1 SIR improvement');
        legend;        subplot(2*packFigNum,1,2*k)
        if sub_case_num*packFigNum<SortedPlotThr
            for cn = 1:sub_case_num
                % plot(T,SIR_Sub(2*cn,:),LineStyle{cn},'DisplayName',['case',num2str(cn)]);
                plot(T,SIR_Sub(2*cn,:),LineStyle((cn-1)*4+1:(cn)*4-1),'DisplayName',['case',num2str(cn)]);
                hold on
            end            
        else
            for cn = 1:min(SortedPlotNum,size(SIR_Sub,1))
                plot(T,SIR_Sub(2*SIR_even_sorted_index(end-cn+1),:),LineStyle((cn-1)*4+1:cn*4-1),'DisplayName',['case',num2str(SIR_even_sorted_index(end-cn+1))]);
                hold on;
            end
            %  plot(T,SIR_Sub(2*BorderEvenIndex),:),LineStyle((cn-1)*4+1:cn*4-1),'DisplayName',['case',num2str(BorderEvenIndex)]);
            %plot(T,SIR_Sub(2*SIR_even_sorted_index(1),:),LineStyle((sub_case_num-1)*4+1:sub_case_num*4-1),'DisplayName',['case',num2str(SIR_even_sorted_index(1))]);
            plot(T,SIR_Sub(2*SIR_even_sorted_index(1),:),LineStyle((sub_case_num-1)*4+1:sub_case_num*4-1),'DisplayName',['case',num2str(SIR_even_sorted_index(1))]);
        end
        xlabel('Time(s)'); ylabel('SIR impovement(dB)'); title('Source2 SIR improvement'); legend;
        h  =gcf; myboldify(h); %,MarkerSize,XLabelFontSize,YLabelFontSize,ZLabelFontSize,FontSize,LineWidth,LegendFontSize,TitleFontSize)
    end
end