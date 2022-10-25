function [SIR_time] = cal_SIR_time(x,s,s_est,tap_Length,mode)
% mode1: 按照0-1,0-2,0-3...的形式计算SIR_time
% mode2: 按照0-1,1-2,2-3...的形式计算SIR_time
L = min(size(s,2), size(s_est,2));
SIR_time = []; 
timeblock_Num = ceil(L / tap_Length);
mic_num = size(x,1);
source_num = size(s_est,1);
source_range = 1:source_num;
for N_tb = 1:timeblock_Num
    if mode == 1
        if N_tb * tap_Length >= L
            tapblock = 1 : L;
        else
            tapblock = 1 : N_tb * tap_Length;
        end
    else
        if N_tb * tap_Length >= L
            tapblock = (N_tb-1)*tap_Length+1 : L;
        else
            tapblock = (N_tb-1)*tap_Length+1 : N_tb*tap_Length;
        end
    end
    if mic_num == source_num
        [~,SIR_in,~,~] = bss_eval_sources(x(:,tapblock), s(:,tapblock));
    else
        SIR_in = 0;
    end
    [~,SIR_out,~,~] = bss_eval_sources(s_est(:,tapblock), s(source_range,tapblock));
    SIR_imp = SIR_out-SIR_in;
    SIR_time= [SIR_time SIR_imp];
end