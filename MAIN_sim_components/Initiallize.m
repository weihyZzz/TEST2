%%% Parameters initialization %%%
global DOA_tik_ratio; global DOA_Null_ratio; global Ratio_Rxx;  global win_exp_ratio; global PowerRatio;global diagonal_method;global epsilon_ratio1; 
global frameNum; global frameStart; global epsilon_start_ratio;global seed; global epsilon; epsilon = 1e-32; global epsilon_ratio; global SubBlockSize; global SB_ov_Size;
global OrderEst; global OutIter_Num; global order_gamma;  global OrderEstUppThr; global OrderEstLowThr;
global GammaRatioSet; global GammaRatio; global GammaRatioThr; global MNMF_p_norm; global MinDisWin;
SIR_case = []; SDR_case = []; SAR_case = []; SNR_case = []; SIR = []; SDR = []; SAR = []; SIR_time_all = []; SIR_time_all = [];SDR_time_all = [];