function H = cal_RTF(F,fs,r,theta)
%%% Calculate Relative Transfer Function %%%
% Ref: A.Brendel, "SPATIALLY INFORMED INDEPENDENT VECTOR ANALYSIS"
% EQN (15)

M = length(r);
N_FFT = 2*(F-1); % FFT Points = 2*(spec_coeff_num-1);
I = length(theta); % 若一个源对应多个DOAs，则全都计算 
% H = zeros(M,F,I);
H = zeros(M,F);
vf = [0:F-1]*fs/N_FFT; % Real frequency corresponding to the frequency indices of STFT;
cs = 340; % Speed of sound, currently set as 340m/s;
for m = 1:M
%     H(m,:) = exp(j*2*pi*vf * r(m) * cos(theta)/cs);
    for i = 1:I
        H(m,:,i) = exp(j*2*pi*vf * r(m) * cos(theta(i))/cs);
    end
end



