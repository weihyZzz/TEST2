function [xR,s,Fss1,mic_pos,theta] = generate_sim_mix(src_comb, sim_mic, room_type, room_imp_on, DebugRato, muteOn, normal_type, deavg, SNR)
%% 生成房间仿真信号
% src_comb代表不同的信号源组合，s1代表target，s2代表interference
% Source combination: 1—female1+female2 ; 2—female1+male ; 3—female1+washer
%                     4—male+washer ; 5—wake up words+female1 ;
%                     6—wake up words+male ; 7—wake up words+washer
%                     8—wake up words+GWN; 9—wake up words+music
% sim_mic:仿真麦克风数目
% room_type:房间类型，在room_imp_setup函数中定义
% room_imp_on:是否使用房间信道仿真，该值为0则直接相加进行混合
% DebugRato:调试用，决定使用多长的信号进行仿真
% muteOn:调试用，把s2从muteStart开始的muteLen长度的信号进行静音
% normaltype: 混合信号归一化类型，0—不归一化，1—功率归一化，2—幅值归一化，3—SNR混合（附带归一化）
% deavg:混合信号去均值，仿真和实录均适用，1开启0关闭
% SNR:目标信号和噪声（或者干扰信号）的混合信噪比，若mix_SNR=0，则为1:1混合，仅在normal_case=3时使用

if src_comb == 8 % 高斯白噪声仿真
    num_points = 160000; % 160000的点数，在采样率为16000相当于10s的语音数据，与其它仿真对齐
    Data = generate_random_data(num_points);
    y1 = audioread('data/xiaoaitongxue_10s.wav');
    y2 = normrnd(-7.269e-06 ,1.449e-02 ,[num_points,1]); %高斯分布
    s1 = y1; s2 = y2;
    Fss1 = 16000; Fss2 = Fss1; room_imp_on = 1;
else
    % 静音部分源信号
    [sf1,sf2] = source_select(src_comb);
    [s1, Fss1] = audioread(sf1);  [s2, Fss2] = audioread(sf2);
    num_points = 160000; % 原始源信号限定在10s
    s1 = s1(1:num_points,1); s2 = s2(1:num_points,1); % 源信号是多路的话取其中一路即可
    if muteOn
        muteLen = Fss2 * 1-1; muteStart = 1;
        s2(find(abs(s2(muteStart:muteStart+muteLen)) > 1e-4) + muteStart - 1) = 0;
    end
end
%% 调整仿真源信号长度和重采样
dataLen = min(length(s1), length(s2));  dataLenRe = dataLen * DebugRato;
s1 = s1(1:dataLen); s2 = s2(1:dataLen);
if dataLenRe <= dataLen
    s1 =s1(1:dataLenRe);  s2 =s2(1:dataLenRe); % 此处建议s2比s1长，要不然会报错
else
    copynum = floor(dataLenRe / dataLen);
    cut_length = mod(dataLenRe, dataLen);
    s1 = [repmat(s1,copynum,1); s1(1:cut_length)];
    s2 = [repmat(s2,copynum,1); s2(1:cut_length)];
end
if Fss2 ~= Fss1
    s2 = resample(s2, Fss1, Fss2); % 采样率不同需要resample，建议Fss1 = 16000
end
s = [s1'; s2']; 
%% 过信道和归一化
if room_imp_on
    [H1,H2,mic_pos,theta] = room_imp_setup(sim_mic,room_type);
    % Scenario : 1—the oldest one with angle ; 2—the oldest one
    %            3—mic>num Scenario; 4—Block-online Scenario 1
    %            5—Block-online Scenario 2; 6—online Scenario(200ms)
    x1R=fftfilt(H1,s1(:,1));     x2R=fftfilt(H2,s2(:,1));
    switch normal_type
        case 0 % no normalization
            xR = (x1R+x2R)';
        case 1 % power normalization
            E_1=sum(x1R.*conj(x1R)); x1R = x1R./sqrt(E_1); 
            E_2=sum(x2R.*conj(x2R)); x2R = x2R./sqrt(E_2);
            xR = (x1R+x2R)';
        case 2 % amplitude normalization
            xR = (x1R+x2R)';
            normCoef = max(max(abs(xR)));
            xR = xR ./ normCoef;
        case 3 % SNR mix (with normalization)
            [xR,~,~,~] = SNRmix(x1R,x2R,SNR);
            xR = xR';
    end
    if deavg 
        xR = (xR.'- mean(xR.')).';
    end        
else
    xR = [(s1+s2)'; (s1+s2)'];
end

end

%%  getComplexGGD得到复高斯分布
function [Z_gen] = getComplexGGD(N,C,c)
Z = zeros(N,1);
cc = c*2;
for i = 1:N %cc = 2 is Gaussian
    Z(i) = gamrnd(2/cc,1)^(1/cc) * exp(j*2*pi*rand);
end
cNorm = (gamma(2/c)/(gamma(1/c))); 
w =Z*(1/sqrt(cNorm)); %Normalize the complex variance||归一化
waug = [w conj(w)]'; 
Z_gen = C^(1/2)*waug;  
end