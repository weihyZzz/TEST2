function [xR,s,Fss1,mic_pos,theta] = generate_sim_mix(src_comb, sim_mic, room_type, room_imp_on, DebugRato, muteOn, normal_type, deavg, SNR)
%% ���ɷ�������ź�
% src_comb����ͬ���ź�Դ��ϣ�s1����target��s2����interference
% Source combination: 1��female1+female2 ; 2��female1+male ; 3��female1+washer
%                     4��male+washer ; 5��wake up words+female1 ;
%                     6��wake up words+male ; 7��wake up words+washer
%                     8��wake up words+GWN; 9��wake up words+music
% sim_mic:������˷���Ŀ
% room_type:�������ͣ���room_imp_setup�����ж���
% room_imp_on:�Ƿ�ʹ�÷����ŵ����棬��ֵΪ0��ֱ����ӽ��л��
% DebugRato:�����ã�����ʹ�ö೤���źŽ��з���
% muteOn:�����ã���s2��muteStart��ʼ��muteLen���ȵ��źŽ��о���
% normaltype: ����źŹ�һ�����ͣ�0������һ����1�����ʹ�һ����2����ֵ��һ����3��SNR��ϣ�������һ����
% deavg:����ź�ȥ��ֵ�������ʵ¼�����ã�1����0�ر�
% SNR:Ŀ���źź����������߸����źţ��Ļ������ȣ���mix_SNR=0����Ϊ1:1��ϣ�����normal_case=3ʱʹ��

if src_comb == 8 % ��˹����������
    num_points = 160000; % 160000�ĵ������ڲ�����Ϊ16000�൱��10s���������ݣ��������������
    Data = generate_random_data(num_points);
    y1 = audioread('data/xiaoaitongxue_10s.wav');
    y2 = normrnd(-7.269e-06 ,1.449e-02 ,[num_points,1]); %��˹�ֲ�
    s1 = y1; s2 = y2;
    Fss1 = 16000; Fss2 = Fss1; room_imp_on = 1;
else
    % ��������Դ�ź�
    [sf1,sf2] = source_select(src_comb);
    [s1, Fss1] = audioread(sf1);  [s2, Fss2] = audioread(sf2);
    num_points = 160000; % ԭʼԴ�ź��޶���10s
    s1 = s1(1:num_points,1); s2 = s2(1:num_points,1); % Դ�ź��Ƕ�·�Ļ�ȡ����һ·����
    if muteOn
        muteLen = Fss2 * 1-1; muteStart = 1;
        s2(find(abs(s2(muteStart:muteStart+muteLen)) > 1e-4) + muteStart - 1) = 0;
    end
end
%% ��������Դ�źų��Ⱥ��ز���
dataLen = min(length(s1), length(s2));  dataLenRe = dataLen * DebugRato;
s1 = s1(1:dataLen); s2 = s2(1:dataLen);
if dataLenRe <= dataLen
    s1 =s1(1:dataLenRe);  s2 =s2(1:dataLenRe); % �˴�����s2��s1����Ҫ��Ȼ�ᱨ��
else
    copynum = floor(dataLenRe / dataLen);
    cut_length = mod(dataLenRe, dataLen);
    s1 = [repmat(s1,copynum,1); s1(1:cut_length)];
    s2 = [repmat(s2,copynum,1); s2(1:cut_length)];
end
if Fss2 ~= Fss1
    s2 = resample(s2, Fss1, Fss2); % �����ʲ�ͬ��Ҫresample������Fss1 = 16000
end
s = [s1'; s2']; 
%% ���ŵ��͹�һ��
if room_imp_on
    [H1,H2,mic_pos,theta] = room_imp_setup(sim_mic,room_type);
    % Scenario : 1��the oldest one with angle ; 2��the oldest one
    %            3��mic>num Scenario; 4��Block-online Scenario 1
    %            5��Block-online Scenario 2; 6��online Scenario(200ms)
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

%%  getComplexGGD�õ�����˹�ֲ�
function [Z_gen] = getComplexGGD(N,C,c)
Z = zeros(N,1);
cc = c*2;
for i = 1:N %cc = 2 is Gaussian
    Z(i) = gamrnd(2/cc,1)^(1/cc) * exp(j*2*pi*rand);
end
cNorm = (gamma(2/c)/(gamma(1/c))); 
w =Z*(1/sqrt(cNorm)); %Normalize the complex variance||��һ��
waug = [w conj(w)]'; 
Z_gen = C^(1/2)*waug;  
end