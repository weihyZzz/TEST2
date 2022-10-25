function azEst = doa_estimation(x,d,nsrc,fs)
[nSample,nChannel]=size(x);
if nChannel == 2 % 2 mic
    micPos = ... 
        [0-d/2,0+d/2;
        0,0;
        0,0];
else
    micPos = ...
        [-d-d/2,0-d/2, 0+d/2, d+d/2; % 4 mic
              0,    0,     0, 0;
              0,    0,     0, 0];
end
% micPos = ... 
% ...%  mic1	 mic2   mic3   mic4   mic5   mic6   mic7  mic8
%     [ 0.037 -0.034 -0.056 -0.056 -0.037  0.034  0.056 0.056;  % x
%       0.056  0.056  0.037 -0.034 -0.056 -0.056 -0.037 0.034;  % y
%     -0.038   0.038 -0.038  0.038 -0.038  0.038 -0.038 0.038]; % z

azBound = [-180 180]; % ��λ��������Χ
elBound = [0];   % ������������Χ����ֻ��ˮƽ�棺��elBound=0;
gridRes = 1;          % ��λ��/�����ǵķֱ���
alphaRes = 5;          % Resolution (? of the 2D reference system defined for each microphone pair

method = 'SRP-PHAT';
wlen = 512;
window = hann(wlen);
noverlap = 0.5*wlen;
nfft = 512;
c = 343;        % ����
freqRange = [];         % �����Ƶ�ʷ�Χ []Ϊ����Ƶ��
pooling = 'max';      % ��ξۺϸ�֡�Ľ��������֡ȡ�������{'max' 'sum'}

%% ��ȡ��Ƶ�ļ�(fix)
%[nSample,nChannel]=size(x);
if nChannel>nSample, error('ERROR:�����ź�ΪnSample x nChannel'); end
[~,nMic,~] = size(micPos);
if nChannel~=nMic, error('ERROR:��˷���Ӧ���ź�ͨ�������'); end
%% �������(fix)
Param = pre_paramInit(c,window, noverlap, nfft,pooling,azBound,elBound,gridRes,alphaRes,fs,freqRange,micPos);
%% ��λ(fix)
specGlobal = doa_srp(x,method, Param);

%% ����Ƕ�
minAngle                   = 10;         % ����ʱ����֮����С�н�
specDisplay                = 0;          % �Ƿ�չʾ�Ƕ���{1,0}

[pfEstAngles,~] = post_findPeaks(specGlobal, Param.azimuth, Param.elevation, Param.azimuthGrid, Param.elevationGrid, nsrc, minAngle, specDisplay);

azEst = pfEstAngles(:,1)';
elEst = pfEstAngles(:,2)';
for i = 1:nsrc
    fprintf('Estimated source %d : \n Azimuth (Theta): %.0f \t Elevation (Phi): %.0f \n\n',i,azEst(i),elEst(i));
end