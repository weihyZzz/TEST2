function spec = srpPhat_spec(X, f, tauGrid)
%[1] Dibiase J H . A High-Accuracy, Low-Latency Technique for Talker Localization inReverberant Environments Using Microphone Arrays. [J]. European Journal of Biochemistry, 2000, 216(1):281-91.
% page:83-94 srpPhat
X1 = X(:,:,1);
X2 = X(:,:,2);
[nbin,nFrames] = size(X1);
ngrid = length(tauGrid);
% tauGrid  (6.2) of [1]
P = X1.*conj(X2);
P = P./abs(P);
spec = zeros(nbin,nFrames,ngrid);
for pkInd = 1:ngrid
    EXP = repmat(exp(-2*1i*pi*tauGrid(pkInd)*f),1,nFrames);% EXP in (6.4) of [1]
    spec(:,:,pkInd) = real(P).*real(EXP) - imag(P).*imag(EXP); % (6.4) of [1]
    % 比直接spec(:,:,pkInd) = real(P.*EXP)计算速度更快
end

end