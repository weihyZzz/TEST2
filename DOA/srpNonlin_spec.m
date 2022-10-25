function spec = srpNonlin_spec(X, f, alpha, tauGrid)
%[1] Dibiase J H . A High-Accuracy, Low-Latency Technique for Talker Localization inReverberant Environments Using Microphone Arrays. [J]. European Journal of Biochemistry, 2000, 216(1):281-91.
% page:83-94 srpPhat
X1 = X(:,:,1);
X2 = X(:,:,2);

[nbin,nFrames] = size(X1);
ngrid = length(tauGrid);

spec = zeros(nbin,nFrames,ngrid);
P = X1.*conj(X2);
P = P./abs(P);
temp = ones(1,nFrames);
for pkInd = 1:ngrid,
    EXP = exp(-2*1i*pi*tauGrid(pkInd)*f);% EXP in (6.4) of [1]
    EXP = EXP(:,temp);
    spec(:,:,pkInd) = 1 - tanh(alpha*sqrt(abs(2-2*real(P.*EXP)))); 
end

end