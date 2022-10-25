function [SDR,SIR,SNR,SAR,perm] = bss_eval_noise(se,s,noise)

if nargin == 3
    noiseout = 1; 
else
    noiseout = 0;
end

[nsrc,nsampl]=size(se);
[nsrc2,nsampl2]=size(s);
if noiseout
    [nsrc3,nsampl3]=size(noise);
end
if nsrc2~=nsrc, error('The number of estimated sources and reference sources must be equal.'); end
if nsampl2~=nsampl, error('The estimated sources and reference sources must have the same duration.'); end
L = 0;
W=ceil(nsampl/20);

% 用不同的窗测试
win = ones(1,W*2);
% win = triang(W*2)';
SDR=zeros(nsrc,nsrc);
SIR=zeros(nsrc,nsrc);
SNR=zeros(nsrc,nsrc);
SAR=zeros(nsrc,nsrc);
for jest=1:nsrc
    for jtrue=1:nsrc
        if noiseout
            [starget,einterf,enoise,eartif]=bss_decomp_tvfilt(se(jest,:),jtrue,s,noise,win,W,L);
            [SDR(jest,jtrue),SIR(jest,jtrue),SNR(jest,jtrue),SAR(jest,jtrue)]=bss_crit(starget,einterf,enoise,eartif);
        else
            [starget,einterf,eartif]=bss_decomp_tvfilt(se(jest,:),jtrue,s,win,W,L);
            [SDR(jest,jtrue),SIR(jest,jtrue),SAR(jest,jtrue)]=bss_crit(starget,einterf,eartif);
        end
    end
end

% 排序
perm=perms(1:nsrc);
nperm=size(perm,1);
meanSIR=zeros(nperm,1);
for p=1:nperm
    meanSIR(p)=mean(SIR((0:nsrc-1)*nsrc+perm(p,:)));
end
[meanSIR,popt]=max(meanSIR);
perm=perm(popt,:).';
SDR=SDR((0:nsrc-1).'*nsrc+perm);
SIR=SIR((0:nsrc-1).'*nsrc+perm);
SAR=SAR((0:nsrc-1).'*nsrc+perm);
if noiseout
    SNR=SNR((0:nsrc-1).'*nsrc+perm);
else
    SNR = zeros(nsrc,1);
end

return;

function w = triang(n)
% TRIANG Triangular window.
if rem(n,2)
% It's an odd length sequence
w = 2*(1:(n+1)/2)/(n+1);
w = [w w((n-1)/2:-1:1)]';
else
% It's even
w = (2*(1:(n+1)/2)-1)/n;
w = [w w(n/2:-1:1)]';
end