function doa = doa_update(W,d,fs)
[N,M,K] = size(W);
A = zeros(N,K,M); % »ìºÏ¾ØÕó£¬ÎªWµÄÄæ
for n = 1:N
    A(n,:,:) = pinv(squeeze(W(n,:,:)));
end
N_FFT = 2*(N-1); % FFT Points = 2*(spec_coeff_num-1);
vf = [0:N-1]*fs/N_FFT;
cs = 343; % Speed of sound, currently set as 343m/s;
dd = 2*pi*vf/cs; 
dd = dd.';
for k = 1:K
    Ak = squeeze(A(:,k,:));
    Ak_ang = angle(Ak);
    d_ang = Ak_ang(:,1)-Ak_ang(:,2);
    ang = d_ang ./ dd;
    avg_ang(k) = mean(ang(2:end));
end
doa = acos(avg_ang / d);