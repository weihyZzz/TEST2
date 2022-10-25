 function [y ,nu1 ,pi1 ,nu2, pi2 ,W ,F] = EM_IVA_noiseFree(x,upS1,upS2,nEM)
% The EM code for IVA algorithm, No Noise! y=Wx

% Inputs:
%   y1(nK,nT)    : FFT coefficients for channel one of the mixed signal;freq*T
%   y2(nK,nT)    : FFT coefficients for channel two of the mixed signal;F*T
%   upS1        : logic, upS1==1, update PDF for source 1;
%   upS2        : logic, upS2==1, update PDF for source 2;
%   nu1(K,nS)   : precision of prior for source 1;covariance freq * conmponent mixture
%   pi1(nS,1)   : weights of SMM for prior of source 1;一个源中两个类所占的比重
%   nu2(K,nS)   : precision of prior for source 2;
%   pi2(nS,1)   : weights of SMM for prior of fsource 2;
%   nEM         : number of EM iterations;
% Outputs:
%   x1(nK,nT)    : estimated FFT coefficients of source 1;
%   x2(nK,nT)    : estimated FFT coefficients of source 2;
%   nu1(K,nS)   : precision of SMM for source 1;
%   pi1(nS,1)   : weights of SMM for source 1;
%   nu2(K,nS)   : precision of SMM for source 2;
%   pi2(nS,1)   : weights of SMM for source 2;
%   W(2,2,nK)   : W=inv(A), the DEmixing matrix, x=W*y;
%   F           : the likelihood value
% 
% Ref[1] An Expectation–Maximization-Based IVA Algorithm for Speech Source Separation Using Student’s t Mixture Model Based Source Priors
% This code is the noise free version of the EM-IVA. 

% %% pre-whitening，令Cxx等于单位矩阵;
% C = cal_Cxx(x);
%     Q_H = zeros(nK,K,K);
%     for n = 1:nK
%         % We will need the inverse square root of Cx
%         [e_vec,e_val] = eig(squeeze(C(n,:,:)),'vector');
%         Q_H = fliplr(e_vec) .* sqrt(flipud(e_val)).';
%         x(n,:,:) = squeeze(x(n,:,:)) * (Q_H^-1).';
%     end
%%
[nK,nT,K]= size(x);Y = zeros(nK,nT,K);
nS1 = K;nS2 = nS1;%仅先实现2*2
y1 = x(:,:,1); y2 = x(:,:,2);
% [nK,nT] = size(y1);% nK：nfreq nT:nframe
% nS1 = size(pi1,1);
% nS2 = size(pi2,1);
v = 4;              % Degree of freedom
d = 512;
% Random Initialization;
% W = randn(2,2,nK);
% Unitary Initialization;
W = zeros(2,2,nK);W(1,1,:) = 1;W(2,2,:) = 1;% (9) of [1]
% Random Initialization;
pi1 = rand(nS1,1);pi1 = pi1./sum(pi1);
pi2 = rand(nS2,1);pi2 = pi2./sum(pi2);
% % initialize by whitening matrix
% for k = 1:nK
%    W(:,:,k) = inv(sqrtm([y1(k,:);y2(k,:)]*[y1(k,:);y2(k,:)]'/nT));
% end

y11 = real(y1.*conj(y1));y22 = real(y2.*conj(y2));y12 = y1.*conj(y2);Sig_11 = zeros(nS1*nS2,nK);Sig_21 = zeros(nS1*nS2,nK);Sig_22 = zeros(nS1*nS2,nK);dS = zeros(nS1*nS2,1);
%% initialization nu1,nu2, eq(8) of [1]
% nu1 = randn(nK,nS1);
% nu2 = randn(nK,nS2);
% for ns1 = 1:nS1
%     nu1(:,ns1) = 1./(sum(y11 * pi1(ns1,:),2) / nT) ;
% end
% for ns2 = 1:nS2
%     nu2(:,ns2) = 1./(sum(y22 * pi2(ns2,:),2) / nT) ;
% end
    nu1(:,1) = 1./(sum(y11 * pi1(1,:),2) / nT) ;%   3 lines under (7) of [1]
    nu2(:,1) = 1./(sum(y11 * pi1(2,:),2) / nT) ;%* 0.1;
    nu1(:,2) = 1./(sum(y22 * pi2(1,:),2) / nT) ;%* 0.1;
    nu2(:,2) = 1./(sum(y22 * pi2(2,:),2) / nT) ;%* 0.1;

%%%%%%%%%%%%%%%%%%%%  EM STEP  %%%%%%%%%%%%%%%%%%
for iEM = 1:nEM
display(['Iteration:  ' num2str(iEM)]);   
W11nu1 = nu1.*repmat(abs(squeeze(W(1,1,:))).^2,1,nS1);% (16) of [1]  W'*phi*W的展开形式
W21nu2 = nu2.*repmat(abs(squeeze(W(2,1,:))).^2,1,nS2);
W12nu1 = nu1.*repmat(abs(squeeze(W(1,2,:))).^2,1,nS1);
W22nu2 = nu2.*repmat(abs(squeeze(W(2,2,:))).^2,1,nS2);
W11W12nu1 = nu1.*repmat(squeeze(W(1,1,:)).*conj(squeeze(W(1,2,:))),1,nS1);
W21W22nu2 = nu2.*repmat(squeeze(W(2,1,:)).*conj(squeeze(W(2,2,:))),1,nS2);
for is1 = 1:nS1
    for is2 = 1:nS2
        is = (is2-1)*nS2+is1;
        Sig_11(is,:) = (W11nu1(:,is1)+W21nu2(:,is2))';
        Sig_21(is,:) = (W11W12nu1(:,is1)+W21W22nu2(:,is2)).';
        Sig_22(is,:) = (W12nu1(:,is1)+W22nu2(:,is2))';
        dS(is) = log(pi1(is1))+log(pi2(is2))+sum(log(nu1(:,is1).*nu2(:,is2)./pi+eps));% (17) of [1]:log(p(qi))yu前面一项的部分
    end
end
f = repmat(dS,1,nT)-Sig_11*y11-2*real(Sig_21*y12)-Sig_22*y22;% (17) of [1]， 把（8）代入（17）
maxf = max(f,[],1);
q = exp(f-repmat(maxf,nS1*nS2,1));% 归一化
z = sum(q,1);% (18).1 of [1]
qs = q./repmat(z,nS1*nS2,1);% (18).2 of [1] 对应式子中的 z(xi)

% Update A and source models.
sumqs = sum(qs,2);% 把概率对帧求和
sumq = reshape(sumqs,nS1,nS2);
qyy11 = ((v/2+d/2)/v)*qs*y11';%（A12）上方Mik公式中的z(si)*(v/2+d/2)/v*xi(k)'*xi(k)
qyy22 = ((v/2+d/2)/v)*qs*y22';% 同上
qyy12 = ((v/2+d/2)/v)*qs*y12';
for k = 1:nK
    dnu = repmat(nu1(k,:)',nS2,1)-reshape(repmat(nu2(k,:),nS1,1),nS1*nS2,1);% v1(k)-v2(k)
    M11 = dnu'*qyy11(:,k);%（A12）上方Mik公式
    M12 = dnu'*qyy12(:,k);
    M22 = dnu'*qyy22(:,k);
    eig1 = (M11+M22)/2-sqrt((M11-M22)^2/4+abs(M12)^2);% (A.17) of [1]对应 式子中beta
    %eigvec = [M12/(eig1-M11);1]/sqrt(abs(M12/(eig1-M11))^2+1); 
    eigvec = [1; (eig1-M11)/(M12)]/sqrt(1+abs((eig1-M11)/(M12))^2);% (26) of [1]
    %[V D] = eig([M11 M12; M12' M22]);
    %W2(:,:,k) = [conj(V(1,1)) conj(V(2,1)); -V(2,1) V(1,1)];
    W(:,:,k) = [conj(eigvec(1)) conj(eigvec(2)); -eigvec(2) eigvec(1)];% (9) of [1]
    %if (iEM>3); dbstop; end
    %% update Y
    if upS1 %&&0
        a1 = abs(W(1,1,k))^2*sum(reshape(qyy11(:,k),nS1,nS2),2);% (27) of [1]z(xij=r)*xi(k)'*W(k)'*W(k)*xi(k)
        a2 = W(1,1,k)*conj(W(1,2,k))*sum(reshape(qyy12(:,k),nS1,nS2),2);
        a3 = abs(W(1,2,k))^2*sum(reshape(qyy22(:,k),nS1,nS2),2);
        nu1(k,:) = ((v/2+d/2)/v)*((sum(sumq,2)./(a1+2*real(a2)+a3+eps))');% (27) of [1]
        pi1 = sum(sumq,2); pi1 = pi1./sum(pi1+eps);% 归一化
   end
   
   if upS2 %&&0
        a1 = abs(W(2,1,k))^2*sum(reshape(qyy11(:,k),nS1,nS2),1);% (27) of [1]z(xij=r)*xi(k)'*W(k)'*W(k)*xi(k)
        a2 = W(2,1,k)*conj(W(2,2,k))*sum(reshape(qyy12(:,k),nS1,nS2),1);
        a3 = abs(W(2,2,k))^2*sum(reshape(qyy22(:,k),nS1,nS2),1);
        nu2(k,:) = ((v/2+d/2)/v)*((sum(sumq,1)./(a1+2*real(a2)+a3+eps)));% (27) of [1]
        pi2 = sum(sumq,1)'; pi2 = pi2./sum(pi2+eps);% 归一化
   end
end

F(iEM) = sum(maxf)+sum(log(z));% % (29) of [1] costfunc
plot(F); %drawnow; 
end

%%%% Signal Estimation
% for nsrc = 1:K
%         Y(:,:,nsrc) = sum(x .* permute(repmat(W(nsrc,:,:),nT,1,1),[3,1,2]), 3);
% end
x1 = repmat(squeeze(W(1,1,:)),1,nT).*y1+repmat(squeeze(W(1,2,:)),1,nT).*y2;
x2 = repmat(squeeze(W(2,1,:)),1,nT).*y1+repmat(squeeze(W(2,2,:)),1,nT).*y2;
y(:,:,1) = x1;y(:,:,2) = x2;
% for k = 1:nS1
%    y(:,:,k) = sum(x .* permute(repmat(permute(W(:,k,:),[3,1,2]),1,1,nT),[1,3,2]), 3);
% end
return;
function Cxx = cal_Cxx(X) % 旧版Cxx计算方法，暂时保留，(9) of [3]
[N, T, M] = size(X);
Cxx = zeros(N, M, M);
for n = 1:N
    Xf = squeeze(X(n,:,:));
    if T == 1
        % 当T=1时，由于squeeze的特性，Xf与[3]中xf相等，直接用原公式
        Cf = Xf * Xf';
    else
        % 本Xf为[3]中xf的转置，因此(xf*xf') = Xf.' * conj(Xf)
        Cf = Xf.' * conj(Xf) / T; 
    end
    Cxx(n,:,:) = Cf;
end

