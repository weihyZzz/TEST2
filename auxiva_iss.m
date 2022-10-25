function [Y,W]=auxiva_iss(X,option)
% [1] R. Scheibler and N. Ono, "FAST AND STABLE BLIND SOURCE SEPARATION
%                               WITH RANK-1 UPDATES"

iter_num = option.iter_num;
X=permute(X,[1,3,2]);% #freq * #frame * #mic
[n_freq, n_src,n_frames]=size(X);
%% initial
X=permute(X,[2,3,1]);
Y=X; % n_src*n_frames*n_freq
r_inv=zeros(n_src,n_frames);
v=zeros(n_freq,n_src);
tic
for iter=1:iter_num
  normy=zeros(n_src,n_frames);
  eps=1e-10;
  for freq_iter=1:n_freq
      normy=normy+abs(Y(:,:,freq_iter)).^2; %(8)of [1]
  end
   r_inv=1./max(eps,normy/n_freq); %% Gauss 
 
  for src_iter=1:n_src
      Y_temp=reshape(permute(Y,[2,1,3]),[n_frames,1,n_src,n_freq]);
      pro1=times(Y,r_inv);
      pro2=reshape(conj(Y_temp(:,:,src_iter,:)),[n_frames,1,n_freq]);
      for i=1:size(Y_temp,4)
        v_num(:,:,i)=pro1(:,:,i)*pro2(:,:,i); % n_src * 1 * n_freq %first line (under for f...)in argorithm 1 of [1]
        v_denom(:,:,i)=r_inv*abs(pro2(:,:,i)).^2; % n_src * 1 * n_freq %second line in argorithm 1 of [1]
      end
      v(:,:)=conj(squeeze(v_num(:,1,:))'./squeeze(v_denom(:,1,:))'); % n_freq*n_src %third line in argorithm 1 of [1] 
      vtemp1=squeeze(v_denom(:,1,:))';
      v(:,src_iter)=v(:,src_iter)-1./sqrt(vtemp1(:,src_iter)); % fourth line in argorithm 1 of [1]
      vtemp2=reshape(permute(v,[2,1]),[n_src,1,n_freq]);
      Y_temp2=permute(Y_temp,[2,1,3,4]) ;
      pro3=reshape(Y_temp2(:,:,src_iter,:),[1,n_frames,n_freq]);
      for i=1:size(vtemp2,3)
        Y(:,:,i)=Y(:,:,i)-vtemp2(:,:,i)*pro3(:,:,i); % last step in algorthm 1 of [1]
      end
  end 
end
for i=1:size(Y,3)
    W(:,:,i)=Y(:,1:n_src,i) *inv(X(:,1:n_src,i));%(4) of [1]
end
% 该算法需要backProjection进行放缩处理
X=permute(X,[3,2,1]);
Y=permute(Y,[3,2,1]);%调整输出维度
Y=backProjection(Y,X(:,:,1));
W=permute(W,[3,1,2]);
 toc
