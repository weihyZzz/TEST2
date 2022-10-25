%% extreme_em_beta
beta_select = [];preframe = 10;s_est = [];
[nmic,nsample] = size(x);
nframe = fix((nsample)/(win_size/2)+1)+1;
for frameflag = 10:10:nframe flag = 0; s_est_set = [];obj_cmp = [];
for beta1 = beta1_case for beta2 = beta2_case for beta1_offset = beta1_offset_case for beta2_offset = beta2_offset_case 
    flag = flag+1;fprintf('case %d\n',flag);
    EMIVA_beta = { [beta1+beta1_offset beta2+beta2_offset];[beta1+beta1_offset beta2+beta2_offset]};                                
    option.EMIVA_beta = EMIVA_beta;    
    if batch_algs == 3 ||  online_algs == 3 % EMIVA
       [s_est_re,label,~, ~,~,obj] = CGGMM_IVA_batch(x(:,1+(win_size/2+1) * (frameflag-10):(win_size/2+1) * frameflag), option); 
    end
    beta_select = [beta_select, EMIVA_beta];
    obj_cmp = [obj_cmp, obj];
    s_est_set = [s_est_set; s_est_re];
end
end   
end
end
[~,loc] = min(obj_cmp);
s_est_part = [s_est_set(2*loc-1,:);s_est_set(2*loc,:)];
s_est = [s_est,s_est_part];
end

% option.EMIVA_beta = EMIVA_beta;