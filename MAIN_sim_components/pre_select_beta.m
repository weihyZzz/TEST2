%pre_select_beta
beta_select = [];obj_cmp = [];preframe = 10;flag = 0; obj=[];
for beta1 = beta1_case for beta2 = beta2_case for beta1_offset = beta1_offset_case for beta2_offset = beta2_offset_case 
    flag = flag+1;
    EMIVA_beta = { [beta1+beta1_offset beta2+beta2_offset];[beta1+beta1_offset beta2+beta2_offset]};                                
    option.EMIVA_beta = EMIVA_beta;
    if batch_algs == 3 %||  online_algs == 3 % EMIVA 
       fprintf('case %d\n',flag);
       [s_est,label,~, ~,~,obj] = CGGMM_IVA_batch(x(:,1:(win_size/2+1) * preframe), option); 
    end
    beta_select = [beta_select, EMIVA_beta];   
    obj_cmp = [obj_cmp,obj];                    
end
end
end
end
if batch_algs == 3 %||
[~,loc] = min(obj_cmp);
EMIVA_beta = beta_select(:,loc);
option.EMIVA_beta = EMIVA_beta;
end
%¼ÇÂ¼beta
fid=fopen('beta_record.txt','a');fprintf(fid,'\n%s\n',filenameTmp);fprintf(fid,'beta = ');fprintf(fid,'%g   ',cell2mat(option.EMIVA_beta(1)));
