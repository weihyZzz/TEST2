%% beta_setup
if auto_beta == 0
beta1_case = [20]/10;           
beta2_case = [20]/10;                    
beta1_offset_case = [-4]/10^1;         
beta2_offset_case = [0]/10^1;
else
beta_dataset = [1.6:0.1:2.4 1.6:0.1:2.4];
beta_set = nchoosek(beta_dataset,2);
beta = beta_set(beta_case,:);
end
             
