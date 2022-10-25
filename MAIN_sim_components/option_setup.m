%%option_setup%%
option.D_open = D_open; option.iter_num = iter_num; option.inner_iter_num = inner_iter_num;
if verbose ==0     option.verbose = false; end
if verbose ==1     option.verbose = true; end
option.forgetting_fac = forgetting_fac; option.select = select; option.ita = ita; option.IVE_method = IVE_method;
option.parti = parti; option.thFactor = thFactor; option.online = online; option.win_size = win_size;
option.win_ana = win_ana; option.win_syn = win_syn; option.Lb = Lb; option.batch_type = batch_type;
option.perm = perm_on; option.whitening_open = whitening_open; option.partisize = partisize;
option.n_orders_online = [n_orders1 n_orders2]; % online n_orders
option.n_orders_batch = cell2mat(n_orders_case(n_orders_num)); option.pre_batch = pre_batch;
% option.EMIVA_beta = EMIVA_beta_case(2*EMIVA_beta_num-1 : 2*EMIVA_beta_num);
option.diagonal_method = diagonal_method; option.gamma = gamma; option.initial = initial;
option.batch_update_num = batch_update_num; option.delta = delta; option.mix_model = mix_model;
option.MNMF_refMic = MNMF_refMic; option.MNMF_nb = MNMF_nb; option.MNMF_it = MNMF_it; option.MNMF_first_batch = MNMF_first_batch; option.MNMF_batch_size = MNMF_batch_size; option.MNMF_rho = MNMF_rho;
option.MNMF_fftSize = MNMF_fftSize; option.MNMF_shiftSize = MNMF_shiftSize; option.ILRMA_init = ILRMA_init; option.ILRMA_type = ILRMA_type; option.ILRMA_nb = ILRMA_nb;
option.MNMF_delta = MNMF_delta; option.ILRMA_it = ILRMA_it; option.ILRMA_normalize = ILRMA_normalize; option.ILRMA_dlratio = ILRMA_dlratio;
option.MNMF_drawConv = MNMF_drawConv;option.sub_beta = sub_beta;option.tMNMF_v = tMNMF_v;option.tMNMF_trial = tMNMF_trial;option.mpdf_pro = mpdf_pro;
option.AR_init_it = AR_init_it;option.AR_tap = AR_tap;option.AR_delay = AR_delay;option.AR_init_SCM = AR_init_SCM;option.AR_internorm = AR_internorm;
option.ILRMA_drawConv = ILRMA_drawConv;  option.EMIVA_max_iter = EMIVA_max_iter;option.EMIVA_ratio = EMIVA_ratio;option.singleout = singleout_case;
option.detect_low = detect_low;option.detect_up = detect_up;option.pmulti = pmulti; option.logp = logp;
option.epsilon_start_ratio = epsilon_start_ratio;option.epsilon = epsilon;option.epsilon_ratio = epsilon_ratio; 
option.AVGMMIVA_max_iter = AVGMMIVA_max_iter;
option.MLDR_fft_size = MLDR_fft_size; option.MLDR_shift_size = MLDR_shift_size; option.MLDR_iteration = MLDR_iteration; option.MLDR_delta = MLDR_delta; 
option.MLDR_epsilon = MLDR_epsilon; option.MLDR_epsilon_hat = MLDR_epsilon_hat; option.MLDR_moving_lambda = MLDR_moving_lambda; 
option.MLDR_moving_average = MLDR_moving_average; option.MLDR_epsilon1 = MLDR_epsilon1; option.MLDR_zeta = MLDR_zeta; option.MLDR_beta = MLDR_beta; 
option.MLDR_rho = MLDR_rho; option.MLDR_epsilon1_hat = MLDR_epsilon1_hat; option.MLDR_gamma_first5 = MLDR_gamma_first5; option.MLDR_gamma_others = MLDR_gamma_others;
<<<<<<< HEAD
option.run_gwpe = run_gwpe;
=======
option.SCM_alpha = SCM_alpha;
>>>>>>> my_addnew-ver
% 对4mic的情况，需要有四个orders，这时候用orders1和orders2重复一次
if sim_mic == 4 && determined option.n_orders_online = [n_orders1 n_orders2 n_orders1 n_orders2]; end
option.nmf_iter_num = nmf_iter_num; option.nmf_fac_num = nmf_fac_num; option.nmf_beta = nmf_beta;
option.prior = cell2mat(prior); option.nmf_p = nmf_p; option.nmfupdate = nmfupdate; option.nmf_b = nmf_b; 
if mix_sim_case option.mic_pos = mic_pos; option.theta = theta; end 
GammaRatio = ones(1,8); option.initial_rand = initial_rand;  option.DOA_init = DOA_init; 
option.deltabg = deltabg; option.project_back = project_back; option.annealing = annealing;
option.prebatch_iter_num = prebatch_iter_num; option.DOA_esti = DOA_esti; option.DOA_update = DOA_update;
option.esti_mic_dist = esti_mic_dist; option.mix_sim = mix_sim; option.deltaf = deltaf;
option.DOA_esti_online = DOA_esti_online;