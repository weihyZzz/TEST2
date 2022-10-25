if online
    % 在线算法(online)
    if online_algs == 1 % AuxIVA
        [s_est,label] = auxiva_audio_bss_online_perm(x,source_num,option); % 带乱序输入的online版本
        %           [s_est] = auxiva_audio_bss_online_single(x,source_num,option);
        %           [s_est,label] = nmfiva_audio_bss_online_perm(x,source_num,option);
    elseif online_algs == 2 % MNMF
        [s_hat,label] = bss_multichannelNMF_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif online_algs == 3 % EMIVA
        tic
        [s_est,label,~, ~,~] = CGGMM_IVA_online_revise(x, option);
        toc
    elseif online_algs == 4 % IVE
        [s_est,label] = auxive_audio_bss_online_perm(x,option); % IVE
    elseif online_algs == 5 % t-mnmf
        [s_hat,label] = t_MNMF_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif online_algs == 6 % fastmnmf
        [s_hat,label] = FastMNMF_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif online_algs == 7 % t_fastmnmf
        [s_hat,label] = t_FastMNMF_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif online_algs == 8 % fastmnmf2
        [s_hat,label] = FastMNMF2_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,2,:)).';
    elseif online_algs == 9 % MLDR
        [s_hat,label] = MLDR_online(x.',option);
        s_est = s_hat.'; source_num = 1;
    elseif online_algs == 10 % tfastmnmf differnt pdf;                                % s1:t-distribution;s2:guass-distribution
        [s_hat,label] = mpdf_FastMNMF_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif online_algs == 11 % tfastmnmf differnt v
        [s_hat,label] = t_FastMNMFdiffv_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif online_algs == 12 % subguass_mnmf
        [s_hat,label] = subguas_FastMNMF_bss_online(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    end
    out_type = 'online';
else
    % 离线算法(batch)
    %         %% preprocess for select beta
    %         pre_select_beta;
    if batch_algs == 1 % AuxIVA
        [s_est,label] = auxiva_audio_bss_batch(x,source_num,option); % IVA batch(including nmfiva)
    elseif batch_algs == 2 % MNMF
        tic
        [s_hat,label] = bss_multichannelNMF_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';toc
    elseif batch_algs == 3 % EMIVA
        %             [pcaData,projectionVectors,eigVal] = whitening_pre(x',2);
        [s_est,label,~, ~,~,~,obj_set] = CGGMM_IVA_batch(x, option);
    elseif batch_algs == 4 % IVE
        [s_est,label] = auxive_audio_bss_batch(x,option); % IVE batch
    elseif batch_algs == 5 % EMIVA_fast
        [s_est,label,~, ~,~] = CGGMM_IVA_batch_fast(x, option);
    elseif batch_algs == 6 % AV-GMM-IVA
        [s_est,label,~, ~,~] = AV_GMM_IVA_batch(x, option);
    elseif batch_algs == 7 % subguassmnmf 未改好
        [s_hat,label] = subguas_FastMNMF_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 8 % t-mnmf
        [s_hat,label,cost] = t_MNMF_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 9 % fastmnmf
        [s_hat,label] = FastMNMF_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 10 % t_fastmnmf
        [s_hat,label] = t_FastMNMF_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 11 % MLDR
        [s_hat,label] = MLDR_batch(x.',option);
        s_est = s_hat.'; source_num = 1;
    elseif batch_algs == 12 % fastmnmf2
        [s_hat,label] = FastMNMF2_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 13 % tfastmnmf differnt v
        [s_hat,label] = t_FastMNMFdiffv_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 14 % tfastmnmf differnt pdf;
        % s1:t-distribution;s2:guass-distribution
        [s_hat,label] = mpdf_FastMNMF_bss_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 15 % tfastmnmf differnt pdf;
        % s1:t-distribution;s2:guass-distribution
        [s_hat,label] = bse_SCM_offline(x.',source_num,option);
        s_est = squeeze(s_hat(:,MNMF_refMic,:)).';
    elseif batch_algs == 16 % AR_FastMNMF2
        [s_hat,label] = AR_FastMNMF2_new(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 17 % AR_FastMNMF1
        [s_hat,label] = AR_FastMNMF(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 18 % AR_t_FastMNMF1
        [s_hat,label] = AR_t_FastMNMF(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 19 % AR_t_FastMNMF2
        [s_hat,label] = AR_t_FastMNMF2(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 20 % FastMNMF_nopart
        [s_hat,label] = FastMNMF_nopart(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 21 % FastMNMF2_nopart
        [s_hat,label] = FastMNMF2_nopart(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 22 % t_FastMNMF_nopart
        [s_hat,label] = t_FastMNMF_nopart(x.',source_num,option);
        s_est = s_hat.';
    elseif batch_algs == 23 % ILRMA_GWPE
        [s_hat,label] = ILRMA_GWPE(x.',source_num,option);
        s_est = s_hat.';
    end
    out_type = 'batch';
end