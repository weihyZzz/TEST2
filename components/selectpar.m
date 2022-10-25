function [spec_indices,par_select] = selectpar(partition, select, parti, XX, K, N, XE_TH)

    if select
        if parti
            for k = 1:K
                blk_index = [];
                for n = 1:partition{k}.num
                    indices = partition{k}.index{n};
                    if mean(mean(abs(XX(indices,:,k^2)))) > XE_TH(k)
                        blk_index = [blk_index n];
                    end
                end
                par_select{k}.index = {partition{k}.index{blk_index}};
                par_select{k}.num = length(blk_index);
                partition_size = cellfun(@(x) length(x), par_select{k}.index);
                par_select{k}.size = partition_size;
                par_select{k}.contrast = partition{k}.contrast;
                par_select{k}.contrast_derivative = partition{k}.contrast_derivative;
                spec_indices{k} = unique([partition{k}.index{blk_index}]);
            end
        else
            for k = 1:K
                blk_index = [];
%                 for nn = 1:N
%                     if mean(abs(XX(nn,:,k^2)),2) > XE_TH(k)
%                         blk_index = [blk_index nn];
%                     end
%                 end
                blk_index1 = find( mean(abs(XX(:,:,k^2)),2) > XE_TH(k) );
                blk_index = blk_index1';
                par_select{k}.index = {blk_index};
                par_select{k}.num = 1;
                partition_size = cellfun(@(x) length(x), par_select{k}.index);
                par_select{k}.size = partition_size;
                par_select{k}.contrast = partition{k}.contrast;
                par_select{k}.contrast_derivative = partition{k}.contrast_derivative;
                spec_indices{k} = unique([partition{k}.index{:}]);
            end
        end
    else
        for k = 1:K
            spec_indices{k} = unique( [partition{k}.index{:}] );
            par_select = {};
        end
    end
end