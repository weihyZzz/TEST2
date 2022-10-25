    function [target_tag, intf_tag] = source_select_new(target_source_num, intf_source_num, varargin)
    %% ����Դ�ź��ļ���
    % ��Ϊ����͹̶�ģʽ�����ģʽ��target_src�����ѡȡK��Դ��intf_src�����ѡȡ��ͬ��L��Դ��Ϊ���ţ�K+L<=7
    % �̶�ģʽ�¸��������indexѡ��Դ�źź͸����ź�
    target_src = {'data/dev1_female3_src_1.wav','data/dev1_female3_src_3.wav',...
        'data/dev1_male3_src_1.wav','data/xiaoaitongxue_10s.wav',...
        'data/wake_up_word_repeated2.wav','data/washer_10s_resample.wav',...
        'data/dev1_wdrums_src_2.wav','data/WGN.wav','data/source_jovi.wav',...
        'data/source_man.wav','data/source_woman.wav','data/source_xvxv.wav'};
    intf_src = target_src;
    
%% Random pick mode
    if nargin < 3 
        target_tag = target_src(randperm(length(target_src),target_source_num)); % ���ѡ��K��Ŀ��Դ
        temp_target = target_tag;
        remain_num = length(intf_src)-length(target_tag);
        for i = 1:remain_num
            same_index = [];
            for j = 1:length(temp_target)
                if strcmp(intf_src{i},temp_target{j}) == 1
                    intf_src(i) = [];
                    same_index = [same_index j];
                end
            end
            temp_target(same_index) = [];
            if isempty(temp_target); break; end
        end
        intf_tag = intf_src(randperm(length(intf_src),intf_source_num));
    else 
%% Fix pick mode
      target_index = varargin{1}; intf_index = varargin{2};
      target_tag = target_src(target_index);
      intf_tag = intf_src(intf_index);      
    end

    %% �ɰ�source_select, �̶�����srcһ��
%     switch type
%         case 1
%             source_file1 = 'data/dev1_female3_src_1.wav';
%             source_file2 = 'data/dev1_female3_src_3.wav';
%             fprintf('female1+female2\n');
%         case 2
%             source_file1 = 'data/dev1_female3_src_1.wav';
%             source_file2 = 'data/dev1_male3_src_1.wav';
%             fprintf('male+female1\n');
%         case 3
%             source_file1 = 'data/dev1_female3_src_1.wav';
%             source_file2 = 'data/washer_10s_resample.wav';      
%             fprintf('female1+washer\n');
%         case 4
%             source_file1 = 'data/dev1_male3_src_1.wav';
%             source_file2 = 'data/washer_10s_resample.wav';
%             fprintf('wuw+washer\n');
%         case 5
%             source_file1 = 'data/xiaoaitongxue_10s.wav';
%             source_file2 = 'data/dev1_female3_src_1.wav';
% %             source_file2 = 'data/wake_up_word_repeated2.wav';
%             fprintf('wuw+female1\n');
%         case 6
%             source_file1 = 'data/wake_up_word_repeated2.wav';
%             source_file2 = 'data/dev1_male3_src_1.wav';
% %             source_file2 = 'data/xiaoaitongxue.wav';
%             fprintf('male+wuw\n');
%         case 7
%             source_file1 = 'data/xiaoaitongxue_10s.wav';
%             source_file2 = 'data/washer_10s_resample.wav';
% %             source_file2 = 'data/wake_up_word_repeated2.wav';
%             fprintf('wuw+washer\n');
%         case 9
%             source_file1 = 'data/xiaoaitongxue_10s.wav';
%             source_file2 = 'data/dev1_wdrums_src_2.wav';
%             fprintf('wuw+music\n');
% %             source_file2 = 'data/dev1_female3_src_1.wav';
% %             source_file2 = 'data/wake_up_word_repeated2.wav';
%     end
        
        
        
        