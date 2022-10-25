    function [source_file1, source_file2] = source_select(type)
    %% ����Դ�ź��ļ���
    % ���Ѷ�source����ͳһ�����֣�source_file1������target source��source_file2������interference 
        
    switch type
        case 1
            source_file1 = 'data/dev1_female3_src_1.wav';
            source_file2 = 'data/dev1_female3_src_3.wav';
            fprintf('female1+female2\n');
        case 2
            source_file1 = 'data/dev1_female3_src_1.wav';
            source_file2 = 'data/dev1_male3_src_1.wav';
            fprintf('male+female1\n');
        case 3
            source_file1 = 'data/dev1_female3_src_1.wav';
            source_file2 = 'data/washer_10s_resample.wav';      
            fprintf('female1+washer\n');
        case 4
            source_file1 = 'data/dev1_male3_src_1.wav';
            source_file2 = 'data/washer_10s_resample.wav';
            fprintf('wuw+washer\n');
        case 5
            source_file1 = 'data/xiaoaitongxue_10s.wav';
            source_file2 = 'data/dev1_female3_src_1.wav';
%             source_file2 = 'data/wake_up_word_repeated2.wav';
            fprintf('wuw+female1\n');
        case 6
            source_file1 = 'data/wake_up_word_repeated2.wav';
            source_file2 = 'data/dev1_male3_src_1.wav';
%             source_file2 = 'data/xiaoaitongxue.wav';
            fprintf('male+wuw\n');
        case 7
            source_file1 = 'data/xiaoaitongxue_10s.wav';
            source_file2 = 'data/washer_10s_resample.wav';
%             source_file2 = 'data/wake_up_word_repeated2.wav';
            fprintf('wuw+washer\n');
        case 9
            source_file1 = 'data/xiaoaitongxue_10s.wav';
            source_file2 = 'data/dev1_wdrums_src_2.wav';
            fprintf('wuw+music\n');
%             source_file2 = 'data/dev1_female3_src_1.wav';
%             source_file2 = 'data/wake_up_word_repeated2.wav';
    end