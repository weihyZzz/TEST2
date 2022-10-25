function [win_ana, win_syn] = ...
    generate_win(win_type, win_size, inc)

    win_type = lower(strtrim(win_type));
    n = [0:win_size-1] - fix(win_size / 2);
    
    if strcmp(win_type, 'hann')    
        win_ana = .5 * (1 + cos(2 * pi * n / win_size));
        
    elseif strcmp(win_type, 'hamming')
        win_ana = 0.54 * ones(1, win_size) ...
                  + 0.46 * cos(2 * pi * n / win_size);
    else
        error('Window type %s not supported!', win_type);
    end
    
    win_syn = generate_win_syn(win_ana, inc);

end

function win_syn = generate_win_syn(win_ana, inc)
    win_size = length(win_ana);
    win_syn = win_ana;  
end
