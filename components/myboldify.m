function myboldify(h)
% myboldify: make lines and text bold
%   myboldify boldifies the current figure
%   myboldify(h) applies to the figure with the handle h
FontSize = 12;
LineWidth =2;
MarkerSize =12;
if nargin < 1
    h = gcf; 
end
ha = get(h, 'Children'); % the handle of each axis

for i = 1:length(ha)
    
    if strcmp(get(ha(i),'Type'), 'axes') % axis format
        set(ha(i), 'FontSize', FontSize);      % tick mark and frame format
        set(ha(i), 'LineWidth', LineWidth);

        set(get(ha(i),'XLabel'), 'FontSize', FontSize);
        %set(get(ha(i),'XLabel'), 'VerticalAlignment', 'top');

        set(get(ha(i),'YLabel'), 'FontSize', FontSize);
        %set(get(ha(i),'YLabel'), 'VerticalAlignment', 'baseline');

        set(get(ha(i),'ZLabel'), 'FontSize', FontSize);
        %set(get(ha(i),'ZLabel'), 'VerticalAlignment', 'baseline');

        set(get(ha(i),'Title'), 'FontSize', FontSize);
        %set(get(ha(i),'Title'), 'FontWeight', 'Bold');
    end
    
    hc = get(ha(i), 'Children'); % the objects within an axis
    for j = 1:length(hc)
        chtype = get(hc(j), 'Type');
        if strcmp(chtype(1:3), 'bar')
            set(hc(j), 'LineWidth', LineWidth);
        elseif strcmp(chtype(1:4), 'text')
            set(hc(j), 'FontSize', FontSize); % 14 pt descriptive labels
        elseif strcmp(chtype(1:4), 'line')
            set(hc(j), 'LineWidth', LineWidth);
            set(hc(j), 'MarkerSize', MarkerSize);
        elseif strcmp(chtype, 'hggroup')
            hcc = get(hc(j), 'Children');
            if strcmp(get(hcc, 'Type'), 'hggroup')
                hcc = get(hcc, 'Children');
            end
            for k = 1:length(hcc) % all elements are 'line'
                set(hcc(k), 'LineWidth', LineWidth);
                set(hcc(k), 'MarkerSize', MarkerSize);
            end
        elseif strcmp(chtype,'errorbar')
            set(hc(j), 'LineWidth', LineWidth);
        end
    end
    
end
end
