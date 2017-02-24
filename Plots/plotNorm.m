% Script to show effect of normalization


clc;
clear all;
close all;
mainPath = '/Users/jannis/Dropbox/GitHub/FacePeeper/Plots';
path = [mainPath,'/normPlot'];
cd(path)

% Classes: 127 - 156 - 347 - 162

filenames = dir('*.jpg');
f = figure('units','normalized','outerposition',[0 0 1 1]);

for k = 1:length(filenames)
    h = subplot(3,4,k);
    p = get(h,'pos');
    img = imread([path,'/',filenames(k).name]);
    imshow(img);
    if k == 1
        text(215,-9,'Face cropped images', 'FontSize',30)
        text(180,126, 'Normalized face cropped images', 'FontSize',30)
        text(135,260, 'Augmented normalized face cropped images', 'FontSize',30)

        
    end
    if k < 5
        p(2) = p(2) - 0.08;
        set(h,'pos',p)
    elseif k < 9
        p(2) = p(2) - 0.04;
        set(h,'pos',p)
    end
    
    switch mod(k,4)
        case 1
            set(h,'pos',p)
        case 2
            p(1) = p(1) - 0.05;
            set(h,'pos',p)
        case 3
            p(1) = p(1) - 0.1;
            set(h,'pos',p)
        case 0
            p(1) = p(1) - 0.15;
            set(h,'pos',p)
    end
                    
end

saveas(f,[mainPath,'/norm.eps']);


    