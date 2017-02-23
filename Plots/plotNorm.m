% Script to show effect of normalization


clc;
clear all;
close all;
path = '/Users/jannis/Desktop/Plots';
cd(path)

filenames = dir('*.jpg');
f = figure('units','normalized','outerposition',[0 0 1 1]);

for k = 1:length(filenames)
    h = subplot(2,4,k);
    p = get(h,'pos')
    img = imread([path,'/',filenames(k).name]);
    imshow(img);
    if k == 1
        text(215,-8,'Face cropped images', 'FontSize',30)
        text(200,132, 'Normalized face cropped images', 'FontSize',30)
    end
    if k < 5
        p(2) = p(2) - 0.1;
        set(h,'pos',p)
    end
        
end


    