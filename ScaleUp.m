

mkdir('112x112');
folders = dir([pwd,'\100x100\']);
for k = 1:length(folders)
    if folders(k).name(1) ~= '.'
        mkdir([pwd,'\112x112\',folders(k).name])
        imNames = dir([pwd,'\100x100\',folders(k).name,'\*.jpg']);
        for l = 1:length(imNames)
            img = imread([pwd,'\100x100\',folders(k).name,'\',imNames(l).name]);
            img = imresize(img,[112,112]);
            imwrite(img,[pwd,'\112x112\',folders(k).name,'\',imNames(l).name,'.jpg']);
        end
    end
end
        