function out = FaceExtracter(folder)
    % This function receives the absolute path to a folder (in Windows
    % notation). The faces in each of the .jpg files within the folder are
    % extracted and saved separately within a new folder.
    close all;
    FDetect = vision.CascadeObjectDetector;
    newFolder = [folder,'_Faces'];
    mkdir(newFolder);
    imNames = dir([folder,'\*.jpg']);
    
    % Go through all jpgs in folderr
    for imCounter = 1:length(imNames)
        
        img = imread(imNames(imCounter).name);
        fP = step(FDetect,img); %facePositions
        
        % Extract each face and save separately
        for k = 1:size(fP,1)
            Face = img(fP(k,2):fP(k,2)+fP(k,4),fP(k,1):fP(k,1)+fP(k,3),:);
            Face = imresize(Face,[100 100]);
            imwrite(Face,[newFolder,'\',imNames(imCounter).name,'_',num2str(k),'.jpg'])
        end
    end
