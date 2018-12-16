%%
clc
clear
close all
path = 'C:\Users\10\Desktop\Segmenation All of good code\segment 96-08-13\2\1';
path1 = 'image';
list = ls(path);
list(1:2,:) =[];
% [foldernames,labels,file_paths] =get_file_paths(path);
add = 1;
acc = 0;
for ii = 1 : size(list,1)
    I = (imread([path '\' list(ii,:)]));
    I1 = I; 
    if size(I,3)>1
        I = rgb2gray(I);
    end
    %     figure(10)
    %     imshow(I,[])
    %     I = adapthisteq(I);
    I = max(I(:))-I;
    I = imsharpen(I);
    tic 
    if ii==21
        nn = 1;
    end 
    if ii==10
        hh = 1; 
    end
    [mserStats,Image,I] = PlateRead(I);
    if size(Image,2)>7
        [Image] = Select8Char(Image);
         mserStats = vertcat(Image.box);
        ocr = [];
        for oi = 1 : size(Image,2)
            ocr = [ocr Image(oi).Ocr];
        end
        index = [1 : size(mserStats,1)];
        if size(mserStats,1)>1
            IExpandedBBoxes = ShowBound(mserStats,I,index);
        else
            IExpandedBBoxes = I1;
        end
        savePath = sprintf('%s - %d_%s.bmp',list(ii,1:end-4),ii,ocr);
        imwrite([IExpandedBBoxes],[path1 '/' savePath], 'bmp');
        figure(1)
        imshow(IExpandedBBoxes,[]);
        pause(0.01)
    end
end
