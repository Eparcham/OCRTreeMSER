clc;
clear;
close all;

directo = 'temp';
addN = 1;
imsize = 40;
[foldernamesF,labels,file_paths ] =get_file_paths(directo);
save foldernamesF foldernamesF
for i = 1 : size(labels,2)
    I = imread(file_paths{i});
    if size(I,3)>1
        I = rgb2gray(I);
    end
    level = graythresh(I);
    I = 1-imbinarize(I,level);
    [L,NUM] = bwlabeln(I); % we can use filter to reduce region
    SegmentInfo = regionprops(L,'Area','BoundingBox');% can reduse filter
    Area = vertcat(SegmentInfo.Area);
    [val,index]=max(Area);
    NesbatFirst = [SegmentInfo(index).BoundingBox(4)/SegmentInfo(index).BoundingBox(3) SegmentInfo(index).BoundingBox(3)/SegmentInfo(index).BoundingBox(4)];
    I=centersquare(I,imsize);
    figure(1);imshow(I)
    ImM{i,1} = I;
    ImM{i,2} = labels(i);
    ImM{i,3} = NesbatFirst;
    ImM{i,4} = foldernamesF{labels(i)};
    i
end
save ImM ImM
