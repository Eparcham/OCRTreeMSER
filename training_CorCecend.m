clc;
clear;
close all;

directo = 'charCheak';
addN = 1;
imsize = 40;
[foldernamesC,labels,file_paths ] =get_file_paths(directo);
save foldernamesC foldernamesC
for i = 1 : size(labels,2)
    I = imread(file_paths{i});
    if size(I,3)>1
        I = rgb2gray(I);
    end
    level = graythresh(I);
    I = 1-imbinarize(I,level);
    I=centersquare(I,imsize);
    figure(1);imshow(I)
    Sec{i,1} = I;
    Sec{i,2} = labels(i);
    i
end
save Sec Sec