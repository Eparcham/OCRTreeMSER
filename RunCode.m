warning('off')
clear
clc
close all
%% input image

[file,path]=uigetfile({'*.jpg;*.bmp;*.png;*.tif'},'Choose an image');
s=[path,file];
I=imread(s);
if size(I,3)>1
    I = rgb2gray(I);
end
% I = max(I(:)) - I;
% I = imsharpen(I);
figure 
imshow(I,[]); 
tic 
[mserStats,Image,I] = PlateRead(I);
toc
index = [1 : size(mserStats,1)];
IExpandedBBoxes = ShowBound(mserStats,I,index);
figure  
imshow(IExpandedBBoxes,[]); 
%% select true selection       
Image = Select8Char(Image);   
%%
%[box,Image,listError] = FindTrueSegment(Image);
box = vertcat(Image.box);
index = [1 : size(box,1)]; 
IExpandedBBoxes = ShowBound(box,I,index);
figure
imshow(IExpandedBBoxes,[]);
ocr = [];
for i = 1 : size(Image,2)
    ocr = [ocr Image(i).Ocr];
end
title(ocr);  
Image = FinalSegmentPlate(Image,I);
