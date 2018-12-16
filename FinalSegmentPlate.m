function Image = FinalSegmentPlate(Image,I)

M = [];
ocr = [];
for i = 1 : size(Image,2)
    Seg = Image(i).ImageTh;
    if i==2
        H = [Image(i).box(2) Image(i).box(4)];
    end
    flag = 0;
    if strcmp(Image(i).Ocr,'B')
        flag = 1;
    elseif (strcmp(Image(i).Ocr,'C') || strcmp(Image(i).Ocr,'S'))
        flag = 1;
    end
    if flag
        box = Image(i).box;
        box(2) = H(1);
        box(4) = H(2);
        T = imcrop(I,box);
        T = logical(T<Image(i).Intnsifiy) ;
        %% morphologi
        BW_out = imfill(T, 'holes');
        BW_out = bwpropfilt(BW_out, 'Area', [20, inf]);
        Image(i).ImageTh = BW_out;
        T = Image(i).ImageTh;
        M = [M T];
    else
        M = [M Seg];
    end
    ocr = [ocr Image(i).Ocr];
end
figure
imshow(M,[]);
title(ocr);