function BW = LocalThroshold(I,nhoodSize,sensitivity)

% bright foreground, map the sensitivity in [0, 1]
I1 = I;
I = im2double(I);
scaleFactor = 0.6 + (1-sensitivity);
T = localMedianThresh(I, nhoodSize, scaleFactor);
T = max(min(T,1),0);
BW = I1 > T.*max(double(I1(:)));
end

function T = localMedianThresh(I, nhoodSize, scaleFactor)
T = scaleFactor*medfilt2(I,nhoodSize,'symmetric');
end

