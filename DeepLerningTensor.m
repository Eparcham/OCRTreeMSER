function [prob,index] = DeepLerningTensor(im,deep)

imread1 = single(im);
f1 = cnnConvolve(imread1,deep.W1,deep.B1);
f2=relu(f1);
f3 = cnnPool([2 2],f2);
f4=cnnConvolve(f3,deep.W2,deep.B2);
f5=relu(f4);
f6 = cnnPool([2 2],f5);
add = 1;
[r,c,h] = size(f6);

f7 = zeros(1,r*c*h);
for k = 1 : r
    for j = 1 : c
        for i = 1 : h
            f7(1,add) = f6(k,j,i);
            add = add + 1;
        end
    end
end

f8=(deep.fc1'*f7') + deep.fcb1';
f9=relu(f8);
f10=(deep.fc2'*f9) + deep.fcb2';
y = Softmax(f10);  
[prob,index]=max(y);

end

function [convolvedFeatures] = cnnConvolve(images,W,b)

[filterDimRow,filterDimCol,channel,numFilters] = size(W);
[imageDimRow, imageDimCol,~, numImages] = size(images);
convDimRow = imageDimRow - filterDimRow + 1;
convDimCol = imageDimCol - filterDimCol + 1;
convolvedFeatures = zeros(convDimRow, convDimCol, numFilters, numImages);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        convolvedImage = zeros(convDimRow, convDimCol);
        for channelNum = 1:channel
            filter = W(:,:,channelNum,filterNum);
            filter = rot90(squeeze(filter),2);
            im = squeeze(images(:, :, channelNum,imageNum));
            convolvedImage = convolvedImage + conv2(im, filter,'valid');
        end
        convolvedImage = convolvedImage + b(filterNum);
        convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
    end
end
end

function [pooledFeatures] = cnnPool(poolDim, convolvedFeatures)


numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDimRow = size(convolvedFeatures, 1);
convolvedDimCol = size(convolvedFeatures, 2);
pooledDimRow = floor(convolvedDimRow / poolDim(1));
pooledDimCol = floor(convolvedDimCol / poolDim(2));
featuresTrim = convolvedFeatures(1:pooledDimRow*poolDim(1),1:pooledDimCol*poolDim(2),:,:);
pooledFeatures = zeros(pooledDimRow, pooledDimCol, numFilters, numImages);

for imageNum = 1:numImages
    for filterNum = 1:numFilters
        features = featuresTrim(:,:,filterNum, imageNum);
        temp = im2col(features, poolDim, 'distinct');
        [m] = max(temp);
        pooledFeatures(:,:,filterNum,imageNum) = reshape(m, size(pooledFeatures,1), size(pooledFeatures,2));
    end
end
end

function [ top ] = relu( bottom )
    top=max(0,bottom);
end

function y = Softmax(x)
  ex = exp(x); 
  y  = ex / sum(ex);
end
