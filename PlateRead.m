function  [mserStats,tree,Fimage] = PlateRead(I)

load ImM
load foldernamesF
% MaxIdx = MultiThroshuld(I,15);
global id th_Hamposhani thValedA2 Hborder display TH_Ocr thRait thSelectH th_Max_Hamposhani imsize thWidth MaxAcceptVal
thRait = 0.6;
Hborder = 0.25;
TH_Ocr = 0.95;
thValedA2 =0.7;  %% th A2
th_Hamposhani = 0.25;  %% hamposhani
th_Max_Hamposhani = 1;
id = 1;
display = 1;
thWidth = 2;
MaxAcceptVal = 0.6;
thSelectH = 0.5;
acceptTH = 0.45;  %% ocr
minW = 0.01 * size(I,2);
maxW = 0.3 * size(I,2);
minH = 0.01 * size(I,1);
maxH = 1 * size(I,1);

TimeStep(1,1) = 1; % index of node
TimeStep(2,1) = 0; % father
TimeStep(3,1) = 0; % Val overlab
TimeStep(4,1) = size(I,2);
TimeStep(5,1) = 0;
tree(1).id = 1; % ID
tree(1).parent = 0; % PARENT
tree(1).Childs = []; % PARENT
tree(1).box = [1 1 size(I,1) size(I,2)];
% se = strel('disk', 1);
imsize = 40;
level = 1;
%% paramter of segment
load foldernamesNN
%for ii = 1 : size(bw,1)%% we must to change rang of this for and exit from loop
% for ii = size(MaxIdx,2):-1:1
%      i = MaxIdx(ii);
%h = test_throshold(I);

MaxIdx = MultiThroshuld(I,15);
%for ii = 200 : -2 : 30
for ii = 1 : size(MaxIdx,2)
     i = MaxIdx(ii);
     im = logical((1-(I>i)));
   % im = logical(1-bw{ii});
    if display
        figure(1)
        cla
        subplot(2,1,1)
        imshow(im,[]);
    end
    [L,NUM] = bwlabeln(im); % we can use filter to reduce region
    SegmentInfo = regionprops(L,'Area','BoundingBox','Image','Centroid','PixelList');% can reduse filter
    for j = 1 : NUM
        ratio1 = SegmentInfo(j).BoundingBox(4) / SegmentInfo(j).BoundingBox(3);
        ratio2 = SegmentInfo(j).BoundingBox(3) / SegmentInfo(j).BoundingBox(4);
        Rait = [ratio1 ratio2];
        %% cheak size of bord
        W = SegmentInfo(j).BoundingBox(3);
        H = SegmentInfo(j).BoundingBox(4);
        cheakSize = (W>=minW && W<=maxW && H>=minH && H<=maxH);
        if (cheakSize) % we can not use it
            im1 = (SegmentInfo(j).Image);
            imtemp = im1;
            imNUM=centersquare(im1,imsize); % very important
            T = double(imNUM);
            xn = [];
            for icor = 1 : size(ImM,1)
                xn=[xn corr2(ImM{icor,1},T)] ;
            end
            [MaxTh,B] = max(xn);
            out=foldernamesF{B(1)}; % ocr
            if B(1)>9
                selectClass = 1;
            else
                selectClass = 2;
            end
            information = [ratio1 ratio2 MaxTh selectClass];
            Pos(1) = SegmentInfo(j).BoundingBox(1) + SegmentInfo(j).BoundingBox(3)/2;
            Pos(2) = SegmentInfo(j).BoundingBox(2) + SegmentInfo(j).BoundingBox(4)/2;
            if (MaxTh>acceptTH)
                if display
                    subplot(2,1,2)
                    cla
                    imshow(imNUM,[]);
                    title(['Ocr is: ',out,'  Val is: ',num2str(MaxTh)]);
                end
                id = id + 1;
                data.Area = SegmentInfo(j).Area;
                data.leval = level;
                data.flag = B(1);
                data.Ocr = out;
                data.Val = MaxTh;
                data.box = SegmentInfo(j).BoundingBox;
                data.Center = Pos;
                data.PixelList = SegmentInfo(j).PixelList;
                data.ImageTh = imtemp;
                data.Intnsifiy = i;
                data.feat = information;
                data.Cat = selectClass; %% if 1 is char if 2 is number
                Nesbat = ImM{B(1),3};
                Accept = (sum(min([Nesbat;Rait]) ./ max([Nesbat;Rait])))/2;
                data.AcceptRait = Accept;
                [TimeStep,tree] = CallTree(tree,data,TimeStep,th_Hamposhani,th_Max_Hamposhani,id);
            end
        end
    end
    level = level + 1;
end

TimeStep(6:8,:) = 0;
[tree] = TraceTree(tree,TimeStep);
[mserStats,tree,Fimage,continueFlag] = PlateSegmentFind(tree,I);
if continueFlag
    %% is ok
else
    mserStats=[];
    tree=[];
    Fimage=[];
end

end

function [tree] = TraceTree(tree,TimeStep)

global display
if display
    flag = vertcat(tree.parent);
    figure
    [x,y,~] = treeplot_p(flag');
    for i = 2 : size(TimeStep,2)
        s = sprintf('  %s - %.2f - %.2f',tree(i).Ocr,tree(i).Val,tree(i).Hight);
        %s = sprintf('%d',tree(i).id);
        %s = sprintf('%s-%0.2f',tree(i).Ocr,tree(i).Val);
        text(x(i),y(i),s);
    end
end
ValFather = TimeStep(5,1);
IdFather = 1;
[TimeStep_new] = PruneTree(TimeStep,ValFather,1,IdFather,tree);
ind = TimeStep_new(7,:)~=0;
tree = tree(ind);
end

function [TimeStep] = PruneTree(TimeStep,ValFather,IdFather,id,tree);
global TH_Ocr
index = tree(id).Childs;
if IdFather==27
    ff = 0;
end
for ii = 1 : size(index,2)
    if (ValFather>(TH_Ocr * TimeStep(5,index(ii)))) % yani pedar balatareh
        [TimeStep] = PruneTree(TimeStep,ValFather,IdFather,index(ii),tree);
    else % farzand balatareh
        TimeStep(7,IdFather) = 0;
        TimeStep(8,IdFather) = 0;
        TimeStep(7,index(ii)) = index(ii);
        TimeStep(8,index(ii)) = TimeStep(5,index(ii));
        [TimeStep] = PruneTree(TimeStep,TimeStep(5,index(ii)),index(ii),index(ii),tree);
    end
end

end

function  [mserStats,Image,bb,continueFlag] = PlateSegmentFind(Image,I)

load Deep
load ImM
load foldernamesNN
mserStats = [];
bb = [];
global imsize Hborder thSelectH thRait thWidth
[teta,selectTochangeDeger,continueFlag] = FindDeger(Image);
if continueFlag
    tform = affine2d([cosd(teta) -sind(teta) 0; sind(teta) cosd(teta) 0; 0 0 1]);
    [bb,ref] = imwarp(I,tform);
    % first change box of image
    add = 1;
    for i = 1 : size(Image,2)
        [x1,y1]=transformPointsForward(tform,Image(i).PixelList(:,1),Image(i).PixelList(:,2));
        x1 = x1 - ref.XWorldLimits(1);
        y1 = y1 - ref.YWorldLimits(1);
        Image(i).PixelList = [x1 y1];
        x1 = round(x1);
        y1 = round(y1);
        x = min(x1);
        y = min(y1);
        w = max(x1) - x;
        h = max(y1) - y;
        
        Image(i).box = [x y w h];
        Image(i).Center = [x + round(w/2) y + round(h/2)];
        im2 = imwarp(Image(i).ImageTh,tform);
        im2=centersquare(im2,imsize);
        T = (im2);
        [trueclass,B] = DeepLerningTensor(T,deep);
        Image(i).ImageTh = im2;
        if (B~=25)
            Image(i).OcrIndex = B;
            Image(i).Ocr =foldernamesNN{B};
            Image(i).Val = trueclass;
            if B>9
                Image(i).Cat = 1;
            else
                Image(i).Cat = 2;
            end
            Rait = [h/w w/h];
            Nesbat = ImM{B(1),3};
            Accept = (sum(min([Nesbat;Rait]) ./ max([Nesbat;Rait])))/2;
            Image(i).AcceptRait = Accept;
            Final(add) = Image(i);
            add = add + 1;
        else
            in =  find(selectTochangeDeger==i);
            if size(in,1)>0
                selectTochangeDeger(in) = [];
            end
        end
    end
    %% if from best 3 selection
  
    box = vertcat(Image.box); 
    info.HightSelect  = mean(box(selectTochangeDeger,4));
    info.select       = selectTochangeDeger;
    info.M            = mean(box(selectTochangeDeger,4)).*thSelectH;
    info.HightM       = max(box(info.select,4)) + Hborder*max(box(info.select,4));
    info.HightMi      = min(box(info.select,2));
    info.templeatSize = [info.HightMi - Hborder*max(box(info.select,4)) info.HightMi + info.HightM];
    info.Width        = max(box(selectTochangeDeger,3)).*thWidth;
    info.Rait         = box(:,3).*box(:,4);
    info.MinVal       = min(info.Rait(selectTochangeDeger,:))*thRait;
    Image = Final; 
    Image = FilterImage(Image,info);
    Image = CallRemoveRepet(Image);
    mserStats = vertcat(Image.box);
    [~,index]= sort(mserStats(:,1));
    mserStats = mserStats(index,:);
    Image = Image(index);
end
end

function [teta,selectTochangeDeger,continueFlag] = FindDeger(Image)

teta = 0;
% first cheak ocr val
Cat = vertcat(Image.Cat);
indexC = find(Cat==2);
continueFlag = 0;
if size(indexC,1)>1
    %     Ones = vertcat(Image(indexC).flag);
    %     ind = Ones ~= 1;
    %     indexC = indexC(ind);
    infoFeat = vertcat(Image(indexC).feat);
    [~,index]= sort(infoFeat(:,3),'descend');
    index = indexC(index);
    if size(index,1)>10
        indexChange = index(1:10);
    else
        indexChange = index(1:end);
    end
    %% secend cheak ertfah
    continueFlag = 1;
    Box = vertcat(Image.box);
    [~,index]= sort(Box(indexChange,4),'descend');
    indexChange = indexChange(index);
    if size(indexChange,1)>5
        indexChange = indexChange(1:5);
    elseif (size(indexChange,1)==4 || size(indexChange,1)==3)
        indexChange = indexChange(1:3);
    elseif size(indexChange,1)<3
        continueFlag = 0;
    end
    %% new filter cheak y 96/08/30
    if continueFlag
        Center = vertcat(Image.Center);
        Center = Center(indexChange,:);
        D = zeros(size(indexChange,1),size(indexChange,1));
        for i = 1 : size(Center,1)
            for j = 1 : size(Center,1)
                D(i,j) = norm(Center(i,2)-Center(j,2));
            end
        end
        Dis = sum(D);
        [~,index] = sort(Dis);
        selectTochangeDeger = indexChange(index(1:3));
        Center    = vertcat(Image.Center);
        finddeger = Center(selectTochangeDeger,:);
        [~,ind]   = sort(finddeger(:,1));
        finddeger = finddeger(ind,:);
        x = finddeger(1,:);
        y = finddeger(end,:);
        m = (y(2) - x(2))/(y(1) - x(1));
        teta = atand(m);
        teta(isnan(teta)) = 0;
    end
end
end

function Image = FilterImage(Image,info)

index = [];
for i = 1 : size(Image,2)
    y = Image(i).box(2);
    h = Image(i).box(4);
    w = Image(i).box(3);
    %if (y>=templeatSize(1) && (y+h)<=templeatSize(2) &&  (Rait(i)>MinVal || M<h || (BestCenter>(Center(i,2)-h)) && (BestCenter<(Center(i,2)+h)))
    if (y>=info.templeatSize(1) && (y+h)<=info.templeatSize(2) && (h*w>info.MinVal || info.M<h) && (info.Width>=w || Image(i).Cat==1))
        index = [index i];
        Image(i).HightSelect = info.HightSelect;
    end
end
Image = Image(index);
box = vertcat(Image.box);
[~,index]= sort(box(:,1));
Image = Image(index);

end
function Image = CallRemoveRepet(Image)

feat = vertcat(Image.feat);
index = [];
for i = 1 : size(Image,2)
    for j = 1 : size(Image,2)
        if i~=j
            [~,~,~,zarib1,zarib2] = FindOverLap(Image(i).box,Image(j).box);
            if (zarib1>0.5 || zarib2>0.5)
                h1 = Image(i).box(4);
                w1 = Image(i).box(3);
                h2 = Image(j).box(4);
                w2 = Image(j).box(3);
                A1 = h1.*w1;
                A2 = h2.*w2;
                [valma,indma] = max([A1 A2]);
                [valmi,indmi] = min([A1 A2]);
                nesbat = valma/valmi;
                if (nesbat<2)
                    RaiteI = feat(i,3).*Image(i).Val;
                    RaiteJ = feat(j,3).*Image(j).Val;
                    if (RaiteI>=RaiteJ)
                        index = [index j];
                    else
                        index = [index i];
                        break;
                    end
                else
                    if indma==1
                        index = [index j];
                    else
                        index = [index i];
                    end
                end
            end
        end
    end
end
index = unique(index);
Image(index) = [];
end

function [TimeStep,tree] = CallTree(tree,data,TimeStep,th_Hamposhani,th_Max_Hamposhani,id)

flag = 1;
if size(TimeStep,2)==1
    [Overlap] = FindOverLap(tree(1).box,data.box);
    data.val = Overlap;
    Sz = size(TimeStep,2);
    tree(Sz+1).Childs = [];
    tree(Sz).Childs = [tree(Sz).Childs id];
    tree(Sz+1).id = id;
    tree(Sz+1).flag = data.flag;
    tree(Sz+1).Hight = data.box(4);
    tree(Sz+1).parent = TimeStep(1,1);
    tree(Sz+1).box=data.box;
    tree(Sz+1).leval=data.leval;
    tree(Sz+1).Ocr=data.Ocr;
    tree(Sz+1).Val=data.Val;
    tree(Sz+1).Center=data.Center;
    tree(Sz+1).PixelList=data.PixelList;
    tree(Sz+1).ImageTh=data.ImageTh;
    tree(Sz+1).Intnsifiy=data.Intnsifiy;
    tree(Sz+1).feat=data.feat;
    tree(Sz+1).Cat = data.Cat;
    tree(Sz+1).Overlap = Overlap;
    tree(Sz+1).AcceptRait = data.AcceptRait;
    %%
    TimeStep(1,Sz+1) = id;
    TimeStep(2,Sz+1) = TimeStep(1,1);
    TimeStep(3,Sz+1) = Overlap;
    TimeStep(4,Sz+1) = data.box(4);
    TimeStep(5,Sz+1) = data.Val;
else
    i = 1;
    ind = tree(i).Childs;
    cheakNode = ind;
    index = 0;
    while flag
        maxval = 0;
        for j = 1 : size(cheakNode,2)
            [Overlap] = FindOverLap(tree(cheakNode(j)).box,data.box);
            if maxval<Overlap
                maxval = Overlap;
                index = cheakNode(j);
            end
            if maxval>th_Max_Hamposhani
                break;
            end
        end
        if (maxval>th_Hamposhani )%&& data.box(4)>(tree(index).box(4) * thValedA2))
            i = index;
            ind = tree(i).Childs;
            cheakNode = ind;
            if size(cheakNode,2)<1
                data.val = maxval;
                Sz = size(TimeStep,2);
                tree(Sz+1).Childs = [];
                tree(i).Childs = [tree(i).Childs id];
                tree(Sz+1).id = id;
                tree(Sz+1).flag = data.flag;
                tree(Sz + 1).Hight = data.box(4);
                tree(Sz + 1).parent = TimeStep(1,i);
                tree(Sz+1).box=data.box;
                tree(Sz+1).leval=data.leval;
                tree(Sz+1).Ocr=data.Ocr;
                tree(Sz+1).Val=data.Val;
                tree(Sz+1).Center=data.Center;
                tree(Sz+1).PixelList=data.PixelList;
                tree(Sz+1).ImageTh=data.ImageTh;
                tree(Sz+1).Intnsifiy=data.Intnsifiy;
                tree(Sz+1).feat=data.feat;
                tree(Sz+1).Cat = data.Cat;
                tree(Sz+1).Overlap = maxval;
                TimeStep(1,Sz+1) = id;
                TimeStep(2,Sz+1) = TimeStep(1,i);
                TimeStep(3,Sz+1) = maxval;
                TimeStep(4,Sz+1) = data.box(4);
                TimeStep(5,Sz+1) = data.Val ;
                tree(Sz+1).AcceptRait = data.AcceptRait;
                flag = 0;
            end
        else
            data.val = maxval;
            Sz = size(TimeStep,2);
            tree(Sz+1).Childs = [];
            tree(i).Childs = [tree(i).Childs id];
            tree(Sz+1).id = id;
            tree(Sz+1).flag = data.flag;
            tree(Sz+1).parent = TimeStep(1,i);
            tree(Sz+1).box=data.box;
            tree(Sz+1).Hight = data.box(4);
            tree(Sz+1).leval=data.leval;
            tree(Sz+1).Ocr=data.Ocr;
            tree(Sz+1).Val=data.Val;
            tree(Sz+1).Center=data.Center;
            tree(Sz+1).PixelList=data.PixelList;
            tree(Sz+1).ImageTh=data.ImageTh;
            tree(Sz+1).Intnsifiy=data.Intnsifiy;
            tree(Sz+1).feat=data.feat;
            tree(Sz+1).Cat = data.Cat;
            tree(Sz+1).Overlap = maxval;
            TimeStep(1,Sz+1) = id;
            TimeStep(2,Sz+1) = TimeStep(1,i);
            TimeStep(3,Sz+1) = maxval;
            TimeStep(4,Sz+1) = data.box(4);
            TimeStep(5,Sz+1) = data.Val;
            tree(Sz+1).AcceptRait = data.AcceptRait;
            flag = 0;
        end
    end
end
Sz = size(TimeStep,2);
[tree,TimeStep] = UpdateTree(tree,TimeStep,Sz);
[tree,TimeStep] = UpdateOverLab(tree,TimeStep);
end

function [tree,TimeStep] = UpdateTree(tree,TimeStep,i)
EndNode = TimeStep(:,i);
ind = find(TimeStep(1,:)==EndNode(2));
FatherHight = TimeStep(4,ind);
if FatherHight>=EndNode(4)
    % is ok
else
    TimeStep(3:end,i) = TimeStep(3:end,ind);
    TimeStep(3:end,ind) = EndNode(3:end);
    
    tempind   = tree(ind);
    tempi     = tree(i);
    tree(ind) = tempi;
    tree(i)   = tempind;
    
    tree(ind).id     = tempind.id;
    tree(ind).parent = tempind.parent;
    tree(ind).Childs = tempind.Childs;
    
    tree(i).id     = tempi.id;
    tree(i).parent = tempi.parent;
    tree(i).Childs = tempi.Childs;
    
    
    [tree,TimeStep] = UpdateTree(tree,TimeStep,ind);
end
end

function [tree,TimeStep] = UpdateOverLab(tree,TimeStep)
for i = 2 : size(TimeStep,2)
    [Overlap] = FindOverLap(tree(TimeStep(1,i)).box,tree(TimeStep(2,i)).box);
    tree(i).Overlap = Overlap;
    TimeStep(3,i) = Overlap;
end
end

function  [zarib,A1,A2,zarib1,zarib2] = FindOverLap(PosI,PosJ);

x  = PosI;
x1 = PosJ;

area = rectint(x,x1);
zarib = (area)/max([x(3)*x(4)  x1(3)*x1(4)]);
zarib1 = area/(x(3)*x(4));
zarib2 = area/(x1(3)*x1(4));

A1 = PosI(4);
A2 = PosJ(4);

end

function [x,y,h] = treeplot_p(p,c,d)

[x,y,h]=treelayout(p);
f = find(p~=0);
pp = p(f);
X = [x(f); x(pp); NaN(size(f))];
Y = [y(f); y(pp); NaN(size(f))];

X = X(:);
Y = Y(:);

if nargin == 1
    n = length(p);
    if n < 500
        plot (x, y, 'ro', X, Y, 'g-');
    else
        plot (X, Y, 'r-');
    end
else
    [~, clen] = size(c);
    if nargin < 3
        if clen > 1
            d = [c(1:clen-1) '-'];
        else
            d = 'r-';
        end
    end
    [~, dlen] = size(d);
    if clen>0 && dlen>0
        plot (x, y, c, X, Y, d);
    elseif clen>0
        plot (x, y, c);
    elseif dlen>0
        plot (X, Y, d);
    else
    end
end
xlabel(['height = ' int2str(h)]);
axis([0 1 0 1]);
end

function CS=centersquare(I,OPTSIZE)

if nargin<2
    OPTSIZE=SQUARE_IMG_SIZE;
end

osz=size(I);
[ms]=max(osz);

sqim=zeros(ms);
y=floor((ms-osz(1))/2);
x=floor((ms-osz(2))/2);
sqim(1+y:y+osz(1),1+x:x+osz(2))=I;

smallersize=floor(OPTSIZE/1.3);
tmp=imresize(sqim,smallersize*ones(1,2),'nearest');

CS=zeros(OPTSIZE,OPTSIZE);
offset=floor((OPTSIZE-smallersize)/2);
range=1+offset:offset+smallersize;
CS(range,range)=tmp;

end


function image=averagefilter(image, varargin)

numvarargs = length(varargin);
if numvarargs > 2
    error('myfuns:somefun2Alt:TooManyInputs', ...
        'requires at most 2 optional inputs');
end

optargs = {[3 3] 0};            % set defaults for optional inputs
optargs(1:numvarargs) = varargin;
[window, padding] = optargs{:}; % use memorable variable names
m = window(1);
n = window(2);

if ~mod(m,2) m = m-1; end       % check for even window sizes
if ~mod(n,2) n = n-1; end

if (~ismatrix(image))            % check for color pictures
    display('The input image must be a two dimensional array.')
    display('Consider using rgb2gray or similar function.')
    return
end

% Initialization.
[rows, columns] = size(image);   % size of the image

% Pad the image.
imageP  = padarray(image, [(m+1)/2 (n+1)/2], padding, 'pre');
imagePP = padarray(imageP, [(m-1)/2 (n-1)/2], padding, 'post');

% Always use double because uint8 would be too small.
imageD = double(imagePP);

% Matrix 't' is the sum of numbers on the left and above the current cell.
t = cumsum(cumsum(imageD),2);

% Calculate the mean values from the look up table 't'.
imageI = t(1+m:rows+m, 1+n:columns+n) + t(1:rows, 1:columns)...
    - t(1+m:rows+m, 1:columns) - t(1:rows, 1+n:columns+n);

% Now each pixel contains sum of the window. But we want the average value.
imageI = imageI/(m*n);

% Return matrix in the original type class.
image = cast(imageI, class(image));
end

function output = niblack(image, varargin)

numvarargs = length(varargin);
if numvarargs > 4
    error('myfuns:somefun2Alt:TooManyInputs', ...
        'Possible parameters are: (image, [m n], k, offset, padding)');
end

optargs = {[3 3] -0.2 0 'replicate'};

optargs(1:numvarargs) = varargin;
[window, k, offset, padding] = optargs{:};

if ~ismatrix(image)
    error('The input image must be a two-dimensional array.');
end

% Convert to double
image = double(image);

% Mean value
mean = averagefilter(image, window, padding);

% Standard deviation
meanSquare = averagefilter(image.^2, window, padding);
deviation = (meanSquare - mean.^2).^0.5;

% Initialize the output
output = zeros(size(image));

% Niblack
output(image > mean + k * deviation - offset) = 1;
end

function [mu,v,p,prb]=MultiThroshuld(ima,k)

ima=double(ima);
ima=ima(:);
mi=min(ima);
ima=ima-mi+1;
m=max(ima);

h=HistP(ima);

x=find(h);
h=h(x);
x=x(:);
h=h(:);

mu=(1:k)*m/(k+1);
v=ones(1,k)*m;
p=ones(1,k)*1/k;

sml = mean(diff(x))/1000;
while(1)
    prb = Distrub(mu,v,p,x);
    scal = sum(prb,2)+eps;
    Like2=sum(h.*log(scal));
    
    for j=1:k
        pp=h.*prb(:,j)./scal;
        p(j) = sum(pp);
        mu(j) = sum(x.*pp)/p(j);
        vr = (x-mu(j));
        v(j)=sum(vr.*vr.*pp)/p(j)+sml;
    end
    p = p + 1e-3;
    p = p/sum(p);
    
    prb = Distrub(mu,v,p,x);
    scal = sum(prb,2)+eps;
    Like1=sum(h.*log(scal));
    if((Like1-Like2)<0.0001)
        break;
    end
%                 clf
%                 plot(x,h);
%                 hold on
%                 plot(x,prb,'g--')
%                 plot(x,sum(prb,2),'r')
%                 pause(0.001);
end
mu=mu+mi-1;

end

function y=Distrub(m,v,g,x)
x=x(:);
m=m(:);
v=v(:);
g=g(:);
for i=1:size(m,1)
    d = x-m(i);
    amp = g(i)/sqrt(1.2*pi*v(i));
    y(:,i) = amp*exp(-0.99 * (d.*d)/v(i));
end
end
function[h]=HistP(datos)
datos=datos(:);
ind= isnan(datos)==1;
datos(ind)=0;
ind= isinf(datos)==1;
datos(ind)=0;
tam=length(datos);
m=ceil(max(datos))+1;
h=zeros(1,m);
for i=1:tam
    f=floor(datos(i));
    if(f>0 && f<(m-1))
        a2=datos(i)-f;
        a1=1-a2;
        h(f)  =h(f)  + a1;
        h(f+1)=h(f+1)+ a2;
    end
end
h=conv(h,[1,2,3,2,1]);
h=h(3:(length(h)-2));
h=h/sum(h);
end



