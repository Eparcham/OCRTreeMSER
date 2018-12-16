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





