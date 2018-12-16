function  Image = Select8Char(Image)

Cat = vertcat(Image.Cat);
index = find(Cat==1);
i = 1;
removeList = [];
while (i<=size(index,1) && size(index,2)>0)
    ind =  index(i);
    if (ind>1 && ind<size(Cat,1))
        in1 = sum(Cat(1:ind-1)==2);
        in2 = sum(Cat(ind+1:end)==2);
        if (in1>=2 && in2>=4)
            % accept
            i = i + 1;
        else
            index(i) = [];
            removeList = [removeList ind];
            i = 1;
        end
    else
        %% reset index of image
        index(i) = [];
        removeList = [removeList ind];
        i = 1;
    end
end
Image(removeList) =[];
Cat = vertcat(Image.Cat);
index = find(Cat==1); 
if size(index,1)>1  % yani bish az 1 char darim dar segement
    % entekhab yeki az hameh
    % bar asaseh visheki nesbat
%     Raite = vertcat(Image.AcceptRait);
    feat = vertcat(Image.Val);
    %Raite = Raite.*feat;
    Raite = feat;
    Raite = Raite(index);
    [~,indexM] = max(Raite);
    index(indexM) = [];
    Image(index) = []; 
end
remove =[];
if size(Image,2)>8
    Cat = vertcat(Image.Cat);
    feat = vertcat(Image.Val);
    Raite = feat;
    index = find(Cat==1);
    in1 = sum(Cat(1:index-1)==2);
    in2 = sum(Cat(index+1:end)==2);
    if in1>2 %% problim is here
        [~,in] = sort(Raite(1:index-1),'descend');
        remove = in(3:end);
    end
    if in2>5
        [~,in] = sort(Raite(index+1:end),'descend');
        remove = [remove' (index + in(6:end))'];
    end
    Image(remove) = [];
end

end