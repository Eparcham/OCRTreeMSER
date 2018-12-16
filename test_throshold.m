function h = test_throshold(im)

j = HistP(im(:));
% figure
% plot(j)
b = medfilt1(j,30);
% hold on
% plot(b,'-r')
%% find shib in plot
m = zeros(size(b,2),2);
for i = 2 : size(b,2)
    m(i,1) = (b(i)-b(i-1));
    m(i,2) = i;
end

mf = zeros(size(m,1),2);
for i = 2 : size(m,1)
    mf(i,1) = (m(i,1)-m(i-1,1));
    mf(i,2) = i;
end
ind = abs(mf(:,1))<1;
add = 2;
hist(1,:) = mf(1,:);
th = 1;
for i = 3 : size(ind,1) 
    if ind(i)==1
        if abs(hist(add-1,2)-mf(i,2))>th
            hist(add,:) = mf(i,:);
            add = add + 1;
        end
    end
end
h = hist(:,2);
%%   
% for i = 1 : size(hist,1)
%     im1 = logical((1-((im)>hist(i,2))));
%     figure(2)
%     imshow(im1,[]);
%     pause(1)
% end 
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
h=h(20:(length(h)-20));
end