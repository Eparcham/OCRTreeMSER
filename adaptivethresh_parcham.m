function bw = adaptivethresh_parcham(im)

fsize = [42];
t     = [1];
add = 1;
for i = 1 : size(fsize,2)
    g = fspecial('gaussian',fsize(i), fsize(i));
    fim = filter2(g, im);
    for j = 1 : size(t,2)
        bw{add,1} = im >( fim-t(j));
%         figure
%         imshow(1-bw{add,1},[]);
%         title(['filter size: ',num2str(fsize(i)),' th: ',num2str(t(j))]);
        add = add + 1;
    end
end

% fsize = [31 41 51 61 71 81 91];
% t = [4];
% for i = 1 : size(fsize,2)
%     g = fspecial('gaussian',2*fsize(i), fsize(i));
%     fim = filter2(g, im);
%     for j = 1 : size(t,2)
%         bw{add,1} = im >( fim-t(j));
% %         figure
% %         imshow(1-bw{add,1},[]);
% %         title(['filter size: ',num2str(fsize(i)),' th: ',num2str(t(j))]);
%         add = add + 1;
%     end
% end

