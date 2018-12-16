function [IExpandedBBoxes] = ShowBound(mserStats,I,index)
temp = mserStats(index,:);
b = (1:size(mserStats,1));
bb1 = [];
for i = 1 : size(b,2)
    list = find(index==b(i));
    if size(list,2)<1
        bb1 = [bb1 i];
    end
end
bboxes1 = mserStats(bb1,:);

bboxes = (temp);
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

expansionAmount = 0.0;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;
 
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
IExpandedBBoxes = insertShape(I,'Rectangle',expandedBBoxes,'LineWidth',1,'color','g');

%%
bboxes = (bboxes1);
xmin = bboxes(:,1);
ymin = bboxes(:,2);
xmax = xmin + bboxes(:,3) - 1;
ymax = ymin + bboxes(:,4) - 1;

expansionAmount = 0.0;
xmin = (1-expansionAmount) * xmin;
ymin = (1-expansionAmount) * ymin;
xmax = (1+expansionAmount) * xmax;
ymax = (1+expansionAmount) * ymax;
 
xmin = max(xmin, 1);
ymin = max(ymin, 1);
xmax = min(xmax, size(I,2));
ymax = min(ymax, size(I,1));

expandedBBoxes = [xmin ymin xmax-xmin+1 ymax-ymin+1];
IExpandedBBoxes = insertShape(IExpandedBBoxes,'Rectangle',expandedBBoxes,'LineWidth',1,'color','r');

%%

end
