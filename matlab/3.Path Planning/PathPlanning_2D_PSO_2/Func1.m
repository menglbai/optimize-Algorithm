%% 计算航迹距离
function z=Func1(Path,Start,End) %Path是一个n行2列的矩阵，存放航迹节点坐标信息；Start和End都是1*2的坐标向量
a=max(size(Path));
z=distance(Path(1,:),Start)+distance(End,Path(a,:));
for j=1:a-1
    z=z+distance(Path(j,:),Path(j+1,:));
end
end