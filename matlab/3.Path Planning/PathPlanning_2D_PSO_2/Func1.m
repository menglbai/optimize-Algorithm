%% ���㺽������
function z=Func1(Path,Start,End) %Path��һ��n��2�еľ��󣬴�ź����ڵ�������Ϣ��Start��End����1*2����������
a=max(size(Path));
z=distance(Path(1,:),Start)+distance(End,Path(a,:));
for j=1:a-1
    z=z+distance(Path(j,:),Path(j+1,:));
end
end