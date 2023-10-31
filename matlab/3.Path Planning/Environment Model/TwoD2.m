N=20;
a=rand(N)>0.35;%生成0的数量不小于35%的N*Nlogical类型数据
figure
axis([0,N,0,N]) %设置图的横纵坐标，N为地图矩阵的行数或列数
for i=1:N 
for j=1:N 
if a(i,j)==1 %1为白色无障碍，0为黑色有障碍
x1=j-1;y1=N-i; %注意矩阵和栅格坐标表示转化
x2=j;y2=N-i; 
x3=j;y3=N-i+1; 
x4=j-1;y4=N-i+1; 
fill([x1,x2,x3,x4],[y1,y2,y3,y4],[1,1,1]); %将1234点所围成的图形进行白色填充
hold on 
else 
x1=j-1;y1=N-i; 
x2=j;y2=N-i; 
x3=j;y3=N-i+1; 
x4=j-1;y4=N-i+1; 
fill([x1,x2,x3,x4],[y1,y2,y3,y4],[0.2,0.2,0.2]); %将1234点所围成的图形进行黑色填充
hold on 
set(gca,'YTickLabel',[20 18 16 14 12 10 8 6 4 2 0]); %使得地图矩阵的行列与正常坐标轴的行列一致
end 
end 
end 
hold on 
