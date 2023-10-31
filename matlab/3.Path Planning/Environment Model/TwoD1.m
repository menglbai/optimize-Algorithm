% a = load('environment.txt');%blacb--barrier occputy 35%.
% a = load('environment6060.mat', 'k');
% a = a.k;
a=rand(10)>0.35; %0的数量不小于35%的20*20logical
n=size(a,1);%获取行数
b=a;
b(end+1,end+1)=0;%pcolor不处理最后一行和最后一列，故这里补一行和一列
figure;
pcolor(b); % 赋予栅格颜色
colormap([0 0 0;1 1 1])%设置颜色[0 0 0]是黑色[1 1 1]是白色
%%
set(gca,'XTick',[],'YTick',[]);  % 设置坐标
axis image xy
 
displayNum(n);%显示栅格中的数值
 
text(1,n+1.5,'START','Color','red','FontSize',10);%显示start字符
text(n+1,1.5,'GOAL','Color','red','FontSize',10);%显示goal字符
 
hold on
%pin strat&goal positon
scatter(1+0.75,n+0.5,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0], 'LineWidth',0.05);%start point
scatter(n+0.75,1+0.5,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0], 'LineWidth',0.05);%goal point
 
hold on