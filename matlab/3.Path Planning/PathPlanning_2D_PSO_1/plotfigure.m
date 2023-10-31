figure(1)
a=[20,40,60,85];%横坐标
b=[0,20,-10,5];%纵坐标
R=[10,12,18,10];%威胁半径
theta=0:pi/100:2*pi;%把0-2pi分成200份
Name=['A','B','C','D'];%给威胁源命名
color=['r','b','c','y'];%给威胁区域上色
for j=1:size(a,2)
    x=a(j)+R(j)*cos(theta);
    y=b(j)+R(j)*sin(theta);
    plot(x,y);
    p(j)=fill(x,y,color(j),'DisplayName',Name(j));
    hold on
end
legend([p(1),p(2),p(3),p(4)])
axis equal           %将横轴纵轴的定标系数设成相同值 ,即单位长度相同
axis([0,100,-50,50]); 
title('\fontname{Times New Roman}PSO\fontname{宋体}规划结果','FontSize',15)
xlabel('x km','FontSize',15,'FontName','Times New Roman');
ylabel('y km','FontSize',15,'FontName','Times New Roman');
set(gca,'FontSize',12,'FontName','Times New Roman');
%% 绘制起始点和目标点，这里以起始点和目标点为横轴
Start=[0,0];
End=[100,0];
DSE=sqrt((Start(1)-End(1))^2+(Start(2)-End(2))^2);
plot(Start(1),Start(2),'r*','HandleVisibility','off');
plot(End(1),End(2),'bd','HandleVisibility','off');
text(Start(1)+1,Start(2),'START','FontName','Times New Roman');
text(End(1)+1,End(2),'GOAL','FontName','Times New Roman');
col=30;
for j=1:col
    X(j)=j*(End(1)-Start(1))/(col+1);
end
g(:,2)=Result_PSO(3).sol;
g(:,1)=X';
g; %最优航迹
h(1,:)=Start;
h(col+2,:)=End;
h(2:col+1,:)=g;
p(5)=plot(h(:,1),h(:,2),'rp','HandleVisibility','off');%绘制航迹点
hh=0:0.1:100;
HH=spline(h(:,1),h(:,2),hh);%三次样条插值得曲线
plot(hh,HH,'-','HandleVisibility','off');
hold on
figure(2)
S=log10(cg_PSO(3).cg);
plot(S);
title('\fontname{Times New Roman}PSO\fontname{宋体}收敛曲线','FontSize',15);
xlabel('\fontname{宋体}迭代次数','FontSize',15);
ylabel('\fontname{宋体}适应度值','FontSize',15);
set(gca,'FontSize',12,'FontName','Times New Roman');