%% 例2 空袭目标威胁评估
clear;
clc;
%% 加载目标威胁评估数据
filename='D:\matlab\ProjectionPursuit\PP_PSO\threat_data.xlsx'; %威胁数据
sheet=4;
A=xlsread(filename,sheet);
format short
% 规范决策矩阵
for j=1:size(A,2)
    Max(j)=max(A(:,j));
    Min(j)=min(A(:,j));
    Max_Min(j)=Max(j)-Min(j);
end
for i=1:size(A,1)
    for j=1:size(A,2)
        A(i,j)=(A(i,j)-Min(j))/Max_Min(j);
    end
end
%% PSO算法参数初始化
N=30;                    	%群体粒子个数
D=size(A,2);                %粒子维数
Nmax=500;                   %最大迭代次数
c1=1.5;                 	%学习因子1
c2=1.5;                 	%学习因子2
Wmax=0.8;               	%惯性权重最大值
Wmin=0.4;               	%惯性权重最小值
Xmax=1;                 	%位置最大值
Xmin=-1;                 	%位置最小值
Vmax=0.1;                 	%速度最大值
Vmin=-0.1;                	%速度最小值
Ntest=30;                   %实验次数
test.g=[];
test.gbest=[];
%Initialize Population Array
pop=repmat(test,Ntest,1);
for jj=1:Ntest
%初始化种群个体位置和速度
x=rand(N,D)*(Xmax-Xmin)+Xmin;
v=rand(N,D)*(Vmax-Vmin)+Vmin;
%初始化个体最优位置和最优值
p=x;
pbest=ones(N,1);
for i=1:N
    pbest(i)=SD_penaltyfunction(A,x(i,:));
end
%初始化全局最优位置和最优值
g=ones(1,D);
gbest=inf;
for i=1:N
    if(pbest(i)<gbest)
        g=p(i,:);
        gbest=pbest(i);
    end
end
gb=ones(1,Nmax);%每次迭代的最优解
%% 按照公式迭代直到满足精度或者最大迭代次数
for i=1:Nmax
    for j=1:N
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%
        if(SD_penaltyfunction(A,x(j,:))<pbest(j))
            p(j,:)=x(j,:);
            pbest(j)=SD_penaltyfunction(A,x(j,:));
        end
        %%%%%%%%%%更新全局最优位置和最优值%%%%%%%%%
        if(pbest(j)<gbest)
            g=p(j,:);
            gbest=pbest(j);
        end
        %%%%%%%%%%计算动态惯性权重值%%%%%%%%%%%
        w=Wmax-(Wmax-Wmin)*i/Nmax;
        %%%%%%%%%%更新位置和速度值%%%%%%%%%%%%%
        v(j,:)=w*v(j,:)+c1*rand*(p(j,:)-x(j,:))+c2*rand*(g-x(j,:));
        x(j,:)=x(j,:)+v(j,:);
        %%%%%%%%%%%边界条件处理%%%%%%%%%%%%%%%
        for ii=1:D
            if (v(j,ii)>Vmax)||(v(j,ii)< Vmin)
                v(j,ii)=rand*(Vmax-Vmin)+Vmin;
            end
            if (x(j,ii)>Xmax)||(x(j,ii)< Xmin)
                x(j,ii)=rand*(Xmax-Xmin)+Xmin;
            end
        end
    end
    %%%%%%%%%%%%记录历代全局最优值%%%%%%%%%%%%%
    gb(i)=gbest;
end
pop(jj).g=g;
pop(jj).gbest=gbest;
end
Gbest=inf;
N_local=0;
for i=1:Ntest
    if pop(i).gbest<Gbest
        Gbest=pop(i).gbest;
    end
end
for index=1:Ntest
    if pop(index).gbest==Gbest
        G=abs(pop(index).g);%最佳投影方向
    break
    end
end
gg=sum(G.^2); %验证模长是否等于1
AA=A*G';
[B,I]=sort(AA,'descend');%目标按威胁程度从大到小排序
for k=1:Ntest
    if floor(pop(k).gbest*10000)>floor(Gbest*10000)
        N_local=N_local+1;
    end
end
figure(1)
plot(gb)
xlabel('迭代次数');
ylabel('适应度值');
title('适应度进化曲线')

figure(2)
G;
TPDA=[0.3034 0.6097 0.5331 0.5102];
MDM=[0.1265 0.5055 0.0271 0.3409];
Y=[];
for i=1:4
    Y(i,1)=G(i);
    Y(i,2)=TPDA(i);
    Y(i,3)=MDM(i);
end
X=1:4;
%画出4组柱状图，宽度1
h=bar(X,Y,1);      
%修改横坐标名称、字体
set(gca,'XTickLabel',{'目标类型','飞行高度','飞行速度','飞临时间'},'FontSize',12,'FontName','宋体');
% 设置柱子颜色,颜色为RGB三原色，每个值在0~1之间即可
set(h(1),'FaceColor',[30,150,252]/255)     
set(h(2),'FaceColor',[162,214,249]/255)    
set(h(3),'FaceColor',[252,243,0]/255)    
% set(h(4),'FaceColor',[255,198,0]/255)    
ylim([0 0.7]);      %y轴刻度
%修改x,y轴标签
ylabel('\fontname{宋体}\fontsize{16}权重');
xlabel('\fontname{宋体}\fontsize{16}指标'); 
%修改图例
legend({'\fontname{Times New Roman}PSO','\fontname{Times New Roman}TPDA','\fontname{Times New Roman}MDM'},'FontSize',12);

MDM_res=[0.6664 0.1935 0.6658 0.1097];
TPDA_AA=[1.9563 0.4038 1.9547 0.3894];
figure(3)
YY=[];
for i=1:4
    YY(i,1)=AA(i); 
    YY(i,2)=TPDA_AA(i);
    YY(i,3)=MDM_res(i);
end
X=1:4;
%画出4组柱状图，宽度1
h=bar(X,YY,1);      
%修改横坐标名称、字体
set(gca,'XTickLabel',{'A','B','C','D'},'FontSize',12,'FontName','Times New Roman');
% 设置柱子颜色,颜色为RGB三原色，每个值在0~1之间即可
set(h(1),'FaceColor',[30,150,252]/255)     
set(h(2),'FaceColor',[162,214,249]/255)    
set(h(3),'FaceColor',[252,243,0]/255)    
% set(h(4),'FaceColor',[255,198,0]/255)    
ylim([0 2]);      %y轴刻度
%修改x,y轴标签
ylabel('\fontname{宋体}\fontsize{16}威胁度');
xlabel('\fontname{宋体}\fontsize{16}目标'); 
%修改图例
legend({'\fontname{Times New Roman}PSO','\fontname{Times New Roman}TPDA','\fontname{Times New Roman}MDM'},'FontSize',12);