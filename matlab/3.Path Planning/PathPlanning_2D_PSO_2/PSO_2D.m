%% 不失一般性，给定起始点、目标点以及障碍信息，总可以构建一个以起始点和目标点连线为x轴的二维平面坐标系。
clear
clc
close
%% 构建环境地图
%% 威胁源信息
A=[25,40,60,80];%横坐标
B=[0,30,-20,15];%纵坐标
R=[10,12,18,10];%威胁半径
theta=0:pi/100:2*pi;%把0-2pi分成200份
color=['r','b','c','y'];%给威胁区域上色
for j=1:size(A,2)
    x=A(j)+R(j)*cos(theta);
    y=B(j)+R(j)*sin(theta);
    plot(x,y);
    fill(x,y,color(j));
    hold on
end
axis equal           %将横轴纵轴的定标系数设成相同值 ,即单位长度相同
axis([0,100,-50,50]); 
xlabel('x km');
ylabel('y km');
%% 绘制起始点和目标点，这里以起始点和目标点为横轴
Start=[0,0];
End=[100,0];
DSE=sqrt((Start(1)-End(1))^2+(Start(2)-End(2))^2);
plot(Start(1),Start(2),'r*');
plot(End(1),End(2),'bd');
text(Start(1)+1,Start(2),'START');
text(End(1)+1,End(2),'GOAL');
%% 初始化参数
L=20;                   	%群体粒子个数，每个粒子代表一条航迹
ndot=12;                    %粒子维数，一条航迹中航迹节点的个数-1
Nmax=100;                   %最大迭代次数
c1=1.5;                 	%学习因子1
c2=1.5;                 	%学习因子2
Wmax=0.9;               	%惯性权重最大值
Wmin=0.4;               	%惯性权重最小值
Xmax=50;                 	%位置最大值，也是x轴上限
Xmin=-50;                  	%位置最小值，也是x轴下限
K=0.1;                      %比例系数
Vmax=K*Xmax;                %速度最大值
Vmin=K*Xmin;                %速度最小值
%% 种群初始化
for j=1:L
    for i=1:ndot+1
    Path(i,1,j)=100/(ndot+2)*i;       %航迹点的横坐标相对明确，将横轴(ndot+2)等分
    Path(i,2,j)=rand*(Xmax-Xmin)+Xmin;%航迹点的纵坐标随机生成
    end
end
V=rand(ndot+1,1,L)*(Vmax-Vmin)+Vmin;  %速度用于调整纵坐标
%初始化个体最优位置（航迹）和最优值（航迹长度）
P=Path;
pbest=ones(L,1);
for i=1:L
    pbest(i)=Func1(Path(:,:,i),Start,End);%Path(:,:,i)代表航迹节点
end
%初始化全局最优位置（航迹）和最优值（航迹长度）
g=ones(ndot,2);
gbest=inf;
for i=1:L
    if(pbest(i)<gbest)
        g=P(:,:,i);
        gbest=pbest(i);
    end
end
gb=ones(1,Nmax);%用于存储每次迭代的最优解
%% 按照公式依次迭代直到满足精度或者迭代次数
for i=1:Nmax
    for j=1:L
        %更新个体最优位置和最优值
        if Func1(Path(:,:,j),Start,End)<pbest(j)
            P(:,:,j)=Path(:,:,j);
            pbest(j)=Func1(Path(:,:,j),Start,End);
        end
        %更新全局最优位置和最优值
        if(pbest(j)<gbest)
            g=P(:,:,j);
            gbest=pbest(j);
        end
        %计算动态惯性权重值
        w=Wmax-(Wmax-Wmin)*i/Nmax;
        %更新位置和速度值，主要更新纵坐标
        V(:,1,j)=w*V(:,1,j)+c1*rand*(P(:,2,j)-Path(:,2,j))+c2*rand*(g(:,2)-Path(:,2,j));
        Path(:,2,j)=Path(:,2,j)+V(:,1,j);
        %边界条件处理
        for ii=1:ndot+1
            if (V(ii,1,j)>Vmax)||(V(ii,1,j)<Vmin)
                V(ii,1,j)=rand*(Vmax-Vmin)+Vmin;
            end
            if (Path(ii,2,j)>Xmax)||(Path(ii,2,j)<Xmin)
                Path(ii,2,j)=rand*(Xmax-Xmin)+Xmin;
            end
        end
        %如果航迹点落在威胁区域内则重新随机纵坐标
        for k=1:ndot+1
            for l=1:max(size(R))
                while (Path(k,1,j)-A(l))^2+(Path(k,2,j)-B(l))^2<=R(l)^2 
                    Path(k,2,j)=rand*(Xmax-Xmin)+Xmin;
                end
            end            
        end
        %对相邻航迹点之间的航迹进行碰撞检测
        for k=1:ndot
            for l=1:max(size(R))
%                 while ((Path(k,1,j)/5+Path(k+1,1,j)*4/5)-A(l))^2+((Path(k,2,j)/5+Path(k+1,2,j)*4/5)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*2/5+Path(k+1,1,j)*3/5)-A(l))^2+((Path(k,2,j)*2/5+Path(k+1,2,j)*3/5)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*3/5+Path(k+1,1,j)*2/5)-A(l))^2+((Path(k,2,j)*3/5+Path(k+1,2,j)*2/5)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*4/5+Path(k+1,1,j)/5)-A(l))^2+((Path(k,2,j)*4/5+Path(k+1,2,j)/5)-B(l))^2<=R(l)^2
%                     Path(k+1,2,j)=rand*(Xmax-Xmin)+Xmin;
%                 end         
               while ((Path(k,1,j)/3+Path(k+1,1,j)*2/3)-A(l))^2+((Path(k,2,j)/3+Path(k+1,2,j)*2/3)-B(l))^2<=R(l)^2 ...
                        || ((Path(k,1,j)*2/3+Path(k+1,1,j)/3)-A(l))^2+((Path(k,2,j)*2/3+Path(k+1,2,j)/3)-B(l))^2<=R(l)^2
                    Path(k+1,2,j)=rand*(Xmax-Xmin)+Xmin;
                end
%                 while ((Path(k,1,j)/4+Path(k+1,1,j)*3/4)-A(l))^2+((Path(k,2,j)/4+Path(k+1,2,j)*3/4)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*2/4+Path(k+1,1,j)/2)-A(l))^2+((Path(k,2,j)*2/4+Path(k+1,2,j)/2)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*3/4+Path(k+1,1,j)/4)-A(l))^2+((Path(k,2,j)*3/4+Path(k+1,2,j)/4)-B(l))^2<=R(l)^2 
%                     Path(k+1,2,j)=rand*(Xmax-Xmin)+Xmin;
%                 end
            end
        end 
    end
    %%%%%%%%%%%%记录历代全局最优值%%%%%%%%%%%%%
    gb(i)=gbest;
end
%如何解决两点连线和障碍物有交集问题以及初始点生成位置是否有待考量
g; %最优航迹
h(1,:)=Start;
h(ndot+3,:)=End;
h(2:ndot+2,:)=g;
plot(h(:,1),h(:,2),'rp');%绘制航迹点
hh=0:0.1:100;
HH=spline(h(:,1),h(:,2),hh);%三次样条插值得曲线
plot(hh,HH,'-');
figure(2)
plot(gb)
xlabel('迭代次数');
ylabel('适应度值');
title('适应度进化曲线');