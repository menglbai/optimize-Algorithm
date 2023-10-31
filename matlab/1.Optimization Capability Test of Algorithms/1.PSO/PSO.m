% 惯性权重线性下降(Linearly Decreasing Inertia Weight, LDW)
function [Best_score,Best_pos,cg_curve]=PSO(N,Max_iteration,Xmin,Xmax,dim,fobj)
%convergence curve收敛曲线
cg_curve=zeros(1,Max_iteration); %历次迭代最优解
%初始化参数
% N=150;                    %粒子个数
% dim=2;                    %粒子维数
% Nmax=1000;                %最大迭代次数
c1=1.5;                 	%学习因子1
c2=1.5;                 	%学习因子2
Wmax=0.8;               	%惯性权重最大值
Wmin=0.4;               	%惯性权重最小值
% Xmax=500;                 %位置最大值
% Xmin=-500;                %位置最小值
k=(rand+1)*0.1;
Vmax=k*Xmax;                %速度最大值
Vmin=-Vmax;                	%速度最小值
%% 初始化种群个体位置和速度
x=rand(N,dim)*(Xmax-Xmin)+Xmin;
v=rand(N,dim)*(Vmax-Vmin)+Vmin;
p=x;
pbest=ones(N,1);
% 计算初始群体的函数值
for i=1:N
    pbest(i)=fobj(x(i,:));
end
% 最优解和最优值
Best_pos=ones(1,dim);
Best_score=inf;
for i=1:N
    if(pbest(i)<Best_score)
        Best_pos=p(i,:);
        Best_score=pbest(i);
    end
end
%% 按照公式依次迭代直到满足精度或者迭代次数
for i=1:Max_iteration
    for j=1:N
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%
        if(fobj(x(j,:))<pbest(j))
            p(j,:)=x(j,:);
            pbest(j)=fobj(x(j,:));
        end
        %%%%%%%%%%更新全局最优位置和最优值%%%%%%%%%
        if(pbest(j)<Best_score)
            Best_pos=p(j,:);
            Best_score=pbest(j);
        end
        %%%%%%%%%%计算动态惯性权重值%%%%%%%%%%%
        w=Wmax-(Wmax-Wmin)*i/Max_iteration;
%       w=0.6;
        %%%%%%%%%%更新位置和速度值%%%%%%%%%%%%%
        v(j,:)=w*v(j,:)+c1*rand*(p(j,:)-x(j,:))+c2*rand*(Best_pos-x(j,:));
        x(j,:)=x(j,:)+v(j,:);
        %%%%%%%%%%%边界条件处理%%%%%%%%%%%%%%%
        for ii=1:dim
            if (v(j,ii)>Vmax)||(v(j,ii)< Vmin)
                v(j,ii)=rand*(Vmax-Vmin)+Vmin;
            end
            if (x(j,ii)>Xmax)||(x(j,ii)< Xmin)
                x(j,ii)=rand*(Xmax-Xmin)+Xmin;
            end
        end
    end
    %%%%%%%%%%%%记录历代全局最优值%%%%%%%%%%%%%
    cg_curve(i)=Best_score;
end
% figure(1)
% plot(gb)
% xlabel('迭代次数');
% ylabel('适应度值');
% title('适应度进化曲线')
end