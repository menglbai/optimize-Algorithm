%% PathPlanning 3D TPDA
clc
clear
close all
%% 三维路径规划模型定义
S = [1, 1, 1];    %起始点
T = [100, 100, 40];%目标点
% 生成山峰地图
mapRange = [100,100,100];% 地图长、宽、高范围
[XX,Y,Z] = defMap(mapRange);
%% 初始参数设置
N = 500;           % 最大迭代次数
M = 30;            % 蜻蜓个体数量
pointNum = 5;      % 每一个个体包含的位置点数量
lb=0;              % 位置下限
ub=100;            % 位置上限
%% 种群初始化 矩阵X(i).pos为蜻蜓个体
X.pos=rand(pointNum,3);
X.fitness=0;
X.DeltaX=[];
X.path=[];
X=repmat(X,M,1);
for i=2:M
    for j=1:pointNum*3
        if X(i-1).pos(j)>0.5
            X(i).pos(j)=2-2*X(i-1).pos(j)+rand/(100*M);
        end
        if X(i-1).pos(j)<=0.5 && X(i-1).pos(j)>=0
            X(i).pos(j)=2*X(i).pos(j)+rand/(100*M);
        end
    end
end
for i=1:M
    for j=1:pointNum*3
        if X(i).pos(j)>1
            X(i).pos(j)=1;
        end
        if X(i).pos(j)<0
            X(i).pos(j)=0;
        end
    end
    X(i).pos=100*X(i).pos;
end
clear i 
clear j 
clear k 
for i=1:M
    X(i).fitness=d(S,X(i).pos(1,:))+d(X(i).pos(pointNum,:),T);
    for j=1:(pointNum-1)
        X(i).fitness=X(i).fitness+d(X(i).pos(j,:),X(i).pos(j+1,:));
    end
end
%% 步长向量
for i=1:M
    X(i).DeltaX=100*rand(pointNum,3);
end
cg_curve=zeros(1,N);
if size(ub,2)==1
    ub=ones(pointNum,3)*ub;
    lb=ones(pointNum,3)*lb;
end
Delta_max=(ub-lb)/10;
% 初始化食物位置（最优值）
Food.fitness=inf;
Food.pos=[];
% 初始化天敌位置（最差值）
Enemy.fitness=-inf;
Enemy.pos=[];
Fitness=zeros(1,M);% 适应度值
P=X;
% for i=1:M
%     P(i).fitness=X(i).fitness;
% end
for i=1:M
[flag,fitness,path] = calFitness(S,T,XX,Y,Z,X(i).pos);
  % 碰撞检测判断
    if flag == 1
        % 若flag=1，表明此路径将与障碍物相交，则增大适应度值
        X(i).fitness = 1000*fitness;
        X(i).path = path;
    else
        % 否则，表明可以选择此路径
        X(i).fitness = fitness;
        X(i).path = path;
    end
end
    
%% 开始迭代
for iter=1:N
    % 更新搜索半径r
    r=(ub-lb)/4+((ub-lb)*(iter/N)*2);
    % 更新各权重系数
    w=0.9-iter*(0.5/N);  
    my_c=0.1-iter*(0.1/(N/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %分离度 
    a=2*rand*my_c; %对齐度 
    c=2*rand*my_c; %内聚度 
    f=2*rand;      %食物吸引力 
    e=my_c;        %敌排斥力 
    %找到食物和天敌
    for i=1:M %首先计算所有个体适应度值
        [flag,fitness,path] = calFitness(S,T,XX,Y,Z,X(i).pos);
  % 碰撞检测判断
    if flag == 1
        % 若flag=1，表明此路径将与障碍物相交，则增大适应度值
        X(i).fitness = 1000*fitness;
        X(i).path = path;
    else
        % 否则，表明可以选择此路径
        X(i).fitness = fitness;
        X(i).path = path;
    end
        Fitness(1,i)=Length(X(i).pos);
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%
        if Length(X(i).pos)<P(i).fitness
            P(i).pos=X(i).pos;
            P(i).fitness=Length(X(i).pos);
        end
        % 计算二阶原点矩
        E_f=sum(Fitness(1,i))/M;
        D_f=sum(Fitness(1,i)-E_f)^2/(M-1);
        E_ff=E_f^2+D_f;
        if Length(X(i).pos)<Food.fitness %寻找每次迭代的最小值
            Food.fitness=Length(X(i).pos);%寻找全局最优位置和全局最优值
            Food.pos=X(i).pos;
        end
        if Length(X(i).pos)>Enemy.fitness %寻找每次迭代的最大值
            if all(all(X(i).pos<ub)) && all(all(X(i).pos>lb))
                Enemy.fitness=Length(X(i).pos);
                Enemy.pos=X(i).pos;
            end
        end
    end    
    %找到每只蜻蜓的邻居
    for i=1:M
        index=0;
        neighbours_num=0;        
%         clear Neighbours_DeltaX
        clear Neighbours_X
        %找到相邻邻居
        for j=1:M
            Dist2Enemy=distance_3D(X(i).pos,X(j).pos);%计算欧氏距离
            if all(all(Dist2Enemy<=r)) && all(all(Dist2Enemy~=0))
                index=index+1;%邻居序号
                neighbours_num=neighbours_num+1;%邻居数量
                Neighbours_X(index).DeltaX=X(j).DeltaX;
                Neighbours_X(index).pos=X(j).pos;
            end
        end        
        % 分离
        S=zeros(pointNum,3);
        if neighbours_num>1
            for k=1:neighbours_num
                S=S+(Neighbours_X(k).pos-X(i).pos);
            end
%             S=-S;
        else % 如果没有邻居
            S=zeros(pointNum,3);
        end        
        % 对齐
        if neighbours_num>1
            A=(sum(Neighbours_X.DeltaX))/neighbours_num;
        else
            A=X(i).DeltaX;
        end        
        % 内聚
        if neighbours_num>1
            C_temp=(sum(Neighbours_X.pos))/neighbours_num;
        else
            C_temp=X(i).pos;
        end        
        C=C_temp-X(i).pos;        
        % 靠近食物
        Dist2Food=distance_3D(X(i).pos,Food.pos);
        if all(Dist2Food<=r)% 食物在邻域内
            F=Food.pos-X(i).pos;
        else
            F=0;%？
        end        
        % 远离天敌
        Dist2Enemy=distance_3D(X(i).pos,Enemy.pos);
        if all(Dist2Enemy<=r)% 天敌在邻域内
            Enemy=Enemy.pos+X(i).pos;
        else
            Enemy=zeros(pointNum,3);
        end
        
        for tt=1:pointNum*3
            if X(i).pos(tt)>ub(tt)%大于上限
                X(i).pos(tt)=lb(tt);
                X(i).DeltaX(tt)=rand;
            end
            if X(i).pos(tt)<lb(tt)
                X(i).pos(tt)=ub(tt);
                X(i).DeltaX(tt)=rand;
            end
        end       
        %% 自适应学习因子
        v=abs(Fitness(1,i)-Food.fitness)/(Food.fitness+eps);
        c_it=1/(1+exp(1)^(-v));
        RR=2*rand-1;% -1和1之间的随机数
        if any(Dist2Food>r) %如果食物位置在邻域外
            %%当有个体与个体i相邻时
            if neighbours_num>1
                % 种群分类和动态学习因子
                if abs(Length(X(i).pos)-E_f)<=E_ff
                for j=1:pointNum*3
                    X(i).DeltaX(j)=w*X(i).DeltaX(j)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if X(i).DeltaX(j)>Delta_max(j)
                        X(i).DeltaX(j)=Delta_max(j);
                    end
                    if X(i).DeltaX(j)<-Delta_max(j)
                        X(i).DeltaX(j)=-Delta_max(j);
                    end
                    X(i).pos(j)=c_it*X(i).pos(j)+X(i).DeltaX(j)+RR*(Food.pos(j)-X(i).pos(j));
                end
                end                
                if Length(X(i).pos)-E_f<-E_ff
                for j=1:pointNum*3
                    X(i).DeltaX(j)=w*X(i).DeltaX(j)+0.2*(rand*A(j,1)+rand*C(j,1)+rand*S(j,1));
                    if X(i).DeltaX(j)>Delta_max(j)
                        X(i).DeltaX(j)=Delta_max(j);
                    end
                    if X(i).DeltaX(j)<-Delta_max(j)
                        X(i).DeltaX(j)=-Delta_max(j);
                    end
                    X(i).pos(j)=c_it*X(i).pos(j)+X(i).DeltaX(j)+RR*(Food.pos(j)-X(i).pos(j));
                end     
                end             
                if Length(X(i).pos)-E_f>E_ff                
                for j=1:pointNum*3
                    X(i).DeltaX(j)=w*X(i).DeltaX(j)+5*(rand*A(j,1)+rand*C(j,1)+rand*S(j,1));
                    if X(i).DeltaX(j)>Delta_max(j)
                        X(i).DeltaX(j)=Delta_max(j);
                    end
                    if X(i).DeltaX(j)<-Delta_max(j)
                        X(i).DeltaX(j)=-Delta_max(j);
                    end
                    X(i).pos(j)=c_it*X(i).pos(j)+X(i).DeltaX(j)+RR*(Food.pos(j)-X(i).pos(j));
                end     
                end                
            else % 当没有任何个体与个体i相邻时 逆向搜索的莱维飞行随机游走
                if abs(Length(X(i).pos)-E_f)<=E_ff
                    X(i).pos=X(i).pos+0.5*Levy(pointNum,3)'.*(Food.pos-X(i).pos)+0.5*Levy(pointNum,3)'.*(P(i).pos-X(i).pos);
                end
                
                if Length(X(i).pos)-E_f<-E_ff                
                    X(i).pos=X(i).pos+Levy(pointNum*3)'.*(P(i).pos-X(i).pos);
                end
             
                if Length(X(i).pos)-E_f>E_ff                
                    X(i).pos=Food.pos+Levy(pointNum*3)'.*(Food.pos-X(i).pos);
                end  
                X(i).DeltaX=0;
            end            
        else % 如果食物位置是相邻蜻蜓位置（该个体较好）
            for j=1:pointNum*3
                X(i).DeltaX(j)=(a*A(j)+c*C(j)+s*S(j)+f*F(j)+e*Enemy(j))+w*X(i).DeltaX(j);
                if X(i).DeltaX(j)>Delta_max(j)
                    X(i).DeltaX(j)=Delta_max(j);
                end
                if X(i).DeltaX(j)<-Delta_max(j)
                    X(i).DeltaX(j)=-Delta_max(j);
                end
                X(i).pos(j)=c_it*X(i).pos(j)+X(i).DeltaX(j)+RR*(Food.pos(j)-X(i).pos(j));
            end 
        end        
        Flag4ub=X(i).pos>ub;
        Flag4lb=X(i).pos<lb;
        %范围大于上限则取上限，范围小于下限则取下限，否则不变
        X(i).pos=(X(i).pos.*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
    end
    Best_score=Food_fitness;
    Best_pos=Food.pos;
    cg_curve(iter)=Best_score;
end