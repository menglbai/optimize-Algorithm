%% 例1 海上无人机蜂群目标威胁评估
clear;
clc;
%% 输入威胁数据并进行归一化处理
filename='D:\matlab\PP\threat_data.xlsx'; %威胁数据
sheet=8;
Threat=xlsread(filename,sheet);
format short
for j=1:size(Threat,2)
    Max(j)=max(Threat(:,j));
    Min(j)=min(Threat(:,j));    Max_Min(j)=Max(j)-Min(j);
end
for i=1:size(Threat,1)
    for j=1:size(Threat,2)
        Threat(i,j)=(Threat(i,j)-Min(j))/Max_Min(j);
    end
end
Ntest=30;   % 实验次数
test.g=[];
test.gbest=[];
pop=repmat(test,Ntest,1);
for jj=1:Ntest
SearchAgents_num=30; % 种群数量
Max_iteration=500;   % 最大迭代次数
dim=size(Threat,2);  % 个体维数
ub=1;                % 解空间上限
lb=-1;               % 解空间下限
%% 种群初始化 Tent混沌映射
X=rand(dim,SearchAgents_num);
x=zeros(1,SearchAgents_num);
x(1)=rand;
for i=2:SearchAgents_num
    if 2*x(i-1)>1
        x(i)=2-2*x(i-1)+rand/SearchAgents_num;
    end
    if 2*x(i-1)<=1 && 2*x(i-1)>=0
        x(i)=2*x(i-1)+rand/SearchAgents_num;
    end
end
for i=1:SearchAgents_num
    if x(i)>1
        x(i)=1;
    end
    if x(i)<0
        x(i)=0;
    end
end
for i=1:SearchAgents_num
    for j=1:dim
    X(j,i)=2*ub*x(i)+lb;
    end
end
% 生成[lb,ub]区间内均匀分布的N个个体
%% 初始化步向量（蜻蜓的飞行方向及步长）
DeltaX=rand(dim,SearchAgents_num)*(ub-lb)+lb;
cg_curve=zeros(1,Max_iteration);
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end
Delta_max=(ub-lb)/10;
% 初始化食物位置（最优值）
Food_fitness=inf;
Food_pos=zeros(dim,1);
% 初始化天敌位置（最差值）
Enemy_fitness=-inf;
Enemy_pos=zeros(dim,1);
Fitness=zeros(1,SearchAgents_num);% 适应度值
P=X;
Pbest=ones(SearchAgents_num,1);
for i=1:SearchAgents_num
    Pbest(i)=SD_penaltyfunction(Threat,X(:,i));
end
%% 开始迭代
for iter=1:Max_iteration
    r=(ub-lb)/4+((ub-lb)*(iter/Max_iteration)*2);
    w=0.9-iter*(0.5/Max_iteration);    
    my_c=0.1-iter*(0.1/(Max_iteration/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %分离权重
    a=2*rand*my_c; %排队权重
    c=2*rand*my_c; %结盟权重
    f=2*rand;      %食物权重
    e=my_c;        %天敌权重
    %找到食物和天敌（最优解和最差解）
    for i=1:SearchAgents_num
        Fitness(1,i)=SD_penaltyfunction(Threat,X(:,i));% 此处X(:,i)为列向量
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%
        if SD_penaltyfunction(Threat,X(:,i))<Pbest(i)
            P(:,i)=X(:,i);
            Pbest(i)=SD_penaltyfunction(Threat,X(:,i));
        end
        % 计算二阶原点矩
        E_f=sum(Fitness)/SearchAgents_num;
        D_f=sum(Fitness-E_f)^2/(SearchAgents_num-1);
        E_ff=E_f^2+D_f;
        if Fitness(1,i)<Food_fitness 
            Food_fitness=Fitness(1,i);
            Food_pos=X(:,i);
        end
        if Fitness(1,i)>Enemy_fitness 
            if all(X(:,i)<ub') && all(X(:,i)>lb') 
                Enemy_fitness=Fitness(1,i);
                Enemy_pos=X(:,i);
            end
        end
    end    
    %找到每只蜻蜓的邻居
    for i=1:SearchAgents_num
        index=0;
        neighbours_num=0;        
        clear Neighbours_DeltaX
        clear Neighbours_X
        %找到相邻邻居
        for j=1:SearchAgents_num
            Dist2Enemy=distance(X(:,i),X(:,j));
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;%邻居序号
                neighbours_num=neighbours_num+1;%邻居数量+1
                Neighbours_DeltaX(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=X(:,j);
            end
        end        
        % 分离
        S=zeros(dim,1);
        if neighbours_num>1
            for k=1:neighbours_num
                S=S+(Neighbours_X(:,k)-X(:,i));
            end
            S=-S;
        else
            S=zeros(dim,1);
        end        
        % 对齐
        if neighbours_num>1
            A=(sum(Neighbours_DeltaX')')/neighbours_num;
        else
            A=DeltaX(:,i);
        end        
        % 内聚
        if neighbours_num>1
            C_temp=(sum(Neighbours_X')')/neighbours_num;
        else
            C_temp=X(:,i);
        end        
        C=C_temp-X(:,i);        
        % 食物
        Dist2Food=distance(X(:,i),Food_pos(:,1));
        if all(Dist2Food<=r)
            F=Food_pos-X(:,i);
        else
            F=0;
        end        
        % 远离天敌
        Dist2Enemy=distance(X(:,i),Enemy_pos(:,1));
        if all(Dist2Enemy<=r)
            Enemy=Enemy_pos+X(:,i);
        else
            Enemy=zeros(dim,1);
        end
        
        for tt=1:dim
            if X(tt,i)>ub(tt)%大于上限
                X(tt,i)=lb(tt);
                DeltaX(tt,i)=rand;
            end
            if X(tt,i)<lb(tt)
                X(tt,i)=ub(tt);
                DeltaX(tt,i)=rand;
            end
        end
        %% 自适应学习因子
        v=abs(Fitness(1,i)-Food_fitness)/(Food_fitness+eps);
        c_it=1/(1+exp(1)^(-v));
        RR=2*rand-1;% -1和1之间的随机数
        if any(Dist2Food>r) %如果食物位置在邻域外
            %%当有个体与个体i相邻时
            if neighbours_num>1
                % 种群分类和动态学习因子
                if abs(SD_penaltyfunction(Threat,X(:,i))-E_f)<=E_ff
                for j=1:dim
                    DeltaX(j,i)=w*DeltaX(j,i)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    X(j,i)=c_it*X(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-X(j,i));
                end
                end                
                if SD_penaltyfunction(Threat,X(:,i))-E_f<-E_ff
                for j=1:dim
                    DeltaX(j,i)=w*DeltaX(j,i)+0.2*(rand*A(j,1)+rand*C(j,1)+rand*S(j,1));
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    X(j,i)=c_it*X(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-X(j,i));
                end     
                end             
                if SD_penaltyfunction(Threat,X(:,i))-E_f>E_ff                
                for j=1:dim
                    DeltaX(j,i)=w*DeltaX(j,i)+5*(rand*A(j,1)+rand*C(j,1)+rand*S(j,1));
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    X(j,i)=c_it*X(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-X(j,i));
                end     
                end                
            else % 当没有任何个体与个体i相邻时 逆向搜索的莱维飞行随机游走
                if abs(SD_penaltyfunction(Threat,X(:,i))-E_f)<=E_ff
                    X(:,i)=X(:,i)+0.5*Levy(dim)'.*(Food_pos-X(:,i))+0.5*Levy(dim)'.*(P(:,i)-X(:,i));
                end
                
                if SD_penaltyfunction(Threat,X(:,i))-E_f<-E_ff                
                    X(:,i)=X(:,i)+Levy(dim)'.*(P(:,i)-X(:,i));
                end
             
                if SD_penaltyfunction(Threat,X(:,i))-E_f>E_ff                
                    X(:,i)=Food_pos+Levy(dim)'.*(Food_pos-X(:,i));
                end  
                DeltaX(:,i)=0;
            end            
        else % 如果食物位置是相邻蜻蜓位置（该个体较好）
            for j=1:dim
                DeltaX(j,i)=(a*A(j,1)+c*C(j,1)+s*S(j,1)+f*F(j,1)+e*Enemy(j,1))+w*DeltaX(j,i);
                if DeltaX(j,i)>Delta_max(j)
                    DeltaX(j,i)=Delta_max(j);
                end
                if DeltaX(j,i)<-Delta_max(j)
                    DeltaX(j,i)=-Delta_max(j);
                end
                X(j,i)=c_it*X(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-X(j,i));
            end 
        end        
        Flag4ub=X(:,i)>ub';
        Flag4lb=X(:,i)<lb';
        %范围大于上限则取上限，范围小于下限则取下限，否则不变
        X(:,i)=(X(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
    end
    Best_score=Food_fitness;
    Best_pos=Food_pos;
    cg_curve(iter)=Best_score;
end
pop(jj).g=Best_pos;
pop(jj).gbest=Best_score;
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
AA=Threat*G;
[B,I]=sort(AA,'descend');%目标按威胁程度从大到小排序
for k=1:Ntest
    if floor(pop(k).gbest*10000)>floor(Gbest*10000)
        N_local=N_local+1;
    end
end