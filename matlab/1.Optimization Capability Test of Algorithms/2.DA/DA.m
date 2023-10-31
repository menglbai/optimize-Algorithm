% dim = 待优化参数个数
% Max_iteration =最大迭代次数
% SearchAgents_num = 蜻蜓数量
% ub=[ub1,ub2,...,ubn] 参数取值上限
% lb=[lb1,lb2,...,lbn] 参数取值下限
% To run DA: [Best_score,Best_pos,cg_curve]=DA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
function [Best_score,Best_pos,cg_curve]=DA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
disp('DA is optimizing your problem');
% 初始化蜻蜓种群 
X=rand(dim,SearchAgents_num)*(ub-lb)+lb;% 每一列为一个个体
% 初始化步向量（蜻蜓的飞行方向及步长）
DeltaX=rand(dim,SearchAgents_num)*(ub-lb)+lb;
cg_curve=zeros(1,Max_iteration);
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end
%初始化蜻蜓邻里半径
Delta_max=(ub-lb)/10;
% 初始化食物位置（最优值）
Food_fitness=inf;
Food_pos=zeros(dim,1);
% 初始化天敌位置（最差值）
Enemy_fitness=-inf;
Enemy_pos=zeros(dim,1);
Fitness=zeros(1,SearchAgents_num); 
for iter=1:Max_iteration
    r=(ub-lb)/4+((ub-lb)*(iter/Max_iteration)*2);
    w=0.9-iter*((0.9-0.4)/Max_iteration);
    my_c=0.1-iter*((0.1-0)/(Max_iteration/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %分离度 0.0013
    a=2*rand*my_c; %对齐度 0.1884
    c=2*rand*my_c; %内聚度 0.1791
    f=2*rand;      %食物吸引力 0.8826
    e=my_c;        %敌排斥力 0.0996
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %找到食物和天敌
    for i=1:SearchAgents_num %首先计算所有目标值
        Fitness(1,i)=fobj(X(:,i)');
        if Fitness(1,i)<Food_fitness %寻找每次迭代的最小值
            Food_fitness=Fitness(1,i);
            Food_pos=X(:,i);
        end
        if Fitness(1,i)>Enemy_fitness %寻找每次迭代的最大值
            if all(X(:,i)<ub') && all( X(:,i)>lb')
                Enemy_fitness=Fitness(1,i);
                Enemy_pos=X(:,i);
            end
        end
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %找到每只蜻蜓的邻居
    for i=1:SearchAgents_num
        index=0;
        neighbours_no=0;        
        clear Neighbours_DeltaX
        clear Neighbours_X
        %找到相邻邻居
        for j=1:SearchAgents_num
            Dist2Enemy=distance(X(:,i),X(:,j));%计算欧氏距离
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;%邻居序号
                neighbours_no=neighbours_no+1;%邻居数量
                Neighbours_DeltaX(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=X(:,j);
            end
        end        
        % 分离 - ％%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.1)
        S=zeros(dim,1);
        if neighbours_no>1
            for k=1:neighbours_no
                S=S+(Neighbours_X(:,k)-X(:,i));
            end
            S=-S;
        else
            S=zeros(dim,1);
        end        
        % 对齐%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.2)
        if neighbours_no>1
            A=(sum(Neighbours_DeltaX')')/neighbours_no;
        else
            A=DeltaX(:,i);
        end        
        % 内聚%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.3)
        if neighbours_no>1
            C_temp=(sum(Neighbours_X')')/neighbours_no;
        else
            C_temp=X(:,i);
        end        
        C=C_temp-X(:,i);        
        % 靠近食物%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.4)
        Dist2Food=distance(X(:,i),Food_pos(:,1));
        if all(Dist2Food<=r)
            F=Food_pos-X(:,i);
        else
            F=0;
        end        
        % 远离天敌%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.5)
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
        
        if any(Dist2Food>r) %如果食物位置不是相邻蜻蜓位置
            %%当有个体与个体 i 相邻时
            if neighbours_no>1
                for j=1:dim
                    DeltaX(j,i)=w*DeltaX(j,i)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    X(j,i)=X(j,i)+DeltaX(j,i);
                end
            else
                % Eq. (3.8)
                %当没有任何个体与个体 i 相邻时
                X(:,i)=X(:,i)+Levy(dim)'.*X(:,i);
                DeltaX(:,i)=0;
            end
        else% 如果食物位置是相邻蜻蜓位置
            for j=1:dim
                % Eq. (3.6)
                DeltaX(j,i)=(a*A(j,1)+c*C(j,1)+s*S(j,1)+f*F(j,1)+e*Enemy(j,1))+w*DeltaX(j,i);
                if DeltaX(j,i)>Delta_max(j)
                    DeltaX(j,i)=Delta_max(j);
                end
                if DeltaX(j,i)<-Delta_max(j)
                    DeltaX(j,i)=-Delta_max(j);
                end
                X(j,i)=X(j,i)+DeltaX(j,i);
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