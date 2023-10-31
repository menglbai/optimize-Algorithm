%% Tent混沌映射+种群分类和动态学习因子+逆向搜索的莱维飞行随机游走
function [Best_score,Best_pos,cg_curve]=TPDA_3D(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
disp('TPDA is optimizing your problem');
%% 种群初始化 Tent混沌映射
% X=zeros(dim,SearchAgents_num);
% X(:,1)=rand(dim,1);
% for i=2:SearchAgents_num
%     for j=1:dim
%         if X(j,i-1)>0.5
%             X(j,i)=2-2*X(j,i-1)+rand/SearchAgents_num;
%         end
%         if 2*X(j,i-1)<=1 && 2*X(j,i-1)>=0
%             X(j,i)=2*X(j,i-1)+rand/SearchAgents_num;
%         end
%     end
% end
dragonfly.pos=[];
dragonfly = repmat(dragonfly,SearchAgents_num,1);
for i = 1:SearchAgents_num
    % 粒子按照正态分布随机生成
    dragonfly(i).pos.x = unifrnd(lb,ub,1,dim);
    dragonfly(i).pos.y = unifrnd(lb,ub,1,dim);
    dragonfly(i).pos.z = unifrnd(lb,ub,1,dim);
    for j=1:dim
        Path(3*j-2,i)=dragonfly(i).pos.x(j);
        Path(3*j-1,i)=dragonfly(i).pos.y(j);
        Path(3*j,i)=dragonfly(i).pos.z(j);
    end
end
%生成[lb,ub]区间内均匀分布的N个个体
%% 初始化步向量（蜻蜓的飞行方向及步长）
DeltaX=rand(dim*3,SearchAgents_num)*(ub-lb)+lb;
cg_curve=zeros(1,Max_iteration);
if size(ub,2)==1
    ub=ones(1,dim*3)*ub;
    lb=ones(1,dim*3)*lb;
end
Delta_max=(ub-lb)/10;
% 初始化食物位置（最优值）
Food_fitness=inf;
Food_pos=zeros(dim*3,1);
% 初始化天敌位置（最差值）
Enemy_fitness=-inf;
Enemy_pos=zeros(dim*3,1);
Fitness=zeros(1,SearchAgents_num);% 适应度值
P=Path;
Pbest=ones(SearchAgents_num,1);
for i=1:SearchAgents_num
    Pbest(i)=fobj(Path(:,i)');
end
%% 开始迭代
for iter=1:Max_iteration
    % 更新搜索半径r
    r=(ub-lb)/4+((ub-lb)*(iter/Max_iteration)*2);
    % 更新各权重系数
    w=0.9-iter*((0.9-0.4)/Max_iteration);  
    my_c=0.1-iter*((0.1-0)/(Max_iteration/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %分离度 
    a=2*rand*my_c; %对齐度 
    c=2*rand*my_c; %内聚度 
    f=2*rand;      %食物吸引力 
    e=my_c;        %敌排斥力 
    %找到食物和天敌
    for i=1:SearchAgents_num %首先计算所有目标值
        Fitness(1,i)=fobj(Path(:,i)');
        %%%%%%%%%%更新个体最优位置和最优值%%%%%%%%%
        if fobj(Path(:,i)')<Pbest(i)
            P(:,i)=Path(:,i);
            Pbest(i)=fobj(Path(:,i)');
        end
        % 计算二阶原点矩
        E_f=sum(Fitness)/SearchAgents_num;
        D_f=sum(Fitness-E_f)^2/(SearchAgents_num-1);
        E_ff=E_f^2+D_f;
        if Fitness(1,i)<Food_fitness %寻找每次迭代的最小值
            Food_fitness=Fitness(1,i);%寻找全局最优位置和全局最优值
            Food_pos=Path(:,i);
        end
        if Fitness(1,i)>Enemy_fitness %寻找每次迭代的最大值
            if all(Path(:,i)<ub') && all(Path(:,i)>lb')
                Enemy_fitness=Fitness(1,i);
                Enemy_pos=Path(:,i);
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
            Dist2Enemy=distance_3D(Path(:,i),Path(:,j));%计算欧氏距离
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;%邻居序号
                neighbours_num=neighbours_num+1;%邻居数量
                Neighbours_DeltaX(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=Path(:,j);
            end
        end        
        % 分离
        S=zeros(dim*3,1);
        if neighbours_num>1
            for k=1:neighbours_num
                S=S+(Neighbours_X(:,k)-Path(:,i));
            end
%             S=-S;
        else % 如果没有邻居
            S=zeros(dim*3,1);
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
            C_temp=Path(:,i);
        end        
        C=C_temp-Path(:,i);        
        % 靠近食物
        Dist2Food=distance_3D(Path(:,i),Food_pos);
        if all(Dist2Food<=r)% 食物在邻域内
            F=Food_pos-Path(:,i);
        else
            F=0;%？
        end        
        % 远离天敌
        Dist2Enemy=distance_3D(Path(:,i),Enemy_pos);
        if all(Dist2Enemy<=r)% 天敌在邻域内
            Enemy=Enemy_pos+Path(:,i);
        else
            Enemy=zeros(dim*3,1);
        end
        
        for tt=1:dim*3
            if Path(tt,i)>ub(tt)%大于上限
                Path(tt,i)=lb(tt);
                DeltaX(tt,i)=rand;
            end
            if Path(tt,i)<lb(tt)
                Path(tt,i)=ub(tt);
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
                if abs(fobj(Path(:,i)')-E_f)<=E_ff
                for j=1:dim*3
                    DeltaX(j,i)=w*DeltaX(j,i)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    Path(j,i)=c_it*Path(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-Path(j,i));
                end
                end                
                if fobj(Path(:,i)')-E_f<-E_ff
                for j=1:dim*3
                    DeltaX(j,i)=w*DeltaX(j,i)+0.2*(rand*A(j,1)+rand*C(j,1)+rand*S(j,1));
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    Path(j,i)=c_it*Path(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-Path(j,i));
                end     
                end             
                if fobj(Path(:,i)')-E_f>E_ff                
                for j=1:dim*3
                    DeltaX(j,i)=w*DeltaX(j,i)+5*(rand*A(j,1)+rand*C(j,1)+rand*S(j,1));
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    Path(j,i)=c_it*Path(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-Path(j,i));
                end     
                end                
            else % 当没有任何个体与个体i相邻时 逆向搜索的莱维飞行随机游走
                if abs(fobj(Path(:,i)')-E_f)<=E_ff
                    Path(:,i)=Path(:,i)+0.5*Levy(dim*3)'.*(Food_pos-Path(:,i))+0.5*Levy(dim*3)'.*(P(:,i)-Path(:,i));
                end
                
                if fobj(Path(:,i)')-E_f<-E_ff                
                    Path(:,i)=Path(:,i)+Levy(dim*3)'.*(P(:,i)-Path(:,i));
                end
             
                if fobj(Path(:,i)')-E_f>E_ff                
                    Path(:,i)=Food_pos+Levy(dim*3)'.*(Food_pos-Path(:,i));
                end  
                DeltaX(:,i)=0;
            end            
        else % 如果食物位置是相邻蜻蜓位置（该个体较好）
            for j=1:dim*3
                DeltaX(j,i)=(a*A(j,1)+c*C(j,1)+s*S(j,1)+f*F(j,1)+e*Enemy(j,1))+w*DeltaX(j,i);
                if DeltaX(j,i)>Delta_max(j)
                    DeltaX(j,i)=Delta_max(j);
                end
                if DeltaX(j,i)<-Delta_max(j)
                    DeltaX(j,i)=-Delta_max(j);
                end
                Path(j,i)=c_it*Path(j,i)+DeltaX(j,i)+RR*(Food_pos(j)-Path(j,i));
            end 
        end        
        Flag4ub=Path(:,i)>ub';
        Flag4lb=Path(:,i)<lb';
        %范围大于上限则取上限，范围小于下限则取下限，否则不变
        Path(:,i)=(Path(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
    end
    Best_score=Food_fitness;
    Best_pos=Food_pos;
    cg_curve(iter)=Best_score;
end