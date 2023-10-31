%% Tent����ӳ��+��Ⱥ����Ͷ�̬ѧϰ����+������������ά�����������
function [Best_score,Best_pos,cg_curve]=TPDA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
disp('TPDA is optimizing your problem');
%% ��Ⱥ��ʼ�� Tent����ӳ��
X=zeros(dim,SearchAgents_num);
X(:,1)=rand(dim,1);
for i=2:SearchAgents_num
    for j=1:dim
        if X(j,i-1)>0.5
            X(j,i)=2-2*X(j,i-1)+rand/SearchAgents_num;
        end
        if 2*X(j,i-1)<=1 && 2*X(j,i-1)>=0
            X(j,i)=2*X(j,i-1)+rand/SearchAgents_num;
        end
    end
end
% dragonfly.pos=[];
% dragonfly = repmat(dragonfly,SearchAgents_num,1);
% for i = 1:SearchAgents_num
%     % ���Ӱ�����̬�ֲ��������
%     dragonfly(i).pos.x = unifrnd(lb,ub,1,dim);
%     dragonfly(i).pos.y = unifrnd(lb,ub,1,dim);
%     dragonfly(i).pos.z = unifrnd(lb,ub,1,dim);
%     for j=1:dim
%         Path(3*j-2,i)=dragonfly(i).pos.x(j);
%         Path(3*j-1,i)=dragonfly(i).pos.y(j);
%         Path(3*j,i)=dragonfly(i).pos.z(j);
%     end
% 
% end
%����[lb,ub]�����ھ��ȷֲ���N������
%% ��ʼ�������������ѵķ��з��򼰲�����
DeltaX=rand(dim,SearchAgents_num)*(ub-lb)+lb;
cg_curve=zeros(1,Max_iteration);
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end
Delta_max=(ub-lb)/10;
% ��ʼ��ʳ��λ�ã�����ֵ��
Food_fitness=inf;
Food_pos=zeros(dim,1);
% ��ʼ�����λ�ã����ֵ��
Enemy_fitness=-inf;
Enemy_pos=zeros(dim,1);
Fitness=zeros(1,SearchAgents_num);% ��Ӧ��ֵ
P=X;
Pbest=ones(SearchAgents_num,1);
for i=1:SearchAgents_num
    Pbest(i)=fobj(X(:,i)');
end
%% ��ʼ����
for iter=1:Max_iteration
    % ���������뾶r
    r=(ub-lb)/4+((ub-lb)*(iter/Max_iteration)*2);
    % ���¸�Ȩ��ϵ��
    w=0.9-iter*((0.9-0.4)/Max_iteration);  
    my_c=0.1-iter*((0.1-0)/(Max_iteration/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %����� 
    a=2*rand*my_c; %����� 
    c=2*rand*my_c; %�ھ۶� 
    f=2*rand;      %ʳ�������� 
    e=my_c;        %���ų��� 
    %�ҵ�ʳ������
    for i=1:SearchAgents_num %���ȼ�������Ŀ��ֵ
        Fitness(1,i)=fobj(X(:,i)');
        %%%%%%%%%%���¸�������λ�ú�����ֵ%%%%%%%%%
        if fobj(X(:,i)')<Pbest(i)
            P(:,i)=X(:,i);
            Pbest(i)=fobj(X(:,i)');
        end
        % �������ԭ���
        E_f=sum(Fitness)/SearchAgents_num;
        D_f=sum(Fitness-E_f)^2/(SearchAgents_num-1);
        E_ff=E_f^2+D_f;
        if Fitness(1,i)<Food_fitness %Ѱ��ÿ�ε�������Сֵ
            Food_fitness=Fitness(1,i);% Ѱ��ȫ������λ�ú�ȫ������ֵ
            Food_pos=X(:,i);
        end
        if Fitness(1,i)>Enemy_fitness %Ѱ��ÿ�ε��������ֵ
            if all(X(:,i)<ub') && all(X(:,i)>lb')
                Enemy_fitness=Fitness(1,i);
                Enemy_pos=X(:,i);
            end
        end
    end    
    %�ҵ�ÿֻ���ѵ��ھ�
    for i=1:SearchAgents_num
        index=0;
        neighbours_num=0;        
        clear Neighbours_DeltaX
        clear Neighbours_X
        %�ҵ������ھ�
        for j=1:SearchAgents_num
            Dist2Enemy=distance(X(:,i),X(:,j));%����ŷ�Ͼ���
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;%�ھ����
                neighbours_num=neighbours_num+1;%�ھ�����
                Neighbours_DeltaX(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=X(:,j);
            end
        end        
        % ����
        S=zeros(dim,1);
        if neighbours_num>1
            for k=1:neighbours_num
                S=S+(Neighbours_X(:,k)-X(:,i));
            end
%             S=-S;
        else % ���û���ھ�
            S=zeros(dim,1);
        end        
        % ����
        if neighbours_num>1
            A=(sum(Neighbours_DeltaX')')/neighbours_num;
        else
            A=DeltaX(:,i);
        end        
        % �ھ�
        if neighbours_num>1
            C_temp=(sum(Neighbours_X')')/neighbours_num;
        else
            C_temp=X(:,i);
        end        
        C=C_temp-X(:,i);        
        % ����ʳ��
        Dist2Food=distance(X(:,i),Food_pos);
        if all(Dist2Food<=r)% ʳ����������
            F=Food_pos-X(:,i);
        else
            F=0;%��
        end        
        % Զ�����
        Dist2Enemy=distance(X(:,i),Enemy_pos);
        if all(Dist2Enemy<=r)% �����������
            Enemy=Enemy_pos+X(:,i);
        else
            Enemy=zeros(dim,1);
        end
        
        for tt=1:dim
            if X(tt,i)>ub(tt)%��������
                X(tt,i)=lb(tt);
                DeltaX(tt,i)=rand;
            end
            if X(tt,i)<lb(tt)
                X(tt,i)=ub(tt);
                DeltaX(tt,i)=rand;
            end
        end       
        %% ����Ӧѧϰ����
        v=abs(Fitness(1,i)-Food_fitness)/(Food_fitness+eps);
        c_it=1/(1+exp(1)^(-v));
        RR=2*rand-1;% -1��1֮��������
        if any(Dist2Food>r) %���ʳ��λ����������
            %%���и��������i����ʱ
            if neighbours_num>1
                % ��Ⱥ����Ͷ�̬ѧϰ����
                if abs(fobj(X(:,i)')-E_f)<=E_ff
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
                if fobj(X(:,i)')-E_f<-E_ff
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
                if fobj(X(:,i)')-E_f>E_ff                
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
            else % ��û���κθ��������i����ʱ ������������ά�����������
                if abs(fobj(X(:,i)')-E_f)<=E_ff
                    X(:,i)=X(:,i)+0.5*Levy(dim)'.*(Food_pos-X(:,i))+0.5*Levy(dim)'.*(P(:,i)-X(:,i));
                end
                
                if fobj(X(:,i)')-E_f<-E_ff                
                    X(:,i)=X(:,i)+Levy(dim)'.*(P(:,i)-X(:,i));
                end
             
                if fobj(X(:,i)')-E_f>E_ff                
                    X(:,i)=Food_pos+Levy(dim)'.*(Food_pos-X(:,i));
                end  
                DeltaX(:,i)=0;
            end            
        else % ���ʳ��λ������������λ�ã��ø���Ϻã�
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
        %��Χ����������ȡ���ޣ���ΧС��������ȡ���ޣ����򲻱�
        X(:,i)=(X(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
    end
    Best_score=Food_fitness;
    Best_pos=Food_pos;
    cg_curve(iter)=Best_score;
end