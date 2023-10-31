%% ADDA: Adaptive learning factor & Differential evolution strategy ��������Ӧѧϰ���ӺͲ�ֽ������Ե������㷨
function [Best_score,Best_pos,cg_curve]=ADDA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
disp('ADDA is optimizing your problem');
%% ��ʼ��������Ⱥ
X=rand(dim,SearchAgents_num)*(ub-lb)+lb;% ÿһ��Ϊһ������
% ��ʼ�������������ѵķ��з��򼰲�����
DeltaX=rand(dim,SearchAgents_num)*(ub-lb)+lb;
cg_curve=zeros(1,Max_iteration);
F_i=0.9;
F_f=0.1;
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end
SearchAgents=(1:SearchAgents_num);
Delta_max=(ub-lb)/10;
% ��ʼ��ʳ��λ�ã�����ֵ��
Food_fitness=inf;
Food_pos=zeros(dim,1);
% ��ʼ�����λ�ã����ֵ��
Enemy_fitness=-inf;
Enemy_pos=zeros(dim,1);
Fitness=zeros(1,SearchAgents_num);% ��Ӧ��ֵ
%% ��ʼ����
for iter=1:Max_iteration
    r=(ub-lb)/4+((ub-lb)*(iter/Max_iteration)*2);
    % ���¸�Ȩ��ϵ��[0.9,0.4]
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
        if Fitness(1,i)<Food_fitness %Ѱ��ÿ�ε�������Сֵ
            Food_fitness=Fitness(1,i);
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
            S=-S;
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
        if all(Dist2Food<=r)
            F=Food_pos-X(:,i);
        else
            F=0;
        end        
        % Զ�����
        Dist2Enemy=distance(X(:,i),Enemy_pos);
        if all(Dist2Enemy<=r)
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
       %%
        if any(Dist2Food>r) %���ʳ��λ�ò�����������λ��
            %%���и��������i����ʱ
            if neighbours_num>1               
                for j=1:dim
                    DeltaX(j,i)=w*DeltaX(j,i)+rand*A(j,1)+rand*C(j,1)+rand*S(j,1);
                    if DeltaX(j,i)>Delta_max(j)
                        DeltaX(j,i)=Delta_max(j);
                    end
                    if DeltaX(j,i)<-Delta_max(j)
                        DeltaX(j,i)=-Delta_max(j);
                    end
                    X(j,i)=c_it*X(j,i)+DeltaX(j,i);
                end                
            else %��û���κθ��������i����ʱ
                X(:,i)=c_it*X(:,i)+Levy(dim)'.*X(:,i);
                DeltaX(:,i)=0;
            end
        else %���ʳ��λ������������λ��
            for j=1:dim
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
       %% ��ֽ�������              
        F_it=F_i+(F_f-F_i)*(Fitness(1,i)-Food_fitness)/(Enemy_fitness-Food_fitness);
        p=SearchAgents(randperm(length(SearchAgents),2));
        H_it=Food_pos+F_it*(X(:,p(1))-X(:,p(2)));        
        PCR=0.6;
        rand_j=randi(SearchAgents_num);
        for kk=1:dim
            wy=rand;
            if wy<=PCR || kk==rand_j
                V(kk)=H_it(kk);
            else
                V(kk)=X(kk,i);
            end
        end
        if fobj(V')<fobj(X(:,i)')
            X(:,i)=V;
        end
        % �߽紦��
        Flag4ub=X(:,i)>ub';
        Flag4lb=X(:,i)<lb';
        %��Χ����������ȡ���ޣ���ΧС��������ȡ���ޣ����򲻱�
        X(:,i)=(X(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
    end
    Best_score=Food_fitness;
    Best_pos=Food_pos;
    cg_curve(iter)=Best_score;
end