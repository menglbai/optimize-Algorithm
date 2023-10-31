%% PathPlanning 3D TPDA
clc
clear
close all
%% ��ά·���滮ģ�Ͷ���
S = [1, 1, 1];    %��ʼ��
T = [100, 100, 40];%Ŀ���
% ����ɽ���ͼ
mapRange = [100,100,100];% ��ͼ�������߷�Χ
[XX,Y,Z] = defMap(mapRange);
%% ��ʼ��������
N = 500;           % ����������
M = 30;            % ���Ѹ�������
pointNum = 5;      % ÿһ�����������λ�õ�����
lb=0;              % λ������
ub=100;            % λ������
%% ��Ⱥ��ʼ�� ����X(i).posΪ���Ѹ���
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
%% ��������
for i=1:M
    X(i).DeltaX=100*rand(pointNum,3);
end
cg_curve=zeros(1,N);
if size(ub,2)==1
    ub=ones(pointNum,3)*ub;
    lb=ones(pointNum,3)*lb;
end
Delta_max=(ub-lb)/10;
% ��ʼ��ʳ��λ�ã�����ֵ��
Food.fitness=inf;
Food.pos=[];
% ��ʼ�����λ�ã����ֵ��
Enemy.fitness=-inf;
Enemy.pos=[];
Fitness=zeros(1,M);% ��Ӧ��ֵ
P=X;
% for i=1:M
%     P(i).fitness=X(i).fitness;
% end
for i=1:M
[flag,fitness,path] = calFitness(S,T,XX,Y,Z,X(i).pos);
  % ��ײ����ж�
    if flag == 1
        % ��flag=1��������·�������ϰ����ཻ����������Ӧ��ֵ
        X(i).fitness = 1000*fitness;
        X(i).path = path;
    else
        % ���򣬱�������ѡ���·��
        X(i).fitness = fitness;
        X(i).path = path;
    end
end
    
%% ��ʼ����
for iter=1:N
    % ���������뾶r
    r=(ub-lb)/4+((ub-lb)*(iter/N)*2);
    % ���¸�Ȩ��ϵ��
    w=0.9-iter*(0.5/N);  
    my_c=0.1-iter*(0.1/(N/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %����� 
    a=2*rand*my_c; %����� 
    c=2*rand*my_c; %�ھ۶� 
    f=2*rand;      %ʳ�������� 
    e=my_c;        %���ų��� 
    %�ҵ�ʳ������
    for i=1:M %���ȼ������и�����Ӧ��ֵ
        [flag,fitness,path] = calFitness(S,T,XX,Y,Z,X(i).pos);
  % ��ײ����ж�
    if flag == 1
        % ��flag=1��������·�������ϰ����ཻ����������Ӧ��ֵ
        X(i).fitness = 1000*fitness;
        X(i).path = path;
    else
        % ���򣬱�������ѡ���·��
        X(i).fitness = fitness;
        X(i).path = path;
    end
        Fitness(1,i)=Length(X(i).pos);
        %%%%%%%%%%���¸�������λ�ú�����ֵ%%%%%%%%%
        if Length(X(i).pos)<P(i).fitness
            P(i).pos=X(i).pos;
            P(i).fitness=Length(X(i).pos);
        end
        % �������ԭ���
        E_f=sum(Fitness(1,i))/M;
        D_f=sum(Fitness(1,i)-E_f)^2/(M-1);
        E_ff=E_f^2+D_f;
        if Length(X(i).pos)<Food.fitness %Ѱ��ÿ�ε�������Сֵ
            Food.fitness=Length(X(i).pos);%Ѱ��ȫ������λ�ú�ȫ������ֵ
            Food.pos=X(i).pos;
        end
        if Length(X(i).pos)>Enemy.fitness %Ѱ��ÿ�ε��������ֵ
            if all(all(X(i).pos<ub)) && all(all(X(i).pos>lb))
                Enemy.fitness=Length(X(i).pos);
                Enemy.pos=X(i).pos;
            end
        end
    end    
    %�ҵ�ÿֻ���ѵ��ھ�
    for i=1:M
        index=0;
        neighbours_num=0;        
%         clear Neighbours_DeltaX
        clear Neighbours_X
        %�ҵ������ھ�
        for j=1:M
            Dist2Enemy=distance_3D(X(i).pos,X(j).pos);%����ŷ�Ͼ���
            if all(all(Dist2Enemy<=r)) && all(all(Dist2Enemy~=0))
                index=index+1;%�ھ����
                neighbours_num=neighbours_num+1;%�ھ�����
                Neighbours_X(index).DeltaX=X(j).DeltaX;
                Neighbours_X(index).pos=X(j).pos;
            end
        end        
        % ����
        S=zeros(pointNum,3);
        if neighbours_num>1
            for k=1:neighbours_num
                S=S+(Neighbours_X(k).pos-X(i).pos);
            end
%             S=-S;
        else % ���û���ھ�
            S=zeros(pointNum,3);
        end        
        % ����
        if neighbours_num>1
            A=(sum(Neighbours_X.DeltaX))/neighbours_num;
        else
            A=X(i).DeltaX;
        end        
        % �ھ�
        if neighbours_num>1
            C_temp=(sum(Neighbours_X.pos))/neighbours_num;
        else
            C_temp=X(i).pos;
        end        
        C=C_temp-X(i).pos;        
        % ����ʳ��
        Dist2Food=distance_3D(X(i).pos,Food.pos);
        if all(Dist2Food<=r)% ʳ����������
            F=Food.pos-X(i).pos;
        else
            F=0;%��
        end        
        % Զ�����
        Dist2Enemy=distance_3D(X(i).pos,Enemy.pos);
        if all(Dist2Enemy<=r)% �����������
            Enemy=Enemy.pos+X(i).pos;
        else
            Enemy=zeros(pointNum,3);
        end
        
        for tt=1:pointNum*3
            if X(i).pos(tt)>ub(tt)%��������
                X(i).pos(tt)=lb(tt);
                X(i).DeltaX(tt)=rand;
            end
            if X(i).pos(tt)<lb(tt)
                X(i).pos(tt)=ub(tt);
                X(i).DeltaX(tt)=rand;
            end
        end       
        %% ����Ӧѧϰ����
        v=abs(Fitness(1,i)-Food.fitness)/(Food.fitness+eps);
        c_it=1/(1+exp(1)^(-v));
        RR=2*rand-1;% -1��1֮��������
        if any(Dist2Food>r) %���ʳ��λ����������
            %%���и��������i����ʱ
            if neighbours_num>1
                % ��Ⱥ����Ͷ�̬ѧϰ����
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
            else % ��û���κθ��������i����ʱ ������������ά�����������
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
        else % ���ʳ��λ������������λ�ã��ø���Ϻã�
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
        %��Χ����������ȡ���ޣ���ΧС��������ȡ���ޣ����򲻱�
        X(i).pos=(X(i).pos.*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;
    end
    Best_score=Food_fitness;
    Best_pos=Food.pos;
    cg_curve(iter)=Best_score;
end