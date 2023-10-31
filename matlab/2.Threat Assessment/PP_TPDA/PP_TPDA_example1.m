%% ��1 �������˻���ȺĿ����в����
clear;
clc;
%% ������в���ݲ����й�һ������
filename='D:\matlab\PP\threat_data.xlsx'; %��в����
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
Ntest=30;   % ʵ�����
test.g=[];
test.gbest=[];
pop=repmat(test,Ntest,1);
for jj=1:Ntest
SearchAgents_num=30; % ��Ⱥ����
Max_iteration=500;   % ����������
dim=size(Threat,2);  % ����ά��
ub=1;                % ��ռ�����
lb=-1;               % ��ռ�����
%% ��Ⱥ��ʼ�� Tent����ӳ��
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
% ����[lb,ub]�����ھ��ȷֲ���N������
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
    Pbest(i)=SD_penaltyfunction(Threat,X(:,i));
end
%% ��ʼ����
for iter=1:Max_iteration
    r=(ub-lb)/4+((ub-lb)*(iter/Max_iteration)*2);
    w=0.9-iter*(0.5/Max_iteration);    
    my_c=0.1-iter*(0.1/(Max_iteration/2));
    if my_c<0
        my_c=0;
    end
    s=2*rand*my_c; %����Ȩ��
    a=2*rand*my_c; %�Ŷ�Ȩ��
    c=2*rand*my_c; %����Ȩ��
    f=2*rand;      %ʳ��Ȩ��
    e=my_c;        %���Ȩ��
    %�ҵ�ʳ�����У����Ž�����⣩
    for i=1:SearchAgents_num
        Fitness(1,i)=SD_penaltyfunction(Threat,X(:,i));% �˴�X(:,i)Ϊ������
        %%%%%%%%%%���¸�������λ�ú�����ֵ%%%%%%%%%
        if SD_penaltyfunction(Threat,X(:,i))<Pbest(i)
            P(:,i)=X(:,i);
            Pbest(i)=SD_penaltyfunction(Threat,X(:,i));
        end
        % �������ԭ���
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
    %�ҵ�ÿֻ���ѵ��ھ�
    for i=1:SearchAgents_num
        index=0;
        neighbours_num=0;        
        clear Neighbours_DeltaX
        clear Neighbours_X
        %�ҵ������ھ�
        for j=1:SearchAgents_num
            Dist2Enemy=distance(X(:,i),X(:,j));
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;%�ھ����
                neighbours_num=neighbours_num+1;%�ھ�����+1
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
        else
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
        % ʳ��
        Dist2Food=distance(X(:,i),Food_pos(:,1));
        if all(Dist2Food<=r)
            F=Food_pos-X(:,i);
        else
            F=0;
        end        
        % Զ�����
        Dist2Enemy=distance(X(:,i),Enemy_pos(:,1));
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
        RR=2*rand-1;% -1��1֮��������
        if any(Dist2Food>r) %���ʳ��λ����������
            %%���и��������i����ʱ
            if neighbours_num>1
                % ��Ⱥ����Ͷ�̬ѧϰ����
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
            else % ��û���κθ��������i����ʱ ������������ά�����������
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
        G=abs(pop(index).g);%���ͶӰ����
    break
    end
end
gg=sum(G.^2); %��֤ģ���Ƿ����1
AA=Threat*G;
[B,I]=sort(AA,'descend');%Ŀ�갴��в�̶ȴӴ�С����
for k=1:Ntest
    if floor(pop(k).gbest*10000)>floor(Gbest*10000)
        N_local=N_local+1;
    end
end