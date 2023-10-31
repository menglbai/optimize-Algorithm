% dim = ���Ż���������
% Max_iteration =����������
% SearchAgents_num = ��������
% ub=[ub1,ub2,...,ubn] ����ȡֵ����
% lb=[lb1,lb2,...,lbn] ����ȡֵ����
% To run DA: [Best_score,Best_pos,cg_curve]=DA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
function [Best_score,Best_pos,cg_curve]=DA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj)
disp('DA is optimizing your problem');
% ��ʼ��������Ⱥ 
X=rand(dim,SearchAgents_num)*(ub-lb)+lb;% ÿһ��Ϊһ������
% ��ʼ�������������ѵķ��з��򼰲�����
DeltaX=rand(dim,SearchAgents_num)*(ub-lb)+lb;
cg_curve=zeros(1,Max_iteration);
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end
%��ʼ����������뾶
Delta_max=(ub-lb)/10;
% ��ʼ��ʳ��λ�ã�����ֵ��
Food_fitness=inf;
Food_pos=zeros(dim,1);
% ��ʼ�����λ�ã����ֵ��
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
    s=2*rand*my_c; %����� 0.0013
    a=2*rand*my_c; %����� 0.1884
    c=2*rand*my_c; %�ھ۶� 0.1791
    f=2*rand;      %ʳ�������� 0.8826
    e=my_c;        %���ų��� 0.0996
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %�ҵ�ʳ������
    for i=1:SearchAgents_num %���ȼ�������Ŀ��ֵ
        Fitness(1,i)=fobj(X(:,i)');
        if Fitness(1,i)<Food_fitness %Ѱ��ÿ�ε�������Сֵ
            Food_fitness=Fitness(1,i);
            Food_pos=X(:,i);
        end
        if Fitness(1,i)>Enemy_fitness %Ѱ��ÿ�ε��������ֵ
            if all(X(:,i)<ub') && all( X(:,i)>lb')
                Enemy_fitness=Fitness(1,i);
                Enemy_pos=X(:,i);
            end
        end
    end    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %�ҵ�ÿֻ���ѵ��ھ�
    for i=1:SearchAgents_num
        index=0;
        neighbours_no=0;        
        clear Neighbours_DeltaX
        clear Neighbours_X
        %�ҵ������ھ�
        for j=1:SearchAgents_num
            Dist2Enemy=distance(X(:,i),X(:,j));%����ŷ�Ͼ���
            if (all(Dist2Enemy<=r) && all(Dist2Enemy~=0))
                index=index+1;%�ھ����
                neighbours_no=neighbours_no+1;%�ھ�����
                Neighbours_DeltaX(:,index)=DeltaX(:,j);
                Neighbours_X(:,index)=X(:,j);
            end
        end        
        % ���� - ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        % ����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.2)
        if neighbours_no>1
            A=(sum(Neighbours_DeltaX')')/neighbours_no;
        else
            A=DeltaX(:,i);
        end        
        % �ھ�%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.3)
        if neighbours_no>1
            C_temp=(sum(Neighbours_X')')/neighbours_no;
        else
            C_temp=X(:,i);
        end        
        C=C_temp-X(:,i);        
        % ����ʳ��%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.4)
        Dist2Food=distance(X(:,i),Food_pos(:,1));
        if all(Dist2Food<=r)
            F=Food_pos-X(:,i);
        else
            F=0;
        end        
        % Զ�����%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Eq. (3.5)
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
        
        if any(Dist2Food>r) %���ʳ��λ�ò�����������λ��
            %%���и�������� i ����ʱ
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
                %��û���κθ�������� i ����ʱ
                X(:,i)=X(:,i)+Levy(dim)'.*X(:,i);
                DeltaX(:,i)=0;
            end
        else% ���ʳ��λ������������λ��
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
        %��Χ����������ȡ���ޣ���ΧС��������ȡ���ޣ����򲻱�
        X(:,i)=(X(:,i).*(~(Flag4ub+Flag4lb)))+ub'.*Flag4ub+lb'.*Flag4lb;        
    end
    Best_score=Food_fitness;
    Best_pos=Food_pos;
    cg_curve(iter)=Best_score;
end