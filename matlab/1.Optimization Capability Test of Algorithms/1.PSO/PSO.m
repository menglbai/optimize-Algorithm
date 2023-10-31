% ����Ȩ�������½�(Linearly Decreasing Inertia Weight, LDW)
function [Best_score,Best_pos,cg_curve]=PSO(N,Max_iteration,Xmin,Xmax,dim,fobj)
%convergence curve��������
cg_curve=zeros(1,Max_iteration); %���ε������Ž�
%��ʼ������
% N=150;                    %���Ӹ���
% dim=2;                    %����ά��
% Nmax=1000;                %����������
c1=1.5;                 	%ѧϰ����1
c2=1.5;                 	%ѧϰ����2
Wmax=0.8;               	%����Ȩ�����ֵ
Wmin=0.4;               	%����Ȩ����Сֵ
% Xmax=500;                 %λ�����ֵ
% Xmin=-500;                %λ����Сֵ
k=(rand+1)*0.1;
Vmax=k*Xmax;                %�ٶ����ֵ
Vmin=-Vmax;                	%�ٶ���Сֵ
%% ��ʼ����Ⱥ����λ�ú��ٶ�
x=rand(N,dim)*(Xmax-Xmin)+Xmin;
v=rand(N,dim)*(Vmax-Vmin)+Vmin;
p=x;
pbest=ones(N,1);
% �����ʼȺ��ĺ���ֵ
for i=1:N
    pbest(i)=fobj(x(i,:));
end
% ���Ž������ֵ
Best_pos=ones(1,dim);
Best_score=inf;
for i=1:N
    if(pbest(i)<Best_score)
        Best_pos=p(i,:);
        Best_score=pbest(i);
    end
end
%% ���չ�ʽ���ε���ֱ�����㾫�Ȼ��ߵ�������
for i=1:Max_iteration
    for j=1:N
        %%%%%%%%%%���¸�������λ�ú�����ֵ%%%%%%%%%
        if(fobj(x(j,:))<pbest(j))
            p(j,:)=x(j,:);
            pbest(j)=fobj(x(j,:));
        end
        %%%%%%%%%%����ȫ������λ�ú�����ֵ%%%%%%%%%
        if(pbest(j)<Best_score)
            Best_pos=p(j,:);
            Best_score=pbest(j);
        end
        %%%%%%%%%%���㶯̬����Ȩ��ֵ%%%%%%%%%%%
        w=Wmax-(Wmax-Wmin)*i/Max_iteration;
%       w=0.6;
        %%%%%%%%%%����λ�ú��ٶ�ֵ%%%%%%%%%%%%%
        v(j,:)=w*v(j,:)+c1*rand*(p(j,:)-x(j,:))+c2*rand*(Best_pos-x(j,:));
        x(j,:)=x(j,:)+v(j,:);
        %%%%%%%%%%%�߽���������%%%%%%%%%%%%%%%
        for ii=1:dim
            if (v(j,ii)>Vmax)||(v(j,ii)< Vmin)
                v(j,ii)=rand*(Vmax-Vmin)+Vmin;
            end
            if (x(j,ii)>Xmax)||(x(j,ii)< Xmin)
                x(j,ii)=rand*(Xmax-Xmin)+Xmin;
            end
        end
    end
    %%%%%%%%%%%%��¼����ȫ������ֵ%%%%%%%%%%%%%
    cg_curve(i)=Best_score;
end
% figure(1)
% plot(gb)
% xlabel('��������');
% ylabel('��Ӧ��ֵ');
% title('��Ӧ�Ƚ�������')
end