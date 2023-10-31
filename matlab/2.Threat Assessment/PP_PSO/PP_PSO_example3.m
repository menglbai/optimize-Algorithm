%% ��3 ������ս�ɻ�Ŀ����в����
clear;
clc;
%% ����Ŀ����в��������
filename='D:\matlab\PP\threat_data.xlsx'; %��в����
sheet=6;
A=xlsread(filename,sheet);
format short
% �淶���߾���
for j=1:size(A,2)
    Max(j)=max(A(:,j));
    Min(j)=min(A(:,j));
    Max_Min(j)=Max(j)-Min(j);
end
for i=1:size(A,1)
    for j=1:size(A,2)
        A(i,j)=(A(i,j)-Min(j))/Max_Min(j);
    end
end
%% PSO�㷨������ʼ��
N=30;                    	%Ⱥ�����Ӹ���
D=size(A,2);                %����ά��
Nmax=500;                   %����������
c1=1.5;                 	%ѧϰ����1
c2=1.5;                 	%ѧϰ����2
Wmax=0.8;               	%����Ȩ�����ֵ
Wmin=0.4;               	%����Ȩ����Сֵ
Xmax=1;                 	%λ�����ֵ
Xmin=-1;                 	%λ����Сֵ
Vmax=0.1;                 	%�ٶ����ֵ
Vmin=-0.1;                	%�ٶ���Сֵ
Ntest=30;                   %ʵ�����
test.g=[];
test.gbest=[];
%Initialize Population Array
pop=repmat(test,Ntest,1);
for jj=1:Ntest
%��ʼ����Ⱥ����λ�ú��ٶ�
x=rand(N,D)*(Xmax-Xmin)+Xmin;
v=rand(N,D)*(Vmax-Vmin)+Vmin;
%��ʼ����������λ�ú�����ֵ
p=x;
pbest=ones(N,1);
for i=1:N
    pbest(i)=SD_penaltyfunction(A,x(i,:));
end
%��ʼ��ȫ������λ�ú�����ֵ
g=ones(1,D);
gbest=inf;
for i=1:N
    if(pbest(i)<gbest)
        g=p(i,:);
        gbest=pbest(i);
    end
end
gb=ones(1,Nmax);%ÿ�ε��������Ž�
%% ���չ�ʽ����ֱ�����㾫�Ȼ�������������
for i=1:Nmax
    for j=1:N
        %%%%%%%%%%���¸�������λ�ú�����ֵ%%%%%%%%%
        if(SD_penaltyfunction(A,x(j,:))<pbest(j))
            p(j,:)=x(j,:);
            pbest(j)=SD_penaltyfunction(A,x(j,:));
        end
        %%%%%%%%%%����ȫ������λ�ú�����ֵ%%%%%%%%%
        if(pbest(j)<gbest)
            g=p(j,:);
            gbest=pbest(j);
        end
        %%%%%%%%%%���㶯̬����Ȩ��ֵ%%%%%%%%%%%
        w=Wmax-(Wmax-Wmin)*i/Nmax;
        %%%%%%%%%%����λ�ú��ٶ�ֵ%%%%%%%%%%%%%
        v(j,:)=w*v(j,:)+c1*rand*(p(j,:)-x(j,:))+c2*rand*(g-x(j,:));
        x(j,:)=x(j,:)+v(j,:);
        %%%%%%%%%%%�߽���������%%%%%%%%%%%%%%%
        for ii=1:D
            if (v(j,ii)>Vmax)||(v(j,ii)< Vmin)
                v(j,ii)=rand*(Vmax-Vmin)+Vmin;
            end
            if (x(j,ii)>Xmax)||(x(j,ii)< Xmin)
                x(j,ii)=rand*(Xmax-Xmin)+Xmin;
            end
        end
    end
    %%%%%%%%%%%%��¼����ȫ������ֵ%%%%%%%%%%%%%
    gb(i)=gbest;
end
pop(jj).g=g;
pop(jj).gbest=gbest;
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
AA=A*G';
[B,I]=sort(AA,'descend');%Ŀ�갴��в�̶ȴӴ�С����
for k=1:Ntest
    if floor(pop(k).gbest*10000)>floor(Gbest*10000)
        N_local=N_local+1;
    end
end
figure(1)
plot(gb)
xlabel('��������');
ylabel('��Ӧ��ֵ');
title('��Ӧ�Ƚ�������')
%% ��ָ��Ȩ�رȽ�
figure(2)
G;
TPDA=[0.2687 0.6711 0.2847 0.6230 0.1016];
Y=[];
for i=1:5
    Y(i,1)=G(i);
    Y(i,2)=TPDA(i);
end
X=1:5;
 %����4����״ͼ�����1
h=bar(X,Y,1);      
 %�޸ĺ��������ơ�����
set(gca,'XTickLabel',{'Ŀ������','��·�ݾ�','Ŀ���ٶ�','����ʱ��','����ǿ��'},'FontSize',12,'FontName','����');
% ����������ɫ,��ɫΪRGB��ԭɫ��ÿ��ֵ��0~1֮�伴��
set(h(1),'FaceColor',[30,150,252]/255)     
% set(h(2),'FaceColor',[162,214,249]/255)    
set(h(2),'FaceColor',[252,243,0]/255)    
% set(h(4),'FaceColor',[255,198,0]/255)    
ylim([0 0.8]);      %y��̶�
%�޸�x,y���ǩ
ylabel('\fontname{����}\fontsize{15}Ȩ��');
xlabel('\fontname{����}\fontsize{15}ָ��'); 
%�޸�ͼ��
legend({'\fontname{Times New Roman}PSO','\fontname{Times New Roman}TPDA'},'FontSize',12);
%% ��Ŀ����в�ȱȽ�
% TPDA�����
TPDA_AA=[1.1069 1.3609 1.2211 0.0339 0.1252];
figure(3)
YY=[];
for i=1:5
    YY(i,1)=AA(i); 
    YY(i,2)=TPDA_AA(i);
end
X=1:5;
%����4����״ͼ�����1
h=bar(X,YY,1);      
%�޸ĺ��������ơ�����
set(gca,'XTickLabel',{'A','B','C','D','E'},'FontSize',15,'FontName','Times New Roman');
% ����������ɫ,��ɫΪRGB��ԭɫ��ÿ��ֵ��0~1֮�伴��
% set(h(1),'FaceColor',[30,150,252]/255)     
set(h(1),'FaceColor',[162,214,249]/255)    
% set(h(1),'FaceColor',[252,243,0]/255)    
set(h(2),'FaceColor',[255,198,0]/255)    
ylim([0 1.5]);      %y��̶�
%�޸�x,y���ǩ
ylabel('\fontname{����}\fontsize{16}��в��');
xlabel('\fontname{����}\fontsize{16}Ŀ��'); 
%�޸�ͼ��
legend({'\fontname{Times New Roman}PSO','\fontname{Times New Roman}TPDA'},'FontSize',12);