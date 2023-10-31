%% ��ʧһ���ԣ�������ʼ�㡢Ŀ����Լ��ϰ���Ϣ���ܿ��Թ���һ������ʼ���Ŀ�������Ϊx��Ķ�άƽ������ϵ��
clear
clc
close
%% ����������ͼ
%% ��вԴ��Ϣ
A=[25,40,60,80];%������
B=[0,30,-20,15];%������
R=[10,12,18,10];%��в�뾶
theta=0:pi/100:2*pi;%��0-2pi�ֳ�200��
color=['r','b','c','y'];%����в������ɫ
for j=1:size(A,2)
    x=A(j)+R(j)*cos(theta);
    y=B(j)+R(j)*sin(theta);
    plot(x,y);
    fill(x,y,color(j));
    hold on
end
axis equal           %����������Ķ���ϵ�������ֵͬ ,����λ������ͬ
axis([0,100,-50,50]); 
xlabel('x km');
ylabel('y km');
%% ������ʼ���Ŀ��㣬��������ʼ���Ŀ���Ϊ����
Start=[0,0];
End=[100,0];
DSE=sqrt((Start(1)-End(1))^2+(Start(2)-End(2))^2);
plot(Start(1),Start(2),'r*');
plot(End(1),End(2),'bd');
text(Start(1)+1,Start(2),'START');
text(End(1)+1,End(2),'GOAL');
%% ��ʼ������
L=20;                   	%Ⱥ�����Ӹ�����ÿ�����Ӵ���һ������
ndot=12;                    %����ά����һ�������к����ڵ�ĸ���-1
Nmax=100;                   %����������
c1=1.5;                 	%ѧϰ����1
c2=1.5;                 	%ѧϰ����2
Wmax=0.9;               	%����Ȩ�����ֵ
Wmin=0.4;               	%����Ȩ����Сֵ
Xmax=50;                 	%λ�����ֵ��Ҳ��x������
Xmin=-50;                  	%λ����Сֵ��Ҳ��x������
K=0.1;                      %����ϵ��
Vmax=K*Xmax;                %�ٶ����ֵ
Vmin=K*Xmin;                %�ٶ���Сֵ
%% ��Ⱥ��ʼ��
for j=1:L
    for i=1:ndot+1
    Path(i,1,j)=100/(ndot+2)*i;       %������ĺ����������ȷ��������(ndot+2)�ȷ�
    Path(i,2,j)=rand*(Xmax-Xmin)+Xmin;%��������������������
    end
end
V=rand(ndot+1,1,L)*(Vmax-Vmin)+Vmin;  %�ٶ����ڵ���������
%��ʼ����������λ�ã�������������ֵ���������ȣ�
P=Path;
pbest=ones(L,1);
for i=1:L
    pbest(i)=Func1(Path(:,:,i),Start,End);%Path(:,:,i)�������ڵ�
end
%��ʼ��ȫ������λ�ã�������������ֵ���������ȣ�
g=ones(ndot,2);
gbest=inf;
for i=1:L
    if(pbest(i)<gbest)
        g=P(:,:,i);
        gbest=pbest(i);
    end
end
gb=ones(1,Nmax);%���ڴ洢ÿ�ε��������Ž�
%% ���չ�ʽ���ε���ֱ�����㾫�Ȼ��ߵ�������
for i=1:Nmax
    for j=1:L
        %���¸�������λ�ú�����ֵ
        if Func1(Path(:,:,j),Start,End)<pbest(j)
            P(:,:,j)=Path(:,:,j);
            pbest(j)=Func1(Path(:,:,j),Start,End);
        end
        %����ȫ������λ�ú�����ֵ
        if(pbest(j)<gbest)
            g=P(:,:,j);
            gbest=pbest(j);
        end
        %���㶯̬����Ȩ��ֵ
        w=Wmax-(Wmax-Wmin)*i/Nmax;
        %����λ�ú��ٶ�ֵ����Ҫ����������
        V(:,1,j)=w*V(:,1,j)+c1*rand*(P(:,2,j)-Path(:,2,j))+c2*rand*(g(:,2)-Path(:,2,j));
        Path(:,2,j)=Path(:,2,j)+V(:,1,j);
        %�߽���������
        for ii=1:ndot+1
            if (V(ii,1,j)>Vmax)||(V(ii,1,j)<Vmin)
                V(ii,1,j)=rand*(Vmax-Vmin)+Vmin;
            end
            if (Path(ii,2,j)>Xmax)||(Path(ii,2,j)<Xmin)
                Path(ii,2,j)=rand*(Xmax-Xmin)+Xmin;
            end
        end
        %���������������в���������������������
        for k=1:ndot+1
            for l=1:max(size(R))
                while (Path(k,1,j)-A(l))^2+(Path(k,2,j)-B(l))^2<=R(l)^2 
                    Path(k,2,j)=rand*(Xmax-Xmin)+Xmin;
                end
            end            
        end
        %�����ں�����֮��ĺ���������ײ���
        for k=1:ndot
            for l=1:max(size(R))
%                 while ((Path(k,1,j)/5+Path(k+1,1,j)*4/5)-A(l))^2+((Path(k,2,j)/5+Path(k+1,2,j)*4/5)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*2/5+Path(k+1,1,j)*3/5)-A(l))^2+((Path(k,2,j)*2/5+Path(k+1,2,j)*3/5)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*3/5+Path(k+1,1,j)*2/5)-A(l))^2+((Path(k,2,j)*3/5+Path(k+1,2,j)*2/5)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*4/5+Path(k+1,1,j)/5)-A(l))^2+((Path(k,2,j)*4/5+Path(k+1,2,j)/5)-B(l))^2<=R(l)^2
%                     Path(k+1,2,j)=rand*(Xmax-Xmin)+Xmin;
%                 end         
               while ((Path(k,1,j)/3+Path(k+1,1,j)*2/3)-A(l))^2+((Path(k,2,j)/3+Path(k+1,2,j)*2/3)-B(l))^2<=R(l)^2 ...
                        || ((Path(k,1,j)*2/3+Path(k+1,1,j)/3)-A(l))^2+((Path(k,2,j)*2/3+Path(k+1,2,j)/3)-B(l))^2<=R(l)^2
                    Path(k+1,2,j)=rand*(Xmax-Xmin)+Xmin;
                end
%                 while ((Path(k,1,j)/4+Path(k+1,1,j)*3/4)-A(l))^2+((Path(k,2,j)/4+Path(k+1,2,j)*3/4)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*2/4+Path(k+1,1,j)/2)-A(l))^2+((Path(k,2,j)*2/4+Path(k+1,2,j)/2)-B(l))^2<=R(l)^2 ...
%                         || ((Path(k,1,j)*3/4+Path(k+1,1,j)/4)-A(l))^2+((Path(k,2,j)*3/4+Path(k+1,2,j)/4)-B(l))^2<=R(l)^2 
%                     Path(k+1,2,j)=rand*(Xmax-Xmin)+Xmin;
%                 end
            end
        end 
    end
    %%%%%%%%%%%%��¼����ȫ������ֵ%%%%%%%%%%%%%
    gb(i)=gbest;
end
%��ν���������ߺ��ϰ����н��������Լ���ʼ������λ���Ƿ��д�����
g; %���ź���
h(1,:)=Start;
h(ndot+3,:)=End;
h(2:ndot+2,:)=g;
plot(h(:,1),h(:,2),'rp');%���ƺ�����
hh=0:0.1:100;
HH=spline(h(:,1),h(:,2),hh);%����������ֵ������
plot(hh,HH,'-');
figure(2)
plot(gb)
xlabel('��������');
ylabel('��Ӧ��ֵ');
title('��Ӧ�Ƚ�������');