figure(1)
a=[20,40,60,85];%������
b=[0,20,-10,5];%������
R=[10,12,18,10];%��в�뾶
theta=0:pi/100:2*pi;%��0-2pi�ֳ�200��
Name=['A','B','C','D'];%����вԴ����
color=['r','b','c','y'];%����в������ɫ
for j=1:size(a,2)
    x=a(j)+R(j)*cos(theta);
    y=b(j)+R(j)*sin(theta);
    plot(x,y);
    p(j)=fill(x,y,color(j),'DisplayName',Name(j));
    hold on
end
legend([p(1),p(2),p(3),p(4)])
axis equal           %����������Ķ���ϵ�������ֵͬ ,����λ������ͬ
axis([0,100,-50,50]); 
title('\fontname{Times New Roman}PSO\fontname{����}�滮���','FontSize',15)
xlabel('x km','FontSize',15,'FontName','Times New Roman');
ylabel('y km','FontSize',15,'FontName','Times New Roman');
set(gca,'FontSize',12,'FontName','Times New Roman');
%% ������ʼ���Ŀ��㣬��������ʼ���Ŀ���Ϊ����
Start=[0,0];
End=[100,0];
DSE=sqrt((Start(1)-End(1))^2+(Start(2)-End(2))^2);
plot(Start(1),Start(2),'r*','HandleVisibility','off');
plot(End(1),End(2),'bd','HandleVisibility','off');
text(Start(1)+1,Start(2),'START','FontName','Times New Roman');
text(End(1)+1,End(2),'GOAL','FontName','Times New Roman');
col=30;
for j=1:col
    X(j)=j*(End(1)-Start(1))/(col+1);
end
g(:,2)=Result_PSO(3).sol;
g(:,1)=X';
g; %���ź���
h(1,:)=Start;
h(col+2,:)=End;
h(2:col+1,:)=g;
p(5)=plot(h(:,1),h(:,2),'rp','HandleVisibility','off');%���ƺ�����
hh=0:0.1:100;
HH=spline(h(:,1),h(:,2),hh);%����������ֵ������
plot(hh,HH,'-','HandleVisibility','off');
hold on
figure(2)
S=log10(cg_PSO(3).cg);
plot(S);
title('\fontname{Times New Roman}PSO\fontname{����}��������','FontSize',15);
xlabel('\fontname{����}��������','FontSize',15);
ylabel('\fontname{����}��Ӧ��ֵ','FontSize',15);
set(gca,'FontSize',12,'FontName','Times New Roman');