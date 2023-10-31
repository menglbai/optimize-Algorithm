clear
clc
%% ��ʾ����������ƣ���x,y�γɸ�����
t=1:100;
[X,Y] =meshgrid(t);
%% ��׼���ν�ģ 
h1=sin(Y+10)+0.2*sin(X)+0.1*cos(0.6*sqrt(X^2+Y^2))+0.1*cos(Y)+0.1*sin(0.1*sqrt(X^2+Y^2));
%��ϵ���趨Ϊa=10,b=0.2,c=0.1,d=0.6,e=0.1,f=0.1,g=0.1
%% ɽ�彨ģ
h=[13 18 15 11];
x0=[30 40 60 80];
y0=[40 70 15 45]; 
xs=[9 8 10 8]; 
ys=[10 8 5 8]; 
for x=1:100 
    for y=1:100 
        for i=1:4
            h2(i)=h(i)*exp(-((x-x0(i))/xs(i))^2-((y-y0(i))/ys(i))^2); 
            h3(x,y)=sum(h2); 
        end
    end
end
%% ��в��ģ
% xx=[10 30 60 60];
% yy=[15 60 30 70];
% a=[12 13 15 6];
% b=[2 5 3 4];
% c=[1 2 2 4];
% d=[2 3 5 1];
% for x=1:100
%     for y=1:100
%             h4(x,y)=a(1)/(b(1)+c(1)*(x-xx(1))^2+d(1)*(y-yy(1))^2);
%             h5(x,y)=a(2)/(b(2)+c(2)*(x-xx(2))^2+d(2)*(y-yy(2))^2);
%             h6(x,y)=a(3)/(b(3)+c(3)*(x-xx(3))^2+d(3)*(y-yy(3))^2);
%             h7(x,y)=a(4)/(b(4)+c(4)*(x-xx(4))^2+d(4)*(y-yy(4))^2);
%     end
% end
%%
rr=[5 5 6 4 20];
aa=[20 30 45 80 70];
bb=[30 65 40 70 70];
cc=[40 15 20 33 0];
[x,y,z]=sphere(50);
for i=1:5
    surf(rr(i)*x+aa(i),rr(i)*y+bb(i),rr(i)*z+cc(i));
    hold on
end
%%
% h=max(max(max(h4,h5),h6),h7);
z=max(h1,h3);
% z=max(h1,h3);
r=max(size(z));
%���ú�������
x=1:r;
y=1:r;
%������X��Y��Zָ����������
mesh(x,y,z);
%����һ����ά����ͼ 
surf(x,y,z);
%��ɫӳ�伴ɫͼ������ǰͼ������ɫͼ����ΪԤ�������ɫͼ֮һ 
colormap; 
%���ɫ��
colorbar; 
%�����޶� 
axis([0,100,0,100,0,50]); 
%�������ͼ�ζ������ɫ��ɫ����ɫ�ʵĲ�ֵ����ʹɫ��ƽ������ 
shading interp;
xlabel('x km');
ylabel('y km');
zlabel('z m');
%%
hold on
s1=sprintf('(%d,%d,%d)',10,20,11);
plot3(10,20,11,'go');
% text(10,20,11,'o','color','g','FontSize',20);
text(6,20,13,'START','color','g','FontSize',15);
s2=sprintf('(%d,%d,%d)',80,80,15);
plot3(80,80,30,'rp');
% text(80,80,15,'*','color','r','FontSize',20);
text(76,80,33,'GOAL','color','r','FontSize',15);