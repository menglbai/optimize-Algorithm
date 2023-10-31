% Tent混沌映射 产生混沌序列
% x(1)=rand
% x(i+1)=(2x(i))mod1+rand/N
clear
close
clc
N=100000;
ub=60;
x=zeros(1,N);
x(1)=rand;
for i=2:N
    if 2*x(i-1)>1
        x(i)=2-2*x(i-1)+rand/N;
    end
    if 2*x(i-1)<=1 && 2*x(i-1)>=0
        x(i)=2*x(i-1)+rand/N;
    end
end
for i=1:N
    if x(i)>1
        x(i)=1;
    end
    if x(i)<0
        x(i)=0;
    end
end
% X=x.*ub*2-ub;% 生成[-ub,ub]区间内均匀分布的N个个体
[Y,I]=sort(x,'descend');
plot(Y,'.')
H=histogram(Y,230);
% H.FaceColor=[0.67 0 1];
% H.EdgeColor='b';
axis([0 1 0 550])
xlabel('Tent chaotic map','Fontname','Times New Roman','Fontsize',16);
% ylabel('个体数量');
