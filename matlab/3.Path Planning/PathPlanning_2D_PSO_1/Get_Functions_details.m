%18个测试函数，用于测试智能优化算法性能
%lb:解空间下限；ub:解空间上限；dim:解空间维数
function [lb,ub,dim,fobj] = Get_Functions_details(F)
D=10;
switch F
    case 'F1'
        fobj = @F1;
        lb=-100;
        ub=100;
        dim=D;
    case'F2'
        fobj=@F2;
        lb=-100;
        ub=100;
        dim=D;
    case'F3'
        fobj=@F3;
        lb=-10;
        ub=10;
        dim=D;
    case'F4'
        fobj=@F4;
        lb=-1.28;
        ub=1.28;
        dim=D;  
     case'F5'
        fobj=@F5;
        lb=-5;
        ub=10;
        dim=D;
    case 'F6'
        fobj = @F6;
        lb=-100;
        ub=100;
        dim=D;
    case 'F7'
        fobj = @F7;
        lb=-100;
        ub=100;
        dim=D;
    case 'F8'
        fobj = @F8;
        lb=-10;
        ub=10;
        dim=D;
    case 'F9'
        fobj = @F9;
        lb=-30;
        ub=30;
        dim=D;  
    case'F10'
        fobj=@F10;
        lb=-10;
        ub=10;
        dim=D;
    case'F11'
        fobj=@F11;
        lb=-100;
        ub=100;
        dim=2;
    case'F12'
        fobj=@F12;
        lb=-5;
        ub=15;
        dim=2;
    case'F13'
        fobj=@F13;
        lb=-500;
        ub=500;
        dim=D;
    case 'F14'
        fobj = @F14;
        lb=-5.12;
        ub=5.12;
        dim=D;
    case'F15'
        fobj=@F15;
        lb=-100;
        ub=100;
        dim=2;
    case'F16'
        fobj=@F16;
        lb=-5;
        ub=5;
        dim=2;
    case'F17'
        fobj = @F17;
        lb=-32.768;
        ub=32.768;
        dim=D;      
    case'F18'
        fobj = @F18;
        lb=-600;
        ub=600;
        dim=D;
    case'F'
        fobj=@F;
        lb=-50;
        ub=50;
        dim=30;
end
end

% F1(Sphere Function,US)
function o = F1(x)
o=sum(x.^2);
end
%F2 (Step Function,US)
function o=F2(x)
col=max(size(x));
sum=0;
for i=1:col
    sum=sum+(floor(x(i)+0.5))^2;
end
o=sum;
end
%F3 (Sum Squares Function,US)
function o=F3(x)
col=max(size(x));
sum=0;
for i=1:col
    sum=sum+i*x(i)^2;
end
o=sum;
end
%F4 (Quartic with Noise,US)
function o=F4(x)
col=max(size(x));
sum=0;
for i=1:col
    sum=sum+i*x(i)^4;
end
o=sum+rand;
end
%F5 (Zakharov Function,UN)
function o=F5(x)
col=max(size(x));
sum1=0;
sum2=0;
for i=1:col
    xi=x(i);
    sum1=sum1+xi^2;
    sum2=sum2+0.5*i*xi;
end
o=sum1+sum2^2+sum2^4;
end
%F6 (Schwefel_1.2,UN)
function o = F6(x)
col=size(x,2);%x的列数
s=0;
for i=1:col
    s=s+sum(x(1:i))^2;
end
o=s;
end
% F7 (Schwefel_2.21,UN)
function o = F7(x)
o=max(abs(x));
end
% F8 (Schwefel_2.22,UN)
function o = F8(x)
o=sum(abs(x))+prod(abs(x));%连加和连乘
end
% F9 (Rosenbrock Function,UN)
function o = F9(x)
col=max(size(x));
sum=0;
for i=1:(col-1)
    xi=x(i);
    xnext=x(i+1);
    new=100*(xnext-xi^2)^2+(xi-1)^2;
    sum=sum+new;
end
o=sum;
end
%F10 (Dixon-Price Function,UN)
function o=F10(x)
col=max(size(x));
sum=(x(1)-1)^2;
for i=2:col
    sum=sum+i*(2*x(i)^2-x(i-1))^2;
end
o=sum;
end
%F11 (Bohachevsky1,MS)
function o=F11(x)
o=x(1)^2+2*x(2)^2-0.3*cos(3*pi*x(1))-0.4*cos(4*pi*x(2))+0.7;
end
%F12 (Booth,MS)
function o=F12(x)
col=max(size(x));
% o=(x(1)+2*x(2)-7)^2+(2*x(1)+x(2)-5)^2;
o=(x(2)-(5.1/(4*pi*pi))*x(1)^2+(5/pi)*x(1)-6)^2+10*(1-1/(8*pi))*cos(x(1))+10;
end
%F13 (Schwefel_226,MS)
function o=F13(x)
col=max(size(x));
sum=0;
for i=1:col
    sum=sum+x(i)*sin(sqrt(abs(x(i))));
end
o=418.9829*d-sum;
end
% F14 (Rastrigin Function,MS)
function o = F14(x)
col=size(x,2);
s=0;
for i=1:col
    s=s+(x(i)^2-10*cos(2*pi*x(i))+10);
end
o=s;
end
%F15 (Schaffer,MN)
function o=F15(x)
o=0.5+((sin(sqrt(x(1)^2+x(2)^2)))^2-0.5)/(1+0.001*(x(1)^2+x(2)^2))^2;
end
%F16 (Six Hump Camel Back,MN)
function o=F16(x)
o=4*x(1)^2-2.1*x(1)^4+(1/3)*x(1)^6+x(1)*x(2)-4*x(2)^2+4*x(2)^4;
end
% F17 (Ackley Function,MN)
function o = F17(x)
col=size(x,2);
o=-20*exp(-0.2*sqrt(sum(x.^2)/col))-exp(sum(cos(2*pi.*x))/col)+20+exp(1);
end
% F18 (Griewank Function,MN)
function o = F18(x)
col=size(x,2);
o=sum(x.^2)/4000-prod(cos(x./sqrt([1:col])))+1;
end
% F
function Length=F(y)
col=max(size(y));
S=[0 0];
T=[100 0];
a=[20,40,60,85];
b=[0,20,-10,5];
r=[10,12,18,10];
x=ones(1,col+1);
for j=1:col
    x(j)=j*(T(1)-S(1))/(col+1);
end
Length=sqrt((x(1)-S(1))^2+(y(1)-S(2))^2)+sqrt((x(col)-T(1))^2+(y(col)-T(2))^2);
for i=1:col-1
    Length=Length+sqrt((y(i)-y(i+1))^2+(x(i)-x(i+1))^2);
end
for i=1:col
    for k=1:max(size(a))
        if (x(i)-a(k))^2+(y(i)-b(k))^2<=r(k)^2
            Length=Length*1000;
            break
        end
    end
end
for i=1:col-1
    for k=1:max(size(a))
%         if (x(i)/3+x(i+1)*2/3-a(k))^2+(y(i)/3+y(i+1)*2/3-b(k))^2<=r(k)^2 ...
%             || (x(i)*2/3+x(i+1)/3-a(k))^2+(y(i)*2/3+y(i+1)/3-b(k))^2<=r(k)^2
%             Length=Length*1000;
%             break
%         end
        if ((x(i)/6+x(i+1)*5/6)-a(k))^2+((y(i)/6+y(i+1)*5/6)-b(k))^2<=r(k)^2 ...
             || ((x(i)*2/6+x(i+1)*4/6)-a(k))^2+((y(i)*2/6+y(i+1)*4/6)-b(k))^2<=r(k)^2 ...
             || ((x(i)*3/6+x(i+1)*3/6)-a(k))^2+((y(i)*3/6+y(i+1)*3/6)-b(k))^2<=r(k)^2 ...
             || ((x(i)*4/6+x(i+1)*2/6)-a(k))^2+((y(i)*4/6+y(i+1)*2/6)-b(k))^2<=r(k)^2 ...
             || ((x(i)*5/6+x(i+1)/6)-a(k))^2+((y(i)*5/6+y(i+1)/6)-b(k))^2<=r(k)^2
             Length=Length*10000;
%             if (xx(jj)-a(k))^2+(yy(jj)-b(k))^2<=r(k)^2
                Length=Length*1000;
                break
        end
    end
    
end
if (x(col)/3+T(1)*2/3-a(max(size(a))))^2+(y(col)/3+T(2)*2/3-b(max(size(a))))^2<=r(max(size(a)))^2 ...
    || (x(col)*2/3+T(1)/3-a(max(size(a))))^2+(y(col)*2/3+T(2)/3-b(max(size(a))))^2<=r(max(size(a)))^2
    Length=Length*1000;
end 
end
