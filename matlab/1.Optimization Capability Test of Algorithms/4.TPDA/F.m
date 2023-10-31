function Length=F(y)
col=max(size(y));
S=[0 0];
T=[100 0];
a=[25,40,60,80];
b=[0,30,-20,15];
r=[10,12,18,10];
x=ones(1,col+1);
for j=1:col
    x(j)=j*(T(1)-S(1))/(col+1);
end
Length=sqrt((x(1)-S(1))^2+(y(1)-s(2))^2)+sqrt((x(col)-T(1))^2+(y(col)-T(2))^2);
for i=1:col-1
    Length=Length+sqrt((y(i)-y(i+1))^2+(x(i)-x(i+1))^2);
end
for i=1:col
    for k=1:max(size(a))
        if (x(i)-a(k))^2+(y(i)-b(k))^2<=r(k)^2
            Length=Length*1000;
            break
        end
        break
    end
    break
end

