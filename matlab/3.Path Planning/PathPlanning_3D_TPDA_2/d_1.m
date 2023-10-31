function f_1=d_1(Y)
[a, ~]=size(Y);
S=[1 1 1];
T=[100 100 40];
sum=d_0(Y(1,:),S)+d_0(Y(a,:),T);
for i=1:a-1
    sum=sum+d_0(Y(i,:),Y(i+1,:));
end
f_1=sum;
end