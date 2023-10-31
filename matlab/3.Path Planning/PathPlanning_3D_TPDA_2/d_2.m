function f_2=d_2(X)
a=length(X);
Y=zeros(a/3,3);
for i=1:a/3
    for j=1:3
        Y(i,j)=X(3*(i-1)+j);
    end
end
f_2=d_1(Y);
end