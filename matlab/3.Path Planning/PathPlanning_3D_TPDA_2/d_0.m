function f_0=d_0(a,b)
sum=0;
for i=1:length(a)
    sum=sum+((a(i)-b(i))^2);
end
f_0=sqrt(sum);
end