function f=Length(X)
S=[0 0 0];
T=[100 100 40];
sum=d(S,X(1,:))+d(X(max(size(X)),:),T);
for i=1:max(size(X))-1
    sum=sum+d(X(i,:),X(i+1,:));
end
f=sum;
end