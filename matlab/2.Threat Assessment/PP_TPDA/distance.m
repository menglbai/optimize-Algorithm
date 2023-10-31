function o=distance(a,b)
for i=1:size(a,1)%a,b为列向量
%     o(1,i)=sqrt((a(i)-b(i))^2);
    o(1,i)=abs(a(i)-b(i));%o为行向量
end
end