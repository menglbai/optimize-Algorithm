function o=distance_3D(a,b)
for i=1:size(a,1)
    for j=1:size(a,2)
%     o(1,i)=sqrt((a(i)-b(i))^2);
    o(i,j)=abs(a(i,j)-b(i,j));
    end
end
end