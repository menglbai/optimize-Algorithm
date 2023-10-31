function o=distance_3D(a,b)
for i=1:size(a,1)
%     o(1,i)=sqrt((a(i)-b(i))^2);
    o(1,i)=abs(a(i)-b(i));
end
end