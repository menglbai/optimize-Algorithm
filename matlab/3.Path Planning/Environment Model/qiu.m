r=[6 10 12 8];
a=[20 30 45 80];
b=[30 65 40 70];
c=[40 15 20 33];
[x,y,z]=sphere(50);
for i=1:4
    surf(r(i)*x+a(i),r(i)*y+b(i),r(i)*z+c(i));
    hold on
end

axis([0,100,0,100,0,50]); 

