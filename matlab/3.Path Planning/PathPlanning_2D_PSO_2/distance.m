%% 计算两个航迹点之间的距离
function d=distance(A,B)
d=sqrt((A(1)-B(1))^2+(A(2)-B(2))^2); %欧氏距离
end