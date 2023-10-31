function displayNum(n)
%输入参数
%矩阵维数----n
%无输出
%%
x_text = 1:n*n;%产生1-400的数值.
%将数值在栅格图上显示出来
for i = 1:n*n
    [row,col] = ind2sub(n,i);
    [array_x,array_y] = arry2orxy(n,row,col);
    text(array_x+0.2,array_y+0.5,num2str(x_text(i)));
end
%验证栅格数值与行列值是否对应
test_num = [32 100 255];
% [row1,col1] = ind2sub(n,test_num1);
% [row2,col2] = ind2sub(n,test_num2);
% [row3,col3] = ind2sub(n,test_num3);
% [array_x1,array_y1] = arry2orxy(n,row1,col1);
% [array_x2,array_y2] = arry2orxy(n,row2,col2);
% [array_x3,array_y3] = arry2orxy(n,row3,col3);
%%
for j=1:3
    [row(j),col(j)]=ind2sub(n,test_num(j));
    [array_x(j),array_y(j)]=arry2orxy(n,row(j),col(j));
end
%%
fprintf('the value %d is on array_x = %d,array_y = %d\n',test_num(1),array_x(1),array_y(1));%显示校对信息，供人工检验.
fprintf('the value %d is on array_x = %d,array_y = %d\n',test_num(2),array_x(2),array_y(2));
fprintf('the value %d is on array_x = %d,array_y = %d\n',test_num(3),array_x(3),array_y(3));
end