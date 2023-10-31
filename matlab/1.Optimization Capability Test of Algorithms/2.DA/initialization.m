% 初始化种群
function Positions=initialization(SearchAgents_num,dim,ub,lb)
Boundary_no= size(ub,2); 
if Boundary_no==1
    ub_new=ones(1,dim)*ub;
    lb_new=ones(1,dim)*lb;
else
     ub_new=ub;
     lb_new=lb;   
end
for i=1:dim
    ub_i=ub_new(i);
    lb_i=lb_new(i);
    Positions(:,i)=rand(SearchAgents_num,1).*(ub_i-lb_i)+lb_i;
end
Positions=Positions';
end