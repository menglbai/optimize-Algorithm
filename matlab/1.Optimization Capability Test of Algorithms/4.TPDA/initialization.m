function Positions=initialization(SearchAgents_num,dim,ub,lb)
col= size(ub,2);
if col==1
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
Positions=Positions';% 每一列为一个个体