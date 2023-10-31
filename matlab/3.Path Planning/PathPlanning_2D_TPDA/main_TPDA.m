clear
close
clc
R=30;% 实验次数
test.sol=[];
test.res=[];
Result_TPDA=repmat(test,R,1);
cg.cg=[];
cg_TPDA=repmat(cg,R,1);
SearchAgents_num=30; % 蜻蜓数量 
Max_iteration=500; % 最大迭代次数
Function_name='F'; % Name of the test function
for r=1:R
tic
% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);%函数相关参数初始化
[Best_score,Best_pos,cg_curve]=TPDA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj);
% figure('Position',[400 400 560 190])
% %Draw search space
% subplot(1,2,1);
% func_plot(Function_name);
% title('Test function')
% xlabel('x_1');
% ylabel('x_2');
% zlabel([Function_name,'( x_1 , x_2 )'])
% grid off
% %Draw objective space
% subplot(1,2,2);
% semilogy(cg_curve,'Color','r')
% title('Convergence curve')
% xlabel('Iteration');
% ylabel('Best score obtained so far');
% axis tight
% grid off
% box on
% legend('DA')
display(['The best solution obtained by TPDA is : ', num2str(Best_pos')]);
display(['The best optimal value of the objective funciton found by TPDA is : ', num2str(Best_score)]);  
toc
Result_TPDA(r).res=Best_score;
Result_TPDA(r).sol=Best_pos;
cg_TPDA(r).cg=cg_curve;
end
Best_TPDA=Result_TPDA(1).res;
for i=2:R
    if Result_TPDA(i).res<Best_TPDA
        Best_TPDA=Result_TPDA(i).res;
    end
end
for i=1:R
    if Result_TPDA(i).res==Best_TPDA
        t=i;
        break
    end
end
Best_TPDA
Worst_TPDA=Result_TPDA(1).res;
for i=2:R
    if Result_TPDA(i).res>Worst_TPDA
        Worst_TPDA=Result_TPDA(i).res;
    end
end
Worst_TPDA
s=0;
for k=1:R
    s=s+Result_TPDA(k).res;
end
Mean_TPDA=s/R
o=0;
for ii=1:R
    o=o+(Result_TPDA(ii).res-Mean_TPDA)^2;
end
Std_TPDA=sqrt(o/(R-1))
C=zeros(1,Max_iteration);
for c=1:R
    C=C+cg_TPDA(c).cg;
end
CG_TPDA=C./R;