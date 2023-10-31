clear
close
clc
R=30;% 实验次数
test.sol=[];
test.res=[];
Result_PSO=repmat(test,R,1);
cg.cg=[];
cg_PSO=repmat(cg,R,1);
N=30;% 粒子数量
Max_iteration=500;% 最大迭代次数
Function_name='F1';
for r=1:R
tic
disp('PSO is optimizing your problem');
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);
[Best_score,Best_pos,cg_curve]=PSO(N,Max_iteration,lb,ub,dim,fobj);

display(['The best solution obtained by PSO is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by PSO is : ', num2str(Best_score)]);  
toc
Result_PSO(r).res=Best_score;
Result_PSO(r).sol=Best_pos;
cg_PSO(r).cg=cg_curve;
end
Best_PSO=Result_PSO(1).res;
for i=2:R
    if Result_PSO(i).res<Best_PSO
        Best_PSO=Result_PSO(i).res;
    end
end
Best_PSO
Worst_PSO=Result_PSO(1).res;
for i=2:R
    if Result_PSO(i).res>Worst_PSO
        Worst_PSO=Result_PSO(i).res;
    end
end
Worst_PSO
s=0;
for k=1:R
    s=s+Result_PSO(k).res;
end
Mean_PSO=s/R
o=0;
for ii=1:R
    o=o+(Result_PSO(ii).res-Mean_PSO)^2;
end
Std_PSO=sqrt(o/(R-1))
C=zeros(1,Max_iteration);
for c=1:R
    C=C+cg_PSO(c).cg;
end
CG_PSO=C./R;

figure('Position',[400 400 560 190])
%Draw search space
subplot(1,2,1);
func_plot(Function_name);
title('Rosenbrock Function','Fontname','Times New Roman','FontSize',20);
xlabel('x_1','Fontname','Times New Roman','FontSize',15);
ylabel('x_2','Fontname','Times New Roman','FontSize',15);
zlabel([Function_name,'( x_1 , x_2 )'],'Fontname','Times New Roman','FontSize',15)
grid off
%Draw objective space
subplot(1,2,2);
semilogy(CG_PSO,'Color','r')
title('Convergence curve')
xlabel('Iteration');
ylabel('Best score obtained so far');
axis tight
grid off
box on
legend('PSO')