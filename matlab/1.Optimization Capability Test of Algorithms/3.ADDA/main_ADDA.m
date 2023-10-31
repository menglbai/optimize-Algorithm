% ADDA�㷨������
clear
close
clc
R=30;% ʵ�����
test.sol=[];
test.res=[];
Result_ADDA=repmat(test,R,1);
cg.cg=[];
cg_ADDA=repmat(cg,R,1);
SearchAgents_num=30; % �������� 
Max_iteration=1000; % ����������
Function_name='F'; % Name of the test function
for r=1:R
tic
% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);%������ز�����ʼ��
[Best_score,Best_pos,cg_curve]=ADDA(SearchAgents_num,Max_iteration,lb,ub,dim,fobj);
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
display(['The best solution obtained by ADDA is : ', num2str(Best_pos')]);
display(['The best optimal value of the objective funciton found by ADDA is : ', num2str(Best_score)]);  
toc
Result_ADDA(r).res=Best_score;
Result_ADDA(r).sol=Best_pos;
cg_ADDA(r).cg=cg_curve;
end
Best_ADDA=Result_ADDA(1).res;
for i=2:R
    if Result_ADDA(i).res<Best_ADDA
        Best_ADDA=Result_ADDA(i).res;
    end
end
Best_ADDA
Worst_ADDA=Result_ADDA(1).res;
for i=2:R
    if Result_ADDA(i).res>Worst_ADDA
        Worst_ADDA=Result_ADDA(i).res;
    end
end
Worst_ADDA
s=0;
for k=1:R
    s=s+Result_ADDA(k).res;
end
Mean_ADDA=s/R
o=0;
for ii=1:R
    o=o+(Result_ADDA(ii).res-Mean_ADDA)^2;
end
Std_ADDA=sqrt(o/(R-1))
C=zeros(1,Max_iteration);
for c=1:R
    C=C+cg_ADDA(c).cg;
end
CG_ADDA=C./R;