% a = load('environment.txt');%blacb--barrier occputy 35%.
% a = load('environment6060.mat', 'k');
% a = a.k;
a=rand(10)>0.35; %0��������С��35%��20*20logical
n=size(a,1);%��ȡ����
b=a;
b(end+1,end+1)=0;%pcolor���������һ�к����һ�У������ﲹһ�к�һ��
figure;
pcolor(b); % ����դ����ɫ
colormap([0 0 0;1 1 1])%������ɫ[0 0 0]�Ǻ�ɫ[1 1 1]�ǰ�ɫ
%%
set(gca,'XTick',[],'YTick',[]);  % ��������
axis image xy
 
displayNum(n);%��ʾդ���е���ֵ
 
text(1,n+1.5,'START','Color','red','FontSize',10);%��ʾstart�ַ�
text(n+1,1.5,'GOAL','Color','red','FontSize',10);%��ʾgoal�ַ�
 
hold on
%pin strat&goal positon
scatter(1+0.75,n+0.5,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0], 'LineWidth',0.05);%start point
scatter(n+0.75,1+0.5,'MarkerEdgeColor',[1 0 0],'MarkerFaceColor',[1 0 0], 'LineWidth',0.05);%goal point
 
hold on