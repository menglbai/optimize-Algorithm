CNT=25;
x=F18(:,1);
y1=F18(:,2);
y2=F18(:,4);
y3=F18(:,6);
y4=F18(:,7);
PSO=log10(y1);
DA=log10(y2);
ADDA=log10(y3);
TPDA=log10(y4);
% PSO=y1;
% DA=y2;
% ADDA=y3;
% TPDA=y4;
k=round(linspace(1,size(x,1),CNT));
hold on
% plot(x,PSO,'color',[0.165,0.333,0.498]);
% plot(x,DA,'color',[0.271,0.737,0.612]);
% plot(x,ADDA,'color',[0.941,0.314,0.463]);
% plot(x,TPDA,'color',[1,0.804,0.431]);
plot(x,PSO,'r');
plot(x,DA,'k');
plot(x,ADDA,'g');
plot(x,TPDA,'b');
plot(x(k),PSO(k),'r>');
plot(x(k),DA(k),'ks');
plot(x(k),ADDA(k),'go');
plot(x(k),TPDA(k),'bp');
title('F18','Fontname','Times New Roman','Fontsize',20)
xlabel('Iteration','Fontname','Times New Roman','Fontsize',16);
set(gca,'XTick',0:100:500);
set(gca,'XTicklabel',{'0','100','200','300','400','500'})
ylabel('\rmlog \itf(x)','Fontname','Times New Roman','Fontsize',16);
legend('PSO','DA','ADDA','TPDA')