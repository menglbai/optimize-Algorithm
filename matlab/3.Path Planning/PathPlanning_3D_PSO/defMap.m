function [X,Y,Z] = defMap(mapRange)
% ��ʼ��������Ϣ
N = 6;                              % ɽ�����
peaksInfo = struct;                 % ��ʼ��ɽ��������Ϣ�ṹ��
peaksInfo.center = [];              % ɽ������
peaksInfo.range = [];               % ɽ������
peaksInfo.height = [];              % ɽ��߶�
peaksInfo = repmat(peaksInfo,N,1);

x=[0.35 0.4 0.55 0.6 0.4 0.8];
y=[0.2 0.4 0.45 0.7 0.8 0.8];
z=[0.3 0.4 0.6 0.3 0.4 0.25];
r=[0.7 1 0.8 0.5 0.4 0.5];

% �������N��ɽ�����������
for i = 1:N
    peaksInfo(i).center = [mapRange(1) * x(i), mapRange(2) * y(i)];
    peaksInfo(i).height = mapRange(3) * z(i);
    peaksInfo(i).range = mapRange*0.1*r(i);
end

%% ��в��ģ
% rr=[5 5 6 4 20];
% aa=[20 30 45 80 70];
% bb=[30 65 40 70 70];
% cc=[40 15 20 33 0];
% [x,y,z]=sphere(50);
% for i=1:5
%     surf(rr(i)*x+aa(i),rr(i)*y+bb(i),rr(i)*z+cc(i));
%     hold on
% end

[a,b,c]=ellipsoid(20,30,50,10,10,10);
% surf(a,b,c);
% hold on

% ����ɽ������ֵ
peakData = [];
for x = 1:mapRange(1)
    for y = 1:mapRange(2)
        sum=0;
        for k=1:N
            h_i = peaksInfo(k).height;
            x_i = peaksInfo(k).center(1);
            y_i = peaksInfo(k).center(2);
            x_si = peaksInfo(k).range(1);
            y_si = peaksInfo(k).range(2);
            sum = sum + h_i * exp(-((x-x_i)/x_si)^2 - ((y-y_i)/y_si)^2);
        end
        peakData(x,y)=sum;
    end
end
% ���������������ڲ�ֵ�ж�·���Ƿ���ɽ�彻��
x = [];
for i = 1:mapRange(1)
    x = [x; ones(mapRange(2),1) * i];
end
y = (1:mapRange(2))';
y = repmat(y,length(peakData(:))/length(y),1);
peakData = reshape(peakData,length(peakData(:)),1);
[X,Y,Z] = griddata(x,y,peakData,linspace(min(x),max(x),100)',linspace(min(y),max(y),100));
end