%指标函数minimize Q(x)=-S(x)*D(x)
function f=SD_penaltyfunction(A,x)
n=size(A,1);
ss=0;
    for j=1:n
        z(j)=A(j,:)*x';
        ss=ss+z(j);
    end
    za=ss/n;
    s=0;
    for k=1:n
        s=s+((z(k)-za).^2)/(n-1);
    end
    s=sqrt(s);
    su=0;
    MAX=0;
    for i=1:n
        for j=1:n
            dis=abs(z(i)-z(j));
            if dis>MAX
                MAX=dis;
            end
        end
    end
    for e=1:n
        for f=1:n
            rcd=abs(z(e)-z(f));
            dd=0.25*MAX-rcd;%密度窗宽半径值R=max(r_ik)/4
            if dd>0
                su=su+dd;
                d=su;
            end
        end
    end    
    r=200;%惩罚系数
    if  sum(x.^2)~=1
        f=-(s*d)+r*(sum(x.^2)-1).^2;
    else
        f=-(s*d);
    end
end