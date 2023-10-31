%指标函数minimize Q(x)=-S(x)*D(x)
function f=SD_penaltyfunction(A,x)
n=size(A,1);
ss=0;
z=ones(1,n);
    for j=1:n
        z(1,j)=A(j,:)*x;
        ss=ss+z(1,j);
    end
    za=ss/n;
    s=0;
    for k=1:n
        s=s+((z(1,k)-za).^2)/(n-1);
    end
    s=sqrt(s);
    su=0;
    MAX=0;
    for i=1:n
        for j=1:n
            dis=abs(z(1,i)-z(1,j));
            if dis>MAX
                MAX=dis;
            end
        end
    end
    for e=1:n
        for f=1:n
            rcd=abs(z(1,e)-z(1,f));
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