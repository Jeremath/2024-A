import numpy as np
import pandas as pd
import math
#参数
n=223
b=1.7
pi=np.pi
t_max=300
v_h=1
d_h=2.86
d_b=1.65
l_h=3.41
l_b=2.2
R=4.5

#计算盘入每一时刻龙头位置
def theta(t):
    if -4*pi*v_h*t/b+(32*pi*0.55/b)**2<0:
        return -1
    theta=np.sqrt(-4*pi*v_h*t/b+(32*pi*0.55/b)**2)
    return theta

#计算盘出每一时刻龙头位置
def theta_out(t):
    theta=np.sqrt(4*pi*v_h*t/b+(2*pi*R/b)**2)
    return theta

#牛顿法解方程,求解各点的位置
def f(theta_now,theta_next,d):
    return (b*b/4/pi/pi)*(theta_now**2+theta_next**2-2*theta_now*theta_next*np.cos(theta_now-theta_next))-d**2
def f_prime(theta_now,theta_next,d):
    return (b*b/4/pi/pi)*(2*theta_next+2*theta_now*(np.cos(theta_now-theta_next)-theta_next*np.sin(theta_now-theta_next)))

def newton_method(f,f_prime,d,theta_now,x0,tol=1e-6,max_iter=100):
    for i in range(max_iter):
        x = x0 - f(theta_now,x0,d) / f_prime(theta_now,x0,d)
        if abs(x - x0) < tol:
            break
        x0 = x
    return x

#考虑矩形的4个角
def corner(x1,y1,x2,y2,l):
    k=(y2-y1)/(x2-x1)
    k_p=-1/k
    db=np.sqrt(1+k**2)*0.15
    b_0=(x1*y2-x2*y1)/(x1-x2)
    b_1=b_0-db
    b_2=b_0+db
    mid_x=(x1+x2)/2
    mid_y=(y1+y2)/2
    c_0=mid_y-k_p*mid_x
    dc=np.sqrt(1+k_p**2)*(l/2)
    c_1=c_0-dc
    c_2=c_0+dc
    x_a=(c_1-b_1)/(k-k_p)
    y_a=k*x_a+b_1
    x_b=(c_1-b_2)/(k-k_p)
    y_b=k*x_b+b_2
    x_c=(c_2-b_1)/(k-k_p)
    y_c=k*x_c+b_1
    x_d=(c_2-b_2)/(k-k_p)
    y_d=k*x_d+b_2
    return x_a,y_a,x_b,y_b,x_c,y_c,x_d,y_d

#判断点是否在矩形内
def is_in_rectangle(x,y,x1,y1,x2,y2,l):
    k=(y2-y1)/(x2-x1)
    b=(x1*y2-x2*y1)/(x1-x2)
    d=np.abs(k*x-y+b)/np.sqrt(1+k**2)
    if d>0.15:
        return False
    else:
        x0=(k*(y-b)+x)/(1+k**2)
        dis_sum=(1+k**2)*(np.abs(x1-x0)+np.abs(x2-x0))
        if dis_sum>l:
            return False
        else:
            return True

#根据掉头半径确定两圆弧
def turning(R):
    theta_turn=2*pi*R/b
    x_1=R*np.cos(theta_turn)
    y_1=R*np.sin(theta_turn)
    x_2=-R*np.cos(theta_turn)
    y_2=-R*np.sin(theta_turn)
    x_mid=(x_1+x_2*2)/3
    y_mid=(y_1+y_2*2)/3
    o_1_x=(x_1+x_mid)/2
    o_1_y=(y_1+y_mid)/2
    o_2_x=(x_2+x_mid)/2
    o_2_y=(y_2+y_mid)/2
    r_2=R/3
    r_1=r_2*2
    return o_1_x,o_1_y,r_1,o_2_x,o_2_y,r_2

#计算掉头空间为R时是否会相撞
#盘出曲线点的递推方程
def g(theta_now,theta_next,d):
    return (b*b/4/pi/pi)*((theta_now-pi)**2+(theta_next-pi)**2-2*(theta_now-pi)*(theta_next-pi)*np.cos(theta_now-theta_next))-d**2
def g_prime(theta_now,theta_next,d):
    return (b*b/4/pi/pi)*(2*(theta_next-pi)+2*(theta_now-pi)*(np.cos(theta_now-theta_next)-(theta_next-pi)*np.sin(theta_now-theta_next)))

def crash(R):
    theta_turn=2*pi*R/b
    x_2=-R*np.cos(theta_turn)
    y_2=-R*np.sin(theta_turn)
    theta_3=newton_method(g,g_prime,d_b,theta_turn+pi,theta_turn+pi*3/2)
    x_3=b*(theta_3-pi)*np.cos(theta_3)/(2*pi)
    y_3=b*(theta_3-pi)*np.sin(theta_3)/(2*pi)
    x_a,y_a,x_b,y_b,x_c,y_c,x_d,y_d=corner(x_2,y_2,x_3,y_3,l_h)
    theta_head=[]
    r_head=[]
    head_x=[]
    head_y=[]
    theta_body_i=np.empty((n,2*t_max))
    r_body_i=np.empty((n,2*t_max))
    body_i_x=np.empty((n,2*t_max))
    body_i_y=np.empty((n,2*t_max))
    t=0
    flag=False
    while True:
        #计算龙头位置
        theta_now=theta(t)
        if theta_now==-1:
            t=t+1
            break
        theta_head.append(theta_now)
        r_head.append(b*theta_now/(2*pi))
        x=r_head[t]*np.cos(theta_now)
        y=r_head[t]*np.sin(theta_now)
        x=round(x,6)
        y=round(y,6)
        head_x.append(x)
        head_y.append(y)
        #计算下一节位置
        theta_next=newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]+pi/2)
        theta_body_i[0][t]=theta_next
        r_body_i[0][t]=theta_next*b/(2*pi)
        x=r_body_i[0][t]*np.cos(theta_next)
        y=r_body_i[0][t]*np.sin(theta_next)
        x=round(x,6)
        y=round(y,6)
        body_i_x[0][t]=x
        body_i_y[0][t]=y
        for i in range(1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            x=round(x,6)
            y=round(y,6)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
        #判断是否与龙头碰撞
        for i in range(1,n-1):
            if(is_in_rectangle(x_a,y_a,body_i_x[i][t],body_i_y[i][t],body_i_x[i+1][t],body_i_y[i+1][t],l_h)):
                flag=True
                break
            elif(is_in_rectangle(x_b,y_b,body_i_x[i][t],body_i_y[i][t],body_i_x[i+1][t],body_i_y[i+1][t],l_h)):
                flag=True
                break
            elif(is_in_rectangle(x_c,y_c,body_i_x[i][t],body_i_y[i][t],body_i_x[i+1][t],body_i_y[i+1][t],l_h)):
                flag=True
                break
            elif(is_in_rectangle(x_d,y_d,body_i_x[i][t],body_i_y[i][t],body_i_x[i+1][t],body_i_y[i+1][t],l_h)):
                flag=True
                break
        t=t+1
    t=t-1
    return flag

left=0
right=4.5
while left<=right:
    mid=(left+right)/2
    if crash(mid):
        left=mid+0.01
    else:
        right=mid-0.01
print(f"掉头半径为{left}时，龙身不会相撞")

#计算掉头的时间
t=0
t_0=0
while True:
    #计算龙头位置
    theta_now=theta(t)
    r_now=b*theta_now/(2*pi)
    if(r_now<=4.5):
        t_0=t
        break
    t=t+1
print(f"掉头的时间为{t_0}")

#计算各点的两圆的弧长
theta_turn=2*pi*R/b
x_turn_in=R*np.cos(theta_turn)
y_turn_in=R*np.sin(theta_turn)
x_turn_out=-R*np.cos(theta_turn)
y_turn_out=-R*np.sin(theta_turn)
o_1_x,o_1_y,r_1,o_2_x,o_2_y,r_2=turning(4.5)
mid_x=(o_1_x+2*o_2_x)/3
mid_y=(o_1_y+2*o_2_y)/3
alpha_c1=pi
alpha_c2=pi
s_1=r_1*alpha_c1
s_2=r_2*alpha_c2



#计算点位置旋转后位置
def rotate_point(o_x, o_y, a, b, r, theta):
    angle = np.arctan2(o_y - b, o_x - a) + theta
    x_prime = a + r * np.cos(angle)
    y_prime = b + r * np.sin(angle)
    return x_prime, y_prime

#计算各点位置速度
theta_head=np.empty(t_max)
r_head=np.empty(t_max)
head_x=np.empty(t_max)
head_y=np.empty(t_max)
theta_body_i=np.empty((n,t_max))
r_body_i=np.empty((n,t_max))
body_i_x=np.empty((n,t_max))
body_i_y=np.empty((n,t_max))
v_body_i=np.empty((n,t_max))
alpha_t_1=2*np.arcsin(d_h/(2*r_1))
alpha_t_2=2*np.arcsin(d_b/(2*r_1))
alpha_t_3=2*np.arcsin(d_h/(2*r_2))
alpha_t_4=2*np.arcsin(d_b/(2*r_2))
for t in range(6,207):
    #盘入阶段
    if t<=t_0:
        #龙头
        theta_now=theta(t)
        theta_head[t]=theta_now
        r_head[t]=b*theta_now/(2*pi)
        head_x[t]=r_head[t]*np.cos(theta_now)
        head_y[t]=r_head[t]*np.sin(theta_now)
        #龙身
        theta_next=newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]+pi/2)
        theta_body_i[0][t]=theta_next
        r_body_i[0][t]=theta_next*b/(2*pi)
        x=r_body_i[0][t]*np.cos(theta_next)
        y=r_body_i[0][t]*np.sin(theta_next)
        body_i_x[0][t]=x
        body_i_y[0][t]=y
        for i in range(1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
        #速度
        m_0=(head_y[t]-body_i_y[0][t])/(head_x[t]-body_i_x[0][t])
        m_now=(theta_head[t]*np.cos(theta_head[t])+np.sin(theta_head[t]))/(-theta_head[t]*np.sin(theta_head[t])+np.cos(theta_head[t]))
        m_next=(theta_body_i[0][t]*np.cos(theta_body_i[0][t])+np.sin(theta_body_i[0][t]))/(-theta_body_i[0][t]*np.sin(theta_body_i[0][t])+np.cos(theta_body_i[0][t]))
        alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
        alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
        v_next=v_h*np.cos(alpha_1)/np.cos(alpha_2)
        v_body_i[0][t]=v_next
        for i in range(1,n):
            m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
            m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
            m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
            alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
            alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
            v_next=v_body_i[i-1][t]*np.cos(alpha_1)/np.cos(alpha_2)
            v_body_i[i][t]=v_next
        # print(v_next,t)
    #圆弧1
    elif t>t_0 and t<=t_0+s_1/v_h:
        #龙头
        s_move=v_h*(t-t_0)
        alpha_move=s_move/r_1
        head_x[t],head_y[t]=rotate_point(o_1_x,o_1_y,x_turn_in,y_turn_in,r_1,-alpha_move)
        #龙身
        num1=math.ceil((alpha_move)/alpha_t_2)
        body_i_x[0][t],body_i_y[0][t]=rotate_point(o_1_x,o_1_y,head_x[t],head_y[t],r_1,alpha_t_1)
        v_body_i[0][t]=v_h
        for i in range(1,num1+1):
            body_i_x[i][t],body_i_y[i][t]=rotate_point(o_1_x,o_1_y,body_i_x[i-1][t],body_i_y[i-1][t],r_1,alpha_t_2)
            v_body_i[i][t]=v_body_i[i-1][t]
        theta_body_i[num1][t]=theta_turn+np.abs(alpha_move-alpha_t_1-num1*alpha_t_2)
        r_body_i[num1][t]=b*theta_body_i[num1][t]/(2*pi)
        body_i_x[num1][t]=r_body_i[num1][t]*np.cos(theta_body_i[num1][t])
        body_i_y[num1][t]=r_body_i[num1][t]*np.sin(theta_body_i[num1][t])
        for i in range(num1+1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
            m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
            m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
            m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
            alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
            alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
            v_next=v_body_i[i-1][t]*np.cos(alpha_1)/np.cos(alpha_2)
            v_body_i[i][t]=v_next
    #圆弧2
    elif t>t_0+s_1/v_h and t<=t_0+s_1/v_h+s_2/v_h:
        #龙头
        s_move=v_h*(t-t_0-s_1/v_h)
        alpha_move=s_move/r_2
        head_x[t],head_y[t]=rotate_point(o_2_x,o_2_y,mid_x,mid_y,r_2,-alpha_move)
        #龙身
        num2=math.ceil((alpha_move)/alpha_t_4)
        body_i_x[0][t],body_i_y[0][t]=rotate_point(o_2_x,o_2_y,head_x[t],head_y[t],r_2,alpha_t_3)
        v_body_i[0][t]=v_h
        for i in range(1,num2+1):
            body_i_x[i][t],body_i_y[i][t]=rotate_point(o_2_x,o_2_y,body_i_x[i-1][t],body_i_y[i-1][t],r_2,alpha_t_4)
            v_body_i[i][t]=v_body_i[i-1][t]
        num3=math.ceil(alpha_c1/alpha_t_2)
        for i in range(num2+1,num2+num3+2):
            body_i_x[i][t],body_i_y[i][t]=rotate_point(o_1_x,o_1_y,mid_x,mid_y,r_1,alpha_t_2)
            v_body_i[i][t]=v_body_i[i-1][t]
        theta_body_i[num2+num3+1][t]=theta_turn+np.abs(alpha_move-alpha_t_3-num2*alpha_t_4)
        r_body_i[num2+num3+1][t]=b*theta_body_i[num2+num3+1][t]/(2*pi)
        body_i_x[num2+num3+1][t]=r_body_i[num2+num3+1][t]*np.cos(theta_body_i[num2+num3+1][t])
        body_i_y[num2+num3+1][t]=r_body_i[num2+num3+1][t]*np.sin(theta_body_i[num2+num3+1][t])
        for i in range(num2+num3+2,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
            m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
            m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
            m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
            alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
            alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
            v_next=v_body_i[i-1][t]*np.cos(alpha_1)/np.cos(alpha_2)
            v_body_i[i][t]=v_next
            #print(v_next,t)
    #盘出阶段：
    else:
        #龙头
        theta_now=theta_out(t-t_0-(s_1+s_2)/v_h)
        theta_head[t]=theta_now
        r_head[t]=b*theta_now/(2*pi)
        head_x[t]=r_head[t]*np.cos(theta_now)
        head_y[t]=r_head[t]*np.sin(theta_now)
        #龙身
        theta_next=newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]-pi/2)
        theta_body_i[0][t]=theta_next
        r_body_i[0][t]=theta_next*b/(2*pi)
        x=r_body_i[0][t]*np.cos(theta_next)
        y=r_body_i[0][t]*np.sin(theta_next)
        body_i_x[0][t]=x
        body_i_y[0][t]=y
        m_0=(head_y[t]-body_i_y[0][t])/(head_x[t]-body_i_x[0][t])
        m_now=(theta_head[t]*np.cos(theta_head[t])+np.sin(theta_head[t]))/(-theta_head[t]*np.sin(theta_head[t])+np.cos(theta_head[t]))
        m_next=(theta_body_i[0][t]*np.cos(theta_body_i[0][t])+np.sin(theta_body_i[0][t]))/(-theta_body_i[0][t]*np.sin(theta_body_i[0][t])+np.cos(theta_body_i[0][t]))
        alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
        alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
        v_next=v_h*np.cos(alpha_1)/np.cos(alpha_2)
        v_body_i[0][t]=v_next
        num4=0
        for i in range(1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]-pi/2)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            if r_body_i[i][t]<R:
                num4=i
                break
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
            m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
            m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
            m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
            alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
            alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
            v_next=v_body_i[i-1][t]*np.cos(alpha_1)/np.cos(alpha_2)
            v_body_i[i][t]=v_next
        body_i_x=np.multiply(body_i_x,-1)
        body_i_y=np.multiply(body_i_y,-1)
        num5=math.ceil(alpha_c2/alpha_t_4)
        for i in range(num4,num4+num5):
            body_i_x[i][t],body_i_y[i][t]=rotate_point(o_2_x,o_2_y,x_turn_out,y_turn_out,r_2,alpha_t_4)
            v_body_i[i][t]=v_body_i[i-1][t]
        num3=math.ceil(alpha_c1/alpha_t_2)
        for i in range(num4+num5,num4+num5+num3):
            body_i_x[i][t],body_i_y[i][t]=rotate_point(o_1_x,o_1_y,mid_x,mid,r_1,alpha_t_2)
            v_body_i[i][t]=v_body_i[i-1][t]
        theta_body_i[num4+num5+num3-1][t]=theta_turn
        r_body_i[num4+num5+num3-1][t]=b*theta_body_i[num4+num5+num3-1][t]/(2*pi)
        body_i_x[num4+num5+num3-1][t]=r_body_i[num4+num5+num3-1][t]*np.cos(theta_body_i[num4+num5+num3-1][t])
        body_i_y[num4+num5+num3-1][t]=r_body_i[num4+num5+num3-1][t]*np.sin(theta_body_i[num4+num5+num3-1][t])
        for i in range(num4+num5+num3,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
            m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
            m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
            m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
            alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
            alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
            v_next=v_body_i[i-1][t]*np.cos(alpha_1)/np.cos(alpha_2)
            v_body_i[i][t]=v_next
            #print(v_next,t)

for t in range(6,207):
    head_x[t]=round(head_x[t],6)
    head_y[t]=round(head_y[t],6)
    for i in range(n):
        body_i_x[i][t]=round(body_i_x[i][t],6)
        body_i_y[i][t]=round(body_i_y[i][t],6)
        v_body_i[i][t]=round(v_body_i[i][t],6)
#输出固定时间，固定位置的位置速度
time_point=[6,56,106,156,206]
num_point=[0,50,100,150,200,222]

for t in time_point:
    print(f"time:{t}")
    print(f"head:x:{head_x[t]},y:{head_y[t]},v:{v_h}")
    for i in num_point:
        print(f"body{i}:x:{body_i_x[i][t]},y:{body_i_y[i][t]},v:{v_body_i[i][t]}")

#将结果写入excel
data=[]
data.append(head_x[6:207])
data.append(head_y[6:207])
for i in range(n):
    data.append(body_i_x[i][6:207])
    data.append(body_i_y[i][6:207])
df=pd.DataFrame(data)
df.to_excel('solution4-1.xlsx')

data_v=[]
for i in range(n):
        data_v.append(v_body_i[i][6:207])
df=pd.DataFrame(data_v)
df.to_excel('solution4-2.xlsx')
