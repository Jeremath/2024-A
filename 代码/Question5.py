import numpy as np
import pandas as pd
#参数
n=223
b=1.7
pi=np.pi
t_max=301
d_h=2.86
d_b=1.65
R=4.5

#计算每一时刻龙头位置
def theta(t,v_h):
    theta=np.sqrt(4*pi*v_h*t/b+(2*pi*R/b)**2)
    return theta

#牛顿法解方程,求解各点的位置
def f(theta_now,theta_next,d,v_h):
    return (b*b/4/pi/pi)*(theta_now**2+theta_next**2-2*theta_now*theta_next*np.cos(theta_now-theta_next))-d**2
def f_prime(theta_now,theta_next,d,v_h):
    return (b*b/4/pi/pi)*(2*theta_next+2*theta_now*(np.cos(theta_now-theta_next)-theta_next*np.sin(theta_now-theta_next)))

def newton_method(f,f_prime,d,theta_now,x0,v_h,tol=1e-6,max_iter=100):
    for i in range(max_iter):
        x = x0 - f(theta_now,x0,d,v_h) / f_prime(theta_now,x0,d,v_h)
        if abs(x - x0) < tol:
            break
        x0 = x
    return x

#计算v为v_h时的最大速度
def v_max(v_test):
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
    while flag==False:
        #计算龙头位置
        theta_now=theta(t,v_test)
        theta_head.append(theta_now)
        r_head.append(b*theta_now/(2*pi))
        x=r_head[t]*np.cos(theta_now)
        y=r_head[t]*np.sin(theta_now)
        head_x.append(x)
        head_y.append(y)
        #计算下一节位置
        theta_next=newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]-pi/2,v_test)
        theta_body_i[0][t]=theta_next
        r_body_i[0][t]=theta_next*b/(2*pi)
        x=r_body_i[0][t]*np.cos(theta_next)
        y=r_body_i[0][t]*np.sin(theta_next)
        body_i_x[0][t]=x
        body_i_y[0][t]=y
        for i in range(1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]-pi/2,v_test)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b/(2*pi)
            if r_body_i[i][t]<R:
                flag=False
                break
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
            flag=True
        t=t+1
    t=t-1
    # print(f"t={t}")
    #计算最大速度
    v_body_i=[]
    m_0=(head_y[t]-body_i_y[0][t])/(head_x[t]-body_i_x[0][t])
    m_now=(theta_head[t]*np.cos(theta_head[t])+np.sin(theta_head[t]))/(-theta_head[t]*np.sin(theta_head[t])+np.cos(theta_head[t]))
    m_next=(theta_body_i[0][t]*np.cos(theta_body_i[0][t])+np.sin(theta_body_i[0][t]))/(-theta_body_i[0][t]*np.sin(theta_body_i[0][t])+np.cos(theta_body_i[0][t]))
    alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
    alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
    v_next=v_test*np.cos(alpha_1)/np.cos(alpha_2)
    v_body_i.append(v_next)
    for i in range(1,n):
        m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
        m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
        m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
        alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
        alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
        v_next=v_body_i[i-1]*np.cos(alpha_1)/np.cos(alpha_2)
        v_body_i.append(v_next)
    return v_body_i[-1]

#二分答案
left=1
right=2
step=0.00001
while left<=right:
    mid=(left+right)/2
    # print(mid)
    if v_max(mid)>2:
        right=mid-step
    else:
        left=mid+step
print(f"龙头最大速度为{right}")





