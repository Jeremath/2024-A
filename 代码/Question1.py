import numpy as np
import pandas as pd
#参数
n=223
b=0.55
pi=np.pi
t_max=301
v_h=1
d_h=2.86
d_b=1.65

#计算每一时刻龙头位置
def theta(t):
    theta=np.sqrt(-4*pi*v_h*t/b+1024*pi*pi)
    return theta

theta_head=[]
r_head=[]
head_x=[]
head_y=[]
for t in range(t_max+1):
    theta_now=theta(t)
    theta_head.append(theta_now)
    r_head.append(b*theta_now/(2*pi))
    x=r_head[t]*np.cos(theta_now)
    y=r_head[t]*np.sin(theta_now)
    x=round(x,6)
    y=round(y,6)
    head_x.append(x)
    head_y.append(y)

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

theta_body_i=np.empty((n,t_max+1))
r_body_i=np.empty((n,t_max+1))
body_i_x=np.empty((n,t_max+1))
body_i_y=np.empty((n,t_max+1))
for t in range(t_max+1):
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
    for t in range(t_max+1):
        theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2)
        theta_body_i[i][t]=theta_next
        r_body_i[i][t]=theta_next*b/(2*pi)
        x=r_body_i[i][t]*np.cos(theta_next)
        y=r_body_i[i][t]*np.sin(theta_next)
        x=round(x,6)
        y=round(y,6)
        body_i_x[i][t]=x
        body_i_y[i][t]=y

#求解各点速度
v_body_i=np.empty((n,t_max+1))
for t in range(t_max):
    m_0=(head_y[t]-body_i_y[0][t])/(head_x[t]-body_i_x[0][t])
    m_now=(theta_head[t]*np.cos(theta_head[t])+np.sin(theta_head[t]))/(-theta_head[t]*np.sin(theta_head[t])+np.cos(theta_head[t]))
    m_next=(theta_body_i[0][t]*np.cos(theta_body_i[0][t])+np.sin(theta_body_i[0][t]))/(-theta_body_i[0][t]*np.sin(theta_body_i[0][t])+np.cos(theta_body_i[0][t]))
    alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
    alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
    v_next=v_h*np.cos(alpha_1)/np.cos(alpha_2)
    v_next=round(v_next,6)
    v_body_i[0][t]=v_next

for i in range(1,n):
    for t in range(t_max):
        m_0=(body_i_y[i][t]-body_i_y[i-1][t])/(body_i_x[i][t]-body_i_x[i-1][t])
        m_now=(theta_body_i[i-1][t]*np.cos(theta_body_i[i-1][t])+np.sin(theta_body_i[i-1][t]))/(-theta_body_i[i-1][t]*np.sin(theta_body_i[i-1][t])+np.cos(theta_body_i[i-1][t]))
        m_next=(theta_body_i[i][t]*np.cos(theta_body_i[i][t])+np.sin(theta_body_i[i][t]))/(-theta_body_i[i][t]*np.sin(theta_body_i[i][t])+np.cos(theta_body_i[i][t]))
        alpha_1=np.arctan(np.abs((m_0-m_now)/(1+m_0*m_now)))
        alpha_2=np.arctan(np.abs((m_0-m_next)/(1+m_0*m_next)))
        v_next=v_body_i[i-1][t]*np.cos(alpha_1)/np.cos(alpha_2)
        v_next=round(v_next,6)
        v_body_i[i][t]=v_next

#输出固定时间，固定位置的位置速度
time_point=[0,60,120,180,240,300]
num_point=[0,50,100,150,200,222]

for t in time_point:
    print(f"time:{t}")
    print(f"head:x:{head_x[t]},y:{head_y[t]},v:{v_h}")
    for i in num_point:
        print(f"body{i}:x:{body_i_x[i][t]},y:{body_i_y[i][t]},v:{v_body_i[i][t]}")

#将结果写入excel
data=[]
data.append(head_x)
data.append(head_y)
for i in range(n):
    data.append(body_i_x[i])
    data.append(body_i_y[i])
df=pd.DataFrame(data)
df.to_excel('solution1-1.xlsx')

data_v=v_body_i
df=pd.DataFrame(data_v)
df.to_excel('solution1-2.xlsx')

