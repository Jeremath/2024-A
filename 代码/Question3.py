import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.seterr(divide='ignore',invalid='ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
#参数
n=223
pi=np.pi
t_max=300
v_h=1
d_h=2.86
d_b=1.65
l_h=3.41
l_b=2.2

#计算每一时刻龙头位置
def theta(t,b):
    theta=np.sqrt(-4*pi*v_h*t/b+(32*pi*0.55/b)**2)
    return theta

#牛顿法解方程,求解各点的位置
def f(theta_now,theta_next,d,b):
    return (b*b/4/pi/pi)*(theta_now**2+theta_next**2-2*theta_now*theta_next*np.cos(theta_now-theta_next))-d**2
def f_prime(theta_now,theta_next,d,b):
    return (b*b/4/pi/pi)*(2*theta_next+2*theta_now*(np.cos(theta_now-theta_next)-theta_next*np.sin(theta_now-theta_next)))

def newton_method(f,f_prime,d,theta_now,x0,b,tol=1e-6,max_iter=100):
    for i in range(max_iter):
        x = x0 - f(theta_now,x0,d,b) / f_prime(theta_now,x0,d,b)
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

#计算碰撞时距离
def dis(b_test):
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
        theta_now=theta(t,b_test)
        theta_head.append(theta_now)
        r_head.append(b_test*theta_now/(2*pi))
        x=r_head[t]*np.cos(theta_now)
        y=r_head[t]*np.sin(theta_now)
        x=round(x,6)
        y=round(y,6)
        head_x.append(x)
        head_y.append(y)
        #计算下一节位置
        theta_next=newton_method(f,f_prime,d_h,theta_head[t],theta_head[t]+pi/2,b_test)
        theta_body_i[0][t]=theta_next
        r_body_i[0][t]=theta_next*b_test/(2*pi)
        x=r_body_i[0][t]*np.cos(theta_next)
        y=r_body_i[0][t]*np.sin(theta_next)
        x=round(x,6)
        y=round(y,6)
        body_i_x[0][t]=x
        body_i_y[0][t]=y
        for i in range(1,n):
            theta_next=newton_method(f,f_prime,d_b,theta_body_i[i-1][t],theta_body_i[i-1][t]+pi/2,b_test)
            theta_body_i[i][t]=theta_next
            r_body_i[i][t]=theta_next*b_test/(2*pi)
            x=r_body_i[i][t]*np.cos(theta_next)
            y=r_body_i[i][t]*np.sin(theta_next)
            x=round(x,6)
            y=round(y,6)
            body_i_x[i][t]=x
            body_i_y[i][t]=y
        #判断是否与龙头碰撞
        x1=head_x[t]
        y1=head_y[t]
        x2=body_i_x[0][t]
        y2=body_i_y[0][t]
        x_a,y_a,x_b,y_b,x_c,y_c,x_d,y_d=corner(x1,y1,x2,y2,l_h)
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
        
        #判断是否与龙身碰撞
        for i in range(0,n-1):
            x1=body_i_x[i][t]
            y1=body_i_y[i][t]
            x2=body_i_x[i+1][t]
            y2=body_i_y[i+1][t]
            x_a,y_a,x_b,y_b,x_c,y_c,x_d,y_d=corner(x1,y1,x2,y2,l_b)
            for j in range(i+2,n-1):
                if(is_in_rectangle(x_a,y_a,body_i_x[j][t],body_i_y[j][t],body_i_x[j+1][t],body_i_y[j+1][t],l_b)):
                    flag=True
                    break
                elif(is_in_rectangle(x_b,y_b,body_i_x[j][t],body_i_y[j][t],body_i_x[j+1][t],body_i_y[j+1][t],l_b)):
                    flag=True
                    break
                elif(is_in_rectangle(x_c,y_c,body_i_x[j][t],body_i_y[j][t],body_i_x[j+1][t],body_i_y[j+1][t],l_b)):
                    flag=True
                    break
                elif(is_in_rectangle(x_d,y_d,body_i_x[j][t],body_i_y[j][t],body_i_x[j+1][t],body_i_y[j+1][t],l_b)):
                    flag=True
                    break
        t=t+1
    t=t-1
    return np.sqrt(head_x[t]**2+head_y[t]**2)

#二分答案
L=0.4
R=0.5
step=0.001
while L<=R:
    mid=(L+R)/2
    if dis(mid)>4.5:
        L=mid+step
    else:
        R=mid-step
print(R)
#遍历
# b_list=np.arange(0,0.55,0.01)
# r_list=[]
# whlie b in b_list:
#     r_list.append(dis(b))
# plt.scatter(b_list,r_list)
# plt.xlabel('螺距d/m')
# plt.ylabel('碰撞半径r/m')
# plt.title('碰撞半径随螺距变化的关系')
# plt.show()
