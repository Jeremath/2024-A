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
        t=t+1
    t=t-1
    return np.sqrt(head_x[t]**2+head_y[t]**2)

#灵敏度分析
b_list=[]
r_list=np.arange(2.5,5.5,0.2)
for r in r_list:
    L=0
    R=r
    step=0.01
    while L<=R:
        mid=(L+R)/2
        # print(mid)
        if dis(mid)>r:
            L=mid+step
        else:
            R=mid-step
    print(f"调头半径为{r}时,最小螺距{R}")
    b_list.append(R)

plt.plot(r_list, b_list, marker='o', markerfacecolor='none', markeredgecolor=(0, 0.57, 0.79), color=(0, 0.57, 0.79))
plt.xlabel('调头半径r$_D$/m',fontsize=12)
plt.ylabel('最小螺距d/m',fontsize=12)
# plt.title('最小螺距随调头半径变化的关系图',fontsize=16)
plt.show()

# 调头半径和最小螺距数据
r_list_result = [2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5, 4.7, 4.9, 5.1, 5.3, 5.5]
b_list_result = [0.5509375, 0.521875, 0.4932812500000001, 0.49187500000000006, 0.49875000000000014, 
          0.48875000000000013, 0.47406250000000016, 0.47000000000000014, 0.44671875000000016, 
          0.45250000000000024, 0.4567187500000002, 0.4409375000000002, 0.44125000000000025, 
          0.43000000000000027, 0.4267968750000002, 0.4167968750000002]

# 计算最小螺距的一阶差分
diff_b_list = np.diff(b_list_result)
diff_r_list = r_list_result[:-1]  # 差分后r_list会少一个元素

# 绘制一阶差分图
plt.plot(diff_r_list, diff_b_list, marker='o', markerfacecolor='none', color=(0, 146/255, 202/255))
plt.xlabel('调头半径r$_D$/m', fontsize=10)
plt.ylabel('一阶差分（最小螺距变化）', fontsize=10)
# plt.title('最小螺距随调头半径变化的一阶差分图', fontsize=14)
# plt.grid(True)
plt.show()