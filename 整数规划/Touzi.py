# 投资收益问题代码
# 导入必要的库
import matplotlib.pyplot as plt #用于绘图
from numpy import ones,diag, c_,zeros #用于创建和操作数组
from scipy.optimize import linprog #用于执行线性规划
#设置matplotlib的参数使其支持LaTeX文本和字体大小
plt. rc('text', usetex=True)
plt.rc('font', size=16)
# 线性规划问题的目标函数系数
c = [-0.05, -0.27, -0.19, -0.185, -0.185]
# 线性不等式约束的系数矩阵(A*x <= b)
# 使用c_来合并数组,zeros创建一个全0的数组作为第一列,
# diag创建一个对角阵
A=c_[zeros(4), diag([0.025, 0.015, 0.055, 0.026]) ]
# 线性等式约束的系数矩阵和右侧的值(Aeq* x=beq)
Aeq=[[1, 1.01, 1.02, 1.045, 1.065]];
beq = [1]
# 初始化参数a,以及两个用于存储结果的空列表
a = 0;
aa = [];
ss = [];
# 循环,a的值从0开始,以0.001的步长增加,直到0.05

while a < 0.05:
    # 创建线性不等式约束的右侧值(b)
    b = ones (4) * a
    # 执行线性规划,得到最优解
    res = linprog (c, A, b, Aeq, beq, bounds=[(0, None), (0, None), (0, None), (0, None), (0, None)])
    #提取线性规划的解向量x和最优值Q
    x = res. x;
    Q = -res.fun
    # 将当前的a值和对应的最优值Q存入列表
    aa. append (a);
    ss. append (Q)
    # a增加0.001
    a = a + 0.001
#绘制结果,a值与最优值Q之间的关系图
plt.plot(aa,ss,'r*') #使用红色星号标记数据点
#设置坐标轴标签,其中a和Q将使用LaTeX格式显示
plt. xlabel ('$a$')
plt. ylabel ('$Q$', rotation=90)
# 保存绘制的图像到文件中,分辨率为500dpi
plt.savefig('figure5_1_1.png', dpi=500)
# 显示图形
plt. show ()