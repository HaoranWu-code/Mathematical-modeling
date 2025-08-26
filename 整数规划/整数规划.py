# 先来求解KK升级的问题
# · Python代码

import numpy as np
from scipy.optimize import linprog

# 目标函数系数,这里取负值因为Linprog默认进行最小化优化
c = [-20, -30, -45]

# 不等式约束的系数矩阵A
A_ub = [
[4, 8,15],
[1, 1, 1]
]
# 不等式约束的右侧值向量b
b_ub = [100, 20]
# 定义域
bounds=[[0,None],[0,None],[0,None]]

# 求解线性规划问题
# 注意:由于Linprog默认是求解最小化问题,我们通过对目标函数系数取负值来转换为最大化问题
result = linprog(c, A_ub, b_ub,bounds=bounds)
result = linprog (c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

print(result)
# 输出结果
print('A、B、C三图分别通关的次数为:')
print(result.x) #解向量

# 目标函数的最大值是最小化问题的相反数
y = -result. fun
print('最终获得的经验为:')
print(y)