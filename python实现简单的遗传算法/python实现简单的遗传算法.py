import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 目标函数（以二元函数为例，寻找最大值）
def objective_function(x, y):
    return np.sin(x) + np.cos(y) + 0.1 * (x + y)


# 遗传算法参数
POPULATION_SIZE = 100  # 种群大小，有100个个体
GENE_LENGTH = 40  # 每个变量的二进制编码长度
GENERATIONS = 100  # 迭代次数，迭代100次
CROSSOVER_RATE = 0.8  # 交叉率
MUTATION_RATE = 0.01  # 变异率
VARIABLES = 2  # 变量个数（多元函数的维度）
X_BOUND = [-10, 10]  # x的取值范围
Y_BOUND = [-10, 10]  # y的取值范围


# 创建个体类
class Individual:
    def __init__(self):
        # 染色体编码（使用二进制编码表示染色体）
        self.chromosome = np.random.randint(2, size=GENE_LENGTH * VARIABLES)  # 随机生成0，1的二进制字符串
        self.fitness = 0

    # 解码
    def decode(self):
        # 将二进制染色体解码为实际变量值
        x_gene = self.chromosome[:GENE_LENGTH]
        y_gene = self.chromosome[GENE_LENGTH:]
        x = X_BOUND[0] + (X_BOUND[1] - X_BOUND[0]) * int(''.join(map(str, x_gene)), 2) / (
                    2 ** GENE_LENGTH - 1)  # 先把二进制编码转化为字符串，然后转化为实际的值，最后进行归一化
        y = Y_BOUND[0] + (Y_BOUND[1] - Y_BOUND[0]) * int(''.join(map(str, y_gene)), 2) / (2 ** GENE_LENGTH - 1)
        return x, y

    # 计算适应度
    def calculate_fitness(self):
        x, y = self.decode()  # 解码
        self.fitness = objective_function(x, y)  # 带入到目标函数，求最大值，所以函数值越大适应度越高
        return self.fitness
    # 初始化种群


def initialize_population():
    return [Individual() for _ in range(POPULATION_SIZE)]  # 生成种群大小的个体


# 按照适应度筛选适应度较大的个体
def selection(population):
    # 轮盘赌选择
    total_fitness = sum(ind.calculate_fitness() for ind in population)  # 计算种群所有个体的适应度之和
    pick = np.random.uniform(0, total_fitness)
    current = 0
    for ind in population:
        current += ind.fitness
        if current > pick:
            return ind  # 如果累加出现大于pick的适应度就返回这个值
    return population[-1]  # 如果所有的累加适应度都小于pick可能是受到浮点数的影响，默认返回最后一个


# 交叉
def crossover(parent1, parent2):
    # 单点交叉
    if np.random.random() < CROSSOVER_RATE:
        point = np.random.randint(1, GENE_LENGTH * VARIABLES - 1)  # 交叉点是随机选取的
        child1 = Individual()
        child2 = Individual()
        child1.chromosome = np.concatenate((parent1.chromosome[:point], parent2.chromosome[point:]))
        child2.chromosome = np.concatenate((parent2.chromosome[:point], parent1.chromosome[point:]))
        return child1, child2
    else:
        return parent1, parent2


# 变异
def mutation(individual):
    # 位翻转变异
    for i in range(len(individual.chromosome)):
        if np.random.random() < MUTATION_RATE:
            individual.chromosome[i] = 1 - individual.chromosome[i]  # 这个点如果是1就变为0，如果是0就变为1
    return individual


# 淘找出适应度最大的个体
def get_best(population):
    best = population[0]
    for ind in population:
        if ind.calculate_fitness() > best.fitness:
            best = ind
    return best


# 遗传算法主体
def genetic_algorithm():
    population = initialize_population()  # 生成初始种群
    best_individual = get_best(population)  # 生成首次最优个体
    best_fitness_history = [best_individual.fitness]  # 记录每次循环的最优个体的适应度

    # 执行迭代次数次循环
    for generation in range(GENERATIONS):
        new_population = []

        # 保留最优个体（精英保留策略）
        new_population.append(best_individual)  # 把上一代最好的个体添加到新种群

        while len(new_population) < POPULATION_SIZE:
            parent1 = selection(population)
            parent2 = selection(population)  # 从种群中按适应度概率选取出两个个体作为父代

            child1, child2 = crossover(parent1, parent2)  # 对两个父代进行交叉获得两个子代

            child1 = mutation(child1)
            child2 = mutation(child2)  # 对两个子代进行变异

            new_population.extend([child1, child2])  # 把子代加入到新种群中

        population = new_population[:POPULATION_SIZE]  # 把新种群代换掉前一个种群
        current_best = get_best(population)  # 找出现种群最优的个体

        if current_best.fitness > best_individual.fitness:
            best_individual = current_best  # 如果现种群的最优个体比之前的最优个体适应度更大就替换掉

        best_fitness_history.append(best_individual.fitness)  # 把这一代最优的个体记录下来

        print(f"Generation {generation}: Best Fitness = {best_individual.fitness:.4f}")  # 打印出目前最优个体的适应度，不想输出太多可以注释掉

    return best_individual, best_fitness_history  # 返回循环得到的最优个体和记录的历史最优个体


# 运行遗传算法
best_individual, fitness_history = genetic_algorithm()
best_x, best_y = best_individual.decode()  # 对得到的最优个体进行解码
print(
    f"\nBest Solution: x = {best_x:.4f}, y = {best_y:.4f}, f(x,y) = {best_individual.fitness:.4f}")  # 打印得到的最优个体对应的取值和函数值

# 可视化结果
fig = plt.figure(figsize=(12, 5))  # 设置画布大小

# 绘制函数曲面
ax1 = fig.add_subplot(121, projection='3d')
x = np.linspace(X_BOUND[0], X_BOUND[1], 50)
y = np.linspace(Y_BOUND[0], Y_BOUND[1], 50)
X, Y = np.meshgrid(x, y)
Z = objective_function(X, Y)
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)  # 作三维函数图像
ax1.scatter(best_x, best_y, best_individual.fitness, color='red', s=100, label='Best Solution')  # 把求到的最优解在函数图上画出来
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('Function and Optimal Solution')
ax1.legend()

# 绘制适应度变化曲线
ax2 = fig.add_subplot(122)
ax2.plot(fitness_history)  # 把历史最优个体的适应度画出来
ax2.set_xlabel('Generation')
ax2.set_ylabel('Best Fitness')
ax2.set_title('Convergence Curve')

plt.tight_layout()
plt.show()
