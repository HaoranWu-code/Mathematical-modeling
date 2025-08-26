# 背包问题：在背包问题（Knapsack
# Problem）中，假设有一个固定容量的背包，以及一组物品，每个物品具有自己的重量和价值。目标是选择一些物品放入背包中，使得背包中物品的总重量不超过背包容量，并且背包中物品的总价值最大化。
#
# 具体来说，我们可以定义背包问题的输入和输出如下：
#
# 输入：
#
# items：一个物品列表，每个物品由其重量和价值组成。例如，items = [(2, 10), (3, 15), (5, 20), (7, 25)]
# 表示共有4个物品，其中第一个物品重量为2，价值为10，第二个物品重量为3，价值为15，以此类推。
# max_weight：背包的最大承载重量。
# 输出：
#
# best_solution：一个长度与物品列表相同的二进制列表，表示选择哪些物品放入背包中。例如，best_solution = [1, 0, 1, 0]
# 表示选择了第一个和第三个物品放入背包中，而忽略了第二个和第四个物品。
# 背包问题的目标是找到一个最佳解（best_solution），使得在满足背包容量限制的前提下，背包中物品的总价值最大化。


import random


# 背包问题遗传算法函数
def genetic_algorithm(items, max_weight, population_size, generations):
    # 初始化种群
    population = generate_population(items, population_size)

    for _ in range(generations):
        # 计算适应度值
        fitness_values = calculate_fitness(population, items, max_weight)

        # 选择父代
        parents = selection(population, fitness_values)

        # 交叉繁殖
        offspring = crossover(parents, population_size)

        # 变异
        mutated_offspring = mutation(offspring)

        # 更新种群
        population = mutated_offspring

    # 计算最终适应度值
    fitness_values = calculate_fitness(population, items, max_weight)

    # 选择最佳解
    best_solution_index = fitness_values.index(max(fitness_values))
    best_solution = population[best_solution_index]

    return best_solution


# 生成初始种群
def generate_population(items, population_size):
    population = []
    for _ in range(population_size):
        solution = [random.randint(0, 1) for _ in range(len(items))]
        population.append(solution)
    return population


# 计算适应度值（背包总价值）
def calculate_fitness(population, items, max_weight):
    fitness_values = []
    for solution in population:
        total_weight = 0
        total_value = 0
        for i in range(len(solution)):
            if solution[i] == 1:
                total_weight += items[i][0]
                total_value += items[i][1]
        if total_weight <= max_weight:
            fitness_values.append(total_value)
        else:
            fitness_values.append(0)
    return fitness_values


# 选择父代
def selection(population, fitness_values):
    parents = []
    total_fitness = sum(fitness_values)

    while len(parents) < len(population):
        selected = random.choices(population, weights=fitness_values)[0]
        parents.append(selected)

    return parents


# 交叉繁殖
def crossover(parents, population_size):
    offspring = []

    while len(offspring) < population_size:
        parent1, parent2 = random.sample(parents, 2)
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        offspring.append(child1)
        offspring.append(child2)

    return offspring


# 变异
def mutation(offspring):
    mutated_offspring = []
    mutation_rate = 0.1

    for solution in offspring:
        mutated_solution = []
        for gene in solution:
            if random.random() < mutation_rate:
                mutated_solution.append(1 if gene == 0 else 0)
            else:
                mutated_solution.append(gene)
        mutated_offspring.append(mutated_solution)

    return mutated_offspring


# 示例问题
items = [(2, 10), (3, 15), (5, 20), (7, 25)]
max_weight = 10
population_size = 50
generations = 100

# 解决背包问题
best_solution = genetic_algorithm(items, max_weight, population_size, generations)
print("Best solution:", best_solution)