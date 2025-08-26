#`Python代码片【子函数排列顺序与主函数的代码执行顺序一致】`.
#```python
import random
import math
import matplotlib.pyplot as plt
import numpy as np
######全局参数###########
population=[[]] #种群
population_size=400 #种群大小
chromosome_length=20 #染色体中基因个数
pc=0.6  #染色体交叉概率
pm=0.01 #染色体变异概率
iteration=200#迭代次数
#######################
########函数图像
def Issue():
    x=np.linspace(-20,20,1000)
    y=np.sin(x)
    plt.plot(x, y, 'r')
    plt.show()
########初始化种群
def Population_origin():
    population=[[]]  #种群 【二维数组】
    for i in range(population_size):
        temp=[]
        for j in range(chromosome_length):  #循环产生0和1直到个体基因，最大值choromosome_length
            temp.append(random.randint(0,1))  #随机产生0整数和1整数    append函数
            population.append(temp)
    return population[1:] #剔除第一空行
########求解显性，解，函数适应值，去负数适应值【概念：基因，显性，解，适应值，去负数适应值】
def Fitness_Function(population):
    character=[]   #基因显性
    for i in range(population_size):
        total=0
        for j in range(chromosome_length):
            total+=population[i][j]*(math.pow(2,j))
        character.append(total) #计算每个个体显性值
    function_value=[]  #适应值
    for i in range(population_size):
        x=character[i]
        function_value.append(math.sin(x))
    fitness_value=[] #去负数适应值
    for i in range(population_size):
        if(function_value[i]>0):
            temp=function_value[i]
        else:
            temp=0.0    #如果适应度小于0,则定为0
        fitness_value.append(temp)
    return fitness_value
########寻找最好的适应度和个体
def Best(population,fitness_value):
    best_individual=[]
    best_fitness=fitness_value[0]
    for i in range(population_size):
        if(fitness_value[i]>best_fitness):
            best_fitness=fitness_value[i] #读取行
            best_individual=population[i] #读取行
    return best_individual,best_fitness
########个体选择  【概念：基因，显性，解，适应值，去负数适应值】
def Selection(population,fitness_value):
    total=0
    for i in range(population_size):
        total+=fitness_value[i]
    normal_fitness=[]
    for i in range(population_size):
        normal_fitness.append(fitness_value[i]/total) #将所有个体的适应度归一化列表
    for i in range(population_size-2,-1,-1):#（计算累计概率）1.计算适应度斐伯纳且列表
        total=0
        j=0
        while(j<=i):
            total+=normal_fitness[j]
            j+=1
        normal_fitness[i]=total
        normal_fitness[population_size-1]=1
    MS=[]
    for i in range(population_size):
        MS.append(random.random())
    new_population=population
    ni=0
    nj=0
    while nj<ni:
        if(MS[nj]<normal_fitness[ni]):
            new_population[nj]=population[ni]  #对原种群，将存活个体进行保存到新种群
            nj+=1
        else:
            ni+=1
    population=new_population  #更新种群，取缔原种群
    return new_population
########染色体交叉
def Crossover(population):
    for i in range(population_size):
        if(random.random()<pc):
            crossover_point=random.randint(0,chromosome_length)
            temp1=[]
            temp2=[]
            temp1.extend(population[i][0:crossover_point])
            temp1.extend(population[i+1][crossover_point:len(population[i])])
            temp2.extend(population[i+1][0:crossover_point])
            temp2.extend(population[i][crossover_point:len(population[i])])
            population[i]=temp1
            population[i+1]=temp2
    new_population=population
    return new_population
########染色体变异
def Mutation(population):
    for i in range(population_size):
        if(random.random()<pm):
            mutation_point=random.randint(0,chromosome_length-1)
            if(population[i][mutation_point]==1):
                population[i][mutation_point]=0
            else:
                population[i][mutation_point]=1
    new_population=population
    return new_population   
########画图
def Draw_plot(results):
    X_axis= []
    Y_axis = []
    for i in range(iteration):
        X_axis.append(i)
        Y_axis.append(results[i][0])
    plt.plot(X_axis, Y_axis)  #x为被迭代到第几次，y为被迭代到该次的最优解
    plt.show()
########'main__':
Issue()# 被求解问题函数图像
results = [[]] #结果
fitness_value = [] #种群对应的适应值 一维
population=Population_origin()    #产生初始化种群，返回初始化种群
for i in range(iteration):
    fitness_value = Fitness_Function(population) #以population（二维（个体，染色体））种群为参
    best_individual, best_fitness = Best(population,fitness_value)
    results.append([best_fitness,best_individual])#不断添加新行
    population=Selection(population,fitness_value)
    population=Crossover(population)
    population=Mutation(population)
results = results[1:]
#显示每次迭代结果
Draw_plot(results)