from sko.PSO import PSO
import matplotlib.pyplot as plt
from DNN import fitModel
from DNN import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sko.GA import GA
import numpy as np

path = "./dataset/qb_lxb_100000.csv"  # 存放文件路径

df = pd.read_csv(path)

print(df[(df['damageLevel'] == 1)].shape)
print(df[(df['damageLevel'] == 2)].shape)
print(df[(df['damageLevel'] == 3)].shape)
print(df[(df['damageLevel'] == 4)].shape)

df1 = df[(df['damageLevel'] == 1)].sample(n=3000, random_state=None)
df2 = df[(df['damageLevel'] == 2)].sample(n=3000, random_state=None)
df3 = df[(df['damageLevel'] == 3)]
df4 = df[(df['damageLevel'] == 4)].sample(n=3000, random_state=None)

df = pd.concat([df1, df2, df3, df4])
df = df.sample(frac=1)
print(df.shape)
df.to_csv('./dataset/qb_lxb_balance_10000.csv')
