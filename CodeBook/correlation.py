import pandas as pd
from scipy.stats import spearmanr, pearsonr

# path='../Evaluation/MNIST/normal/summary.csv'
# df=pd.read_csv(path)
#
# print(df)
# stats=[]
# for i in ('NC0.1','NC0.3','NC0.5','NC0.7', 'NC0.9', 'TKNC', 'TKNP', 'KMNC', 'NBC', 'SNAC'):
#     for j in ('mean', 'max', 'min'):
#         stats.append('ft'+'_'+i+'_'+j)
# print(stats)
# df_nc=df[stats]
# df_nc.dropna()
# print(df_nc[:10])
dir = "../figure/para.csv"
df = pd.read_csv(dir)
print(df)

for i in range(2,7):
    print(df.columns.values[i])
    pearson = pearsonr(df['para'], df.iloc[:, i])
    spearman = spearmanr(df['para'], df.iloc[:, i])
    print('pearson:', pearson)
    print('spearman:', spearman, '\n')



