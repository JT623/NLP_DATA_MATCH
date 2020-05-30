import pandas as pd
import numpy as np
df =np.loadtxt('predict.txt')
print(len(df))
k=0
p=0
for i in df:
    if i == 0:
        k =k+1
    if i ==1:
        p=p+1
print(k)
print(p)
