import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

df = pd.read_csv("responses.csv")
df.shape

df = df.iloc[:, 76:133]
df.head(5)

df = df.dropna()
#...............................................................................................
#Encode categorical data
from sklearn.preprocessing import LabelEncoder

df = df.apply(LabelEncoder().fit_transform)
df

from factor_analyzer import FactorAnalyzer         # pip install factor_analyzer 
fa = FactorAnalyzer(rotation="varimax")
fa.fit(df) 

# Check Eigenvalues
ev, v = fa.get_eigenvalues()
ev

# Create scree plot using matplotlib
plt.scatter(range(1,df.shape[1]+1),ev)
plt.plot(range(1,df.shape[1]+1),ev)
plt.title('Scree Plot')
plt.xlabel('Factors')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()

fa = FactorAnalyzer(5, rotation="varimax")
fa.fit(df)
AF = fa.loadings_
AF = pd.DataFrame(AF)
AF.index = df.columns
AF

F = AF.unstack()
F = pd.DataFrame(F).reset_index()
F = F.sort_values(['level_0',0], ascending=False).groupby('level_0').head(5)    # Top 5 
F = F.sort_values(by="level_0")
F.columns=["FACTOR","Variable","Varianza_Explica"]
F = F.reset_index().drop(["index"],axis=1)
F

F = F.pivot(columns='FACTOR')["Variable"]
F.apply(lambda x: pd.Series(x.dropna().to_numpy()))