import numpy as np
import pandas as pd

values = pd.Series(['apple', 'orange', 'apple', 'apple'] * 2)
pd.unique(values)

pd.value_counts(values)

values = pd.Series([0,1,0,0]*2)
dim = pd.Series(['apple', 'orange'])

values
dim

dim.take(values)
dim.take(values)

fruits = ['apple', 'orange', 'apple', 'apple'] * 2
N = len(fruits)
df = pd.DataFrame({'fruit': fruits,
                   'basket_id': np.arange(N),
                   'count': np.random.randint(3, 15, size=N),
                   'weight': np.random.uniform(0, 4, size=N)},
                  columns=['basket_id', 'fruit', 'count', 'weight'])
df

fruit_cat = df['fruit'].astype('category')

fruit_cat.values.categories
fruit_cat.values.codes

df['fruit'] = fruit_cat

df
pd.get_dummies(df['fruit'])

pd.concat([df, pd.get_dummies(df['fruit'])], axis=1)
