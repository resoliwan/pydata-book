import pandas as pd
import numpy as np

string_data = pd.Series(['a','b',np.nan,'a'])
string_data[0] = None
string_data.isnull()
string_data.notnull()

from numpy import nan as NA

data = pd.Series([1, NA, 3.5, NA, 7])

data.dropna()
data[data.notnull()]

data = pd.DataFrame([
    [1,6,3],
    [1,NA,NA],
    [NA,NA,NA],
    [NA,6.3, 3]
    ])

data.dropna()

data.dropna(how='all')

data[4] = NA

data.dropna(axis=1, how='all')
df = pd.DataFrame(np.random.randn(7,3))
df.iloc[:4,1] = NA
df.iloc[:2,2] = NA

df.dropna(thresh=2)

df.fillna(0)

df.fillna({1:0.5}, inplace=True)

df = pd.DataFrame(np.random.randn(6,3))
df.iloc[2:,1] = NA
df.iloc[4:,2] = NA
df

df.fillna(method='ffill', limit=1)

df.fillna(data.mean())




df.fillna(0)

data = pd.DataFrame({'k1': ['one','two']*3+['two'], 'k2': [1,1,2,3,3,4,4]})
data
data.duplicated()
data.drop_duplicates()

data['v1'] = range(7)
data

data.duplicated(['k1'])

data.drop_duplicates(['k1','k2'], keep='last')
data.drop_duplicates(['k1','k2'])

data = pd.DataFrame({'food': ['bacon', 'pulled pork','bacon','Pastrami','corned beef', 'Bacon', 'pastrami','honey ham', 'nova lox'],
    'ounce': [4,3,12,6,7.5,8,3,5,6]})

meat_to_animal = {
        'bacon': 'pig',
        'pulled pork': 'pig',
        'pastrami': 'cow',
        'corned beef': 'cow',
        'honey ham': 'pig',
        'nova lox': 'salmon'
        }

data['src'] = data['food'].map(lambda x: meat_to_animal[x.lower()])
data
