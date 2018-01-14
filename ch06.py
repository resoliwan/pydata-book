import pandas as pd

df = pd.read_csv('./examples/ex1.csv')

df.columns

df.head()
df.describe()

df = pd.read_table('./examples/ex1.csv', delimiter=',')
df

names = ['a','b','c','d','message']
df = pd.read_csv('./examples/ex2.csv', header=None, names=names, index_col=['message'])
df

df = pd.read_csv('./examples/csv_mindex.csv',index_col=['key1', 'key2'])
df

list(open('./examples/ex3.txt'))

pd.read_table('./examples/ex3.txt', sep='\s+')

df = pd.read_fwf('./examples/ex3.txt', index_col=0)

df = pd.read_csv('./examples/ex4.csv', skip_rows=[0,2,3])
df

santinels = {'message': 'world', 'something': 'one'}
df = pd.read_csv('./examples/ex5.csv', na_values=santinels)

converters = {
        'a': lambda x: x + 'a',
        'b': lambda x: x + 'b'
        }

df = pd.read_csv('./examples/ex0.csv', parse_dates=['dt'], converters=converters, verbose=True)
df

df = pd.read_csv('./examples/ex0.csv', thousands=',')
df



