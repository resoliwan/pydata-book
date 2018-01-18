import pandas as pd
from pandas import Series, DataFrame
import numpy as np

obj = pd.Series([1,2,3,4])
obj.values
obj.index

obj2 = pd.Series([1,2,3,4], index=['a','b','c','d'])
obj2
obj2.values
obj2.index
obj2['a']
obj2['a'] = 3
obj2[['a','b','c']]
obj2[obj2> 2] * 2
np.exp(obj2)

'b' in obj2

'e' in obj2

sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = pd.Series(sdata)

states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = pd.Series(sdata, index=states)
obj4

pd.isnull(obj4)

pd.notnull(obj4)

obj3 + obj4

obj4.name = 'population'
obj4.index.name = 'state'
obj4
obj3

obj
obj.index = ['a', 'b', 'c', 'd']
obj
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002, 2003],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9, 3.2]}

frame = pd.DataFrame(data)
frame.head(1)

frame2 = pd.DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                              index=['one', 'two', 'three', 'four', 'five', 'six'])
frame2 
frame2.columns
frame2['state']
frame2.state
frame2.loc['three']
frame2['debt'] = 16.5
frame2
frame2['debt'] = np.arange(0., 6.)

frame2

val = pd.Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
val
frame2['debt'] = val
frame2

frame2['eastern'] = frame2.state == 'Ohio'
frame2

del frame2['eastern']
frame2

frame2.columns
pop = {
        'Nevada': {2001: 2.4, 2002: 2.9},
        'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}
      }
frame3 = pd.DataFrame(pop)
frame3

frame4 = frame3.T
frame4.columns

frame5 =  pd.DataFrame(pop, index=[2001,2002,2003])
frame5

pdata = {'Ohio': frame3['Ohio'],
         'Nevada': frame3['Nevada']}
pd.DataFrame(pdata)

frame3.index.name = 'year'
frame3.columns.name = 'state'
frame3

frame3.values

frame2.values

obj = pd.Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index
index[1:]
# index[1] = 'd'
labels = pd.Index(np.arange(3))
labels

obj2 = pd.Series([1.4, 1.5, 1.6], index=labels)
obj2
obj2.index is labels

frame3.columns

'Ohio' in frame3.columns
2003 in frame3.index

dup_labels = pd.Index(['a', 'a'])
dup_labels

obj = pd.Series([4.5, 7.2, -5.3, 3.6], index=['a', 'b','c','d'])
obj
obj2 = obj.reindex(['b','c','e'])
obj2

obj3 = pd.Series(['b','p','y'], index=[0,2,4])
obj3

obj3.reindex(range(6), method='ffill')

frame = pd.DataFrame(np.arange(9).reshape(3,3), index=['a','c','d'], columns=['Ohio','Texas','California'])
frame

frame.reindex(['a','b','c','d'])

frame.reindex(columns=['Texas','Utah','California'])

frame.loc[['a','b'], ['Texas', 'Utah']]

obj =  pd.Series(np.arange(5.), index=['a','b','c','d','e'])
obj

new_obj = obj.drop('c')
new_obj
obj.drop(['d','a'])

data = pd.DataFrame(np.arange(16).reshape(4,4), index=['Ohio','Colorado','Utah','New York'], columns=['one','two','three','four'])
data.drop(['Colorado', 'Ohio'])
data.drop('two', axis=1, inplace=True)
data
obj = pd.Series(np.arange(4.), index=['a','b','c','d'])
obj
obj['b']
obj[1]
obj[2:4]
obj[['a','b','c']]
obj[[1, 3]]
obj[obj < 2]
obj
obj['a':'c'] = 4
obj

data = pd.DataFrame(np.arange(16).reshape(4,4), index=['Ohio','Colorado','Utah','New York'], columns=['one','two','three','four'])
data
data[['two','three']]
data[:2]
data[data['three'] > 5]
data[data < 5] = 0
data.loc['Colorado', ['two','three']]
data.loc[['Colorado','Ohio'],'two']
data.iloc[:3,[3,0,1]]

(data.iloc[:,:3]).iloc[:1,:2]

ser = pd.Series(np.arange(3.))
ser
ser.iloc[-1]

ser2 = pd.Series(np.arange(3.), index=['a','b','c'])
ser2
ser2[-2]

s1 = pd.Series([7.3,-2.5,3.4,1.5], index=['a','c','d','e'])
s2 = pd.Series([-2.1,3.6,-1.5,4,3.1], index=['a','c','e','f','g'])
s1
s2
s1 + s2

df1 = pd.DataFrame(np.arange(9.).reshape(3,3), index=['Ohio','Texas','Colorado'], columns=['b','c','d'])
df2 = pd.DataFrame(np.arange(12.).reshape(4,3), index=['Utah','Ohio','Texas','Oregon'], columns=['b','d','e'])
df1
df2

df1 + df2

df1 = pd.DataFrame({'A': [1, 2]})
df2 = pd.DataFrame({'B': [3, 4]})
df1
df2
df1 - df2

list('abcd')

df1 = pd.DataFrame(np.arange(12.).reshape(3,4), columns=list('abcd'))
df2 = pd.DataFrame(np.arange(20.).reshape(4,5), columns=list('abcde'))
df1
df2.loc[1, 'b'] = np.nan
df2

df1.add(df2, fill_value=0)

df1
1 /df1
df1.rdiv(1)

df1.reindex(columns=df2.columns, fill_value=0)

arr = np.arange(12.).reshape((3, 4))
arr
arr[0]

arr - arr[0]

frame = pd.DataFrame(np.arange(12.).reshape((4,3)), index=['Utah','Ohio','Texas','Oregon'], columns=list('bde'))
frame
series = frame.iloc[0]
series
frame - series
series2 = pd.Series(range(3), index=list('bef'))
series2
frame - series2

series3 = frame['d']
series3
frame 

frame - series3
frame.sub(series3, axis=0)

frame = pd.DataFrame(np.arange(12).reshape(4,3), index=['Utah','Ohio','Texas','Oregon'], columns=list('bde'))
frame
np.abs(frame)

f = lambda x: x.max()
frame
frame.apply(f, axis=1)

def f(x):
    return pd.Series([x.min(), x.max()], index=['min', 'max'])

frame.apply(f)

format = lambda x: '%.2f' % x
frame.applymap(format)

frame['e'].map(format)

obj = pd.Series(np.arange(4), index=list('dabc'))
obj
obj.sort_index()

frame = pd.DataFrame(np.arange(8).reshape(2,4), index=['three','one'], columns=list('dabc'))
frame.sort_index()
frame.sort_index(axis=1, ascending=False)

obj = pd.Series([4,7,-3,2])
obj.sort_values()

obj = pd.Series([4, np.nan, 7, np.nan, -3,2])
obj.sort_values()

frame = pd.DataFrame({'b':[4,7,-3,2], 'a': [0,1,0,1]})
frame
frame.sort_values(by=['a','b'])

obj = pd.Series([7,-5,7,4])
obj
obj.rank()

# obj = pd.Series([1,1,1,1])
obj = pd.Series([2,2,2])
obj.rank(pct=True)
obj = pd.Series([1,1,1])
obj.rank(pct=True)

arr = np.arange(4)
arr
arr[::-1]

frame= pd.DataFrame({'b': np.arange(4), 'a': np.arange(4)[::-1], 'c': [1,2,3,1]})

frame.rank(axis=1)

obj = pd.Series(range(5), index=list('aabbc'))
obj.index.is_unique
obj['a']
obj['c']

df = pd.DataFrame(np.random.randn(3,3), index=list('aab'))
df
df.loc['a']
df.loc['b']

df = pd.DataFrame([[1.4, np.nan], [7.10, -4.5], [np.nan, np.nan], [0.75, -1.3]], index=list('abcd'), columns=['one','two'])
df

df.sum(axis=1)
df.mean(axis=1, skipna=False)

df.idxmax()

df.cumsum()

df = pd.DataFrame({'a': range(4), 'b': list('abcd'), 'c':range(4)[::-1]})
df

df.rename(columns={'a':'e', 'x':'t'})
df.describe()

df.count()
df.quantile()

obj = pd.Series(['a','a','b','c'] * 4)
obj.describe()

obj = pd.Series([1,2,3,4])
obj.diff()

obj.quantile([.5])

2 + 1 * 0.5

obj = pd.Series([15,20,35,40,50])
obj.quantile([.25,.5,.75,.90])



obj = pd.Series([1,2,3,4])
obj.diff()


obj = pd.Series([1,2,3,4])
obj.diff()


price = pd.read_pickle('examples/yahoo_price.pkl')
volume = pd.read_pickle('examples/yahoo_volume.pkl')
price.head()

volume.head()

returns = price.pct_change()

returns['MSFT'].corr(returns['IBM'])

returns['MSFT'].cov(returns['IBM'])

returns.corr()

returns.corrwith(returns['IBM'])

fruits = pd.Series(['a','b','b'])
fruits_cat = fruits.astype('category')

fruits_cat.values.codes


