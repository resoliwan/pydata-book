import pandas as pd
import numpy as np
import sys
from pprint import pprint

data = pd.read_csv('./examples/ex5.csv')

data.to_csv('./examples/out.csv', na_rep='NULL', index=False)

data.to_csv(sys.stdout, na_rep='NULL', index=False, columns=['a','b'])

dates = pd.date_range('1/1/2000', periods=7)

ts = pd.Series(np.arange(7), index=dates)
ts.to_csv(sys.stdout)

import csv
f = open('./examples/ex7.csv')

reader = csv.reader(f)

for line in reader:
  print(line)



with open('./examples/ex7.csv') as f:
  lines = list(csv.reader(f))
  header, values = lines[0], lines[1:]
  data_dict = {h: v for h, v in zip(header, zip(*values))}
  print('data_dict', data_dict)

  # print(lines)

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 30, "pets": ["Zeus", "Zuko"]},
              {"name": "Katie", "age": 38,
               "pets": ["Sixes", "Stache", "Cisco"]}]
}
"""
import json

result = json.loads(obj)

result

asjson = json.dumps(result)
asjson


siblings = pd.DataFrame(result['siblings'], columns=['name', 'age'])
siblings 

data = pd.read_json('examples/example.json')
data

data = """
{
    "_id": {
        "$oid": "5a39a80a5c0d5d000117f1c2"
    },
    "target": "cslogs",
    "type": 150,
    "date": {
        "$date": "2017-12-20T00:00:10.848Z"
    },
    "accountid": "5a35d6baf150d10001fe2836",
    "platform": 1,
    "opts": {
        "igr": false,
        "vip": false,
        "client_ip": "0000.0000.0000.0000",
        "account": {
            "owner": "0000",
            "name": "0000",
            "level": 8,
            "exp": 250,
            "cash": 0,
            "money": 2625,
            "account_type": 1
        },
        "session_id": "5a39a80afb17c60001f34eea",
        "date": "2017-12-20T00:00:10.848Z",
        "accountid": "5a35d6baf150d10001fe2836",
        "type": 150,
        "platform": 1
    }
}
"""

df = pd.read_json('./examples/150.json', orient='records')

from pandas.io.json import json_normalize
from flatten_json import flatten_json

flat = flatten_json(json.loads(data), separator='.')
[flat, flat]

df = json_normalize([flat, flat])
column_map = {
    'accountid': 'aid',
    'type': 'log_type',
    'date.$date': 'log_dt',
    'opts.platform': 'platform',
    'opts.store': 'market_type',
    'opts.igr': 'igr_state',
    'opts.vip': 'vip_state'
    }
df.rename(columns=column_map, inplace=True)
df.head()

df.head(1)

df.columns

df = pd.DataFrame([['a','b'], ['c','d']], index=['row1', 'row2'], columns=['col1', 'col2'])
df

df.to_json(orient='split')

pd.read_json(_, orient='split')

df.to_json(orient='index')
pd.read_json(_, orient='index')

from pandas.io.json import json_normalize
from flatten_json import flatten_json


data = [
    {'id': 1, 'name': {'first': 'Coleen', 'last': 'Volk'}},
    {'name': {'given': 'Mose', 'family': 'Regner'}},
    {'name': {'given': ['Mose', 'hello'], 'family': 'Regner'}},
    {'id': 2, 'name': 'Faye Raker'}
    ]

flat = flatten_json(data[2])
json_normalize(flat)


json_normalize(data)

data = [{'state': 'Florida',
         'shortname': 'FL',
         'info': {
              'governor': 'Rick Scott'
         },
         'counties': [{'name': 'Dade', 'population': 12345},
                     {'name': 'Broward', 'population': 40000},
                     {'name': 'Palm Beach', 'population': 60000}]},
        {'state': 'Ohio',
         'shortname': 'OH',
         'info': {
              'governor': 'John Kasich'
         },
         'counties': [{'name': 'Summit', 'population': 1234},
                      {'name': 'Cuyahoga', 'population': 1337}]}]

json_normalize(data, 'counties', ['state', ['info','governor']])

df.to_json(orient='table')

f = pd.DataFrame({'a': np.random.randn(100)})

store = pd.HDFStore('mydata.h5')

store['obj1'] = f
store['obj_col'] = f['a']

store.close()

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



