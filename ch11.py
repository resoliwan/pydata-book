from datetime import datetime
import pandas as pd

now = datetime.now()
now
now.year, now.month, now.day, now.hour, now.minute, now.second, now.timestamp

delta = datetime(2017, 1, 7) - datetime(2018,1,7,23,59,59)
delta

start = datetime(2017,1,7) + delta
start

stamp = datetime(2011, 1, 3)
stamp

str(stamp)

stamp.strftime('%Y-%m-%d %A')

value='2011-01-03'
datetime.strptime(value, '%Y-%m-%d')

datestrs = ['7/6/2011', '8/6/2011']
[datetime.strptime(x, '%m/%d/%Y') for x in datestrs]

from dateutil.parser import parse

parse('2011-01-03')
parse('Jan 31, 1997 10:45 PM')

parse('6/12/2011', dayfirst=True)

datestrs = ['2011-07-06 12:00:00','2011-09-06 12:00:00']
idx = pd.to_datetime(datestrs + [None])
idx
idx[2]
pd.isnull(idx)

from datetime import datetime
import numpy as np

dates =dates = [datetime(2011,1,2),
                datetime(2011,1,5),
                datetime(2011,1,7),
                datetime(2011,1,8),
                datetime(2011,1,10),
                datetime(2011,1,12)]
ts = pd.Series(np.random.randn(6), index=dates)
ts.index
ts.values

ts + ts[::2]

ts.index.dtype

stamp = ts.index[0]
stamp

stamp = ts.index[2]
stamp 

ts[stamp]
ts['2011-01-12']
ts['2011/01/12']

ts = pd.Series(np.random.randn(1000), index=pd.date_range('2011-01-01', periods=1000))
ts['2011-05-01']
ts[datetime(2011,1,7)]
ts['2011-05-01':'2011-05-02']


dates = pd.date_range('1/1/2000', periods=10,freq='W-WED')
dates

dates = pd.DatetimeIndex(['1/1/2000','1/2/2000','1/2/2000','1/2/2000', '2/3/2000'])
dup_ts = pd.Series(np.arange(5), index=dates)

dup_ts.index.is_unique

dup_ts['2000/01/01']
dup_ts['2000/01/02']
grouped = dup_ts.groupby(level=0)
grouped.sum()

