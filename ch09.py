import pandas as pd
import numpy as np

val = 'a,b, guido'
pieces = [x.strip() for x in val.split(',')]
'::'.join(pieces)

val.index(':')
val.find(':')
val.count(',')
val.replace(',', '')
