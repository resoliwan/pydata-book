import pandas as pd
import numpy as np

file_prefix = 'head_'
# file_prefix = ''

unames = ['user_id', 'gender', 'age', 'occupation', 'zip-code']
users = pd.read_table('./datasets/movielens/users.dat', sep='::', names=unames, header=None)

users[:4]


rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('./datasets/movielens/' + file_prefix + 'ratings.dat', sep='::', names=rnames, header=None)

ratings[:4]

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('./datasets/movielens/movies.dat', sep='::', names=mnames, header=None)

ages_data = {
    'code': [1, 18 ,25 ,35 ,45 ,50, 56],
    'cate': ["Under 18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"]
    }
ages = pd.DataFrame(ages_data)
ages.head()

data = pd.merge(pd.merge(ratings, users), movies)

data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')

pd.pivot_table()
data.pivot_table('rating', index='title', columns=['gender', 'age'], aggfunc='mean')
