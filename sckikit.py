from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit([1,2,2,6])
le.classes_
le.transform([1, 1, 2, 6])
le.inverse_transform([0,0,1,2])

le = preprocessing.LabelEncoder()
le.fit(['paris', 'tokyo'])
le.transform(['paris'])


