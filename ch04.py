import numpy as np

data = np.random.randn(2,3)
data
data.shape
data.dtype

data1 = [.1,2,3]
data1

arr1 = np.array(data1)
arr1
arr1.dtype

data2 = [[1,2,3], [4,5,6]]
data2 
arr2 = np.array(data2)
arr2.ndim
arr2.shape
arr2.dtype

np.zeros(10)

np.ones(10)

np.empty(10)

np.arange(10)

data2 = [[1,2], [2,3]]
np.ones_like(data2)

np.full(10, -1)
np.eye(5)

np.array([1,2,3], dtype=np.float64)

a = np.array([1,2,3])
a.dtype
b = a.astype(np.float64)
b.dtype

a = np.array([1.1, 2.2])
b = a.astype(np.int32)
b

c = np.array(['1.1', '2.2'])
c.dtype
d = c.astype(np.float64)
d.dtype
d

arr = np.array([[1,2,3],[4,5,6]])
arr
arr * arr
arr - arr

1 / arr

arr ** 2

arr = np.array([[1,2,3],[4,5,6]])
arr2 = np.array([[1,2,3],[5,6,7]])
arr < arr2

arr = np.arange(10)
arr
arr[5:8]
arr[5:8] = 12

arr_slice = arr[5:8]
arr_slice
arr_slice[1] = 12345
arr

c = arr[5:8].copy()
c[1] = 0
arr

arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[2]
arr2d[1][0]
arr2d[1,0]

arr3d = np.array([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
arr3d
old_values = arr3d[0].copy()
arr3d[0] = 42
arr3d
arr3d[0] = old_values
arr3d
arr3d[0, 1]

arr = np.array([0,1,2,3,4,5,6,7,8,9])
arr[1:6]


arr2d = np.array([[1,2,3],[4,5,6],[7,8,9]])
arr2d[:2, 1:]

arr2d[1, :2]

arr2d[:2,2]

arr2d[:,:1]

arr2d[:2, 1:] = 0

arr2d[1, :2].shape

arr2[1:2, :2].shape

names = np.array(['a','b','c','a','d'])
data = np.random.randn(5,4)
data
names == 'a'
data[names == 'a', 2:]
data[names == 'a', 3:]
data[names == 'a', 3]

data[names != 'a', 3]
data[~(names == 'a'), 3]
cond = names == 'a'
data[~cond]

mask = (names == 'a') | (names == 'b')
mask

mask = (names == 'a') & (names == 'b')
mask

data[data < 0] = 0
data

arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
arr

arr[[4, 3, 0, 6]]

arr[[-3, -5, -7]]

arr = np.arange(32).reshape((8, 4))
arr
arr[[1,2],[1,2]]

arr[[1,2]][:, [0, 3]]

arr = np.arange(15).reshape((3, 5))
arr.T

arr = np.ones((3,1))
arr
np.dot(arr.T, arr)

arr = np.arange(8).reshape((2, 2, 2))
arr
arr.transpose((0, 2, 1))

arr = np.ones((1,2,3))
arr.shape
arr2 = arr.transpose()
arr2.shape
arr3 = arr.transpose(1,2,0)
arr3.shape

arr
arr.swapaxes(1,2)

arr = np.arange(10)
arr
np.sqrt(arr)
np.exp(arr)

arr = np.arange(8)
arr2 = np.ones(8)
np.maximum(arr, arr2)

divmod(7,2)
arr = np.array([1.1, 1.2])
remainer, whole_part = np.modf(arr)
remainer
whole_part

arr = np.arange(10, dtype=np.float64)
arr
np.sqrt(arr, arr)
np.min()

nx, ny = (3, 2)
nx
x = np.linspace(0, 1, nx)
x
y = np.linspace(0, 1, ny)
y

xv, yv = np.meshgrid(x,y)
xv
yv

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
z = np.sqrt(xs** 2 + ys**2)

import matplotlib.pyplot as plt
plt.imshow(z, cmap=plt.cm.gray)
plt.colorbar()
plt.show(block=False)

plt.close()

xarr = np.array([1.1, 1.2])
yarr = np.array([2.1, 2.2])
cond = np.array([True, False])

[ x if c else y for x, y , c in zip(xarr, yarr, cond)]

np.where(cond, xarr, yarr)

arr = np.random.randn(2,2)
arr
arr2 = np.full((2,2), 2)
# arr2
arrm2 = np.full((2,2), -2)
np.where(arr > 0, 2, arr2)

arr = np.random.randn(5, 4)

arr = np.array([[1,1,1],[2,2,2]])
arr
#Computes the statistic over the given axis
arr.sum(axis=0)
arr.sum(axis=1)
arr.mean(axis=0)
arr.mean(axis=1)

arr = np.array([0,1,2])
arr.cumsum()

arr = np.array([[0,1,2],[3,4,5],[6,7,8]])
arr.cumsum(0)
arr.cumsum(1)

arr = np.array([0,1,2])
np.cumprod(arr)

np.mean(arr)
arr.sum()

arr = np.array([1,2,3])
np.cumprod(arr)

arr = np.random.randn(100)
(arr > 0).sum()

bools = np.array([False, True, False])
bools.any()
bools.all()

arr = np.random.randn(6)
arr
arr.sort()
arr

large_arr = np.random.randn(1000)
large_arr.sort()
large_arr[int(0.05 * len(large_arr))]

names = np.array(['a', 'b', 'c', 'd'])
np.unique(names)

np.in1d([[1,2],[3,4]],[1,2,3])
a = [1,2,3]
b = [3,4,5]
np.intersect1d(a, b)
np.union1d(a, b)
np.in1d(a, b)
np.setdiff1d(a,b)
np.setxor1d(a,b)

arr = np.arange(10)
np.save('some_array', arr)
b  = np.load('some_array.npy')
b

np.savez('array_test.npz', a=a, b=b)
arch = np.load('array_test.npz')
arch['b']
arch['a']

x = np.array([[1,2,3],[4,5,6]])
y = np.ones((3,1))
np.dot(x, y)
x @ y

from numpy.linalg import inv, qr
X = np.random.randn(5,5)
mat = X.T.dot(X)
mat
inv(mat)
mat.dot(inv(mat))

import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
walk

import random
position = 0
walk = [position]
steps = 1000
for i in range(steps):
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)

plt.plot(walk[:100])
plt.show(block=False)

plt.close()

steps = (np.random.rand(100) > 0.5)
steps = np.where(steps, 1, -1)
steps[0] = 0
steps
steps = steps.cumsum()
plt.plot(steps)
plt.show(block=False)

plt.close()

draws = np.random.randint(0, 2, size=(10, 100))
steps = np.where(draws > 0, 1, -1)
steps.shape
walk = steps.cumsum(axis=1)
walk.shape

plt.plot(walk)
plt.show(block=False)

plt.close()

(np.abs(walk) >= 10).argmax()

draws = np.random.randint(0, 2, size=(10, 100))
steps = np.where(draws > 0, 1, -1)
steps.shape
walk = steps.cumsum(axis=1)


hist10 = (np.abs(walk) > 10).any(1)

crossing_time = walk[hist10].argmax(1)
crossing_time.mean()



