

#1)
import numpy as np

#2)
arr = np.zeros(10)

#3)
arr = np.ones(10)

#4)
arr = np.ones(10)*5

#5)
mat = np.ones((10,10))

#6)
arr = np.arange(10,50)


#7)
arr = np.arange(9)

# print(arr)

mat = arr.reshape((3,3))
mat = np.reshape(arr, (3,3))


#8)
arr = np.random.randn(10000)

print(np.mean(arr))
print(np.std(arr))

#9)
arr = np.linspace(0.01,1.0,100)
# print(arr)
A = arr.reshape((10,10))

A = np.reshape(np.linspace(0.01,1.0,100),(10,10))
# print(A)

#10)
B = A[6:,:4]
print(B)

#11)
C = A[2:4,:]
# ou C = A[[2,3],:]

#12)
D = C.dot(A)

print(D)

#13
print(A[1,2])

#14)
print(np.mean(A))

#15)
print(np.mean(A,1))

#16)

print(np.mean(A,0))

#17
A[:,1] = 2

#18
np.savetxt("matA.csv",A, delimiter = ",")
