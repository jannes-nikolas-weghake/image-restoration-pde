import numpy as np
from helper_functions import *
import navier_stokes as nv
import solvers as s
from PIL import Image
from timeit import Timer
n = 100
I = np.random.random((n,n))
size_x, size_y = I.shape

def test_loop():
    laplace = np.zeros((size_x, size_y))

    for j, k in product(range(1,size_x-1), range(1,size_y-1)):
        #(sum, edges) = sum_neighbours(I, j, k)
        #laplace[j, k] = sum - (4 - edges) * I[j][k]
        laplace[j, k] = I[j+1,k] + I[j-1,k] + I[j,k+1] + I[j, k-1] - 4 * I[j][k]

    return laplace

def test_vectorized():
    I1 = np.zeros((size_x+2, size_y+2))
    I1[1:-1, 1:-1] = I
    laplace = I1[2:,1:-1]+I1[:-2,1:-1] + I1[1:-1,2:] + I1[1:-1,:-2] - 4 * I

    return laplace

def sum_neighbours(I, j, k):

    sum = 0
    edges = 0
    if j>0:
        sum += I[j - 1][k]
    else:
        edges += 1
    try:
        sum += I[j + 1][k]
    except IndexError:
        edges += 1
    if k >0:
        sum += I[j][k - 1]
    else:
        edges += 1
    try:
        sum += I[j][k + 1]
    except IndexError:
        edges += 1

    return sum, edges
timer = Timer(test_loop)
print(f"Loop time: {timer.timeit(100)}")

timer = Timer(test_vectorized)
print(f"Vectorized time: {timer.timeit(100)}")
# timer = Timer("""I1 = np.zeros((1000+2, 1000+2))""", setup="""import numpy as np""")
# print(f"Time: {timer.timeit(100)}")



