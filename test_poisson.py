import numpy as np
import navier_stokes as nv
import helper_functions as hf
import matplotlib.pyplot as plt
from helper_functions import *
import solvers as s
from PIL import Image
from itertools import product


logo_path = "pictures\logo.png"
cat_path = "pictures\cat.png"
tortoise_path = "pictures\small_tortoise.png"
logo = np.asarray(Image.open(logo_path).convert("L")) / 255
cat = np.asarray(Image.open(cat_path).convert("L")) / 255
tortoise = np.asarray(Image.open(tortoise_path).convert("L")) / 255

I = cat
#I = np.arange(100).reshape((10,10))/99

omegas = nv.calculate_laplace_matrix(I)

hf.plot_two_greyscales(I, omegas, pause=False)

pattern = np.ones(I.shape).astype(bool)
#size_x, size_y = I.shape
#for j, k in product(range(I.shape[0]), range(I.shape[1])):
#    if j <3 or k <3 or j > size_x-3 or k > size_y > 3:
#        pattern[j,k] = False

pattern = generate_random_rectangle_pattern(np.shape(I)[0], np.shape(I)[1], 5, 5, 0.3)
#plot_from_greyscale(pattern)
(distorted, pattern) = delete_with_pattern(I, pattern)
plot_from_greyscale(distorted)
I_restored = nv.solve_poisson(distorted, pattern.astype(bool), omegas, num_iterations=30, plot=False)

hf.plot_two_greyscales(I, I_restored, pause=False)















# omegas = np.random.random((10, 10))


# image = np.random.random((10, 10))
# mask = np.zeros((10, 10))
# for i in range(10):
#     mask[5][i] = 1
#     mask[4][i] = 1


# (distorted, mask) = hf.delete_with_pattern(image, mask)

# recovered = nv.solve_poisson(distorted, mask, omegas)
# hf.plot_two_greyscales(image, distorted)


# recovered_omegas = nv.calculate_laplace_matrix(recovered)

# hf.plot_two_greyscales(
#     omegas,
#     recovered_omegas,
#     pause=False
# )
