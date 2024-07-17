import numpy as np
from itertools import product
import helper_functions as hf


def solve_lagrange(distorted, pattern, num_iterations=200, safeplots=False):
    omega = 1.9
    (size_x, size_y) = np.shape(distorted)
    out = np.copy(distorted)
    last = np.zeros(np.shape(distorted))
    distorted_pixel = np.average(pattern) * size_x * size_y

    for iteration in range(num_iterations):
        
        if safeplots ==True:

            filename = "movie_lagrange/lagrange" + str(iteration + 100) + ".png"
            title = "Lagrange, " + str(iteration)
            hf.plot_from_greyscale(out, filename, title)

        for j, k in product(range(size_x), range(size_y)):
            if pattern[j][k] == True:
                sum = out[j - 1][k] + out[j + 1][k] + out[j][k - 1] + out[j][k + 1]
                out[j][k] = (1 - omega) * out[j][k] + omega * sum / 4

        if iteration < 10 or iteration % 2 == 1:
            continue
        else:
            temp = 0
            for j, k in product(range(size_x), range(size_y)):
                if pattern[j][k] == True:
                    temp += abs(out[j][k] - last[j][k])

            if temp / distorted_pixel < 0.001:
                return out

            last = np.copy(out)

    return out


def solve_euler_diffusion(distorted, pattern,safeplots=False):
    delta_t = 0.1  # alias h
    diffusion_const = 1
    lattice_const = 1

    (size_x, size_y) = np.shape(distorted)
    num_iterations = 200
    out = np.copy(distorted)
    last = np.zeros(np.shape(distorted))
    distorted_pixel = np.average(pattern) * size_x * size_y

    for iteration in range(num_iterations):
        # if iteration % 5 == 0:
        # print(f"current iteration", iteration)

        if safeplots ==True:

            filename = "movie_diffusionV1/DiffusionV1" + str(iteration + 100) + ".png"
            title = "DiffusionV1, " + str(iteration)
            hf.plot_from_greyscale(out, filename, title)

        for j, k in product(range(size_x), range(size_y)):
            if pattern[j][k] == True:
                sum = 0
                edges = 0
                try:
                    sum += out[j - 1][k]
                except IndexError():
                    edges += 1
                try:
                    sum += out[j + 1][k]
                except IndexError():
                    edges += 1
                try:
                    sum += out[j][k - 1]
                except IndexError():
                    edges += 1
                try:
                    sum += out[j][k + 1]
                except IndexError():
                    edges += 1

                out[j][k] = (
                    last[j][k]
                    + diffusion_const
                    * delta_t
                    * (sum - (4 - edges) * last[j][k])
                    / lattice_const**2
                )

        if iteration < 10 or iteration % 2 == 1:
            continue
        else:
            temp = 0
            for j, k in product(range(size_x), range(size_y)):
                if pattern[j][k] == True:
                    temp += abs(out[j][k] - last[j][k])

            if temp / distorted_pixel < 0.001:
                # print(f"finished after", iteration, "iterations")
                return out

            last = np.copy(out)

    # print(f"finished after", num_iterations, " iterations, not converged")
    return out
