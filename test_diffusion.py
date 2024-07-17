import diffusion as d3
import numpy as np
import helper_functions as hf
from itertools import product
from PIL import Image
import solvers as s
import navier_stokes as nv

tortoise_path = "pictures\small_tortoise.png"
tortoise = np.asarray(Image.open(tortoise_path).convert("L")) / 255
original_image = tortoise

color_tortoise_path = "color pictures\small_tortoise.png"
color_tortoise = np.asarray(Image.open(color_tortoise_path).convert("RGB")) / 255
original_color_image = color_tortoise



def circle(size_x, size_y, center_x, center_y, radius):
    out = np.zeros((size_x, size_y))
    for i, j in product(range(size_x), range(size_y)):
        if (i - center_x) ** 2 + (j - center_y) ** 2 - radius**2 < 0.1:
            out[i][j] = 1

    return out


def four_circles():
    image = np.zeros((200, 200))
    image = np.add(image, circle(200, 200, 55, 55, 35))
    image = np.add(image, circle(200, 200, 145, 55, 35))
    image = np.add(image, circle(200, 200, 145, 145, 35))
    image = np.add(image, circle(200, 200, 55, 145, 35))
    image = image * 0.75

    pattern = circle(200, 200, 100, 100, 45)
    return image, pattern


def half_half():
    image = np.zeros((100, 100))
    for i, j in product(range(100), range(100)):
        if i < 50:
            image[i][j] = 0.8
        else:
            image[i][j] = 0.4
    pattern = circle(100, 100, 50, 50, 10)

    return image, pattern


def testdiffusion2():
    
    pattern = hf.generate_grid(
        np.shape(original_image)[0], np.shape(original_image)[1]
    )

    (distorted, pattern) = hf.delete_with_pattern(original_image, pattern)
    # hf.plot_two_greyscales(pattern, distorted)

    d3.TV_inpainting(distorted, pattern, delta_t=1/4, T=200)



    





#true_color_reconstruction()
testdiffusion2()



hf.to_movie('old/test','movie_diffusion.avi')


# score 
