import numpy as np
import navier_stokes as nv
import helper_functions as hf
import matplotlib.pyplot as plt
from helper_functions import *
import solvers as s
from PIL import Image

def test_navier_stokes(image, pattern=None, save_breakpoints=True, forward=False):
    if pattern is None:
        pattern = generate_random_rectangle_pattern(
            np.shape(image)[0], np.shape(image)[1], 1, 1, 0.4
        )
        plot_from_greyscale(pattern)
        (distorted, pattern) = delete_with_pattern(image, pattern)
    else:
        distorted = np.copy(image)
    plot_from_greyscale(distorted, title=f"Damaged image")
    recovered = nv.solve_Navier_Stokes(distorted, pattern.astype(bool), max_iter=50000, delta_t=.01,
                                       breakpoint=10000, g=1, nu=2)
    discrepancy = hf.discrepancy_score(recovered, image, pattern)
    print('Discrepancy score: ', discrepancy)
    plot_from_greyscale(recovered, title=f"Restored image (Discrepancy: {discrepancy:.2f})", filename="Figures/NV/recovered.png")
    return

def test_laplace(original_image, pattern=None):
    if pattern is None:
        pattern = generate_random_rectangle_pattern(
            np.shape(original_image)[0], np.shape(original_image)[1], 15, 15, 0.2
        )
        (distorted, pattern) = delete_with_pattern(original_image, pattern)
    else:
        distorted = original_image
    plot_from_greyscale(distorted, title=f"Damaged image")
    recovered = s.solve_lagrange(distorted, pattern, num_iterations=600)
    discrepancy = hf.discrepancy_score(recovered, original_image, pattern)
    print('Discrepancy score: ', discrepancy)
    plot_from_greyscale(recovered, title=f"Restored image (Discrepancy: {discrepancy:.2f})")
    
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

image = np.zeros((20,20))
pattern = np.zeros(image.shape)
# image[10:,:] = 1
for j,k in product(range(8,13),range(20)):
    image[j,k] = 1
distorted = np.copy(image)
for j,k in product(range(5,16),range(5,16)):
    distorted[j,k] = 0
    pattern[j,k] = 1

# for j,k in product(range(8,13),range(20)):
#     image[j,k] = 1

# image = circle(21,21,10,10,5)
# pattern = np.zeros((21,21))
# for j,k in product(range(3,8),range(3,19)):
#     pattern[j,k] = 1

# distorted = np.copy(image)
# distorted[pattern.astype(bool)] = 0
# hf.plot_from_greyscale(image, title='Original image')
# hf.plot_from_greyscale(pattern, title='Mask image')
# test_navier_stokes(distorted, pattern=pattern, forward=False)
#test_laplace(distorted, pattern)

fish_path = "pictures\sfish.png"
fish_text_path = "pictures\sfish_text.png"
fish = np.asarray(Image.open(fish_path).convert("L")) / 255
fish_text = np.asarray(Image.open(fish_text_path).convert("L")) / 255

fish_pattern = np.zeros(fish.shape).astype(bool)
fish_pattern[fish-fish_text >0] = True
fish_pattern[fish-fish < 0] = True
hf.plot_from_greyscale(fish, title='Original image')
hf.plot_from_greyscale(fish_pattern, title='Mask image')
plt.show()

test_navier_stokes(fish_text, pattern=fish_pattern, forward=False)
# test_laplace(fish_text, pattern=fish_pattern)
# test_laplace(fish_text, pattern=fish_pattern)