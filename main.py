import numpy as np
from helper_functions import *
import navier_stokes as nv
import solvers as s
from PIL import Image

# {
# More ideas:
# Image compression with discrepancy score for picture (sparse matrices)
# image upscaling
# use nbc boundary conditions
# }
""""new score: isophotone direction"""

logo_path = "pictures\logo.png"
cat_path = "pictures\cat.png"
cat_eye_path = "pictures\cat_eye.png"
cat_eye_small_path = "pictures\cat_eye_small.png"
cat_edge_path = "pictures\cat_edge.png"
tortoise_path = "pictures\small_tortoise.png"
logo = np.asarray(Image.open(logo_path).convert("L")) / 255
cat = np.asarray(Image.open(cat_path).convert("L")) / 255
cat_eye = np.asarray(Image.open(cat_eye_path).convert("L")) / 255
cat_eye_small = np.asarray(Image.open(cat_eye_small_path).convert("L")) / 255
cat_edge = np.asarray(Image.open(cat_edge_path).convert("L")) / 255
tortoise = np.asarray(Image.open(tortoise_path).convert("L")) / 255


original_image = tortoise


def test_laplace():
    pattern = generate_random_rectangle_pattern(
        np.shape(original_image)[0], np.shape(original_image)[1], 15, 15, 0.2
    )
    (distorted, pattern) = delete_with_pattern(original_image, pattern)
    recovered = s.solve_lagrange(distorted, pattern)
    plot_from_greyscale(recovered)


def test_navier_stokes():
    pattern = generate_random_rectangle_pattern(
        np.shape(original_image)[0], np.shape(original_image)[1], 15, 15, 0.05
    )
    (distorted, pattern) = delete_with_pattern(original_image, pattern)
    recovered = nv.solve_Navier_Stokes(
        distorted,
        pattern.astype(bool),
        max_iter=10000,
        delta_t=0.01,
        breakpoint=200,
        g=1,
        nu=2,
    )
    plot_two_greyscales(distorted, recovered, pause=False)


def test_both():
    pattern = generate_random_rectangle_pattern(
        np.shape(original_image)[0], np.shape(original_image)[1], 8, 8, 0.2
    )
    (distorted, pattern) = delete_with_pattern(original_image, pattern)

    recovered = s.solve_lagrange(
        np.copy(distorted), np.copy(pattern), num_iterations=400
    )
    w = nv.calculate_laplace_matrix(recovered)
    plot_from_greyscale(recovered, filename="Figures/lagrange")
    recovered = nv.solve_Navier_Stokes(
        distorted,
        pattern.astype(bool),
        max_iter=100000,
        delta_t=0.01,
        breakpoint=10000,
        g=1,
        nu=2,
        save_breakpoints=True
    )
    plot_two_greyscales(
        distorted, recovered, pause=False, filename="Figures/nv_recovered.png"
    )


def test_iso_score():
    pattern = generate_random_rectangle_pattern(
        np.shape(original_image)[0], np.shape(original_image)[1], 8, 8, 0.2
    )
    (distorted, pattern) = delete_with_pattern(original_image, pattern)

    recovered = s.solve_lagrange(np.copy(distorted), np.copy(pattern), num_iterations=400)
    isophone_score(original_image, recovered, pattern)


test_iso_score()
# test_navier_stokes()
# test_laplace()
#test_both()
# plot_from_greyscale(original_image)

# # pattern = generate_random_pattern(np.shape(cat)[0], np.shape(cat)[1], 0.5)
# pattern = generate_random_rectangle_pattern(
#     np.shape(original_image)[0], np.shape(original_image)[1], 15, 15, 0.1
# )


# (distorted, pattern) = delete_with_pattern(original_image, pattern)
# plot_from_greyscale(pattern)
# recovered = nv.solve_Navier_Stokes(distorted, pattern, max_iter=10)
# # plot_three_greyscales(pattern, original_image, recovered, pause=False)
# plot_from_greyscale(recovered)


# # upscale = upscaleimage_x(original_image)
# # plot_three_greyscales(
# #     original_image,
# #     upscale[0],
# #     upscale[1],
# #     name="Figures/three_greyscales_upscale.png",
# #     pause=False,
# # )


# # distorted = delete_with_pattern(original_image, pattern)
# # recovered = s.solve_lagrange(upscale[0], upscale[1])
# # upscale = upscaleimage_y(np.copy(recovered))
# # recovered = s.solve_lagrange(upscale[0], upscale[1])
# # recovered=s.solve_euler_diffusion(distorted,pattern)
# # print(f"discrepancy score:", discrepancy_score(original_image, recovered, pattern))
# # plot_three_greyscales(original_image, upscale[0], recovered, pause=False)
