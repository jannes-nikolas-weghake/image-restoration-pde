import numpy as np
import navier_stokes as nv
import helper_functions as hf
import matplotlib.pyplot as plt
from helper_functions import *
import solvers as s
from PIL import Image

tortoise_path = "pictures\small_tortoise.png"
tortoise = np.asarray(Image.open(tortoise_path).convert("L")) / 255
original_image = tortoise

color_tortoise_path = "color pictures\small_tortoise.png"
color_tortoise = np.asarray(Image.open(color_tortoise_path).convert("RGB")) / 255
original_color_image = color_tortoise

color_tortoise_path = "color pictures\small_tortoise_paint.png"
color_tortoise = np.asarray(Image.open(color_tortoise_path).convert("RGB")) / 255
original_color_image = color_tortoise

color_tortoise_text_path = "color pictures\small_tortoise_paint_text.png"
color_tortoise_text = (
    np.asarray(Image.open(color_tortoise_text_path).convert("RGB")) / 255
)
text_color_image = color_tortoise_text


def testnv(max_iter=100):
    # pattern = hf.generate_random_rectangle_pattern(
    #     np.shape(original_image)[0], np.shape(original_image)[1], 15, 15, 0.05
    # )
    pattern = hf.generate_grid(np.shape(original_image)[0], np.shape(original_image)[1])

    (distorted, pattern) = hf.delete_with_pattern(original_image, pattern)
    # hf.plot_two_greyscales(pattern, distorted)

    recovered = nv.solve_Navier_Stokes(
        distorted,
        pattern.astype(bool),
        max_iter=1000,
        delta_t=0.01,
        breakpoint=10000,
        g=1,
        nu=2,
        save_breakpoints=True,
        pause_time=0.1,
        forward=False,
    )
    plot_two_greyscales(distorted, recovered, pause=False)
    return


def true_color_reconstruction(YIQ=False, max_iter=1000, distorted=None, pattern=None):
    if distorted is None:
        pattern = hf.generate_grid(
            np.shape(original_image)[0], np.shape(original_image)[1]
        )
        image = original_color_image
    else:
        image = distorted
    if YIQ == True:
        image = hf.RGB_to_YIQ(image)
        Y = image[:, :, 0]
        I = image[:, :, 1]
        Q = image[:, :, 2]

        (Y, pattern) = hf.delete_with_pattern_inverted(Y, pattern)
        (I, pattern) = hf.delete_with_pattern_inverted(I, pattern)
        (Q, pattern) = hf.delete_with_pattern_inverted(Q, pattern)
        image[:, :, 0] = nv.solve_Navier_Stokes(
            Y,
            pattern.astype(bool),
            max_iter=max_iter,
            delta_t=0.01,
            breakpoint=10000,
            g=1,
            nu=2,
            save_breakpoints=True,
            pause_time=0.1,
            forward=False,
        )
        image[:, :, 1] = nv.solve_Navier_Stokes(
            I,
            pattern.astype(bool),
            max_iter=max_iter,
            delta_t=0.01,
            breakpoint=10000,
            g=1,
            nu=2,
            save_breakpoints=True,
            pause_time=0.1,
            forward=False,
        )
        image[:, :, 2] = nv.solve_Navier_Stokes(
            Q,
            pattern.astype(bool),
            max_iter=max_iter,
            delta_t=0.01,
            breakpoint=10000,
            g=1,
            nu=2,
            save_breakpoints=True,
            pause_time=0.1,
            forward=False,
        )
        image = hf.YIQ_to_RGB(image)

    else:
        plot_from_greyscale(distorted, title=f"Damaged image")
        R = image[:, :, 0]
        G = image[:, :, 1]
        B = image[:, :, 2]
        if distorted is None:
            (R, pattern) = hf.delete_with_pattern(R, pattern)
            (G, pattern) = hf.delete_with_pattern(G, pattern)
            (B, pattern) = hf.delete_with_pattern(B, pattern)
        image[:, :, 0] = nv.solve_Navier_Stokes(
            R,
            pattern.astype(bool),
            max_iter=max_iter,
            delta_t=0.01,
            breakpoint=10000,
            g=1,
            nu=2,
            save_breakpoints=True,
            pause_time=0.1,
            forward=False,
        )
        plot_from_greyscale(image[:, :, 0])
        image[:, :, 1] = nv.solve_Navier_Stokes(
            G,
            pattern.astype(bool),
            max_iter=max_iter,
            delta_t=0.01,
            breakpoint=10000,
            g=1,
            nu=2,
            save_breakpoints=True,
            pause_time=0.1,
            forward=False,
        )
        plot_from_greyscale(image[:, :, 1])
        image[:, :, 2] = nv.solve_Navier_Stokes(
            B,
            pattern.astype(bool),
            max_iter=max_iter,
            delta_t=0.01,
            breakpoint=10000,
            g=1,
            nu=2,
            save_breakpoints=True,
            pause_time=0.1,
            forward=False,
        )
        plot_from_greyscale(image[:, :, 2])

    return image


pattern = np.zeros(original_color_image.shape[:-1]).astype(bool)
difference = np.abs(original_color_image - text_color_image)
total_difference = difference[:, :, 0] + difference[:, :, 1] + difference[:, :, 2]
pattern[total_difference > 0] = True
print(pattern)

hf.plot_from_greyscale(original_color_image, title="Original image")
hf.plot_from_greyscale(pattern, title="Mask image")
distorted = text_color_image
restored_image = true_color_reconstruction(distorted=distorted, pattern=pattern)
plot_from_greyscale(restored_image)
print("Min", np.min(restored_image))
print("Max ", np.max(restored_image))
discrepancy = hf.discrepancy_score(original_image, restored_image, pattern)
print("Discrepancy score: ", discrepancy)
plot_from_greyscale(
    restored_image, title=f"Restored image (Discrepancy: {discrepancy:.2f})"
)

# testnv()

# gifs
# score
