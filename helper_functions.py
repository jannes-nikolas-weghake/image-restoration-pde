import numpy as np
from random import *
from itertools import product
import matplotlib.pyplot as plt
import solvers as s
import diffusion as d

import navier_stokes as nv


def generate_random_rectangle_pattern(size_x, size_y, rect_x, rect_y, coverage=0.1):
    out = np.zeros((size_x, size_y))
    rect_count = int(size_x * size_y * coverage / (rect_x * rect_y))

    for i in range(rect_count):
        (x, y) = (int(random() * (size_x - rect_x)), int(random() * (size_y - rect_y)))

        for x1, y1 in product(range(rect_x), range(rect_y)):
            out[x + x1][y + y1] = 1

    return out


def discrepancy_score(restored, original, pattern):
    if not np.shape(original) == np.shape(pattern):
        raise Exception("shape of original and pattern need to match")

    if not np.shape(restored) == np.shape(pattern):
        raise Exception("shape of restored and pattern need to match")

    (size_x, size_y) = np.shape(original)
    n = np.average(pattern) * size_x * size_y

    I_mean = 0
    counter = 0
    for x, y in product(range(size_x), range(size_y)):
        if pattern[x][y] == True:
            I_mean += original[x][y]

    I_mean = I_mean / n

    sigma_square = 0

    for x, y in product(range(size_x), range(size_y)):
        if pattern[x][y] == True:
            sigma_square += (original[x][y] - I_mean) ** 2

    sigma_square = sigma_square / (n - 1)

    chi_square = 0
    for x, y in product(range(size_x), range(size_y)):
        if pattern[x][y] == True:
            chi_square += (original[x][y] - restored[x][y]) ** 2

    chi_square = chi_square / (n * sigma_square)

    return chi_square


def generate_grid(size_x, size_y, thickness=1):
    out = np.zeros((size_x, size_y))
    if thickness == 1:
        for i, j in product(range(size_x), range(size_y)):
            if i % 15 == 5 or j % 15 == 5:
                out[i][j] = True
            else:
                out[i][j] = False
    else:
        for i, j in product(range(size_x), range(size_y)):
            if i % 15 == 5 or j % 15 == 5:
                for k in range(-int(thickness / 2), int(thickness / 2)):
                    if i + k < size_x and j + k < size_y:
                        out[i + k][j + k] = True
            else:
                out[i][j] = False
    return out



def generate_random_pattern(size_x, size_y, coverage=0.1):
    out = np.zeros((size_x, size_y))
    for i, j in product(range(size_x), range(size_y)):
        if random() < coverage:
            out[i][j] = True
    return out


def delete_with_pattern(original, pattern):
    (size_x, size_y) = np.shape(original)
    out = np.copy(original)
    if not np.shape(original) == np.shape(pattern):
        raise Exception("shape of original and pattern need to match")

    for i, j in product(range(np.shape(original)[0]), range(np.shape(original)[1])):
        if pattern[i][j] == True:
            if not (
                i != 0
                and j != 0
                and i != 1
                and j != 1
                and i != size_x - 1
                and j != size_y - 1
                and i != size_x - 2
                and j != size_y - 2
            ):
                pattern[i][j] = False
            else:
                out[i][j] = np.random.random()

    return (out, pattern)


def plot_three_greyscales(
    original, pattern, distorted, pause=True, name="Figures/three_greyscales.png"
):
    figure, axis = plt.subplots(1, 3)
    axis[0].imshow(original, cmap=plt.get_cmap("gray"))
    axis[1].imshow(pattern, cmap=plt.get_cmap("gray"))
    axis[2].imshow(distorted, cmap=plt.get_cmap("gray"))
    axis[0].axis("off")
    axis[1].axis("off")
    axis[2].axis("off")

    plt.savefig(name, dpi=400, bbox_inches="tight")
    if pause:
        plt.show(block=False)
        plt.pause(17)
        plt.close()
    else:
        plt.show()


def plot_two_greyscales(
    original, distorted, pause=True, filename="Figures/two_greyscales.png"
):
    figure, axis = plt.subplots(1, 2)
    axis[0].imshow(original, cmap=plt.get_cmap("gray"))
    axis[1].imshow(distorted, cmap=plt.get_cmap("gray"))
    plt.show(block=False)
    plt.savefig(filename, dpi=400, bbox_inches="tight")
    if pause:
        plt.show(block=False)
        plt.pause(17)
        plt.close()
    else:
        plt.show()


def plot_from_greyscale(
    matrix, filename="Figures/greyscale.png", pause_time=0, title="",show=False
):
    if len(matrix.shape) == 3:
        plt.imshow(matrix)
    else:
        plt.imshow(matrix, cmap=plt.get_cmap("gray"))
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.title(title)
    plt.savefig(filename, dpi=400, bbox_inches="tight")
    if show:
        if pause_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(pause_time)
            plt.close()


def plot_from_color(matrix, filename="Figures/color.png", title="", show=False):
    plt.figure()
    plt.imshow(matrix)
    plt.title(title)
    plt.savefig(filename, dpi=400, bbox_inches="tight")
    if show:
        plt.show()
    plt.close('all')


# main idea first use checkerboard pattern in x direction after reconstruction in y direction
# test if better result compared to checker board in both direction
def upscaleimage_x(original):
    distorted = np.zeros((np.shape(original)[0] * 2, np.shape(original)[1]))
    pattern = np.zeros((np.shape(original)[0] * 2, np.shape(original)[1]))

    (size_x, size_y) = np.shape(distorted)
    for i, j in product(range(size_x), range(size_y)):
        if i % 2 == 0 and j % 2 == 0:
            distorted[i][j] = original[int(i / 2)][j]
        elif i % 2 == 1 and j % 2 == 1:
            distorted[i][j] = original[int(i / 2)][j]
        else:
            pattern[i][j] = 1

    return (distorted, pattern)


def upscaleimage_y(original):
    distorted = np.zeros((np.shape(original)[0], np.shape(original)[1] * 2))
    pattern = np.zeros((np.shape(original)[0], np.shape(original)[1] * 2))

    (size_x, size_y) = np.shape(distorted)
    for i, j in product(range(size_x), range(size_y)):
        if i % 2 == 0 and j % 2 == 0:
            distorted[i][j] = original[i][int(j / 2)]
        elif i % 2 == 1 and j % 2 == 1:
            distorted[i][j] = original[i][int(j / 2)]
        else:
            pattern[i][j] = 1

    return (distorted, pattern)


def RGB_to_YIQ(matrix):
    (size_x, size_y, _) = np.shape(matrix)
    conversion_matrix = np.array(
        [[0.299, 0.587, 0.114], [0.5959, -0.2746, -0.3213], [0.2115, -0.5227, 0.3112]]
    )

    for i, j in product(range(size_x), range(size_y)):
        matrix[i][j] = np.matmul(conversion_matrix, matrix[i][j])

    return matrix


def to_movie(foldername, video_name):
    import cv2
    import os

    image_folder = foldername
    video_name = video_name

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


def YIQ_to_RGB(matrix):
    (size_x, size_y, _) = np.shape(matrix)
    conversion_matrix = np.array(
        [[1, 0.956, 0.619], [1, -0.272, -0.647], [1, -1.106, 1.703]]
    )

    for i, j in product(range(size_x), range(size_y)):
        matrix[i][j] = np.matmul(conversion_matrix, matrix[i][j])
    return matrix


def isophone_score(original, reconstructed, pattern, title,plot=False):
    (m, n) = np.shape(original)
    num_distorted_pixel = 0

    I_temp = np.zeros((m + 2, n + 2))

    I_temp[1 : 1 + m, 1 : 1 + n] = original
    I_x = I_temp[2 : 2 + m, 1 : 1 + n] - original
    I_y = I_temp[1 : 1 + m, 2 : 2 + n] - original

    iso_org = np.zeros((m, n, 2))
    iso_org[:, :, 0] = -I_y
    iso_org[:, :, 1] = I_x

    I_temp = np.zeros((m + 2, n + 2))

    I_temp[1 : 1 + m, 1 : 1 + n] = reconstructed
    I_x = I_temp[2 : 2 + m, 1 : 1 + n] - reconstructed
    I_y = I_temp[1 : 1 + m, 2 : 2 + n] - reconstructed

    iso_rec = np.zeros((m, n, 2))
    iso_rec[:, :, 0] = -I_y
    iso_rec[:, :, 1] = I_x
    score = 0

    for i, j in product(range(m), range(n)):
        if pattern[i][j] == True:
            temp = np.sqrt(iso_org[i, j, 0] ** 2 + iso_org[i, j, 1] ** 2) * np.sqrt(
                iso_rec[i, j, 0] ** 2 + iso_rec[i, j, 1] ** 2
            )
            if temp != 0:
                num_distorted_pixel += 1
                score += (
                    iso_org[i, j, 0] * iso_rec[i, j, 0]
                    + iso_org[i, j, 1] * iso_rec[i, j, 1]
                ) / temp

    score = score / num_distorted_pixel

    if plot:
        plot_vectorfield(iso_rec[::-1, :, 0], iso_rec[:-1:, :, 1],title)

    return score


def plot_vectorfield(direction_x, direction_y,title):
    (m, n) = np.shape(direction_x)
    plt.figure()
    original_x, original_y = np.linspace(0, m, m), np.linspace(0, n, n)
    for i, j in product(range(m), range(n)):
        if i == m - 1 or j == n - 1:
            direction_x[i][j] = 0
            direction_y[i][j] = 0

    plt.quiver(
        original_x, original_y, direction_x[:, :], direction_y[:, :], color="g"
    )
    plt.axis("off")
    plt.savefig("vectorfields/"+title+".png", dpi=400, bbox_inches="tight")
    

def true_color_reconstruction(solver, original_color_image, pattern_kind, pattern=0):
    if pattern_kind == "grid" and np.mean(pattern) == 0:
        pattern = generate_grid(
            np.shape(original_color_image)[0],
            np.shape(original_color_image)[1],
            thickness=8,
        )

    elif pattern_kind == "noise" and np.mean(pattern) == 0:
        pattern = generate_random_pattern(
            np.shape(original_color_image)[0], np.shape(original_color_image)[1], 0.5
        )

    elif pattern_kind == "rectangles" and np.mean(pattern) == 0:
        pattern = generate_random_rectangle_pattern(
            np.shape(original_color_image)[0],
            np.shape(original_color_image)[1],
            8,
            8,
            0.2,
        )
    elif np.mean(pattern) != 0:
        pattern_kind = "special"

    image = np.copy(original_color_image)

    (image[:, :, 0], pattern) = delete_with_pattern(image[:, :, 0], pattern)
    (image[:, :, 1], pattern) = delete_with_pattern(image[:, :, 1], pattern)
    (image[:, :, 2], pattern) = delete_with_pattern(image[:, :, 2], pattern)
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]

    temp1 = (
        discrepancy_score(image[:, :, 0], original_color_image[:, :, 0], pattern)
        + discrepancy_score(image[:, :, 1], original_color_image[:, :, 1], pattern)
        + discrepancy_score(image[:, :, 2], original_color_image[:, :, 2], pattern)
    ) / 3

    temp2 = (
        isophone_score(image[:, :, 0], original_color_image[:, :, 0], pattern,"")
        + isophone_score(image[:, :, 1], original_color_image[:, :, 1], pattern,"")
        + isophone_score(image[:, :, 2], original_color_image[:, :, 2], pattern,"")
    ) / 3
    temp1 = int(temp1 * 10000) / 10000
    temp2=int(temp2*100000)/100000
   
    title="discrepancy score: " + str(temp1)+"\n"+"average isophote: "+str(temp2)
    plot_from_color(
        image, filename="presentation/" + pattern_kind + "/" + "distorted.png",title=title
    )

    if solver == "diffusionV2":
        image[:, :, 0] = d.TV_inpainting(R, pattern, delta_t=0.3, T=400)
        image[:, :, 1] = d.TV_inpainting(G, pattern, delta_t=0.3, T=400)
        image[:, :, 2] = d.TV_inpainting(B, pattern, delta_t=0.3, T=400)

    elif solver == "lagrange":
        image[:, :, 0] = s.solve_lagrange(R, pattern, 400)
        image[:, :, 1] = s.solve_lagrange(G, pattern, 400)
        image[:, :, 2] = s.solve_lagrange(B, pattern, 400)

    elif solver == "diffusion":
        image[:, :, 0] = s.solve_euler_diffusion(R, pattern)
        image[:, :, 1] = s.solve_euler_diffusion(G, pattern)
        image[:, :, 2] = s.solve_euler_diffusion(B, pattern)

    elif solver == "navier-stokes":
        image[:, :, 0] = nv.solve_Navier_Stokes(R, pattern, max_iter=5000)
        image[:, :, 1] = nv.solve_Navier_Stokes(G, pattern, max_iter=5000)
        image[:, :, 2] = nv.solve_Navier_Stokes(B, pattern, max_iter=5000)

    temp1 = (
        discrepancy_score(image[:, :, 0], original_color_image[:, :, 0], pattern)
        + discrepancy_score(image[:, :, 1], original_color_image[:, :, 1], pattern)
        + discrepancy_score(image[:, :, 2], original_color_image[:, :, 2], pattern)
    ) / 3

    temp2 = (
        isophone_score(image[:, :, 0], original_color_image[:, :, 0], pattern,"")
        + isophone_score(image[:, :, 1], original_color_image[:, :, 1], pattern,"")
        + isophone_score(image[:, :, 2], original_color_image[:, :, 2], pattern,"")
    ) / 3
    temp1 = int(temp1 * 10000) / 10000
    temp2=int(temp2*100000)/100000
    print(
        f"average discrepancy score for",
        solver,
        "and",
        pattern_kind,
        "",
        temp1,
        
    )
    title="discrepancy score: " + str(temp1)+"\n"+"average isophote: "+str(temp2)
    filename = "presentation/" + pattern_kind + "/" + solver + ".png"
    plot_from_color(image, filename=filename, title=str(title))

    return pattern


# discrepancy score
