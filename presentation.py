import diffusion as d3
import numpy as np
import helper_functions as hf
from itertools import product
from PIL import Image
import solvers as s
import navier_stokes as nv


color_tortoise_path = "color pictures\small_tortoise.png"
color_tortoise = np.asarray(Image.open(color_tortoise_path).convert("RGB")) / 255
original_color_image = color_tortoise

fish = "pictures\scale.png"
fish = np.asarray(Image.open(fish).convert("L")) / 255

tortoise_path = "pictures\small_tortoise.png"
tortoise = np.asarray(Image.open(tortoise_path).convert("L")) / 255

original_image=fish

rectangle_pattern= hf.generate_random_rectangle_pattern(
            np.shape(original_color_image)[0], np.shape(original_color_image)[1],8,8, 0.4
        )

def test_solvers(pattern_kind="noise"):
    if pattern_kind=="rectangle":
        hf.true_color_reconstruction("lagrange", original_color_image, pattern_kind="",pattern=rectangle_pattern)
        hf.true_color_reconstruction("diffusion", original_color_image, pattern_kind="",pattern=rectangle_pattern)
        hf.true_color_reconstruction("diffusionV2", original_color_image, pattern_kind="",pattern=rectangle_pattern)
        hf.true_color_reconstruction("navier-stokes", original_color_image, pattern_kind="",pattern=rectangle_pattern)

    else:
        hf.true_color_reconstruction("lagrange", original_color_image, pattern_kind=pattern_kind)
        hf.true_color_reconstruction("diffusion", original_color_image, pattern_kind=pattern_kind)
        hf.true_color_reconstruction("diffusionV2", original_color_image, pattern_kind=pattern_kind)
        hf.true_color_reconstruction("navier-stokes", original_color_image, pattern_kind=pattern_kind)

def vectorfields():
    pattern=hf.generate_random_rectangle_pattern(
            np.shape(original_image)[0], np.shape(original_image)[1],4,4, 0.4
        )
    
    (distorted,pattern)=hf.delete_with_pattern(original_image,pattern)
    recovered=s.solve_euler_diffusion(distorted,pattern)
    hf.plot_from_greyscale(original_image,"vectorfields/original_pic")
    hf.plot_from_greyscale(recovered,"vectorfields/recovered_pic")
    hf.plot_from_greyscale(distorted,"vectorfields/distorted_pic")
    hf.isophone_score(original_image,original_image,pattern,"original image",plot=True)
    hf.isophone_score(original_image,distorted,pattern,"distorted image",plot=True)
    hf.isophone_score(original_image,recovered,pattern,"recovered image",plot=True)

def movie_generation():
    (size_x,size_y)=np.shape(tortoise)
    pattern=hf.generate_random_pattern(size_x,size_y,0.2)
    (distorted,pattern)=hf.delete_with_pattern(tortoise,pattern)
    out=s.solve_euler_diffusion(distorted,pattern,safeplots=True)


#movie_generation()
#hf.to_movie("movie_DiffusionV1","movie_diffusionV1.avi")
#hf.to_movie("movie_lagrange","movie_lagrange.avi")
vectorfields()
#test_solvers()




