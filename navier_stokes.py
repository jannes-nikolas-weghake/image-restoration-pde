import numpy as np
from itertools import product
import matplotlib.pyplot as plt
#from helper_functions import plot_from_greyscale
from solvers import solve_lagrange

def plot_from_greyscale(
    matrix, filename="Figures/greyscale.png", pause_time=0, title=""
):
    if len(matrix.shape) == 3:
        plt.imshow(matrix)
    else:
        plt.imshow(matrix, cmap=plt.get_cmap("gray"))
    plt.xlabel("Pixel")
    plt.ylabel("Pixel")
    plt.title(title)
    plt.savefig(filename, dpi=400, bbox_inches="tight")
    if pause_time == 0:
        plt.show()
    elif pause_time > 0:
        plt.show(block=False)
        plt.pause(pause_time)
        plt.close()

def solve_Navier_Stokes(distorted, pattern, delta_t=0.01, max_iter=1000, g=1, nu=2, breakpoint=500, save_breakpoints=True, pause_time=-1, forward=False,output=False, plot_omega=True):
    (size_x, size_y) = np.shape(distorted)
    pattern=pattern.astype(bool)
    distorted2 = np.copy(distorted)
    # set pixels that are in the pattern and at the edge of the pattern to the value of the neighboring pixel, which is not part of the pattern to prevent a sharp edge
    #plot_from_greyscale(distorted2, filename='Figures/NV/distorted.png', pause_time=2)
    distorted2 = prevent_sharp_edge_and_average(distorted2, pattern)
    #plot_from_greyscale(distorted2, pause_time=2)

    # make initial guess using lagrange
    # distorted2 = solve_lagrange(distorted2, pattern)

    # choose some omega:
    #omegas = calculate_initial_omegas(distorted2, pattern)
    #omegas = solve_lagrange(omegas, pattern, num_iterations=400)
    
    #distorted2 = solve_poisson(distorted2, pattern, omegas)
    omegas = calculate_laplace_matrix(distorted2)
    omegas = prevent_sharp_edge_omegas(omegas, pattern)
    if output: print(distorted2.shape, omegas.shape)
    if output:print('Plot initial omega guess')
    if output:plot_from_greyscale(omegas, pause_time=2)

    for i in range(max_iter):
        if i >0:
            if output:print(f'(Navier-Stokes solver):                             Iteration {i}')
            # start by solving the Poisson Equation Laplace(I) = omega to get I
            distorted2 = solve_poisson(distorted2, pattern, omegas)
            omegas = calculate_laplace_matrix(distorted2)

        # calculate the velocity using the updated I
        velocities = calculate_velocities(distorted2, forward=forward)

        # calculate omega for the next time step using explicit Euler
        if not forward:
            omegas = calculate_omegas(omegas, velocities, pattern, g, nu, delta_t)
        else:
            omegas = calculate_omegas_forward(omegas, velocities, pattern, g, nu, delta_t)
        
        if i % breakpoint == 0:
            if output: print(f"(Navier-Stokes solver) Currently at iteration {i}")
            if save_breakpoints:
                filename = f"Figures/NV/nv_recovered_iteration_{i:{0}{int(np.log10(max_iter)+1)}}.png"
            else:
                filename = f"Figures/NV/nv_recovered_temp.png"
            plot_from_greyscale(distorted2, filename=filename, pause_time=pause_time)
            if plot_omega:
                if save_breakpoints:
                    filename = f"Figures/NV/nv_omega_iteration_{i:{0}{int(np.log10(max_iter)+1)}}.png"
                else:
                    filename = f"Figures/NV/nv_omega_temp.png"
                plot_from_greyscale(omegas, filename=filename, pause_time=pause_time)

    return solve_poisson(distorted2, pattern, omegas)

def calculate_omegas(omega, velocities, pattern, g, nu, delta_t):
    """Formula for g=1"""
    g=1
    (size_x, size_y) = np.shape(omega)
    #updated_omega = np.zeros((size_x, size_y))
    updated_omega = np.copy(omega)

    laplace_omega = calculate_laplace_matrix(omega)
    gradient_omega = calculate_gradient_omega(omega)

    mean_1 = 0
    mean_2 = 0
    component1_temp = velocities * gradient_omega
    component1 = - (component1_temp[0]+ component1_temp[1])
    component2 = laplace_omega
    # print(f"", component1, component2)
    temp2 = delta_t * (component1 + g * nu * component2)

    update = omega + temp2
    # apply update to pattern pixels
    updated_omega = omega + pattern * (update - omega)

    #print('Mean 1: ', np.mean(component1))
    #print('Mean 2: ', np.mean(component2))
    return updated_omega


def calculate_omegas_old(omega, velocities, pattern, g, nu, delta_t,):
    """Slightly wrong formula used"""
    (size_x, size_y) = np.shape(omega)
    #updated_omega = np.zeros((size_x, size_y))
    updated_omega = np.copy(omega)

    laplace_omega = calculate_laplace_matrix(omega)
    gradient_omega = calculate_gradient_omega(omega, pattern)

    mean_1 = 0
    mean_2 = 0
    for j, k in product(range(size_x), range(size_y)):
        if pattern[j][k] == True:
            component1 = -(
                velocities[j][k][0] * gradient_omega[j][k][0]
                + velocities[j][k][1] * gradient_omega[j][k][1]
            )
            component2 = (1 / 8) * (
                (omega[j, k] - omega[-2 + j, k])
                * np.sqrt(
                    (omega[-1 + j, -1 + k] - omega[-1 + j, 1 + k]) ** 2
                    + (omega[-2 + j, k] - omega[j, k]) ** 2
                )
                + (omega[j, k] - omega[j, -2 + k])
                * np.sqrt(
                    (omega[j, -2 + k] - omega[j, k]) ** 2
                    + (omega[-1 + j, -1 + k] - omega[1 + j, -1 + k]) ** 2
                )
                + (omega[j, 2 + k] - omega[j, k])
                * np.sqrt(
                    (omega[j, k] - omega[j, 2 + k]) ** 2
                    + (omega[-1 + j, 1 + k] - omega[1 + j, 1 + k]) ** 2
                )
                + (-omega[j, k] + omega[2 + j, k])
                * np.sqrt(
                    (omega[1 + j, -1 + k] - omega[1 + j, 1 + k]) ** 2
                    + (omega[j, k] - omega[2 + j, k]) ** 2
                )
            )
            # print(f"", component1, component2)
            mean_1 += np.abs(component1)
            mean_2 += np.abs(component2)
            temp2 = delta_t * (component1 + g * nu * (component2))

            # print(f"", omega[j][k], component2)
            update = omega[j][k] + temp2
            #if np.abs(update) < 3:
            #updated_omega[j][k] = update
            updated_omega[j][k] = update
            #else:
            #    print(f'High omega {update}')
            #    updated_omega[j][k] = 0

    mean_1 = mean_1 /(size_x*size_y)
    mean_2 = mean_2 /(size_x*size_y)
    print('Mean 1: ', mean_1)
    print('Mean 2: ', mean_2)
    return updated_omega

def calculate_omegas_forward(omega, velocities, pattern, g, nu, delta_t):
    (size_x, size_y) = np.shape(omega)
    #updated_omega = np.zeros((size_x, size_y))
    updated_omega = np.copy(omega)

    mean_1 = 0
    mean_2 = 0
    for j, k in product(range(size_x), range(size_y)):
        if pattern[j][k] == True:
            component1 = -(
                velocities[j][k][0] * (omega[j+1,k]- omega[j,k])
                + velocities[j][k][1] * (omega[j,k+1] - omega[j,k])
            )

            squareroot = np.sqrt((omega[j+1,k]-omega[j,k])**2 + (omega[j,k+1]-omega[j,k])**2)
            prefactor = (omega[j+1,k]+omega[j,k+1]-2*omega[j,k])/squareroot
            x_component = prefactor * (omega[j+2,k]+omega[j+1,k+1]+2*omega[j,k]-3*omega[j+1,k]-omega[j,k+1])*(omega[j+1,k]-omega[j,k]) + squareroot * (omega[j+2,k]+omega[j,k]-2*omega[j+1,k])
            y_component = prefactor * (omega[j,k+2]+omega[j+1,k+1]+2*omega[j,k]-3*omega[j,k+1]-omega[j+1,k])*(omega[j,k+1]-omega[j,k]) + squareroot * (omega[j,k+2]+omega[j,k]-2*omega[j,k+1])
            
            component2 = x_component + y_component
            # print(f"", component1, component2)
            mean_1 += np.abs(component1)
            mean_2 += np.abs(component2)
            #component1 = 0
            temp2 = delta_t * (component1 + g * nu * (component2))

            # print(f"", omega[j][k], component2)
            update = omega[j][k] + temp2
            #if np.abs(update) < 3:
            #updated_omega[j][k] = update
            updated_omega[j][k] = update
            #else:
            #    print(f'High omega {update}')
            #    updated_omega[j][k] = 0
    #updated_omega = omega
    mean_1 = mean_1 /(size_x*size_y)
    mean_2 = mean_2 /(size_x*size_y)
    print('Mean 1: ', mean_1)
    print('Mean 2: ', mean_2)
    return updated_omega


def calculate_initial_omegas(I, pattern):
    """Calculate initial omegas by setting them to mean(I), if they are part of the pattern,
    else they are calculated using Laplace(I)"""
    (size_x, size_y) = I.shape
    omegas = np.zeros((size_x, size_y))
    # value = np.mean(I)
    value = 0
    # for j, k in product(range(size_x), range(size_y)):
    #     if not pattern[j,k] == True:
    #         (sum, edges) = sum_neighbours(I, j, k)
    #         omegas[j, k] = sum - (4 - edges) * I[j][k]
    omegas = calculate_laplace_matrix(I)
    omegas[pattern] = value
    # mean_value = np.mean(omegas[np.invert(pattern)])
    # max_value = np.max(omegas[np.invert(pattern)])
    # min_value = np.min(omegas[np.invert(pattern)])
    # for j, k in product(range(size_x), range(size_y)):
    #     if pattern[j][k] == True:
    #             omegas[j, k] = (np.random.random()*(max_value-min_value)+min_value)*0.5
    return omegas
    


def calculate_laplace_matrix(I):
    (size_x, size_y) = np.shape(I)

    I1 = np.zeros((size_x+2, size_y+2))
    I1[1:-1, 1:-1] = I
    laplace = I1[2:,1:-1]+I1[:-2,1:-1] + I1[1:-1,2:] + I1[1:-1,:-2] - 4 * I

    #laplace = np.zeros((size_x, size_y))
    # for j, k in product(range(size_x), range(size_y)):
    #     (sum, edges) = sum_neighbours(I, j, k)
    #     laplace[j, k] = sum - (4 - edges) * I[j][k]

    return laplace


def solve_poisson(I, pattern, omegas, omega1=1.9, threshold=0.01, num_iterations = 10000, plot=False,output=False):
    """Solve Laplace(I) = omega for the positions given in pattern"""

    (size_x, size_y) = np.shape(I)
    distorted_pixel = np.average(pattern) * size_x * size_y
    
    I_updated = np.copy(I)
    #I_updated[pattern] = np.mean(I_updated[np.invert(pattern)])
    last = np.copy(I_updated)
    I1 = np.zeros((size_x+2, size_y+2))

    for iteration in range(num_iterations):
        if plot:
            plot_from_greyscale(I_updated)

        I1[1:-1, 1:-1] = I_updated
        update = 1/4 * (I1[2:,1:-1] + I1[:-2,1:-1] + I1[1:-1,2:] + I1[1:-1,:-2] - omegas)
        # set update for pattern pixels
        I_updated = I_updated + pattern * (update - I_updated)


        # for j, k in product(range(size_x), range(size_y)):
        #     if pattern[j][k] == 1:
        #         (sum, edges) = sum_neighbours(last, j, k)
        #         update_value = (sum - omegas[j][k]) / (4-edges)
        #         I_updated[j][k] = update_value
        
        if iteration % 5 != 0:
            continue
        else:
            temp = 0
            # for j, k in product(range(size_x), range(size_y)):
            #     if pattern[j][k] == True:
            #         temp += abs(I_updated[j][k] - last[j][k])
            temp = np.sum(I_updated - last)
            if output:print(f"(Poisson solver) Change in I: {temp / distorted_pixel}")
            if temp / distorted_pixel < threshold:
                if output:print(f"(Poisson solver) Finished after", iteration, "iterations")
                return I_updated
            last = np.copy(I_updated)
        
    
    raise RuntimeError(f"(Poisson solver) No convergence after ", iteration, "iterations")
    # print(f"(Poisson solver) No convergence after ", iteration, "iterations")
    # return I_updated


def calculate_velocities(I, forward=False):
    """calculate the perpendicular gradient of the intensity"""

    (size_x, size_y) = np.shape(I)
    I1 = np.zeros((size_x+2, size_y+2))
    I1[1:-1,1:-1] = I
    # symmetric derivatives
    grad_x = (I1[2:,1:-1] - I1[:-2,1:-1])/2
    grad_y = (I1[1:-1,2:] - I1[1:-1,:-2])/2
    
    velocities = np.stack((-grad_y, grad_x))


    # velocities = np.zeros((size_x, size_y, 2))
    # last = np.zeros(2)

    # for j, k in product(range(size_x), range(size_y)):
    #     if j > 0 and k >0 and j+1 < size_x and k+1 < size_y:
    #         if not forward:
    #             # symmetric
    #             velocities[j][k][0] = I[j][k - 1] - I[j][k + 1] / 2
    #             velocities[j][k][1] = I[j + 1][k] - I[j - 1][k] / 2
    #         else:
    #             # forward
    #             velocities[j][k][0] = -(I[j][k +1] - I[j][k])
    #             velocities[j][k][1] = I[j + 1][k] - I[j][k]

    #     else:
    #         velocities[j][k] = last
    #     last = velocities[j][k]

    return velocities


def calculate_gradient_omega(omega):
    (size_x, size_y) = np.shape(omega)
    omega1 = np.zeros((size_x+2, size_y+2))
    omega1[1:-1,1:-1] = omega
    # symmetric derivatives
    grad_omega_x = (omega1[2:,1:-1] - omega1[:-2,1:-1])/2
    grad_omega_y = (omega1[1:-1,2:] - omega1[1:-1,:-2])/2

    grad_omega = np.stack((grad_omega_x, grad_omega_y))

    # grad_omega = np.zeros((size_x, size_y, 2))
    # for j, k in product(range(size_x), range(size_y)):
    #     if pattern[j][k] == True:
    #         grad_omega[j][k][0] = omega[j + 1][k] - omega[j - 1][k]
    #         grad_omega[j][k][1] = omega[j][k + 1] - omega[j][k - 1]

    return grad_omega


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

def prevent_sharp_edge_and_average(distorted, pattern):
    (size_x, size_y) = distorted.shape
    mean_value = np.mean(distorted[np.invert(pattern)])
    min_value = np.min(distorted[np.invert(pattern)])
    max_value = np.max(distorted[np.invert(pattern)])
    #print('Mean value: ', mean_value)
    for j,k in product(range(size_x), range(size_y)):
        edge = False
        if pattern[j,k] == True:
            for l in [-1,1]:
                if pattern[j+l,k] == False:
                    distorted[j,k] = distorted[j+l,k]
                    edge = True
                elif pattern[j,k+l] == False:
                    distorted[j,k] = distorted[j,k+l]
                    edge = True
            
            if not edge:
                distorted[j,k] = mean_value
                # distorted[j,k] = 0
                # distorted[j,k] = (np.random.random()*(max_value-min_value))/2 + min_value
    return distorted

def prevent_sharp_edge_omegas(omegas, pattern):
    (size_x, size_y) = omegas.shape
    for j,k in product(range(size_x), range(size_y)):
        edge = False
        if pattern[j,k] == True:
            for l in [-1,1]:
                if pattern[j+l,k] == False:
                    omegas[j+l,k] = omegas[j+2*l,k]
                    edge = True
                elif pattern[j,k+l] == False:
                    omegas[j,k+l] = omegas[j,k+2*l]
                    edge = True
    return omegas