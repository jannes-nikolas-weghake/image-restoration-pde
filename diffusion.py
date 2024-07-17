import numpy as np
import cv2
import matplotlib.pyplot as plt


def TV_inpainting(I, pattern, T=100, delta_t=0.33):

    #{inverted pattern}
    (m, n) = I.shape
    t = 0

    ones = np.ones(pattern.shape)
    while t <= T:
        I_temp = np.zeros((m + 2, n + 2))

        I_temp[1 : 1 + m, 1 : 1 + n] = I
        I_x = I_temp[2 : 2 + m, 1 : 1 + n] - I
        I_y = I_temp[1 : 1 + m, 2 : 2 + n] - I
        I_xx = I_temp[2 : 2 + m, 1 : 1 + n] - 2 * I + I_temp[0:m, 1 : 1 + n]
        I_yy = I_temp[1 : 1 + m, 2 : 2 + n] - 2 * I + I_temp[1 : 1 + m, 0:n]
        I_xy = (
            I_temp[2 : 2 + m, 2 : 2 + n]
            + I_temp[0:m, 0:n]
            - I_temp[0:m, 2 : 2 + n]
            - I_temp[2 : 2 + m, 0:n]
        ) / 4.0
        I += (
            delta_t
            * (pattern)
            * (I_xx * I_y**2 - 2 * I_x * I_y * I_xy + I_yy * I_x**2)
            / (0.01 + (I_x**2 + I_y**2) ** 1.5)
        )

        # if int(t / delta_t)%2==0:
        #     plt.imshow(I, cmap="gray")
        #     plt.axis("off")
        #     plt.title("t/dt = " + str(int(t / delta_t)))
        #     plt.savefig("old/test/" + (str(int(t / delta_t)+100)) + ".png")
        #     plt.close()

        t += delta_t
        # {lieber ein rauschen benutzen}
    plt.imshow(I, cmap="gray")
    plt.title("t/dt = " + str(int(t / delta_t)))
    plt.axis("off")
    plt.savefig("old/test/tv_" + "final" + ".png")
    plt.close()

    return I
