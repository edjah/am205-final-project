"""
Feel free to modify the main() function in this demo file to play around
with fitting your own differential equations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diff_eq import SecondOrderDiffEq


def load_image(filename):
    img = np.asarray(Image.open(filename).convert('L'))
    return torch.tensor(img)


def show_image(tensor, title=None):
    arr = np.array(tensor.detach())
    plt.title(title)
    plt.imshow(arr)
    plt.show()


def my_diffeq(weights, *, constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy):
    """
    Define this differentiation equation using the weights and the provided
    arguments to be any arbitrary function that uses differentiable pytorch
    operations.

    The weights will be learned via the SecondOrderDiffEq.fit() method and
    will be passed to this function as a torch.tensor().

    You must pass the exact number of weights that you're using to the
    SecondOrderDiffEq() constructor.

    Example implementation #1 | simple linear PDE | requires 3 weights
    ---------------------------------------------------------------
    alpha, beta, gamma = weights
    return alpha*f_x + beta*f_y - gamma*f


    Example implementation #2: all quadratic terms | requires 54 weights
    --------------------------------------------------------------------
    tot = 0
    w_idx = 0
    terms = [constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy]
    for i in range(len(terms)):
        tot += weights[w_idx] * terms[i]
        w_idx += 1

    for i in range(len(terms)):
        for j in range(i, len(terms)):
            tot += weights[w_idx] * terms[i] * terms[j]
            w_idx += 1

    return tot
    """

    # TODO: replace me
    alpha, beta, gamma = weights
    return alpha*f_x + beta*f_y - gamma*f


def main():
    # get deterministic results
    torch.manual_seed(205)

    # set up the distance between points
    h = 0.01

    # specify a true differential equation
    def true_fn(x, y):
        x = x * h
        y = y * h
        return np.exp(x) + np.exp(y)

    # generate a sample image
    orig_image = torch.zeros((10, 10))
    for y in range(10):
        for x in range(10):
            orig_image[y][x] = true_fn(x, y)

    # show the original image
    show_image(orig_image, title='original image')

    # fit the differential equation
    diff_eq = SecondOrderDiffEq(my_diffeq, 3, h=h)
    diff_eq.fit(orig_image)
    print('diff_eq weights:', diff_eq.weights)

    # solve on the original boundary conditions
    solved_orig_image = diff_eq.solve(orig_image)
    show_image(solved_orig_image, title='reconstructed image from boundaries')

    # generate a a larger image with the same diff_eq
    large_image = torch.zeros((100, 100))
    for y in range(100):
        for x in range(100):
            large_image[y][x] = true_fn(x, y)

    # showing the larger image
    show_image(large_image, title='larger image')

    # solve on the boundary conditions of the larger image
    solved_large_image = diff_eq.solve(large_image)
    show_image(solved_large_image, title='reconstructed larger image from boundaries')


if __name__ == '__main__':
    main()
