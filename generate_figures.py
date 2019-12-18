import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from diff_eq import SecondOrderDiffEq


def load_image(filename):
    img = np.asarray(Image.open(filename).convert('L'))
    return torch.tensor(img).double()


def show_image(tensor, title=None):
    arr = np.array(tensor.detach())
    plt.title(title)
    plt.imshow(arr)
    plt.show()


def make_figure_1(h):
    torch.manual_seed(205)

    def diff_eq1(weights, *, constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy):
        alpha, beta, gamma = weights
        return alpha*f_x + beta*f_y - gamma*f

    def true_fn(x, y):
        x *= h
        y *= h
        return np.exp(x) + np.exp(y)

    # generating a small image
    orig_img = torch.zeros((20, 10))
    for y in range(20):
        for x in range(10):
            orig_img[y][x] = true_fn(x, y)

    # fitting the differential equation to the smaller image
    d = SecondOrderDiffEq(diff_eq1, num_weights=3, h=h)
    d.fit(orig_img)

    print('diff_eq weights:', d.weights)

    # small image
    solved_orig_img = d.solve(orig_img)

    # extrapolated larger image
    larger_img = torch.zeros((100, 100))
    for y in range(100):
        for x in range(100):
            larger_img[y][x] = true_fn(x, y)

    solved_larger_img = d.solve(larger_img)

    # plotting
    fig, ax = plt.subplots(2, 2)

    ax[0][0].imshow(orig_img, extent=[0, 0.1, 0.2, 0])
    ax[0][0].set_title('Original Image')

    ax[0][1].imshow(solved_orig_img, extent=[0, 0.1, 0.2, 0])
    ax[0][1].set_title('DiffEq Solution')

    ax[1][0].imshow(larger_img, extent=[0, 1.0, 1.0, 0])
    ax[1][0].set_title('Larger Image')

    ax[1][1].imshow(solved_larger_img, extent=[0, 1.0, 1.0, 0])
    ax[1][1].set_title('DiffEq Solution Larger')

    plt.show()


def make_figure_2(h):
    torch.manual_seed(205)

    def diff_eq1(weights, *, constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy):
        result = 0
        terms = [constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy]
        for i in range(len(terms)):
            result += weights[i] * terms[i]
        return result

    def true_fn(x, y):
        x *= h
        y *= h
        return np.exp(x) + np.exp(y)

    # generating a small image
    orig_img = torch.zeros((20, 10))
    for y in range(20):
        for x in range(10):
            orig_img[y][x] = true_fn(x, y)

    # fitting the differential equation to the smaller image
    d = SecondOrderDiffEq(diff_eq1, num_weights=9, h=h)
    d.fit(orig_img)

    print('diff_eq weights:', d.weights)

    # small image
    solved_orig_img = d.solve(orig_img)

    # extrapolated larger image
    larger_img = torch.zeros((100, 100))
    for y in range(100):
        for x in range(100):
            larger_img[y][x] = true_fn(x, y)

    solved_larger_img = d.solve(larger_img)

    # plotting
    fig, ax = plt.subplots(2, 2)

    ax[0][0].imshow(orig_img, extent=[0, 0.1, 0.2, 0])
    ax[0][0].set_title('Original Image')

    ax[0][1].imshow(solved_orig_img, extent=[0, 0.1, 0.2, 0])
    ax[0][1].set_title('DiffEq Solution')

    ax[1][0].imshow(larger_img, extent=[0, 1.0, 1.0, 0])
    ax[1][0].set_title('Larger Image')

    ax[1][1].imshow(solved_larger_img, extent=[0, 1.0, 1.0, 0])
    ax[1][1].set_title('DiffEq Solution Larger')

    plt.show()


def make_figure_3(h):
    torch.manual_seed(205)

    def diff_eq1(weights, *, constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy):
        c1, c2, c3 = weights
        return c1*f_x - c2*torch.cos(c3 * f)

    def true_fn(x, y):
        x *= h
        y *= h

        num = 1 + np.exp(2 + 2*x)
        den = np.sqrt(2 + 2*np.exp(4 + 4*x))
        return np.arccos(num / den)

    # generating a small image
    orig_img = torch.zeros((3, 20))
    for y in range(1, 2):
        for x in range(20):
            orig_img[y][x] = true_fn(x, y)

    # fitting the differential equation to the smaller image
    d = SecondOrderDiffEq(diff_eq1, num_weights=3, h=h)
    d.fit(orig_img)
    print('diff_eq weights:', d.weights)

    # small image
    solved_orig_img = d.solve(orig_img)
    print(solved_orig_img)

    # extrapolated larger image
    larger_img = torch.zeros((3, 100))
    for y in range(1, 2):
        for x in range(100):
            larger_img[y][x] = true_fn(x, y)

    solved_larger_img = d.solve(larger_img, num_epochs=10000)

    # plotting
    fig, ax = plt.subplots(2, 2)

    ax[0][0].plot(np.arange(20)*h, orig_img[1])
    ax[0][0].set_title('Original Function')

    ax[0][1].plot(np.arange(20)*h, solved_orig_img[1])
    ax[0][1].set_title('DiffEq Solution')

    ax[1][0].plot(np.arange(100)*h, larger_img[1])
    ax[1][0].set_title('Larger Function')

    ax[1][1].plot(np.arange(100)*h, solved_larger_img[1])
    ax[1][1].set_title('DiffEq Solution Larger')

    plt.show()


def make_figure_4(h):
    torch.manual_seed(205)

    def diff_eq1(weights, *, constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy):
        c1, c3 = weights
        return c1*f_x - torch.cos(c3 * f)

    def true_fn(x, y):
        x *= h
        y *= h

        num = 1 + np.exp(2 + 2*x)
        den = np.sqrt(2 + 2*np.exp(4 + 4*x))
        return np.arccos(num / den)

    # generating a small image
    orig_img = torch.zeros((3, 20))
    for y in range(1, 2):
        for x in range(20):
            orig_img[y][x] = true_fn(x, y)

    # fitting the differential equation to the smaller image
    d = SecondOrderDiffEq(diff_eq1, num_weights=2, h=h)
    d.fit(orig_img)
    print('diff_eq weights:', d.weights)

    # small image
    solved_orig_img = d.solve(orig_img)
    print(solved_orig_img)

    # extrapolated larger image
    larger_img = torch.zeros((3, 100))
    for y in range(1, 2):
        for x in range(100):
            larger_img[y][x] = true_fn(x, y)

    solved_larger_img = d.solve(larger_img, num_epochs=10000)

    # plotting
    fig, ax = plt.subplots(2, 2)

    ax[0][0].plot(np.arange(20)*h, orig_img[1])
    ax[0][0].set_title('Original Function')

    ax[0][1].plot(np.arange(20)*h, solved_orig_img[1])
    ax[0][1].set_title('DiffEq Solution')

    ax[1][0].plot(np.arange(100)*h, larger_img[1])
    ax[1][0].set_title('Larger Function')

    ax[1][1].plot(np.arange(100)*h, solved_larger_img[1])
    ax[1][1].set_title('DiffEq Solution Larger')

    plt.show()


def make_figure_5(h):
    torch.manual_seed(205)

    def diff_eq1(weights, *, constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy):
        result = 0
        w_idx = 0
        terms = [constant, x, y, f, f_x, f_y, f_xx, f_xy, f_yy]
        for i in range(len(terms)):
            result += torch.sqrt(1 + (terms[i] * weights[w_idx]) ** 2)
            w_idx += 1

        return result

    orig_img = load_image('images/gray08.jpg')

    # fitting the differential equation to the smaller image
    d = SecondOrderDiffEq(diff_eq1, num_weights=9, h=h)
    d.fit(orig_img, stagnation_period=300, num_epochs=10000)

    print('diff_eq weights:', d.weights)

    # small image
    solved_orig_img = d.solve(orig_img, stagnation_period=300, num_epochs=10000)

    # plotting
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(orig_img, extent=[0, 1.0, 1.0, 0])
    ax[0].set_title('Original Image')

    ax[1].imshow(solved_orig_img, extent=[0, 1.0, 1.0, 0])
    ax[1].set_title('DiffEq Solution')

    plt.show()


def main():
    print('Generating figure 1\n*******************')
    make_figure_1(h=0.01)

    print('Generating figure 2\n*******************')
    make_figure_2(h=0.01)

    print('Generating figure 3\n*******************')
    make_figure_3(h=0.01)

    print('Generating figure 4\n*******************')
    make_figure_4(h=0.01)

    print('Generating figure 5\n*******************')
    make_figure_5(h=0.01)


if __name__ == '__main__':
    main()
