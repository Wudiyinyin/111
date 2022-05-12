# Copyright (c) 2021 Li Auto Company. All rights reserved.

import math
from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.spatial.transform import Rotation as Rot

from gmm_fitting import GaussianMixture

# matplotlib.use('Agg')


def test_points_gmm_fitting():
    n, d = 300, 2

    # generate some data points ..
    data = torch.Tensor(n, d).normal_()
    # .. and shift them around to non-standard Gaussians
    data[:n // 2] -= 1
    data[:n // 2] *= sqrt(3)
    data[n // 2:] += 1
    data[n // 2:] *= sqrt(2)

    # Next, the Gaussian mixture is instantiated and ..
    n_components = 2
    # model = GaussianMixture(n_components, d, covariance_type="diag")
    model = GaussianMixture(n_components, d, covariance_type="full")
    model.fit(data)
    # .. used to predict the data points as they where shifted
    y = model.predict(data)

    plot_points(data, y)
    plt.show()

    # x, label = model.sample(4)
    # print(x, label)


def test_gaussian1d_gmm_fitting():
    # (n, d) -> (n, 1, d)
    x_mu = torch.FloatTensor([[-5.], [1.2], [5.], [5.5]]).unsqueeze(1)
    # (n, d, d) -> (n, 1, d, d)
    x_var = torch.FloatTensor([[[0.2]], [[0.5]], [[0.5]], [[0.2]]]).unsqueeze(1)
    # (n, 1)
    x_p = torch.FloatTensor([[0.4], [0.2], [0.2], [0.2]])

    n, _, d = x_mu.shape
    n_components = 2

    model = GaussianMixture(n_components, d, covariance_type="full", init_params='multipath')
    model.fit(x_mu, x_var, x_p, dist_th=1.0)

    print("\nx_mu:", x_mu.shape)
    print("\nx_var:", x_var.shape)
    print("\nx_p:", x_p.shape)

    print("\nmodel.mu", model.mu.shape, "\n", model.mu)
    print("\nmodel.sigma", model.var.shape, "\n", model.var)
    print("\nmodel.pi", model.pi.shape, "\n", model.pi)

    # (1, k, d) -> (k, 1, d)
    estimate_mu = model.mu.transpose(0, 1)
    # (1, k, d, d) -> (k, 1, d, d)
    estimate_var = model.var.transpose(0, 1)
    # (1, k, 1) -> (k, 1)
    estimate_pi = model.pi[0]

    print("\nestimate mu", estimate_mu.shape, "\n", estimate_mu)
    print("\nestimate var", estimate_var.shape, "\n", estimate_var)
    print("\nestimate pi", estimate_pi.shape, "\n", estimate_pi)

    plot_gmm1d(x_mu.numpy(), x_var.numpy(), x_p.numpy(), 'k')
    plot_gmm1d(estimate_mu.numpy(), estimate_var.numpy(), estimate_pi.numpy(), color='g', high=0.1)

    plt.show()


def test_gaussian2d_gmm_fitting():
    # (n, d) -> (n, 1, d)
    x_mu = torch.FloatTensor([[0, -5.], [0, -3.2], [0, 3.5], [0, 5.0]]).unsqueeze(1)
    # x_mu = torch.FloatTensor([[0, -5.], [0, -2.2], [0, 3.5], [0, 5.0]]).unsqueeze(1)
    # (n, d, d) -> (n, 1, d, d)
    x_var = torch.FloatTensor([[[0.2, 0], [0, 0.2]], [[0.2, 0], [0, 0.2]], [[0.2, 0], [0, 0.2]],
                               [[0.2, 0], [0, 0.2]]]).unsqueeze(1)
    # (n, 1)
    x_p = torch.FloatTensor([[0.2], [0.2], [0.2], [0.2]])

    n, _, d = x_mu.shape
    n_components = 2

    model = GaussianMixture(n_components, d, covariance_type="full", init_params='multipath')
    model.fit(x_mu, x_var, x_p, dist_th=1.0)

    print("\nx_mu:", x_mu.shape)
    print("\nx_var:", x_var.shape)
    print("\nx_p:", x_p.shape)

    print("\nmodel.mu", model.mu.shape, "\n", model.mu)
    print("\nmodel.sigma", model.var.shape, "\n", model.var)
    print("\nmodel.pi", model.pi.shape, "\n", model.pi)

    # (1, k, d) -> (k, 1, d)
    estimate_mu = model.mu.transpose(0, 1)
    # (1, k, d, d) -> (k, 1, d, d)
    estimate_var = model.var.transpose(0, 1)
    # (1, k, 1) -> (k, 1)
    estimate_pi = model.pi[0]

    print("\nestimate mu", estimate_mu.shape, "\n", estimate_mu)
    print("\nestimate var", estimate_var.shape, "\n", estimate_var)
    print("\nestimate pi", estimate_pi.shape, "\n", estimate_pi)

    plot_gmm2d(x_mu.numpy(), x_var.numpy(), x_p.numpy(), color='k')
    plot_gmm2d(estimate_mu.numpy(), estimate_var.numpy(), estimate_pi.numpy(), color='g')

    plt.show()


def test_gaussiannd_gmm_fitting():
    # (n, d) -> (n, 1, d)
    x_mu = torch.FloatTensor([[0, -5., 1, -5, 2, -5, 3, -5], [0, -3.5, 1, -3.5, 2, -3.5, 3, -3.5],
                              [0, 3.5, 1, 3.5, 2, 3.5, 3, 3.5], [0, 5.0, 1, 5.0, 2, 5.0, 3, 5.0]]).unsqueeze(1)
    # x_mu = torch.FloatTensor([[0, -5., 1, -5, 2, -5, 3, -5], [0, -3.5, 1, -3.5, 2, -3.5, 3, -3.5],
    #                           [0, 4.5, 1, 4.5, 2, 4.5, 3, 4.5], [0, 5.0, 1, 5.0, 2, 5.0, 3, 5.0]]).unsqueeze(1)
    # x_mu = torch.FloatTensor([[0, -5., 1, -5, 2, -5, 3, -5], [0, -4.5, 1, -4.5, 2, -4.5, 3, -4.5],
    #                           [0, 4.5, 1, 4.5, 2, 4.5, 3, 4.5], [0, 5.0, 1, 5.0, 2, 5.0, 3, 5.0]]).unsqueeze(1)
    # x_mu = torch.FloatTensor([[0, -5., 1, -5, 2, -5, 3, -5], [0, -2, 1, -2, 2, -2, 3, -2], [0, 2, 1, 2, 2, 2, 3, 2],
    #                           [0, 5.0, 1, 5.0, 2, 5.0, 3, 5.0]]).unsqueeze(1)
    # (n, d, d) -> (n, 1, d, d)
    x_var = torch.FloatTensor([[[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]],
                               [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]],
                               [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]],
                               [[0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                           0.2]]]).unsqueeze(1)
    # (n, 1)
    x_p = torch.FloatTensor([[0.2], [0.2], [0.2], [0.2]])

    n, _, d = x_mu.shape
    n_components = 2

    model = GaussianMixture(n_components, d, covariance_type="full", init_params='multipath')
    model.fit(x_mu, x_var, x_p, dist_th=1.0)

    estimate_mu = model.mu.transpose(0, 1)
    estimate_var = model.var.transpose(0, 1)
    estimate_pi = model.pi.transpose(0, 1)

    print("\nx_mu:", x_mu.shape)
    print("\nx_var:", x_var.shape)
    print("\nx_p:", x_p.shape)

    print("\nmodel.mu", model.mu.shape, "\n", model.mu)
    print("\nmodel.sigma", model.var.shape, "\n", model.var)
    print("\nmodel.pi", model.pi.shape, "\n", model.pi)

    # (1, k, d) -> (k, 1, d)
    estimate_mu = model.mu.transpose(0, 1)
    # (1, k, d, d) -> (k, 1, d, d)
    estimate_var = model.var.transpose(0, 1)
    # (1, k, 1) -> (k, 1)
    estimate_pi = model.pi[0]

    print("\nestimate mu", estimate_mu.shape, "\n", estimate_mu)
    print("\nestimate var", estimate_var.shape, "\n", estimate_var)
    print("\nestimate pi", estimate_pi.shape, "\n", estimate_pi)

    for i in range(0, x_mu.shape[-1], 2):
        plot_gmm2d(x_mu[:, :, i:i + 2].numpy(), x_var[:, :, i:i + 2, i:i + 2].numpy(), x_p[:, :].numpy(), color='k')

    for m in range(0, estimate_mu.shape[0], 1):
        for i in range(0, estimate_mu.shape[-1], 2):
            # plot_gmm2d(estimate_mu[:, :, i:i+2].numpy(), estimate_var[:, :, i:i+2, i:i+2].numpy(),
            # estimate_pi[:, :].numpy(), color='g')
            plot_covariance_ellipse(estimate_mu[m, 0, i:i + 2].unsqueeze(-1).numpy(), estimate_var[m, 0, i:i + 2,
                                                                                                   i:i + 2].numpy())
    plt.show()


def plot_gmm1d(x_mu, x_var, x_p, color='r', high=0):
    '''
    x_mu: (n, 1, d)
    x_var: (n, 1, d, d)
    x_p: (n, 1)
    '''

    data = np.sort(np.linspace(-10, 10, 1000).tolist())
    plt.scatter(x_mu[:, 0, 0].tolist(), [high] * x_mu.shape[0], color=color)

    prob = np.sum([x_p[i, 0] * gaussian_1d(data, x_mu[i, 0, 0], x_var[i, 0, 0, 0]) for i in range(x_mu.shape[0])],
                  axis=0)
    plt.plot(data, prob, color=color)
    plt.ylim(-0.1, 1)

    figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()
    figManager.resize(*figManager.window.maxsize())
    plt.xlabel('x')
    plt.ylabel('y')


def plot_gmm2d(x_mu, x_var, x_p, color='r', lines=3, linewidth=0.5):
    '''
    plot_contourf
    x_mu: (n, 1, d)
    x_var: (n, 1, d, d)
    x_p: (n, 1)
    '''

    n = 256
    x_min = -10
    x_max = 10
    y_min = -10
    y_max = 10
    x = np.linspace(x_min, x_max, n)
    y = np.linspace(y_min, y_max, n)
    # X: (n, n) Y: (n, n)
    X, Y = np.meshgrid(x, y)
    # (n*n, 2)
    xy = np.c_[X.reshape(-1), Y.reshape(-1)]

    # gaussian_nd: (n*n, 2) -> (n*n, 1) -> (n*n)
    # stack( [[n*n], [n*n], ...] ) -> (n*n, m)
    prob_raw = np.stack([x_p[i, 0] * gaussian_nd(xy, x_mu[i, 0, :], x_var[i, 0, :, :]) for i in range(x_mu.shape[0])],
                        axis=-1)
    # (n*n, m) -> (n*n)
    prob = np.sum(prob_raw, axis=-1)
    prob = prob.reshape(X.shape)

    C = plt.contour(X, Y, prob, lines, colors=color, linewidth=linewidth)
    plt.clabel(C, inline=True, fontsize=10)
    plt.scatter(x_mu[:, 0, 0], x_mu[:, 0, 1], color=color)


def plot_covariance_ellipse(xEst, PEst):  # pragma: no cover
    '''
    xEst: (2, 1)
    PEst: (2, 2)
    '''
    Pxy = PEst[0:2, 0:2]
    eigval, eigvec = np.linalg.eig(Pxy)

    if eigval[0] >= eigval[1]:
        bigind = 0
        smallind = 1
    else:
        bigind = 1
        smallind = 0

    t = np.arange(0, 2 * math.pi + 0.1, 0.1)
    a = math.sqrt(eigval[bigind])
    b = math.sqrt(eigval[smallind])
    x = [a * math.cos(it) for it in t]
    y = [b * math.sin(it) for it in t]
    angle = math.atan2(eigvec[1, bigind], eigvec[0, bigind])
    rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
    fx = rot @ (np.array([x, y]))
    px = np.array(fx[0, :] + xEst[0, 0]).flatten()
    py = np.array(fx[1, :] + xEst[1, 0]).flatten()
    plt.plot(px, py, "--r")


def plot_points(data, y):
    sns.set(style="white", font="Arial")
    colors = sns.color_palette("Paired", n_colors=12).as_hex()

    n = y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875 * 4, 4))
    ax.set_facecolor('#bbbbbb')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # plot the locations of all data points ..
    for i, point in enumerate(data.data):
        if i <= n // 2:
            # .. separating them by ground truth ..
            ax.scatter(*point, color="#000000", s=3, alpha=.75, zorder=n + i)
        else:
            ax.scatter(*point, color="#ffffff", s=3, alpha=.75, zorder=n + i)

        if y[i] == 0:
            # .. as well as their predicted class
            ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[1])
        else:
            ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[5])

    handles = [
        plt.Line2D([0], [0], color='w', lw=4, label='Ground Truth 1'),
        plt.Line2D([0], [0], color='black', lw=4, label='Ground Truth 2'),
        plt.Line2D([0], [0], color=colors[1], lw=4, label='Predicted 1'),
        plt.Line2D([0], [0], color=colors[5], lw=4, label='Predicted 2')
    ]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    # plt.savefig("example.pdf")
    # plt.savefig("example.svg")


def gaussian_1d(x, u, var):
    sigma = np.sqrt(var)
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-1 * np.power(x - u, 2) / (2 * sigma**2))


def gaussian_nd(x, u, var):
    '''
    x: (n, d)
    mu: (d,)
    var: (d, d)
    '''
    return 1.0 / (np.power(2 * np.pi, x.shape[1] / 2) * np.sqrt(np.linalg.det(var))) * np.exp(
        np.sum(-0.5 * (x - u).dot(np.linalg.inv(var)) * (x - u), axis=1))


if __name__ == "__main__":
    # test_points_gmm_fitting()
    # test_gaussian1d_gmm_fitting()
    # test_gaussian2d_gmm_fitting()
    test_gaussiannd_gmm_fitting()
