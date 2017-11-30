import math

import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.plotting import figure, gridplot
from bokeh.io import show


def plot_model_outputs(model, data, resolution=70):
    box = box_for_data(data)
    grid = grid_from_box(box, resolution)
    radius = suggested_radius(box, resolution)

    losses = get_losses(model, grid)

    probs = [math.exp(-loss) for loss, llh, kld in losses]
    fig_px = plot_elbo(grid, probs, radius)
    plot_datapoints(fig_px, data)

    klds = [kld for loss, llh, kld in losses]
    fig_klds = plot_klds(grid, klds, radius)
    plot_datapoints(fig_klds, data)

    llhs = [math.exp(llh) for loss, llh, kld in losses]
    fig_llhs = plot_llhs(grid, llhs, radius)
    plot_datapoints(fig_llhs, data)

    fig_encs = plot_encodings(model, data)

    gp = gridplot([
        [fig_px, fig_encs],
        [fig_llhs, fig_klds],
    ])
    show(gp)


def plot_elbo(grid, values, radius):
    fig = figure(
        title='Lower bound of p(x)',
        width=400,
        height=400,
    )
    plot_grid_values(fig, grid=grid, values=values, radius=radius)

    return fig


def plot_klds(grid, values, radius):
    fig = figure(
        title='KLD( q(z|x) || p(z) )',
        width=400,
        height=400,
    )
    plot_grid_values(fig, grid=grid, values=values, radius=radius)

    return fig


def plot_llhs(grid, values, radius):
    fig = figure(
        title='Estimated p(x|z)',
        width=400,
        height=400,
    )
    plot_grid_values(fig, grid=grid, values=values, radius=radius)

    return fig


def plot_encodings(model, data):
    model.eval()
    X = Variable(torch.from_numpy(data).contiguous().float())
    _, _, z_mu, z_logvar = model(X)

    z_mu = z_mu.data.numpy()
    z_stddev = (z_logvar * 0.5).exp().data.numpy()

    fig = figure(
        title='Encodings in latent space',
        width=400,
        height=400,
    )

    plot_gaussians(fig, z_mu, z_stddev)
    plot_gaussians(fig, z_mu, 2.0*z_stddev, alpha=0.09)
    plot_datapoints(fig, z_mu)

    return fig


def get_losses(model, grid):
    losses = []
    model.eval()
    for X in grid:
        X = Variable(torch.from_numpy(X).contiguous().float())
        obs_mu, obs_logvar,  mu, logvar = model(X.view(1, -1))
        loss = model.loss(X, obs_mu, obs_logvar, mu, logvar)
        loss = tuple(l.data.numpy()[0] for l in loss[:-1])
        losses.append(loss)

    return losses


def plot_grid_values(fig, grid, values, radius):
    # TODO use image instead of fig.circle()
    values_norm = mpl.colors.Normalize()(values)
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b))
        for r, g, b, _ in 255*plt.cm.Greys(values_norm)
    ]

    fig.circle(grid[:, 0], grid[:, 1], fill_color=colors, line_color=colors, radius=radius)


def plot_gaussians(fig, mus, stddevs, color='black', alpha=0.3):
    fig.ellipse(
        mus[:, 0], mus[:, 1],
        width=stddevs[:, 0], height=stddevs[:, 1],
        fill_alpha=alpha, fill_color=color,
        line_color=None,
    )


def plot_datapoints(fig, data, color='red'):
    fig.cross(data[:, 0], data[:, 1], line_color=color)


def grid_from_box(box, resolution):
    x_min = box[0][0]
    x_max = box[1][0]
    x_range = (x_max - x_min)
    x_res = x_range/resolution
    y_min = box[0][1]
    y_max = box[1][1]
    y_range = (y_max - y_min)
    y_res = y_range/resolution

    grid = np.mgrid[x_min:x_max:x_res, y_min:y_max:y_res].reshape(2, -1).T
    return grid


def suggested_radius(box, resolution):
    x_min = box[0][0]
    x_max = box[1][0]
    x_range = (x_max - x_min)
    y_min = box[0][1]
    y_max = box[1][1]
    y_range = (y_max - y_min)

    return min(x_range, y_range)/resolution/1.3


def box_for_data(data, spacing=0.1):
    mins = data.min(axis=0) - spacing
    maxs = data.max(axis=0) + spacing

    return tuple(mins), tuple(maxs)
