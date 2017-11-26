import math

import numpy as np
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib as mpl
from bokeh.plotting import figure, gridplot
from bokeh.io import show

from vae import NB_SAMPLES  # TODO completely encapsulate this in model


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
    circle_radius = min(x_range, y_range)/resolution/1.3
    return grid, circle_radius


def plot_loss(model, fig, box, nb_samples, resolution):
    xy, radius = grid_from_box(box, resolution)

    losses = []
    model.eval()
    for X in xy:
        X = Variable(torch.from_numpy(X).contiguous().float())
        X = torch.stack([X]*nb_samples)
        obs_mu, obs_logvar,  mu, logvar = model(X)
        loss, _, _ = model.loss(X, obs_mu, obs_logvar, mu, logvar)
        losses.append(math.exp(-loss.data.numpy()[0]))

    print(len(losses), type(losses[0]))

    losses_norm = mpl.colors.Normalize()(losses)
    colors = [
        "#%02x%02x%02x" % (int(r), int(g), int(b))
        for r, g, b, _ in 255*plt.cm.Greys(losses_norm)
    ]

    fig.circle(xy[:, 0], xy[:, 1], fill_color=colors, line_color=colors, radius=radius)


def plot_datapoints(fig, data, color='red'):
    fig.cross(data[:, 0], data[:, 1], line_color=color)


def box_for_data(data, spacing=0.1):
    mins = data.min(axis=0) - spacing
    maxs = data.max(axis=0) + spacing

    return tuple(mins), tuple(maxs)


def plot_gaussians(fig, mus, variances, color='black', alpha=0.3):
    fig.ellipse(
        mus[:, 0], mus[:, 1],
        width=variances[:, 0], height=variances[:, 1],
        fill_alpha=alpha, fill_color=color,
        line_color=None,
    )


def plot_encodings(model, data):
    model.eval()
    X = Variable(torch.from_numpy(data).contiguous().float())
    _, _, z_mu, z_logvar = model(X)

    z_mu = z_mu.data.numpy()
    z_var = z_logvar.exp().data.numpy()

    fig = figure(
        title='Encodings in latent space',
        width=400,
        height=400,
    )

    plot_gaussians(fig, z_mu, z_var)
    plot_gaussians(fig, z_mu, 2.0*z_var, alpha=0.15)
    plot_datapoints(fig, z_mu)

    return fig


def plot_model_outputs(model, data, nb_samples=NB_SAMPLES, resolution=150):
    box = box_for_data(data)
    fig_px = figure(
        title='Lower bound of p(x)',
        width=400,
        height=400,
    )
    plot_loss(model, fig_px, box=box, nb_samples=nb_samples, resolution=resolution)
    plot_datapoints(fig_px, data)

    fig_encs = plot_encodings(model, data)

    gp = gridplot([[fig_px, fig_encs]])
    show(gp)
