
import numpy as np


def plot_normal(fig, mean, cov, alpha=1., color='blue'):
    'Plot a Normal density'

    # Eigenvalue decomposition of the covariance matrix.
    Q, R = np.linalg.qr(cov)

    # Angle of the rotation.
    angle =  - np.arccos(Q[0, 0])
    evals = np.abs(np.diag(R))

    fig.ellipse(x=mean[0], y=mean[1],
            width=4 * np.sqrt(evals[0]),
            height=4 * np.sqrt(evals[1]),
            angle=angle, alpha=.5 * alpha, color=color)
    fig.cross(mean[0], mean[1], color=color, alpha=alpha)
    fig.ellipse(x=mean[0], y=mean[1],
            width=2 * np.sqrt(evals[0]),
            height=2 * np.sqrt(evals[1]),
            angle=angle, alpha=alpha, color=color)


def plot_gmm(fig, gmm, alpha=1., color='blue'):
    'Plot a Normal density'
    for weight, comp in zip(gmm.weights, gmm.modelset):
        plot_normal(fig, comp.mean.numpy(), comp.cov.numpy(),
            alpha * weight.numpy(), color)

def plot_hmm(fig, hmm, alpha=1., color='blue'):
    'Plot a Normal density'
    for comp in hmm.modelset:
        plot_normal(fig, comp.mean.numpy(), comp.cov.numpy(),
            alpha, color)
        

def plot_latent_model(fig, latent_model, alpha=1., color='blue'):
    if 'Mixture' in str(type(latent_model)):
        plot_gmm(fig, latent_model, alpha, color)
    elif 'Normal' in str(type(latent_model)):
        plot_normal(fig, latent_model.mean.numpy(),
                    latent_model.cov.numpy(), alpha, color)
    else:
        raise ValueError

