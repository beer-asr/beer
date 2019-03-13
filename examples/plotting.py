
import numpy as np

def plot_shaded_area(fig, xy1, xy2, **kwargs):
    upper_band = np.append(xy1[:,0], xy2[:, 0][::-1])
    lower_band = np.append(xy1[:,1], xy2[:, 1][::-1])
    fig.patch(upper_band, lower_band, **kwargs)
    
def create_upper_semicircle(radius, npoints):
    angles = np.linspace(0, np.pi, npoints)
    x, y = radius * np.cos(angles), radius * np.sin(angles)
    return np.c_[x, y]

def create_lower_semicircle(radius, npoints):
    xy = create_upper_semicircle(radius, npoints)
    return np.c_[xy[:, 0], -xy[:, 1]]

def plot_circle(fig, radius, npoints, tensor_metric=np.eye(2), transform=None, 
                **kwargs):
    xy1 = create_upper_semicircle(radius, npoints // 2) @ tensor_metric
    xy2 = create_lower_semicircle(radius, npoints // 2) @ tensor_metric
    if transform is not None:
        xy1 = transform(xy1)
        xy2 = transform(xy2)
    plot_shaded_area(fig, xy1, xy2, **kwargs)
    
def plot_covariance(fig, covariance, n_std_dev=2, npoints=100, transform=None, 
                    **kwargs):
    tensor_metric = np.linalg.cholesky(covariance)
    for std_dev in range(n_std_dev, 0, -1):
        plot_circle(fig, std_dev, npoints, tensor_metric=tensor_metric.T, 
                    transform=transform, **kwargs)

def plot_normal(fig, mean, cov, n_std_dev=2, npoints=100, **kwargs):
    'Plot a Normal density'
        
    def transform(xy):
        return mean + xy
    plot_covariance(fig, cov, n_std_dev, npoints, transform=transform,
                    **kwargs)


def plot_gmm(fig, gmm, n_std_dev=2, npoints=100, alpha=1., colors=None, **kwargs):
    'Plot a Normal density'
    weights = gmm.weights.value().detach()
    for weight, comp in zip(weights, gmm.modelset):

        plot_normal(fig, comp.mean.detach().numpy(), comp.cov.detach().numpy(),
            n_std_dev, npoints, alpha=alpha * weight.detach().numpy(), **kwargs)

def plot_hmm(fig, hmm, n_std_dev=2, npoints=100, colors=None, **kwargs):
    'Plot a Normal density'
    if colors is None:
        colors = [None] * hmm.graph.n_states
    for comp, color in zip(hmm.modelset, colors):
        if color is not None:
            kwargs['color'] = color
        plot_normal(fig, comp.mean.numpy(), comp.cov.numpy(),
                    n_std_dev, npoints, **kwargs)
        