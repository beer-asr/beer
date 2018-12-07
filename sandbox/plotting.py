
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

def plot_normal(fig, mean, cov, n_std_dev=2, npoints=100, transform=None, **kwargs):
    'Plot a Normal density'
        
    if transform is None:
        transform = lambda x: x
    def new_transform(xy):
        return mean + transform(xy)
    plot_covariance(fig, cov, n_std_dev, npoints, transform=new_transform,
                    **kwargs)


def plot_gmm(fig, gmm, n_std_dev=2, npoints=100, alpha=1., colors=None, **kwargs):
    'Plot a Normal density'
    if colors is None:
        colors = ['blue'] * len(gmm.modelset)
    for weight, comp, color in zip(gmm.weights, gmm.modelset, colors):
        kwargs['color'] = color
        plot_normal(fig, comp.mean.numpy(), comp.cov.numpy(),
            n_std_dev, npoints, alpha=alpha * weight.numpy(), **kwargs)

def plot_hmm(fig, hmm, n_std_dev=2, npoints=100, **kwargs):
    'Plot a Normal density'
    for comp in hmm.modelset:
        plot_normal(fig, comp.mean.numpy(), comp.cov.numpy(),
                    n_std_dev, npoints, **kwargs)
        