import numpy as np
import torch 


def move_to_device(data, device):
    new_data = (t.to(device) for t in data)
    return new_data


def get_data_below_percentile(X_nxp: np.array, y_n: np.array, percentile: float = 80, n_sample: int = None, seed: int = None):

    #
    # Find the data below the percentile
    #
    perc = np.percentile(y_n, percentile)
    idx = np.where(y_n <= perc)[0]
    print("Max label in training data: {:.1f}. {}-th percentile label: {:.1f}".format(np.max(y_n), percentile, perc))

    #
    # Subsample if so specified
    #
    if n_sample is not None and n_sample < idx.size:
        np.random.seed(seed)
        idx = np.random.choice(idx, size=n_sample, replace=False)

    return X_nxp[idx], y_n[idx], idx


def compute_bin_weights(y, K_factor=0.1, N_bins=64, temp=0.1):
    #
    # Compute offset K for the denominator
    #
    N = y.shape[0]
    K = K_factor * N 

    #
    # Divide the range of y into bins
    #
    y_min = y.min()
    y_max = y.max()
    y = y[..., None]
    steps = np.arange(N_bins + 1) / N_bins
    limits = y_min + (y_max - y_min) * steps 
    lower_limits = limits[:-1].reshape(1, -1)
    upper_limits = limits[1:].reshape(1, -1)

    #
    # Calculate stats of all bins
    #
    bins = 1. * (lower_limits <= y) * (y <= upper_limits)
    bin_counts = bins.sum(0, keepdims=True)
    mid_points = (lower_limits + upper_limits) / 2
    mid_points = mid_points

    #
    # Compute weights 
    #
    weights = np.exp((mid_points - y_max) / temp) * bin_counts / (bin_counts + K)

    return torch.from_numpy(lower_limits), torch.from_numpy(upper_limits), torch.from_numpy(weights).reshape(1, -1)


def assign_bin_weights(y, bins):
    #
    # Extract the characterics from bins
    #
    lower_limits, upper_limits, weights = bins 
    y = y.reshape(-1, 1)

    #
    # Get bin allocation
    #
    which_bin = 1. * (lower_limits <= y) * (y <= upper_limits)
    
    #
    # Compute the weights for each example
    #
    weights = (which_bin * weights).sum(-1)

    return weights


