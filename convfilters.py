import numpy as np
import skimage.filters as filters
import scipy.signal as signal
import scipy.ndimage as ndimage

def apply_kernels(data, kernels, crop=True):
    """
    Apply a list of convolution kernels of shape (w,w) to a dataset of shape
    (n,d,d,...)
    """
    maxsize = max(k.shape[0] for k in kernels)
    pkernels = []
    for k in kernels:
        size = k.shape[0]
        padw = (maxsize-size)/2
        pkernels.append(np.pad(k,padw,'constant'))
    stackkernels = np.stack(pkernels, axis=2)

    data_extend = data[...,np.newaxis]
    kernels_extend = stackkernels[np.newaxis,...]
    while kernels_extend.ndim < data_extend.ndim:
        kernels_extend = kernels_extend[...,np.newaxis,:]

    parts = []
    for i in range(0,data_extend.shape[0],20):
        convres = signal.fftconvolve(data_extend[i:i+20,...],kernels_extend)
        parts.append(convres)

    res = np.concatenate(parts, axis=0)
    if crop:
        padw = (res.shape[1] - data.shape[1])/2
        res = res[:,padw:-padw,padw:-padw,...]

    return res

def apply_gabor_kernels(data, freqs, angle_num):
    """
    Apply gabor kernels
    freqs: List of frequencies
    angle_num: Number of angles to use
    """
    kernels = [filters.gabor_kernel(f, i*np.pi/angle_num) for f in freqs for i in range(angle_num)]
    convres = apply_kernels(data, kernels)
    absres = np.abs(convres)
    return absres

def apply_gaussian_kernels(data, sigmas):
    """
    Apply gaussian kernels
    sigmas: List of sigmas
    """
    parts = []
    for sigma in sigmas:
        cparts = []
        for i in range(0,data.shape[0],20):
            cp1 = ndimage.filters.gaussian_filter1d(data[i:i+20], sigma, 1)
            cp2 = ndimage.filters.gaussian_filter1d(cp1, sigma, 2)
            cparts.append(cp2)
        parts.append(np.concatenate(cparts, axis=0))

    result = np.stack(parts, axis=-1)
    return result

def process_demo(data):
    data1 = expand_data(data)
    data2 = apply_gabor_kernels(data1, [0.05], 3)
    data3 = apply_gaussian_kernels(data2, [20, 40])
    data3p = np.concatenate([data2[...,np.newaxis], data3], axis=-1)
    data4 = flatten_data(data3p)
    return data4

def expand_data(data):
    return data.reshape((data.shape[0], 128, 128, -1))

def flatten_data(data):
    return data.reshape((data.shape[0],-1))
