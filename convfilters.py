import numpy as np
import skimage.filters as filters
import scipy.signal as signal

def apply_kernels(data, kernels):
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

    print data_extend.shape
    print kernels_extend.shape

    parts = []
    for i in range(0,data_extend.shape[0],20):
        convres = signal.fftconvolve(data_extend[i:i+20,...],kernels_extend)
        parts.append(convres)

    return np.concatenate(parts, axis=0)

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
