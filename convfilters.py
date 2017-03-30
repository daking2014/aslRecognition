import numpy as np
import skimage.filters as filters
import scipy.signal as signal
import scipy.ndimage as ndimage
import cv2
import argparse
from util import resizeData
import gc

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

def process_gabor_scales(data):

    cdata = expand_data(data) # n x 128 x 128 x d
    cres = None
    for scale in range(3):
        gabored = apply_gabor_kernels(cdata, [0.1], 4) # n x 128 x 128 x d x 4
        gabored_flat = gabored.reshape(gabored.shape[:-2] + (-1,))
        if cres is None:
            cres = gabored_flat
        else:
            gauss = apply_gaussian_kernels(cres, [2])
            gauss = gauss.reshape(gauss.shape[:-2] + (-1,))
            cres = np.concatenate([gauss, gabored_flat], axis=-1)

        cdata = resizeData(cdata, cdata.shape[1]/2)
        cres = resizeData(cres, cres.shape[1]/2)

    return flatten_data(cres)

def expand_data(data, w=128):
    return data.reshape((data.shape[0], w, w, -1))

def flatten_data(data):
    return data.reshape((data.shape[0],-1))

def main(preprocessed, outfile, key):
    zfile = np.load(preprocessed)
    data = zfile[key]
    procs = []
    for i in range(0,data.shape[0],10):
        processed = process_gabor_scales(data[i:i+10])
        processed = processed*10 # Scale up features to be approximately [0,1]
        procs.append(processed)
        gc.collect()
    processed = np.concatenate(procs,0)
    print "Processed data -> shape", processed.shape
    np.save(outfile, processed)

parser = argparse.ArgumentParser(description='Apply Gabor filters to the depth of a thing')
parser.add_argument('preprocessed', help='Preprocessed image zip file')
parser.add_argument('outfile', help='File to write')
parser.add_argument('--key', default="depth", help='Key to access')

if __name__ == '__main__':
    namespace = parser.parse_args()
    args = vars(namespace)
    main(**args)
