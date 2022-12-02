import numpy as np
import scipy.signal as fft

#https://numbersmithy.com/2d-and-3d-convolutions-using-numpy/

def checkShape(var, kernel):
    '''Check shapes for convolution
    Args:
        var (ndarray): 2d or 3d input array for convolution.
        kernel (ndarray): 2d or 3d convolution kernel.
    Returns:
        kernel (ndarray): 2d kernel reshape into 3d if needed.
    '''
    var_ndim = np.ndim(var)
    kernel_ndim = np.ndim(kernel)
    if var_ndim not in [2, 3]:
        raise Exception("<var> dimension should be in 2 or 3.")
    if kernel_ndim not in [2, 3]:
        raise Exception("<kernel> dimension should be in 2 or 3.")
    if var_ndim < kernel_ndim:
        raise Exception("<kernel> dimension > <var>.")
    if var_ndim == 3 and kernel_ndim == 2:
        kernel = np.repeat(kernel[:, :, None], var.shape[2], axis=2)
    return kernel

def padArray(var, pad1, pad2=None):
    '''Pad array with 0s
    Args:
        var (ndarray): 2d or 3d ndarray. Padding is done on the first 2 dimensions.
        pad1 (int): number of columns/rows to pad at left/top edges.
    Keyword Args:
        pad2 (int): number of columns/rows to pad at right/bottom edges.
            If None, same as <pad1>.
    Returns:
        var_pad (ndarray): 2d or 3d ndarray with 0s padded along the first 2
            dimensions.
    '''
    if pad2 is None:
        pad2 = pad1
    if pad1+pad2 == 0:
        return var
    var_pad = np.zeros(tuple(pad1+pad2+np.array(var.shape[:2])) + var.shape[2:])
    var_pad[pad1:-pad2, pad1:-pad2] = var
    return var_pad

def pickStrided(var, stride):
    '''Pick sub-array by stride
    Args:
        var (ndarray): 2d or 3d ndarray.
        stride (int): stride/step along the 1st 2 dimensions to pick
            elements from <var>.
    Returns:
        result (ndarray): 2d or 3d ndarray picked at <stride> from <var>.
    '''
    if stride < 0:
        raise Exception("<stride> should be >=1.")
    if stride == 1:
        result = var
    else:
        result = var[::stride, ::stride, ...]
    return result

def conv3D(var, kernel, stride=1, pad=0):
      '''3D convolution using scipy.signal.fftconvolve.
      Args:
          var (ndarray): 2d or 3d array to convolve along the first 2 dimensions.
          kernel (ndarray): 2d or 3d kernel to convolve. If <var> is 3d and <kernel>
              is 2d, create a dummy dimension to be the 3rd dimension in kernel.
      Keyword Args:
          stride (int): stride along the 1st 2 dimensions. Default to 1.
          pad (int): number of columns/rows to pad at edges.
      Returns:
          conv (ndarray): convolution result.
      '''
      stride = int(stride)
      kernel = checkShape(var, kernel)
      if pad > 0:
          var_pad = padArray(var, pad, pad)
      else:
          var_pad = var
      conv = fft.fftconvolve(var_pad, kernel, mode='valid')
      if stride > 1:
          conv = pickStrided(conv, stride)
      return conv