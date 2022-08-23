import jax.numpy as np

__all__ = ["get_GE", "get_RGE", "get_RWGE", "get_Rvec", "get_Rmask"]

def get_GE(array):
    grads_vec = np.array(np.gradient(array))
    return np.hypot(grads_vec[0], grads_vec[1])

def get_RGE(array):
    Rvec = get_Rvec(array.shape[0]) 
    Rvec += 1e-8 # extra added for gradient stability
    grads_vec = np.array(np.gradient(array))
    
    xnorm = Rvec[1]*grads_vec[0]
    ynorm = Rvec[0]*grads_vec[1]
    return np.square(xnorm + ynorm)

def get_RWGE(array):
    Rvec = get_Rvec(array.shape[0])
    Rmag = np.hypot(Rvec[0], Rvec[1])
    Rnorm = Rvec/(Rmag+1e-8)
    
    grads_vec = np.array(np.gradient(array))
    
    xnorm = Rnorm[1]*grads_vec[0]
    ynorm = Rnorm[0]*grads_vec[1]
    return np.square(xnorm + ynorm)
    
def get_Rvec(npix):
    c = npix//2
    xs = np.arange(-c, c)
    Rvec = np.array(np.meshgrid(xs, xs))
    return Rvec + 1e-12

def get_Rmask(npix, rmin, rmax, shift=0.5):
    c = npix//2
    xs = np.arange(-c, c) + shift
    YY, XX = np.meshgrid(xs, xs)
    RR = np.hypot(XX, YY)
    return ((RR < rmax) & (RR > rmin)).astype(float)


