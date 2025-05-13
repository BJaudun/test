import numpy as np
from scipy.sparse import diags

from .. import numerical


def matrix_kinetic_operator(N, h):
    diag = -2.0 * np.ones(N)
    off_diag = 1.0 * np.ones(N - 1)
    
    D2 = diags([diag, off_diag, off_diag], [0, 1, -1], shape=(N, N)) / h**2
    
    T = -0.5 * D2
    
    return T

def get_exchange(nx,h):
    energy=-3./4.*(3./np.pi)**(1./3.)*numerical.num_integral(nx,h)
    potential=-(3./np.pi)**(1./3.)*nx**(1./3.)
    return energy, potential 

def get_hartree(nx : float,x : float ,h,eps=1e-1):
    energy=np.sum(nx[None,:]*nx[:,None]*h**2/np.sqrt((x[None,:]-x[:,None])**2+eps)/2)
    potential=np.sum(nx[None,:]*h/np.sqrt((x[None,:]-x[:,None])**2+eps),axis=-1)
    return energy, potential


