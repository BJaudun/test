import numpy as np
from scipy.sparse import diags, kron

from .. import numerical


def matrix_kinetic_operator(N, h):
    main_diag = -np.ones(N)
    off_diag = np.ones(N - 1)
    
    D = diags([main_diag, off_diag], [0, 1], shape=(N, N)) / h  
    
    Dx = kron(np.eye(N), D)
    Dy = kron(D, np.eye(N))
    
    Dx2 = -Dx @ Dx.T
    Dy2 = -Dy @ Dy.T
    
    return Dx2, Dy2

