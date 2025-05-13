import numpy as np
from scipy.sparse import diags

from . import energy1D
from .. import numerical
from ..constant import Parameters


import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

def self_consistency(energy_tolerance, electrons, dom_start, dom_end, grid_points):
    para = Parameters(energy_tolerance, electrons, dom_start, dom_end, grid_points)
    
    nx = np.ones(para.N)
    
    previous_energy = -np.inf
    energy_dif = np.inf
    
    T = energy1D.matrix_kinetic_operator(para.N, para.h)  

    while abs(energy_dif) > para.energy_tolerance:
        
        ex_en, ex_pot = energy1D.get_exchange(nx, para.h)
        ha_en, ha_pot = energy1D.get_hartree(nx, para.x, para.h)
        
        
        total_potential = ex_pot + ha_pot + para.x * para.x
        V = diags(total_potential)
        
       
        hamiltonian = T + V
        
        
        num_states = len(para.occupation)  
        epsilon_n, psi_gn = eigsh(hamiltonian, k=num_states, which='SA')  
        
        energy_dif = epsilon_n[0] - previous_energy
        previous_energy = epsilon_n[0]
        
        nx = np.zeros(para.N)
        for i in range(num_states):
            psi = psi_gn[:, i]
            nx += para.occupation[i] * numerical.sq_mod(psi, para.h)

    return nx, epsilon_n, psi_gn



self_consistency(10e-5,17,200,-5,5)