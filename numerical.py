def num_integral(fx,h):
    return h*(1/2)*(2*sum(fx)-fx[0]-fx[-1])

def norm_integral(fx,h):
    integral = num_integral(fx,h)
    norm_fx = fx / integral
    return norm_fx

def sq_mod(fx,h):
    norm_psi_sq = norm_integral(fx**2,h)
    return norm_psi_sq 