from pydrake.all import *
import time
from scipy import sparse
from scipy.sparse.linalg import spsolve

def linesearch(z, delta_z, merit_func, max_ls_iters=10, verbose=False):
    alpha = 1

    for i in range(max_ls_iters):
        if merit_func(z + alpha*delta_z) < merit_func(z):
            return alpha
        alpha = alpha/2
        if verbose:
            print(f"linesearch: iters = {i} \t alpha = {alpha} \t psi = {merit_func(z + alpha*delta_z)}")
    
    print("Linesearch Failed :(")

def newton(residual, z_guess, verbose = False, max_iters=200, tol=1e-5):
    z = np.copy(z_guess)
    r = residual(z)
    if np.linalg.norm(r, np.inf) < tol:
        return z
    for iters in range(max_iters):
        loop_start = time.perf_counter()
        dr_dz = jacobian(residual, z)

        # plt.imshow(dr_dz != 0)
        if np.any(np.isnan(dr_dz)):
            raise Exception(f"dr_dz has nan: {dr_dz}")
        
        delta_z = -np.linalg.lstsq(dr_dz, r, rcond=None)[0]

        alpha=1
        alpha = linesearch(z, delta_z, lambda z_: np.linalg.norm(residual(z_)), verbose=verbose)
        if alpha is None:
            return z
        z = z + alpha*delta_z

        r = residual(z)

        if verbose:
            # print(f"iter = {iters} \t t = {time.perf_counter()-start_time:1.3f} \t r = {np.linalg.norm(r, np.inf):1.5f} \t cond(dr_dz) = {np.linalg.cond(dr_dz)} \t alpha = {alpha} \t dr_dz: {dr_dz.shape}")
            print(f"iter = {iters} \t r = {np.linalg.norm(r, np.inf):1.5f} \t alpha = {alpha} \t dr_dz: {dr_dz.shape}")

        if np.linalg.norm(r, np.inf) < tol:
            break
        
    
    if iters == max_iters-1:
        print(f"Residual for iter = {iters} did not converge, ||r|| = {np.linalg.norm(r, np.inf)}")
    return z


def implicit_function_kkt_gradient(kkt_func, w, theta):
    # import matplotlib.pyplot as plt
    dr_dw = jacobian(lambda dw : kkt_func(dw, theta), w)
    dr_dtheta = jacobian(lambda dtheta : kkt_func(w, dtheta), theta)
    # print(dr_dw.shape)
    # print(dr_dtheta.shape)
    # print(1.0 - np.count_nonzero(dr_dw)/float(dr_dw.size))
    # print(1.0 - np.count_nonzero(dr_dtheta)/float(dr_dtheta.size))

    # plt.imshow(dr_dw)
    # plt.show()
    dr_dw_sp = sparse.csc_matrix(dr_dw)
    dr_dtheta_sp = sparse.csc_matrix(dr_dtheta)
    # s = time.perf_counter()
    return -spsolve(dr_dw_sp, dr_dtheta_sp).toarray()
    # print(time.perf_counter()-s)
    # s = time.perf_counter()
    # -np.linalg.solve(dr_dw, dr_dtheta)
    # print(time.perf_counter()-s)

