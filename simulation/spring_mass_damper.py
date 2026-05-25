import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import block_diag


def continuous_spring_mass_damper_dynamics(params):
    m, k, b, n = params["m"], params["k"], params["b"], params["n"]
    
    Ac = []
    Bc = []
    for i in range(n):
        Ac.append(np.array([[0, 1],[-k[i] / m[i], -b[i] / m[i]]]))
        Bc.append(np.array([[0, 0], [k[i] / m[i], b[i] / m[i]]]))

    return block_diag(*Ac), block_diag(*Bc)


def discrete_spring_mass_damper_dynamics_FE(params):
    # Forward Euler discretization (bad)
    h = params["h"]
    Ac, Bc = continuous_spring_mass_damper_dynamics(params)
    return (np.eye(2) + h * Ac), h * Bc


def discrete_spring_mass_damper_dynamics_RK4(params):
    h = params["h"]
    A, B = continuous_spring_mass_damper_dynamics(params)

    I = np.eye(A.shape[0])
    A2 = A @ A
    A3 = A2 @ A
    A4 = A3 @ A

    # Ad = I + hA + h^2/2 A^2 + h^3/6 A^3 + h^4/24 A^4
    Ad = I + h * A + (h**2 / 2.0) * A2 + (h**3 / 6.0) * A3 + (h**4 / 24.0) * A4

    # Bd = h(I + h/2 A + h^2/6 A^2 + h^3/24 A^3) B
    Bd = (h * I + (h**2 / 2.0) * A + (h**3 / 6.0) * A2 + (h**4 / 24.0) * A3) @ B

    return Ad, Bd


def interleave_columns(M):
    """
    Given M with shape (N, 2n), return matrix with columns interleaved.
    """
    N, cols = M.shape
    n = cols // 2
    first, second = M[:, :n], M[:, n:]
    # Stack [a1, b1, a2, b2, ...]
    interleaved = np.empty((N, cols), dtype=M.dtype)
    interleaved[:, 0::2] = first
    interleaved[:, 1::2] = second
    return interleaved


def deinterleave_columns(M):
    """
    Given interleaved M with shape (N, 2n), recover original order.
    """
    N, cols = M.shape
    n = cols // 2
    first = M[:, 0::2]
    second = M[:, 1::2]
    return np.hstack([first, second])

def spring_mass_simulate(x0, reference, stiffness, damping, dt, N):
    assert reference.shape[0] + 1 == N
    assert reference.shape[1] == x0.shape[1]
    assert reference.shape[1] // 2 == stiffness.shape[0] == damping.shape[0]
    n = reference.shape[1] // 2
    params = {"h": dt, "k": stiffness, "b": damping, "m": np.ones(n), "n": n}
    Ad, Bd = discrete_spring_mass_damper_dynamics_RK4(params)
    
    x0_reshuffle = interleave_columns(x0)
    reference_reshuffle = interleave_columns(reference)

    X = np.zeros((n*2, N))
    X[:, 0] = x0_reshuffle 
    for k in range(N - 1):
        x_ref = reference_reshuffle[k]
        X[:, k + 1] = Ad @ X[:, k] + Bd @ x_ref.T
    
    return deinterleave_columns(X.T)[1:]


if __name__ == "__main__":
    n = 2
    N = 100
    dt = 0.01
    x0 = np.zeros((1, n * 2))
    ref = np.zeros((N - 1, n * 2))  # positions, velocities
    ref[:, :n] = 1
    stiffness = np.ones(n) * 100
    damping = np.ones(n) * 10

    X = spring_mass_simulate(x0, ref, stiffness, damping, dt, N)
    

    plt.plot(X[:])
    plt.show()
