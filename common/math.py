import numpy as np
from scipy.signal import butter, lfilter, filtfilt
from pydrake.all import *

def rot_log_error(R_WB:RotationMatrix, R_WB_des:RotationMatrix):
    """
    Returns eR = log(R_des^T R) as a 3-vector (axis * angle).
    Works for float and AutoDiff types if R_* are Drake RotationMatrix_[T].
    """
    R_err : RotationMatrix = R_WB_des.matrix().T @ R_WB.matrix()
    if R_err.dtype == float:
        aa = RotationMatrix(R_err).ToAngleAxis()
    else:
        aa = RotationMatrix_[AutoDiffXd](R_err).ToAngleAxis()
    return aa.axis() * aa.angle()   # (3,)

def angle_between_three_pts_2d(a, b, c):
    u = b-a
    v = c-b
    cross = u[0]*v[1] - u[1]*v[0]
    dot   = u @ v
    return np.arctan2(cross, dot)

def angle_between_three_pts_2d_normal(a, b, c):
    u = b-a
    v = c-b
    u = u/np.linalg.norm(u)
    v = v/np.linalg.norm(v)

    cross = u[0]*v[1] - u[1]*v[0]
    dot   = u @ v
    return np.arctan2(cross, dot)


def cross_2d(u, v):
    pass

def ortho_vec_2d(v):
    v = np.cross(np.r_[v, 0], np.array([0,0,1]))[:2]
    v /= np.linalg.norm(v)
    return v

def low_pass(d, freq=0.05):
    b, a = butter(2, freq, btype="low")
    return filtfilt(b, a, d, axis=0)

def make_bezier(control_points,end_time) -> BezierCurve:
    if type(control_points.flatten()[0])==Variable:
        BezierCurve = BezierCurve_[Expression]
    elif control_points.dtype == object:
        BezierCurve = BezierCurve_[AutoDiffXd]
    else:
        BezierCurve = BezierCurve_[float]
    return BezierCurve(0, end_time, control_points)

def sample_bezier_and_derivative(knot_points: np.ndarray, nu, N, end_time=None):
    if type(knot_points.flatten()[0])==Variable or type(knot_points.flatten()[0])==Expression:
        BezierCurve = BezierCurve_[Expression]
    elif knot_points.dtype == object:
        BezierCurve = BezierCurve_[AutoDiffXd]
    else:
        BezierCurve = BezierCurve_[float]
    if end_time is None:
        end_time = 1.0
    if knot_points.ndim == 1:
        control_points = knot_points.reshape((nu, -1), order="F")
    else:
        control_points = knot_points.reshape((nu, -1))
    spline = BezierCurve(0, end_time, control_points)
    ts = np.linspace(0, end_time, N)

    return np.vstack(
        [spline.vector_values(ts), spline.MakeDerivative(1).vector_values(ts)]
    ).T

def sample_bezier_and_two_derivative(knot_points: np.ndarray, nu, N, end_time=None):
    if type(knot_points.flatten()[0])==Variable or type(knot_points.flatten()[0])==Expression:
        BezierCurve = BezierCurve_[Expression]
    elif knot_points.dtype == object:
        BezierCurve = BezierCurve_[AutoDiffXd]
    else:
        BezierCurve = BezierCurve_[float]
    if end_time is None:
        end_time = 1.0
    if knot_points.ndim == 1:
        control_points = knot_points.reshape((nu, -1), order="F")
    else:
        control_points = knot_points.reshape((nu, -1))
    spline = BezierCurve(0, end_time, control_points)
    ts = np.linspace(0, end_time, N)

    return np.vstack(
        [spline.vector_values(ts), spline.MakeDerivative(1).vector_values(ts), spline.MakeDerivative(2).vector_values(ts)]
    ).T
    

def translation_zrot_to_transform(xyz, z_rot):
    return RigidTransform(RollPitchYaw([0,0,z_rot]), xyz)

def transform_3d_points(points, X :RigidTransform):
    # points : (N, M, 3)
    points = np.asarray(points)
    R = X.rotation().matrix()
    t = X.translation()
    return points @ R.T + t

def polar_coord_3d(u, v, l):
    z = np.sin(v)*l
    h = np.cos(v)*l
    x = np.cos(u)*h
    y = np.sin(u)*h

    return np.array([x,y,z])

def cart_to_polar_3d(p):
    u = np.arctan2(p[1], p[0])
    v = np.arctan2(p[2], np.sqrt(p[0]**2 + p[1]**2))
    return np.array([u, v])

def angle_wrap(x):
    while abs(x) > np.pi:
        if x > np.pi:
            x -= np.pi*2
        elif x < -np.pi:
            x += np.pi*2
    return x

def angle_wrap_vec(x_):
    x_vec = np.copy(x_)
    for i, x in enumerate(x_vec):
        while abs(x) > np.pi:
            if x > np.pi:
                x -= np.pi*2
            elif x < -np.pi:
                x += np.pi*2
        x_vec[i] = x
    return x_vec


def finite_diff_central(T, X):
    X_dot = np.zeros_like(X)
    for i in range(1, X.shape[0]-1):
        X_dot[i] = (X[i+1] - X[i-1])/(T[i+1]-T[i-1])

    X_dot[0] = X_dot[1]
    X_dot[-1] = X_dot[-2]
    return X_dot

def finite_diff_jacobian(F, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    f0 = np.asarray(F(x), dtype=float)
    n = x.size
    m = f0.size
    J = np.empty((m, n), dtype=float)

    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = h
        f_plus  = F(x + dx)
        f_minus = F(x - dx)
        J[:, i] = (np.asarray(f_plus) - np.asarray(f_minus)) / (2.0 * h)

    return J


def angular_velocity(x_a, x_b, x_c, v_a, v_b, v_c, eps=1e-10):
    """
    Compute scalar angular velocity (dθ/dt) between segments
    x_a-x_b and x_c-x_b.

    Parameters
    ----------
    x_a, x_b, x_c : (3,) array_like
        3D positions
    v_a, v_b, v_c : (3,) array_like
        3D velocities
    eps : float
        Small regularization to avoid division by zero

    Returns
    -------
    float
        Signed scalar angular velocity (rad/s)
    """
    # Relative vectors
    r1 = x_a - x_b
    r2 = x_c - x_b

    dr1 = v_a - v_b
    dr2 = v_c - v_b

    # Cross product
    cross = np.cross(r1, r2)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < eps:
        return 0.0  # nearly collinear

    term = (dr1 / (np.dot(r1, r1) + eps)
            - dr2 / (np.dot(r2, r2) + eps))

    theta_dot = np.dot(cross, term) / cross_norm

    return theta_dot


def resample_linear_interpolate(times, samples, resample_times):
    traj : PiecewisePolynomial = PiecewisePolynomial.FirstOrderHold(times, samples.T)
    return traj.vector_values(resample_times).T

def apply_transform_to_piecewise_pose(
    X_WF: PiecewisePose, X_FP: RigidTransform
) -> PiecewisePose:
    """
    Compose a constant transform X_FP (expressed in the pose frame F) with a
    PiecewisePose X_WF to produce X_WP(t) = X_WF(t) @ X_FP.
    The returned trajectory preserves X_WF's knot times with linear interpolation.
    """
    times = np.array(X_WF.get_segment_times())
    poses = [X_WF.GetPose(t) @ X_FP for t in times]
    return PiecewisePose.MakeLinear(times.tolist(), poses)


import numpy as np

class TrapezoidalMotionProfile:
    #      1          2
    #      ___________
    #     /           \
    #    /             \
    #   /               \
    #  /                 \
    # init                end
    def __init__(self, path_length, max_velocity, max_acceleration):
        self.path_length = path_length
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        if path_length < max_velocity**2 / max_acceleration:
            self.t_1 = self.t_2 = np.sqrt(path_length / (max_acceleration))
            self.t_end = self.t_1 * 2
        else:
            self.t_1 = max_velocity / max_acceleration
            self.t_2 = self.t_1 + (path_length - self.t_1 * max_velocity) / max_velocity
            self.t_end = self.t_1 + self.t_2

    def state(self, t):
        if t < 0:
            return 0, 0, 0
        if t < self.t_1:
            return (
                (t**2) * self.max_acceleration / 2,
                t * self.max_acceleration,
                self.max_acceleration,
            )
        if t < self.t_2:
            return (
                (self.t_1**2) * self.max_acceleration / 2
                + (t - self.t_1) * self.max_velocity,
                self.max_velocity,
                0,
            )
        if t <= self.t_end:
            return (
                (self.t_1**2) * self.max_acceleration / 2
                + (self.t_2 - self.t_1) * self.max_velocity
                + (self.t_1**2) * self.max_acceleration / 2
                - ((self.t_end - t) ** 2) * self.max_acceleration / 2,
                (self.t_end - t) * self.max_acceleration,
                -self.max_acceleration,
            )
        return 0, 0, 0
