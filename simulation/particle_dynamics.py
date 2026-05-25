from pydrake.all import *
from common.params import ParticleRopeParams
import viser
import trimesh


def stiffness_force_3d(u, x, params:ParticleRopeParams):
    """
    Bending stiffness forces from an energy-based formulation.

    For each consecutive triplet (x_a, x_b, x_c), define the bending angle
        θ = angle(x_b - x_a, x_c - x_b) ∈ [0, π]
    and the per-joint bending energy
        E = 0.5 * k * θ^2.

    The returned forces are -∂E/∂x applied to the particle positions `x`.
    The first "joint" uses a virtual point one link-length behind the fingertip
    along the fingertip x-axis (u[3:6]) so that θ=0 corresponds to the rope
    aligning with the hand direction.
    """
    k = params.stiffness
    l = params.l
    nq, num_particles = params.nq, params.num_particles

    if x.dtype == object:
        F = np.zeros(nq, dtype=x.dtype)
    elif u.dtype == object:
        F = np.zeros(nq, dtype=u.dtype)
    elif type(k) == Variable:
        F = np.zeros(nq, dtype=type(k))
    else:
        F = np.zeros(nq, dtype=np.float64)

    eps = 1e-12
    near_colinear_sin2_tol = 1e-8

    def _joint_stiffness(joint_index: int):
        return k[joint_index]

    def _triplet_bending_forces(x_a, x_b, x_c, k_joint):
        """
        Forces (f_a, f_b, f_c) that correspond to -∂/∂x of 0.5*k*θ^2, where
        θ = angle(x_b - x_a, x_c - x_b).
        """
        g = x_b - x_a
        n = x_c - x_b

        g2 = np.dot(g, g) + eps
        n2 = np.dot(n, n) + eps

        bend_axis = np.cross(g, n)
        bend_axis2 = np.dot(bend_axis, bend_axis)
        bend_axis_norm = np.sqrt(bend_axis2 + eps)
        # Smoothly attenuate near-colinear triplets to avoid hard branch
        # discontinuities that can break Newton line search.
        colinear_scale = bend_axis2 / (bend_axis2 + near_colinear_sin2_tol * g2 * n2)

        # θ = atan2(||g×n||, g·n). For non-collinear triplets that reach here,
        # bend_axis_norm with eps keeps this well-defined and AutoDiff-friendly.
        theta = np.arctan2(bend_axis_norm, np.dot(g, n))

        bend_axis_hat = bend_axis / bend_axis_norm
        tau = -k_joint * colinear_scale * theta * bend_axis_hat

        f_a = np.cross(tau, g) / g2
        f_c = np.cross(tau, n) / n2
        f_b = -(f_a + f_c)
        return f_a, f_b, f_c

    # Joint 0: (virtual point behind hand, hand, particle 0)
    x_hand = u[:3]
    hand_x_axis = u[3:6]
    x_a = x_hand - hand_x_axis * l
    x_b = x_hand
    x_c = x[:3]
    _, _, f_c = _triplet_bending_forces(x_a, x_b, x_c, _joint_stiffness(0))
    F[:3] += f_c

    if num_particles > 1:
        # Joint 1: (hand, particle 0, particle 1)
        x_a = x_hand
        x_b = x[:3]
        x_c = x[3:6]
        _, f_b, f_c = _triplet_bending_forces(x_a, x_b, x_c, _joint_stiffness(1))
        F[:3] += f_b
        F[3:6] += f_c

        # Joints 2..: (particle i, particle i+1, particle i+2)
        for i in range(num_particles - 2):
            joint_index = i + 2
            x_a = x[i * 3 : (i + 1) * 3]
            x_b = x[(i + 1) * 3 : (i + 2) * 3]
            x_c = x[(i + 2) * 3 : (i + 3) * 3]
            f_a, f_b, f_c = _triplet_bending_forces(
                x_a, x_b, x_c, _joint_stiffness(joint_index)
            )
            F[i * 3 : (i + 1) * 3] += f_a
            F[(i + 1) * 3 : (i + 2) * 3] += f_b
            F[(i + 2) * 3 : (i + 3) * 3] += f_c

    if params.fixed_end:
        x_fixed = params.fixed_end_pose.translation()
        if num_particles == 1:
            # End joint: (hand, particle 0, fixed)
            x_a = x_hand
            x_b = x[:3]
            x_c = x_fixed
            _, f_b, _ = _triplet_bending_forces(
                x_a, x_b, x_c, _joint_stiffness(num_particles)
            )
            F[:3] += f_b
        elif num_particles > 1:
            # End joint: (particle N-2, particle N-1, fixed)
            x_a = x[(num_particles - 2) * 3 : (num_particles - 1) * 3]
            x_b = x[(num_particles - 1) * 3 : (num_particles) * 3]
            x_c = x_fixed
            f_a, f_b, _ = _triplet_bending_forces(
                x_a, x_b, x_c, _joint_stiffness(num_particles)
            )
            F[(num_particles - 2) * 3 : (num_particles - 1) * 3] += f_a
            F[(num_particles - 1) * 3 : (num_particles) * 3] += f_b


            x_a = x[(num_particles - 1) * 3 : (num_particles) * 3]
            x_b = x_fixed
            x_c = x_fixed - params.fixed_end_pose.rotation().matrix()[:, 0]*l
            f_a, _, _ = _triplet_bending_forces(
                x_a, x_b, x_c, _joint_stiffness(num_particles)
            )
            F[(num_particles - 1) * 3 : (num_particles) * 3] += f_a

    return F

def damping_force_3d(u, x, v, params:ParticleRopeParams):
    """
    Bending damping forces from a Rayleigh dissipation formulation.

    For each consecutive triplet (x_a, x_b, x_c), define the relative bend rate
        θ̇ = (ω_n - ω_g) · b̂
    where ω_g and ω_n are the angular velocities of the segment vectors
    g = x_b - x_a and n = x_c - x_b, and b̂ is the unit bend axis.

    With per-joint dissipation
        D = 0.5 * b * θ̇^2,
    the returned forces correspond to
        F = -∂D/∂ẋ
    on the particle positions `x`.
    """
    b = params.damping
    l = params.l
    nq, num_particles = params.nq, params.num_particles

    if v.dtype == object:
        F = np.zeros(nq, dtype=v.dtype)
    elif u.dtype == object:
        F = np.zeros(nq, dtype=u.dtype)
    elif type(b) == Variable:
        F = np.zeros(nq, dtype=type(b))
    else:
        F = np.zeros(nq, dtype=np.float64)

    eps = 1e-12
    near_colinear_sin2_tol = 1e-8

    def _joint_damping(joint_index: int):
        return b[joint_index]

    def _triplet_bending_damping_forces(x_a, x_b, x_c, v_a, v_b, v_c, b_joint):
        g = x_b - x_a
        n = x_c - x_b
        g_dot = v_b - v_a
        n_dot = v_c - v_b

        g2 = np.dot(g, g) + eps
        n2 = np.dot(n, n) + eps

        bend_axis = np.cross(g, n)
        bend_axis2 = np.dot(bend_axis, bend_axis)
        bend_axis_norm = np.sqrt(bend_axis2 + eps)
        colinear_scale = bend_axis2 / (bend_axis2 + near_colinear_sin2_tol * g2 * n2)
        bend_axis_hat = bend_axis / bend_axis_norm

        omega_g = np.cross(g, g_dot) / g2
        omega_n = np.cross(n, n_dot) / n2
        theta_dot = np.dot(omega_n - omega_g, bend_axis_hat)

        tau = -b_joint * colinear_scale * theta_dot * bend_axis_hat
        f_a = np.cross(tau, g) / g2
        f_c = np.cross(tau, n) / n2
        f_b = -(f_a + f_c)
        return f_a, f_b, f_c

    x_hand = u[:3]
    hand_x_axis = u[3:6]

    v_hand = u[6:9] if u.shape[0] >= 9 else np.zeros(3, dtype=F.dtype)
    omega_hand = u[9:12] if u.shape[0] >= 12 else np.zeros(3, dtype=F.dtype)

    # Joint 0: (virtual point behind hand, hand, particle 0)
    x_a = x_hand - hand_x_axis * l
    x_b = x_hand
    x_c = x[:3]
    v_b = v_hand
    # x_a moves rigidly with the hand orientation.
    v_a = v_hand - np.cross(omega_hand, hand_x_axis) * l
    v_c = v[:3]
    _, _, f_c = _triplet_bending_damping_forces(
        x_a, x_b, x_c, v_a, v_b, v_c, _joint_damping(0)
    )
    F[:3] += f_c

    if num_particles > 1:
        # Joint 1: (hand, particle 0, particle 1)
        x_a = x_hand
        x_b = x[:3]
        x_c = x[3:6]
        v_a = v_hand
        v_b = v[:3]
        v_c = v[3:6]
        _, f_b, f_c = _triplet_bending_damping_forces(
            x_a, x_b, x_c, v_a, v_b, v_c, _joint_damping(1)
        )
        F[:3] += f_b
        F[3:6] += f_c

        # Joints 2..: (particle i, particle i+1, particle i+2)
        for i in range(num_particles - 2):
            joint_index = i + 2
            x_a = x[i * 3 : (i + 1) * 3]
            x_b = x[(i + 1) * 3 : (i + 2) * 3]
            x_c = x[(i + 2) * 3 : (i + 3) * 3]
            v_a = v[i * 3 : (i + 1) * 3]
            v_b = v[(i + 1) * 3 : (i + 2) * 3]
            v_c = v[(i + 2) * 3 : (i + 3) * 3]
            f_a, f_b, f_c = _triplet_bending_damping_forces(
                x_a, x_b, x_c, v_a, v_b, v_c, _joint_damping(joint_index)
            )
            F[i * 3 : (i + 1) * 3] += f_a
            F[(i + 1) * 3 : (i + 2) * 3] += f_b
            F[(i + 2) * 3 : (i + 3) * 3] += f_c

    if params.fixed_end:
        x_fixed = params.fixed_end_pose.translation()
        v_fixed = np.zeros(3, dtype=F.dtype)
        if num_particles == 1:
            # End joint: (hand, particle 0, fixed)
            x_a = x_hand
            x_b = x[:3]
            x_c = x_fixed
            v_a = v_hand
            v_b = v[:3]
            v_c = v_fixed
            _, f_b, _ = _triplet_bending_damping_forces(
                x_a, x_b, x_c, v_a, v_b, v_c, _joint_damping(num_particles)
            )
            F[:3] += f_b
        elif num_particles > 1:
            # End joint: (particle N-2, particle N-1, fixed)
            x_a = x[(num_particles - 2) * 3 : (num_particles - 1) * 3]
            x_b = x[(num_particles - 1) * 3 : (num_particles) * 3]
            x_c = x_fixed
            v_a = v[(num_particles - 2) * 3 : (num_particles - 1) * 3]
            v_b = v[(num_particles - 1) * 3 : (num_particles) * 3]
            v_c = v_fixed
            f_a, f_b, _ = _triplet_bending_damping_forces(
                x_a, x_b, x_c, v_a, v_b, v_c, _joint_damping(num_particles)
            )
            F[(num_particles - 2) * 3 : (num_particles - 1) * 3] += f_a
            F[(num_particles - 1) * 3 : (num_particles) * 3] += f_b

    return F


def cylinder_collision_force(
    u,
    x,
    X_WF: RigidTransform,
    collision_cylinder: Cylinder,
    params: ParticleRopeParams,
    contact_model="linear",
):
    """
    contact_model can be either "linear" or "log"
    """

    nq, num_particles = params.nq, params.num_particles
    collision_stiffness = 100000.0

    if x.dtype == object:
        F = np.zeros(nq, dtype=x.dtype)
    elif u.dtype == object:
        F = np.zeros(nq, dtype=u.dtype)
    else:
        F = np.zeros(nq, dtype=np.float64)

    cylinder_radius = collision_cylinder.radius()
    cylinder_start = X_WF.translation()
    cylinder_end = X_WF @ np.array([-collision_cylinder.length(), 0, 0])
    cylinder_axis = cylinder_end - cylinder_start
    cylinder_axis_len2 = np.dot(cylinder_axis, cylinder_axis)
    eps = 1e-12

    for i in range(num_particles):
        if i == 0:
            x_a = u[:3]
        else:
            x_a = x[(i-1)*3:(i)*3]
        x_b = x[i*3:(i+1)*3]
        link_vec = x_b - x_a
        link_len2 = np.dot(link_vec, link_vec)

        if link_len2 <= eps or cylinder_axis_len2 <= eps:
            continue

        # Closest points between rope link segment and cylinder axis segment.
        w0 = x_a - cylinder_start
        a = link_len2
        b = np.dot(link_vec, cylinder_axis)
        c = cylinder_axis_len2
        d = np.dot(link_vec, w0)
        e = np.dot(cylinder_axis, w0)
        denom = a * c - b * b

        if np.abs(denom) > eps:
            s = (b * e - c * d) / denom
        else:
            s = 0.0
        s = max(0.0, min(1.0, s))

        t = (b * s + e) / c
        t = max(0.0, min(1.0, t))

        # Reproject once after clamping.
        s = (b * t - d) / a
        s = max(0.0, min(1.0, s))
        t = (b * s + e) / c
        t = max(0.0, min(1.0, t))

        p_link = x_a + s * link_vec
        p_axis = cylinder_start + t * cylinder_axis

        radial_vec = p_link - p_axis
        radial_dist2 = np.dot(radial_vec, radial_vec)
        radial_tol2 = eps * eps
        if radial_dist2 > radial_tol2:
            radial_dist = np.sqrt(radial_dist2)
            normal = radial_vec / radial_dist
        else:
            radial_dist = 0.0
            axis_unit = cylinder_axis / np.sqrt(cylinder_axis_len2)
            trial = np.array([1.0, 0.0, 0.0])
            if np.abs(np.dot(trial, axis_unit)) > 0.9:
                trial = np.array([0.0, 0.0, 1.0])
            normal = trial - np.dot(trial, axis_unit) * axis_unit
            normal = normal / (np.sqrt(np.dot(normal, normal)) + eps)

        penetration = cylinder_radius - radial_dist
        if penetration <= 0:
            continue
        if contact_model == "linear":
            force_magnitude = collision_stiffness * penetration
        else:
            force_magnitude = collision_stiffness * (np.exp(penetration) - 1.0)
        f_link = force_magnitude * normal

        w_a = 1.0 - s
        w_b = s
        if i > 0:
            F[(i-1)*3:(i)*3] += w_a * f_link
        F[i*3:(i+1)*3] += w_b * f_link

    return F
