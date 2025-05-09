# utils.py
import numpy as np

# --- Quaternion Operations ---
def quaternion_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    res = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(res)
    if norm > 1e-9:
        res /= norm
    else:
        res = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) # Safety for zero norm
    return res

def quaternion_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

def quaternion_to_rotation_matrix(q): # Body to World (R_wb)
    q_norm = np.asarray(q, dtype=float)
    norm_val = np.linalg.norm(q_norm)
    if norm_val < 1e-9:
        q_norm = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q_norm = q_norm / norm_val

    w, x, y, z = q_norm
    
    mat = np.array([
        [w*w + x*x - y*y - z*z,   2*(x*y - w*z),          2*(x*z + w*y)],
        [2*(x*y + w*z),           w*w - x*x + y*y - z*z,  2*(y*z - w*x)],
        [2*(x*z - w*y),           2*(y*z + w*x),          w*w - x*x - y*y + z*z]
    ], dtype=float)
    
    if not np.all(np.isfinite(mat)):
        return np.identity(3, dtype=float)
    return mat

# Used by Madgwick for f_g and f_m (R_bw = R_wb.T)
# (q represents R_wb, body to world)
def R_bw_from_q(q_arr): # World to Body (R_bw = R_wb^T)
    qw, qx, qy, qz = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
    R_bw = np.array([
        [qw**2 + qx**2 - qy**2 - qz**2,   2*(qx*qy + qw*qz),             2*(qx*qz - qw*qy)],
        [2*(qx*qy - qw*qz),               qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz + qw*qx)],
        [2*(qx*qz + qw*qy),               2*(qy*qz - qw*qx),             qw**2 - qx**2 - qy**2 + qz**2]
    ], dtype=float)
    return R_bw

def np_encoder(obj):
    """Custom encoder for JSON to handle numpy arrays and floats."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")