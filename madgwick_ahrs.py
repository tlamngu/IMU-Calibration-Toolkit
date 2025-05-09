# madgwick_ahrs.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Re-use quaternion helpers (can be moved to a common utility file later)
def quaternion_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    res = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(res)
    if norm > 1e-9: res /= norm # Ensure unit quaternion
    else: res = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) # Safety for zero norm
    return res

def quaternion_conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)

# Helper: quaternion_to_rotation_matrix_T (R_bw = R_wb.T)
# Used for f_g and f_m consistent with Madgwick's paper
# (q represents R_wb, body to world)
def R_bw_from_q(q_arr): # World to Body (R_bw = R_wb^T)
    qw, qx, qy, qz = q_arr[0], q_arr[1], q_arr[2], q_arr[3]
    R_bw = np.array([
        [qw**2 + qx**2 - qy**2 - qz**2,   2*(qx*qy + qw*qz),             2*(qx*qz - qw*qy)],
        [2*(qx*qy - qw*qz),               qw**2 - qx**2 + qy**2 - qz**2, 2*(qy*qz + qw*qx)],
        [2*(qx*qz + qw*qy),               2*(qy*qz - qw*qx),             qw**2 - qx**2 - qy**2 + qz**2]
    ], dtype=float)
    return R_bw


class MadgwickAHRS:
    def __init__(self, settings_manager, sample_period=1/100.0, beta=0.1):
        self.settings = settings_manager
        self.sample_period = sample_period 
        self.beta = beta 
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) # w, x, y, z

        mag_ref_ned = np.array(self.settings.get("mag_ref_vector_ned"), dtype=float)
        norm_mag_ref = np.linalg.norm(mag_ref_ned)
        if norm_mag_ref > 1e-6:
            self.mag_ref_normalized_ned = mag_ref_ned / norm_mag_ref
        else:
            logger.warning("MadgwickAHRS: Magnetometer reference vector norm is near zero. Using default [1,0,0].")
            self.mag_ref_normalized_ned = np.array([1.0, 0.0, 0.0], dtype=float) 
        logger.info(f"MadgwickAHRS initialized with sample_period: {self.sample_period}, beta: {self.beta}, mag_ref_normalized_ned: {self.mag_ref_normalized_ned.tolist()}")


    def update_sample_period(self, dt):
        if dt > 0:
            self.sample_period = dt
        # else:
            # logger.debug(f"MadgwickAHRS: Received non-positive dt: {dt}, sample_period not updated.")


    def reset_state(self, q_init=None):
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q_init is None else np.array(q_init, dtype=float)
        norm_q = np.linalg.norm(self.quaternion)
        if norm_q > 1e-9: self.quaternion /= norm_q
        else: self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        logger.info(f"MadgwickAHRS state reset. Initial quaternion: {self.quaternion.tolist()}")


    def update(self, gyro_rad_s, accel_m_s2_calibrated, mag_uT_calibrated):
        q = self.quaternion # q_w, q_x, q_y, q_z
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        gyro = np.asarray(gyro_rad_s, dtype=float)
        accel = np.asarray(accel_m_s2_calibrated, dtype=float)
        mag = np.asarray(mag_uT_calibrated, dtype=float)

        if np.linalg.norm(accel) < 1e-6: 
            # logger.debug("MadgwickAHRS: Accelerometer norm near zero, skipping update.")
            return 
        accel_norm = accel / np.linalg.norm(accel)

        use_mag = False
        if np.linalg.norm(mag) > 1e-6:
            mag_norm = mag / np.linalg.norm(mag)
            use_mag = True
        # else:
            # logger.debug("MadgwickAHRS: Magnetometer norm near zero, not using in this update.")


        # Quaternion rate of change from gyroscope
        # dq/dt = 0.5 * q * omega_quat (where omega_quat = [0, gx, gy, gz])
        q_dot_omega = 0.5 * quaternion_multiply(q, np.array([0, gyro[0], gyro[1], gyro[2]], dtype=float))

        # --- Accelerometer Gradient Descent Part ---
        # Objective function: f_g = R_bw * [0,0,1]^T (predicted gravity in body frame)
        f_g = np.array([
            2*(qx*qz - qw*qy), 
            2*(qy*qz + qw*qx), 
            qw**2 - qx**2 - qy**2 + qz**2 
        ], dtype=float)
        
        # Jacobian J_g = df_g / dq
        J_g = 2 * np.array([ 
            [-qy,  qz, -qw, qx], 
            [ qx,  qw,  qz, qy], 
            [ qw, -qx, -qy, qz]  
        ], dtype=float) 
        
        error_accel = f_g - accel_norm
        gradient_step_accel = J_g.T @ error_accel
        
        q_dot_gradient = np.zeros(4, dtype=float)
        if np.linalg.norm(gradient_step_accel) > 1e-9 : 
             q_dot_gradient = gradient_step_accel / np.linalg.norm(gradient_step_accel)


        if use_mag:
            # --- Magnetometer Gradient Descent Part ---
            b_world = self.mag_ref_normalized_ned
            bx, by, bz = b_world[0], b_world[1], b_world[2]

            _R_bw = R_bw_from_q(q) 
            f_m = _R_bw @ b_world
            
            # Jacobian J_m = df_m / dq.
            # J_m_corrected[row_idx, col_idx] = d(f_m[row_idx]) / d(q[col_idx])
            # q = [qw, qx, qy, qz]
            J_m_corrected = np.array([
                [2*(qw*bx + qz*by - qy*bz), 2*(qx*bx + qy*by + qz*bz), 2*(-qy*bx + qx*by - qw*bz), 2*(-qz*bx + qw*by + qx*bz)],
                [2*(-qz*bx + qw*by + qx*bz),2*(qy*bx - qx*by + qw*bz), 2*(qx*bx + qy*by + qz*bz),  2*(-qw*bx - qz*by + qy*bz)], # J_m[1,1] (d(f_m[1])/dqx) corrected, J_m[1,3] (d(f_m[1])/dqz) corrected
                [2*(qy*bx - qx*by + qw*bz), 2*(qz*bx - qw*by - qx*bz), 2*(qw*bx + qz*by - qy*bz),  2*(qx*bx + qy*by + qz*bz)]  # J_m[2,1] (d(f_m[2])/dqx) corrected, J_m[2,2] (d(f_m[2])/dqy) CORRECTED from original
            ], dtype=float)
            # Original J_m_corrected[1,1] was 2*(qy*bx - qx*by - qw*bz) - Corrected to +qw*bz
            # Original J_m_corrected[1,3] was 2*(qw*bx - qz*by + qy*bz) - Corrected (this was in comment already)
            # Original J_m_corrected[2,1] was 2*(qz*bx + qw*by - qx*bz) - Corrected to -qw*by
            # Original J_m_corrected[2,2] was 2*(qw*bx - qy*by - qz*bz) - Corrected to 2*(qw*bx + qz*by - qy*bz)


            error_mag = f_m - mag_norm
            gradient_step_mag = J_m_corrected.T @ error_mag
            
            if np.linalg.norm(gradient_step_mag) > 1e-9:
                q_dot_gradient += gradient_step_mag / np.linalg.norm(gradient_step_mag)
        
        if use_mag and np.linalg.norm(q_dot_gradient) > 1e-9 :
             q_dot_gradient = q_dot_gradient / np.linalg.norm(q_dot_gradient)
        
        q_dot = q_dot_omega - self.beta * q_dot_gradient
        
        self.quaternion = self.quaternion + q_dot * self.sample_period
        norm_q = np.linalg.norm(self.quaternion)
        if norm_q > 1e-9: self.quaternion /= norm_q
        else: self.quaternion = np.array([1.0,0.0,0.0,0.0], dtype=float)


    def get_orientation_quaternion(self):
        return self.quaternion.copy()

    def get_gyro_bias(self): 
        return np.zeros(3, dtype=float)