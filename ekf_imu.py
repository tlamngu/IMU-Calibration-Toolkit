# ekf_imu.py
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Re-use quaternion helpers
def quaternion_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=float); q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    res = np.array([w, x, y, z], dtype=float)
    norm = np.linalg.norm(res)
    if norm > 1e-9: res /= norm
    else: res = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) 
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
        logger.warning("Quaternion to rotation matrix resulted in non-finite values. Returning identity.")
        return np.identity(3, dtype=float)
    return mat

class EKF_IMU:
    def __init__(self, settings_manager):
        self.settings = settings_manager
        self.gravity_mag = self.settings.get("gravity")
        
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) 
        self.gyro_bias = np.array([0.0, 0.0, 0.0], dtype=float)
        
        self.P = np.eye(7, dtype=float)
        initial_orientation_cov = self.settings.get("ekf_initial_orientation_cov")
        initial_gyro_bias_cov = self.settings.get("ekf_initial_gyro_bias_cov")

        if isinstance(initial_orientation_cov, (int, float)):
            self.P[0:4, 0:4] *= initial_orientation_cov
        else:
            logger.warning(f"EKF: ekf_initial_orientation_cov is not a number: {initial_orientation_cov}. Using default P for orientation.")

        if isinstance(initial_gyro_bias_cov, (int, float)):
            self.P[4:7, 4:7] *= initial_gyro_bias_cov
        else:
             logger.warning(f"EKF: ekf_initial_gyro_bias_cov is not a number: {initial_gyro_bias_cov}. Using default P for gyro bias.")

        self.mag_ref_vector_ned = np.array(self.settings.get("mag_ref_vector_ned"), dtype=float)
        self._update_noise_matrices() 

        self.enable_prediction = True
        logger.info(f"EKF_IMU initialized. Gravity: {self.gravity_mag}, Mag Ref NED: {self.mag_ref_vector_ned.tolist()}")
        logger.info(f"EKF_IMU initial P (diag): {np.diag(self.P).tolist()}")


    def set_prediction_enabled(self, enabled: bool):
        self.enable_prediction = enabled
        if not enabled:
            logger.info("EKF prediction step DISABLED.")
        else:
            logger.info("EKF prediction step ENABLED.")


    def _update_noise_matrices(self):
        q_gyro_var = self.settings.get("ekf_q_gyro_noise") 
        q_bias_var = self.settings.get("ekf_q_gyro_bias_noise") 
        
        if not isinstance(q_gyro_var, (int, float)): 
            logger.warning(f"EKF: ekf_q_gyro_noise invalid type ({type(q_gyro_var)}), using default 0.001")
            q_gyro_var = 0.001 
        if not isinstance(q_bias_var, (int, float)): 
            logger.warning(f"EKF: ekf_q_gyro_bias_noise invalid type ({type(q_bias_var)}), using default 0.00001")
            q_bias_var = 0.00001 

        self.Q_proc_continuous = np.diag([q_gyro_var, q_gyro_var, q_gyro_var, 
                                          q_bias_var, q_bias_var, q_bias_var])

        r_accel_var = self.settings.get("ekf_r_accel_noise")
        r_mag_var = self.settings.get("ekf_r_mag_noise")

        if not isinstance(r_accel_var, (int, float)): 
            logger.warning(f"EKF: ekf_r_accel_noise invalid type ({type(r_accel_var)}), using default 0.1")
            r_accel_var = 0.1 
        if not isinstance(r_mag_var, (int, float)): 
            logger.warning(f"EKF: ekf_r_mag_noise invalid type ({type(r_mag_var)}), using default 0.5")
            r_mag_var = 0.5 

        self.R_accel = np.eye(3, dtype=float) * r_accel_var
        self.R_mag = np.eye(3, dtype=float) * r_mag_var
        # logger.debug(f"EKF noise matrices updated. Q_proc_continuous (diag): {np.diag(self.Q_proc_continuous)}, R_accel (diag): {np.diag(self.R_accel)}, R_mag (diag): {np.diag(self.R_mag)}")


    def reset_state(self, q_init=None, bias_init=None):
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float) if q_init is None else np.array(q_init, dtype=float)
        norm_q_val = np.linalg.norm(self.q)
        if norm_q_val > 1e-9: self.q /= norm_q_val
        else: self.q = np.array([1.0,0.0,0.0,0.0], dtype=float)

        self.gyro_bias = np.zeros(3, dtype=float) if bias_init is None else np.array(bias_init, dtype=float)
        
        self.P = np.eye(7, dtype=float)
        initial_orientation_cov = self.settings.get("ekf_initial_orientation_cov")
        initial_gyro_bias_cov = self.settings.get("ekf_initial_gyro_bias_cov")

        if isinstance(initial_orientation_cov, (int, float)):
            self.P[0:4, 0:4] *= initial_orientation_cov
        if isinstance(initial_gyro_bias_cov, (int, float)):
            self.P[4:7, 4:7] *= initial_gyro_bias_cov

        self._update_noise_matrices() 
        self.enable_prediction = True 
        logger.info(f"EKF state and covariance reset. q: {self.q.tolist()}, bias: {self.gyro_bias.tolist()}. Prediction enabled.")
        logger.info(f"EKF P after reset (diag): {np.diag(self.P).tolist()}")


    def predict(self, gyro_rad_s, dt):
        if not self.enable_prediction:
            # logger.debug("EKF prediction skipped as it's disabled.")
            return 

        if dt <= 0: 
            # logger.debug(f"EKF predict: Invalid dt ({dt}), skipping prediction.")
            return

        # self._update_noise_matrices() # Noise matrices are updated at init, reset, and if settings change explicitly

        gyro_corrected = np.asarray(gyro_rad_s, dtype=float) - self.gyro_bias
        
        delta_angle = gyro_corrected * dt
        angle_norm = np.linalg.norm(delta_angle)
        
        delta_q_body = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if angle_norm > 1e-9: 
            axis = delta_angle / angle_norm
            dq_w = np.cos(angle_norm / 2.0)
            dq_xyz = axis * np.sin(angle_norm / 2.0)
            delta_q_body = np.array([dq_w, dq_xyz[0], dq_xyz[1], dq_xyz[2]], dtype=float)
        
        q_prev = self.q.copy() 
        self.q = quaternion_multiply(q_prev, delta_q_body) 
        
        F = np.eye(7, dtype=float)
        
        dq0, dq1, dq2, dq3 = delta_q_body
        F_qq = np.array([
            [dq0, -dq1, -dq2, -dq3],
            [dq1,  dq0,  dq3, -dq2], 
            [dq2, -dq3,  dq0,  dq1], 
            [dq3,  dq2, -dq1,  dq0]  
        ], dtype=float)
        F[0:4, 0:4] = F_qq
        
        _qw, _qx, _qy, _qz = q_prev 
        
        _Xi_q_prev_standard = 0.5 * np.array([
            [-_qx, -_qy, -_qz],  
            [ _qw, -_qz,  _qy],  
            [ _qz,  _qw, -_qx],  
            [-_qy,  _qx,  _qw]   
        ], dtype=float)
        
        F_qb_term = -_Xi_q_prev_standard * dt 
        F[0:4, 4:7] = F_qb_term
        
        G = np.zeros((7, 6), dtype=float)
        G[0:4, 0:3] = _Xi_q_prev_standard 
        G[4:7, 3:6] = np.eye(3, dtype=float) 
        
        Q_noise_variances_dt = self.Q_proc_continuous * dt 
        Q_d = G @ Q_noise_variances_dt @ G.T
                                                     
        self.P = F @ self.P @ F.T + Q_d
        self.P = 0.5 * (self.P + self.P.T) 


    def update_accel(self, accel_m_s2_calibrated):
        if not np.all(np.isfinite(accel_m_s2_calibrated)): 
            logger.debug("EKF update_accel: Non-finite accelerometer data received.")
            return
        accel_norm_val = np.linalg.norm(accel_m_s2_calibrated)
        if accel_norm_val < 1e-6: 
            # logger.debug("EKF update_accel: Accelerometer norm near zero, skipping update.")
            return 
        
        R_wb = quaternion_to_rotation_matrix(self.q) 
        g_world_ref = np.array([0, 0, 1.0], dtype=float) 
        accel_pred_dir = R_wb.T @ g_world_ref 
        accel_measured_dir = accel_m_s2_calibrated / accel_norm_val

        H_k = np.zeros((3, 7), dtype=float) 
        qw, qx, qy, qz = self.q
        g_norm_val = 1.0 
        
        H_k[0,0] = -2*qy*g_norm_val       
        H_k[0,1] =  2*qz*g_norm_val       
        H_k[0,2] = -2*qw*g_norm_val       
        H_k[0,3] =  2*qx*g_norm_val       

        H_k[1,0] =  2*qx*g_norm_val       
        H_k[1,1] =  2*qw*g_norm_val       
        H_k[1,2] =  2*qz*g_norm_val       
        H_k[1,3] =  2*qy*g_norm_val       
        
        H_k[2,0] =  2*qw*g_norm_val       
        H_k[2,1] = -2*qx*g_norm_val       
        H_k[2,2] = -2*qy*g_norm_val       
        H_k[2,3] =  2*qz*g_norm_val       

        S_k = H_k @ self.P @ H_k.T + self.R_accel 
        try:
            K_k = self.P @ H_k.T @ np.linalg.inv(S_k)
        except np.linalg.LinAlgError:
            logger.warning("EKF update_accel: S_k matrix is singular, cannot compute Kalman gain.")
            return

        innovation = accel_measured_dir - accel_pred_dir 
        delta_x = K_k @ innovation 
        
        delta_q_update = delta_x[0:4]
        self.q = self.q + delta_q_update 
        norm_q_val = np.linalg.norm(self.q)
        if norm_q_val > 1e-9: self.q /= norm_q_val
        else: self.q = np.array([1.0,0.0,0.0,0.0], dtype=float)
        
        self.gyro_bias += delta_x[4:7]

        I_KH = np.eye(7, dtype=float) - K_k @ H_k
        self.P = I_KH @ self.P @ I_KH.T + K_k @ self.R_accel @ K_k.T 
        self.P = 0.5 * (self.P + self.P.T) 

    def update_mag(self, mag_uT_calibrated): 
        if not np.all(np.isfinite(mag_uT_calibrated)): 
            logger.debug("EKF update_mag: Non-finite magnetometer data received.")
            return
        mag_norm_val = np.linalg.norm(mag_uT_calibrated)
        if mag_norm_val < 1e-6: 
            # logger.debug("EKF update_mag: Magnetometer norm near zero, skipping update.")
            return

        mag_measured_dir = mag_uT_calibrated / mag_norm_val

        mag_ref_norm_val = np.linalg.norm(self.mag_ref_vector_ned)
        if mag_ref_norm_val < 1e-6: 
            logger.warning("EKF update_mag: Magnetometer reference vector NED is near zero. Skipping mag update.")
            return 
        mag_ref_dir_ned = self.mag_ref_vector_ned / mag_ref_norm_val

        R_wb = quaternion_to_rotation_matrix(self.q)
        mag_pred_dir = R_wb.T @ mag_ref_dir_ned 

        H_k = np.zeros((3, 7), dtype=float) 
        qw, qx, qy, qz = self.q
        mx_d, my_d, mz_d = mag_ref_dir_ned 

        H_k[0,0] = 2*qw*mx_d + 2*qz*my_d - 2*qy*mz_d
        H_k[0,1] = 2*qx*mx_d + 2*qy*my_d + 2*qz*mz_d
        H_k[0,2] = -2*qy*mx_d + 2*qx*my_d - 2*qw*mz_d
        H_k[0,3] = -2*qz*mx_d + 2*qw*my_d + 2*qx*mz_d
        
        H_k[1,0] = -2*qz*mx_d + 2*qw*my_d + 2*qx*mz_d
        H_k[1,1] =  2*qy*mx_d - 2*qx*my_d - 2*qw*mz_d 
        H_k[1,2] =  2*qx*mx_d + 2*qy*my_d + 2*qz*mz_d
        H_k[1,3] =  2*(-qw*mx_d - qz*my_d + qy*mz_d) # CORRECTED: d(f_m_y)/dqz (sign of qw*mx_d and qz*my_d)

        H_k[2,0] =  2*qy*mx_d - 2*qx*my_d + 2*qw*mz_d
        H_k[2,1] =  2*qz*mx_d - 2*qw*my_d - 2*qx*mz_d # CORRECTED: sign of qw*my_d term to be -
        H_k[2,2] =  2*qw*mx_d + 2*qz*my_d - 2*qy*mz_d
        H_k[2,3] =  2*qx*mx_d + 2*qy*my_d + 2*qz*mz_d
        
        # My verification of H_k[2,1] = d(f_m_z)/dqx:
        # R_bw[2,0] = 2*(qxqz + qwqy); d/dqx = 2qz
        # R_bw[2,1] = 2*(qyqz - qwqx); d/dqx = -2qw
        # R_bw[2,2] = qw^2 - qx^2 - qy^2 + qz^2; d/dqx = -2qx
        # d(f_m_z)/dqx = 2*qz*mx_d - 2*qw*my_d - 2*qx*mz_d = 2*(qz*mx_d - qw*my_d - qx*mz_d). This correction is good.

        S_k_mag = H_k @ self.P @ H_k.T + self.R_mag
        try:
            K_k_mag = self.P @ H_k.T @ np.linalg.inv(S_k_mag)
        except np.linalg.LinAlgError:
            logger.warning("EKF update_mag: S_k_mag matrix is singular, cannot compute Kalman gain.")
            return

        innovation_mag = mag_measured_dir - mag_pred_dir 
        delta_x_mag = K_k_mag @ innovation_mag
        
        delta_q_update_mag = delta_x_mag[0:4]
        self.q = self.q + delta_q_update_mag 
        norm_q_val = np.linalg.norm(self.q)
        if norm_q_val > 1e-9: self.q /= norm_q_val
        else: self.q = np.array([1.0,0.0,0.0,0.0], dtype=float)

        self.gyro_bias += delta_x_mag[4:7]

        I_KH_mag = np.eye(7, dtype=float) - K_k_mag @ H_k
        self.P = I_KH_mag @ self.P @ I_KH_mag.T + K_k_mag @ self.R_mag @ K_k_mag.T 
        self.P = 0.5 * (self.P + self.P.T)

    def get_orientation_quaternion(self):
        return self.q.copy()

    def get_gyro_bias(self):
        return self.gyro_bias.copy()