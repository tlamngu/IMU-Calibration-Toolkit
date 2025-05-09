# ellipsoid_fitting.py
import numpy as np
from scipy.optimize import least_squares
import traceback
import logging

logger = logging.getLogger(__name__)

def ellipsoid_residuals(params, data):
    cx, cy, cz = params[0:3]
    M = np.array([
        [params[3], params[4], params[5]],
        [params[4], params[6], params[7]],
        [params[5], params[7], params[8]]
    ], dtype=float)
    center = np.array([cx, cy, cz], dtype=float)
    epsilon_reg = 1e-9
    M_reg = M + np.identity(3) * epsilon_reg
    residuals = np.zeros(len(data))
    for i, point in enumerate(data):
        diff = point - center
        val = diff.T @ M_reg @ diff
        residuals[i] = val - 1.0
    return residuals

def fit_ellipsoid(mag_data_np):
    if mag_data_np.shape[0] < 20:
        logger.warning("Not enough mag data for fitting (need at least 20 points). Received %d points.", mag_data_np.shape[0])
        return None, None
    mag_data_np = np.asarray(mag_data_np, dtype=float)
    if not np.all(np.isfinite(mag_data_np)):
         logger.warning("Non-finite values in mag data for fitting. Cannot proceed.")
         return None, None

    initial_center = np.mean(mag_data_np, axis=0)
    diffs = mag_data_np - initial_center
    avg_radius_sq = np.mean(np.sum(diffs**2, axis=1))
    if avg_radius_sq < 1e-6 or not np.isfinite(avg_radius_sq): 
        logger.warning(f"Average radius squared is very small or non-finite ({avg_radius_sq}). Using default 1.0 for initialization.")
        avg_radius_sq = 1.0
    
    inv_avg_radius_sq = 1.0 / avg_radius_sq # avg_radius_sq is now guaranteed > 0

    initial_params = np.array([
        initial_center[0], initial_center[1], initial_center[2],
        inv_avg_radius_sq, 0, 0, inv_avg_radius_sq, 0, inv_avg_radius_sq
    ], dtype=float)
    if not np.all(np.isfinite(initial_params)):
        logger.warning("Non-finite initial params for ellipsoid fitting due to input data issues. Using default initial parameters.")
        initial_params = np.array([0,0,0, 1,0,0, 1,0, 1], dtype=float)
    else:
        logger.info(f"Ellipsoid fitting initial params: {initial_params.tolist()}")

    try:
        logger.info("Starting least_squares optimization for ellipsoid fitting.")
        result = least_squares(ellipsoid_residuals, initial_params, args=(mag_data_np,),
                               method='lm', verbose=0, ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=3000)
        logger.info(f"least_squares optimization finished. Success: {result.success}, Status: {result.status}, Message: {result.message}")
        
        if not result.success or result.status <= 0:
            logger.warning(f"Ellipsoid fitting optimization failed. Status: {result.status}, Msg: {result.message}. Falling back to min/max method for hard iron.")
            min_vals = np.min(mag_data_np, axis=0); max_vals = np.max(mag_data_np, axis=0)
            hard_iron_offset = (max_vals + min_vals) / 2.0
            soft_iron_matrix = np.identity(3, dtype=float)
            return hard_iron_offset, soft_iron_matrix

        fitted_params = result.x
        logger.info(f"Ellipsoid fitting fitted_params: {fitted_params.tolist()}")
        hard_iron_offset = fitted_params[0:3]
        M = np.array([
            [fitted_params[3], fitted_params[4], fitted_params[5]],
            [fitted_params[4], fitted_params[6], fitted_params[7]],
            [fitted_params[5], fitted_params[7], fitted_params[8]]
        ], dtype=float)
        
        if not np.all(np.isfinite(M)):
            logger.warning("Resulting matrix M from fitting contains non-finite values. Using identity for soft iron.")
            soft_iron_matrix = np.identity(3, dtype=float)
            return hard_iron_offset, soft_iron_matrix
            
        try:
            eigenvalues = np.linalg.eigvalsh(M)
            logger.info(f"Eigenvalues of M: {eigenvalues.tolist()}")
            if np.any(eigenvalues <= 1e-9) or not np.all(np.isfinite(eigenvalues)):
                logger.warning(f"Degenerate ellipsoid matrix M (eigenvalues: {eigenvalues.tolist()}). Using identity for soft iron.")
                soft_iron_matrix = np.identity(3, dtype=float)
            else:
                M_reg = M + np.identity(3) * 1e-9 # Regularize for Cholesky
                logger.info("Attempting Cholesky decomposition for soft iron matrix.")
                L = np.linalg.cholesky(M_reg)
                soft_iron_matrix = L.T 
                if not np.all(np.isfinite(soft_iron_matrix)):
                    logger.warning("Non-finite soft iron matrix after Cholesky decomposition. Using identity.")
                    soft_iron_matrix = np.identity(3, dtype=float)
                else:
                    logger.info("Cholesky decomposition successful. Soft iron matrix calculated.")
        except np.linalg.LinAlgError as e:
            logger.warning(f"LinAlgError during soft iron matrix calculation (e.g., Cholesky failed): {e}. Using identity for soft iron.")
            soft_iron_matrix = np.identity(3, dtype=float)

        if not np.all(np.isfinite(hard_iron_offset)): 
            logger.warning("Non-finite hard iron offset after fitting. Resetting to zero.")
            hard_iron_offset = np.zeros(3)
        # Soft iron matrix already checked for finite values in its calculation block
            
        return hard_iron_offset, soft_iron_matrix
        
    except Exception as e:
        logger.error(f"Unexpected error during ellipsoid fitting process: {e}", exc_info=True)
        # traceback.print_exc() # exc_info=True in logger.error handles this
        try:
            min_vals = np.min(mag_data_np, axis=0); max_vals = np.max(mag_data_np, axis=0)
            hard_iron_offset = (max_vals + min_vals) / 2.0
            if not np.all(np.isfinite(hard_iron_offset)): 
                logger.warning("Fallback hard iron offset non-finite, using zeros.")
                hard_iron_offset = np.zeros(3)
        except Exception as fallback_e:
            logger.error(f"Error in fallback hard iron calculation: {fallback_e}. Using zeros.")
            hard_iron_offset = np.zeros(3)
        soft_iron_matrix = np.identity(3, dtype=float)
        logger.info("Returning default hard_iron_offset and identity soft_iron_matrix due to exception.")
        return hard_iron_offset, soft_iron_matrix