import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch
#MAE for all comparison models
def evaluate_mae_smart(y_true_scaled, y_pred, target_cols, scaler, model_label="Model"):
    """
    Unified Smart Evaluator:
    1. Unscales y_test (since it's currently 0-1).
    2. Checks if y_pred is scaled or raw.
    3. Calculates MAE in kg/m³.
    """
    # Step 1: Ensure y_test is a numpy array and unscale it
    if torch.is_tensor(y_true_scaled):
        y_true_np = y_true_scaled.cpu().numpy()
    else:
        y_true_np = np.array(y_true_scaled)
    
    y_true_raw = scaler.inverse_transform(y_true_np)

    # Step 2: Handle Predictions
    if torch.is_tensor(y_pred):
        y_pred_np = y_pred.cpu().numpy()
    else:
        y_pred_np = np.array(y_pred)
    
    # RESHAPE CHECK: Ensure y_pred is 2D (Samples, 7)
    if y_pred_np.ndim == 1:
        y_pred_np = y_pred_np.reshape(y_true_np.shape)

    # THE SMART CHECK:
    # If values are < 10, they are Scaled. If > 10, they are Raw.
    if np.max(y_pred_np) < 10:
        print(f">>> {model_label}: Scaled predictions detected. Unscaling...")
        y_pred_raw = scaler.inverse_transform(y_pred_np)
    else:
        print(f">>> {model_label}: Raw predictions detected. Calculating directly...")
        y_pred_raw = y_pred_np

    # Step 3: Compute and Print Results
    print(f"\n--- {model_label} MAE Results ---")
    print(f"{'Task Name':<20} | {'MAE (kg/m³)':<15}")
    print("-" * 45)

    results_list = []
    for i, name in enumerate(target_cols):
        mae = mean_absolute_error(y_true_raw[:, i], y_pred_raw[:, i])
        results_list.append({"Task": name, "MAE": mae})
        print(f"{name:<20} | {mae:.4f}")

    mean_mae = np.mean([r['MAE'] for r in results_list])
    print("-" * 45)
    print(f"{'Average Mean':<20} | {mean_mae:.4f}\n")
    
    return pd.DataFrame(results_list)

mae_rf_df = evaluate_mae_smart(y_test, rf_predict, target_names, scaler2, "Random Forest")
mae_svm_df = evaluate_mae_smart(y_test, svm_pred, target_names, scaler2, "SVM")
mae_xgb_df = evaluate_mae_smart(y_test, xgb_predict_1, target_names, scaler2, "XGBoost")
mae_linear_df = evaluate_mae_smart(y_test, linear_reg_pred, target_names, scaler2, "Linear Regression")

