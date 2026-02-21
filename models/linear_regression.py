#Perculair libraries 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Model definition and training
base_model = LinearRegression()
multi_output_model = MultiOutputRegressor(base_model)
multi_output_model.fit(x_train, y_train)
linear_reg_pred = multi_output_model.predict(x_test)

#RMSE 

def evaluate_linear_reg_performance(y_true_data, y_pred_data, target_names):
    # 1. Verification: If y_true_data is a function, we attempt to find the data
    if callable(y_true_data):
        print("!!! WARNING: The variable passed as y_true is a FUNCTION, not data.")
        print("Please check your notebook for a line like 'def y_test():' or similar.")
        return

    # 2. Force conversion to 2D NumPy arrays to enable [:, i] indexing
    y_t = np.atleast_2d(np.array(y_true_data))
    y_p = np.atleast_2d(np.array(y_pred_data))

    # 3. Align shapes if necessary
    if y_p.shape != y_t.shape and y_p.size == y_t.size:
        y_p = y_p.reshape(y_t.shape)

    print(f"{'Task Name':<20} | {'a20-Index':<12} | {'RMSE':<10}")
    print("-" * 50)
    
    a20_scores = []
    rmse_scores = []

    for i, name in enumerate(target_names):
        # Now indexing is safe because y_t is a guaranteed NumPy array
        true_col = y_t[:, i]
        pred_col = y_p[:, i]
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        
        # Calculate a20-index
        mask = (true_col != 0)
        ratios = pred_col[mask] / true_col[mask]
        a20 = np.mean((ratios >= 0.8) & (ratios <= 1.2))
        
        a20_scores.append(a20)
        rmse_scores.append(rmse)
        
        print(f"{name:<20} | {a20:.4f}      | {rmse:.4f}")

    print("-" * 50)
    print(f"{'Average (Mean)':<20} | {np.mean(a20_scores):.4f}      | {np.mean(rmse_scores):.4f}")

# --- EXECUTION ---
target_names = ['Cement', 'Furnace_Slag', 'Fly_ash', 'Water_content', 
                'Admixture_content', 'Coarse_agg', 'Fine_agg']

# IMPORTANT: Check if y_test is your actual label array. 
# If it prints a warning, you need to find where your labels are stored.
evaluate_linear_reg_performance(y_test, linear_reg_pred, target_names)

#MAE
def evaluate_mtl_mae_smart(model, test_loader, target_cols, scaler):
    """
    Collects MTL predictions, applies the Smart Check for scaling,
    and calculates MAE in kg/m³.
    """
    model.eval()
    all_preds = []
    all_trues = []

    # 1. Collect data from DataLoader
    with torch.no_grad():
        for x_batch, y_batch_dict in test_loader:
            preds_dict = model(x_batch)
            
            # Stack the 7 tasks into a single row per sample
            batch_preds = torch.stack([preds_dict[name].flatten() for name in target_cols], dim=1)
            batch_trues = torch.stack([y_batch_dict[name].flatten() for name in target_cols], dim=1)
            
            all_preds.append(batch_preds.cpu().numpy())
            all_trues.append(batch_trues.cpu().numpy())

    # 2. Convert to large 2D arrays (Samples, 7)
    y_pred_np = np.vstack(all_preds)
    y_true_np = np.vstack(all_trues)

    # 3. Always unscale y_true (y_test is scaled)
    y_true_raw = scaler.inverse_transform(y_true_np)

    # 4. THE SMART CHECK for Predictions
    if np.max(y_pred_np) < 10:
        print(">>> MTL Model: Scaled predictions detected. Unscaling to kg/m³...")
        y_pred_raw = scaler.inverse_transform(y_pred_np)
    else:
        print(">>> MTL Model: Raw predictions detected. Calculating directly...")
        y_pred_raw = y_pred_np

    # 5. Compute MAE per Task
    print(f"\n{'Output Task':<20} | {'MAE (kg/m³)':<15}")
    print("-" * 45)

    results_data = []
    for i, name in enumerate(target_cols):
        mae = mean_absolute_error(y_true_raw[:, i], y_pred_raw[:, i])
        results_data.append({"Task": name, "MAE": mae})
        print(f"{name:<20} | {mae:.4f}")

    mean_mae = np.mean([r['MAE'] for r in results_data])
    print("-" * 45)
    print(f"{'MTL System Mean':<20} | {mean_mae:.4f}\n")
    
    return pd.DataFrame(results_data)

# --- EXECUTION ---
mae_mtl_df = evaluate_mtl_mae_smart(model, test_loader, target_names, scaler2)
