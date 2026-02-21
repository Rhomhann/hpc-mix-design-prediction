#Perculair library
from sklearn.ensemble import RandomForestRegressor
#Model setup and training
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(x_train, y_train)
rf_predict = rf_model.predict(x_test)

#RMSE 
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_rf_performance(y_true_data, y_pred_data, target_names):
    """
    Bulletproof evaluator for Random Forest Regressor.
    Ensures data is correctly formatted as 2D arrays before indexing.
    """
    # 1. Safety Check: Stop if y_test is being passed as a function
    if callable(y_true_data):
        raise TypeError("y_test is being passed as a function. Ensure you pass the variable containing your data.")

    # 2. Force conversion to 2D NumPy arrays
    # This fixes the '0-dimensional array' and 'not subscriptable' errors
    y_t = np.atleast_2d(np.array(y_true_data))
    y_p = np.atleast_2d(np.array(y_pred_data))

    # 3. Shape Alignment: Ensure pred matches true (Samples, Tasks)
    if y_p.shape != y_t.shape and y_p.size == y_t.size:
        y_p = y_p.reshape(y_t.shape)

    print(f"{'Task Name':<20} | {'a20-Index':<12} | {'RMSE':<10}")
    print("-" * 50)
    
    a20_scores = []
    rmse_scores = []

    for i, name in enumerate(target_names):
        # Indexing guaranteed 2D arrays
        true_col = y_t[:, i]
        pred_col = y_p[:, i]
        
        # 1. Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        
        # 2. Calculate a20-index (±20% range)
        # Use a mask to avoid division by zero
        mask = (true_col != 0)
        ratios = pred_col[mask] / true_col[mask]
        a20 = np.mean((ratios >= 0.8) & (ratios <= 1.2))
        
        a20_scores.append(a20)
        rmse_scores.append(rmse)
        
        print(f"{name:<20} | {a20:.4f}      | {rmse:.4f}")

    print("-" * 50)
    print(f"{'Average (Mean)':<20} | {np.mean(a20_scores):.4f}      | {np.mean(rmse_scores):.4f}")

# --- EXECUTION ---
target_names = [
    'Cement', 'Furnace_Slag', 'Fly_ash', 'Water_content', 
    'Admixture_content', 'Coarse_agg', 'Fine_agg'
]

# Run the evaluation for Random Forest
evaluate_rf_performance(y_test, rf_predict, target_names)

#MAE
import numpy as np
from sklearn.metrics import mean_absolute_error

def evaluate_rf_mae(y_true_data, y_pred_data, target_names):
    """
    Calculates the Mean Absolute Error (MAE) for each task of the Random Forest model.
    MAE represents the average deviation in original units (kg/m³).
    """
    # 1. Safety Check: Verify y_test isn't a function
    if callable(y_true_data):
        raise TypeError("y_test is being passed as a function. Pass the actual data variable.")

    # 2. Force conversion to 2D NumPy arrays
    y_t = np.atleast_2d(np.array(y_true_data))
    y_p = np.atleast_2d(np.array(y_pred_data))

    # 3. Shape Alignment
    if y_p.shape != y_t.shape and y_p.size == y_t.size:
        y_p = y_p.reshape(y_t.shape)

    print(f"\n{'Task Name':<20} | {'MAE Score (kg/m³)':<15}")
    print("-" * 40)
    
    mae_scores = []

    for i, name in enumerate(target_names):
        # Indexing the numpy arrays for the specific task
        true_col = y_t[:, i]
        pred_col = y_p[:, i]
        
        # Calculate MAE
        mae = mean_absolute_error(true_col, pred_col)
        mae_scores.append(mae)
        
        print(f"{name:<20} | {mae:.4f}")

    print("-" * 40)
    print(f"{'Average (Mean)':<20} | {np.mean(mae_scores):.4f}")

# --- EXECUTION ---
target_names = [
    'Cement', 'Furnace_Slag', 'Fly_ash', 'Water_content', 
    'Admixture_content', 'Coarse_agg', 'Fine_agg'
]

evaluate_rf_mae(y_test, rf_predict, target_names)
