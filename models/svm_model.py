Perculiar libraries
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

svm_model = MultiOutputRegressor(SVR(kernel='rbf'))

# Train the model
svm_model.fit(x_train, y_train)

# Predict
svm_pred = svm_model.predict(x_test)

#RMSE
def evaluate_svm_performance(y_true_data, y_pred_data, target_names):
    """
    Bulletproof evaluator for SVM MultiOutputRegressor.
    Handles potential function/data variable name conflicts.
    """
    # 1. Safety Check: Ensure we aren't indexing a function
    if callable(y_true_data):
        raise TypeError("y_test is being passed as a function. Pass the actual data variable.")

    # 2. Force conversion to 2D NumPy arrays
    # This prevents the "too many indices" error if data is 0-dim or 1-dim
    y_t = np.atleast_2d(np.array(y_true_data))
    y_p = np.atleast_2d(np.array(y_pred_data))

    # 3. Shape Alignment: MultiOutputRegressor usually returns (samples, tasks)
    # If the shapes are flipped or flattened, this ensures they match
    if y_p.shape != y_t.shape and y_p.size == y_t.size:
        y_p = y_p.reshape(y_t.shape)

    print(f"{'Task Name':<20} | {'a20-Index':<12} | {'RMSE':<10}")
    print("-" * 50)
    
    a20_list = []
    rmse_list = []

    for i, name in enumerate(target_names):
        # Slicing the guaranteed 2D arrays
        true_col = y_t[:, i]
        pred_col = y_p[:, i]
        
        # 1. Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        rmse_list.append(rmse)
        
        print(f"{name:<20} | {a20:.4f}      | {rmse:.4f}")

    print("-" * 50)
    print(f"{'Average/Mean':<20} | {np.mean(a20_list):.4f}      | {np.mean(rmse_list):.4f}")

# --- EXECUTION ---
# Ensure y_test is your actual label variable and not a function name
target_names = ['Cement', 'Furnace_Slag', 'Fly_ash', 'Water_content', 
                'Admixture_content', 'Coarse_agg', 'Fine_agg']

evaluate_svm_performance(y_test, svm_pred, target_names)


