#Perculiar library
import xgboost as xgb

#Model setup and training
xgb_linear = xgb.XGBRegressor(objective = "reg:squarederror" , booster = "gblinear")
xgb_linear.fit(x_train, y_train)
xgb_predict = xgb_linear.predict(x_test)

# RMSE
ef evaluate_xgb_performance(y_true_data, y_pred_data, target_names):
    # 1. Verification Step: Check if we accidentally passed a function
    if callable(y_true_data):
        raise TypeError("y_test is a function/method. Ensure you are passing the variable containing your data labels.")

    # 2. Force conversion to Numpy Arrays
    # We use np.array(list(...)) as a fallback to ensure 0-d arrays aren't created
    y_t = np.atleast_2d(np.array(y_true_data))
    y_p = np.atleast_2d(np.array(y_pred_data))

    # If XGBoost flattened the output, reshape it back to (Samples, Tasks)
    if y_p.shape[0] != y_t.shape[0] and y_p.size == y_t.size:
        y_p = y_p.reshape(y_t.shape)

    print(f"{'RMSE':<10}")
    #print("-" * 50)
  
    rmse_scores = []

    for i, name in enumerate(target_names):
        # Indexing guaranteed 2D numpy arrays
        true_col = y_t[:, i]
        pred_col = y_p[:, i]
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        rmse_scores.append(rmse)
        
        print(f"{name:<20} | {a20:.4f}      | {rmse:.4f}")

    print("-" * 50)
    print(f"{np.mean(rmse_scores):.4f}")

# --- EXECUTION ---
# Before running, check your variable names! 
# If 'y_test' is your data, use it. If 'y_test' is a function, find your data variable.
evaluate_xgb_performance(y_test, xgb_predict_1, target_names)
