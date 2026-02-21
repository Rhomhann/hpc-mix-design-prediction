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
    
    rmse_scores = []

    for i, name in enumerate(target_names):
        # Now indexing is safe because y_t is a guaranteed NumPy array
        true_col = y_t[:, i]
        pred_col = y_p[:, i]
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(true_col, pred_col))
        rmse_scores.append(rmse)
        

    print("-" * 50)
    print(f"{np.mean(rmse_scores):.4f}")

# --- EXECUTION ---
target_names = ['Cement', 'Furnace_Slag', 'Fly_ash', 'Water_content', 
                'Admixture_content', 'Coarse_agg', 'Fine_agg']

# IMPORTANT: Check if y_test is your actual label array. 
# If it prints a warning, you need to find where your labels are stored.
evaluate_linear_reg_performance(y_test, linear_reg_pred, target_names)


