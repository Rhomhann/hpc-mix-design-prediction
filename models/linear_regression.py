#Perculair libraries-refer to deep_learning.py for more libraries

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Model definition and training
base_model = LinearRegression()
multi_output_model = MultiOutputRegressor(base_model)
multi_output_model.fit(x_train, y_train)
linear_reg_pred = multi_output_model.predict(x_test)
