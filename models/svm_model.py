Perculiar libraries
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

svm_model = MultiOutputRegressor(SVR(kernel='rbf'))

# Train the model
svm_model.fit(x_train, y_train)

# Predict
svm_pred = svm_model.predict(x_test)
