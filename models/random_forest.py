#Perculair library
from sklearn.ensemble import RandomForestRegressor
#Model setup and training
rf_model = RandomForestRegressor(n_estimators=1000, random_state=42)
rf_model.fit(x_train, y_train)
rf_predict = rf_model.predict(x_test)
