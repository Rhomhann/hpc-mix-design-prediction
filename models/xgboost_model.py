#Perculair libraries-refer to deep_learning.py for more libraries
import xgboost as xgb

#Model setup and training
xgb_linear = xgb.XGBRegressor(objective = "reg:squarederror" , booster = "gblinear")
xgb_linear.fit(x_train, y_train)
xgb_predict = xgb_linear.predict(x_test)
