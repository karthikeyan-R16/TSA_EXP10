# Exp.no: 10   IMPLEMENTATION OF SARIMA MODEL
## Developed by : Karthikeyan R
## Reg No: 212222240045
## Date : 

### AIM:
To implement SARIMA model using python.
### ALGORITHM:
1. Explore the dataset
2. Check for stationarity of time series
3. Determine SARIMA models parameters p, q
4. Fit the SARIMA model
5. Make time series predictions and Auto-fit the SARIMA model
6. Evaluate model predictions
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/content/rainfall.csv')
data['date'] = pd.to_datetime(data['date'])

# Assuming 'rainfall' is the column you want to analyze, 
# replace 'temp' with 'rainfall' in the following lines:
plt.plot(data['date'], data['rainfall'])  # Changed 'temp' to 'rainfall'
plt.xlabel('Date')
plt.ylabel('Rainfall')  # Changed 'Temperature' to 'Rainfall'
plt.title('Rainfall Time Series')  # Changed 'Temperature' to 'Rainfall'
plt.show()

def check_stationarity(timeseries):
    # Perform Dickey-Fuller test
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

check_stationarity(data['rainfall'])  # Changed 'temp' to 'rainfall'

plot_acf(data['rainfall'])  # Changed 'temp' to 'rainfall'
plt.show()
plot_pacf(data['rainfall'])  # Changed 'temp' to 'rainfall'
plt.show()

sarima_model = SARIMAX(data['rainfall'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))  # Changed 'temp' to 'rainfall'
sarima_result = sarima_model.fit()

train_size = int(len(data) * 0.8)
train, test = data['rainfall'][:train_size], data['rainfall'][train_size:]  # Changed 'temp' to 'rainfall'

sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

predictions = sarima_result.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)

mse = mean_squared_error(test, predictions)
rmse = np.sqrt(mse)
print('RMSE:', rmse)

plt.plot(test.index, test, label='Actual')
plt.plot(test.index, predictions, color='red', label='Predicted')
plt.xlabel('Date')
plt.ylabel('Rainfall')  # Changed 'Temperature' to 'Rainfall'
plt.title('SARIMA Model Predictions')
plt.legend()
plt.show()
```
### OUTPUT:

![image](https://github.com/user-attachments/assets/6f85be51-032a-40d1-a5a7-959e5dee48ec)


![image](https://github.com/user-attachments/assets/8bd6828c-21fe-44f4-9770-7b4715d6f29e)


![image](https://github.com/user-attachments/assets/f5cabdfd-aff9-4164-8aa1-61ac1238ccd2)

![image](https://github.com/user-attachments/assets/1cf70c37-54fd-499f-bff2-9e66f3e42a4d)





### RESULT:
Thus the program run successfully based on the SARIMA model.
