from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Load the data
data = pd.read_csv('your_data.csv')

# Convert timestamp columns to datetime and set as index
data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
data.set_index('datetime', inplace=True)

# Split data into training and testing sets
train = data.loc[data.index < '2024-03-01']
test = data.loc[data.index >= '2024-03-01']

def determine_p_d_q(data):
    # Perform ADF test for stationarity
    result = adfuller(data)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')
    
    # Determine d
    if result[1] < 0.05:
        print('Data is stationary.')
        d = 0
    else:
        print('Data is not stationary. Differencing is required.')
        # Try differencing the data and perform ADF test again
        differenced_data = data.diff().dropna()
        d = 1 + determine_p_d_q(differenced_data)

    # Plot ACF and PACF
    plot_acf(data, lags=50)
    plot_pacf(data, lags=50)
    plt.show()

    # Determine p and q based on ACF and PACF plots
    p = 0
    q = 0
    # Analyze ACF and PACF plots to determine p and q
    # (This part needs to be filled based on visual inspection)

    return p, d, q

# Fit ARIMA model

def train_arima_and_evaluate(train_data, test_data, order):
    # Fit ARIMA model
    model = ARIMA(train_data, order=order)
    model_fit = model.fit()

    # Make predictions
    predictions = model_fit.forecast(steps=len(test_data))[0]

    # Evaluate model (calculate RMSE)
    rmse = np.sqrt(np.mean((test_data - predictions)**2))
    plt.plot(np.arrange(len(test_data)),test_data, label='Actual')
    plt.plot(np.arrange(len(test_data)), predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.title('Temperature Prediction for March 2024')
    plt.legend()
    plt.show()
    return rmse

# Example usage
# Assuming 'train' and 'test' are pandas Series containing the training and testing data, respectively
# Assuming 'order' is a tuple representing the order of the ARIMA model (p, d, q)


# Make predictions
#predictions = model_fit.forecast(steps=len(test))

# Evaluate model (e.g., calculate RMSE)
#rmse = ((test['temperature'] - predictions)**2).mean()**0.5

# Visualize predictions vs. actual values

def arima_fitting(variable):
    p, d, q = determine_p_d_q(variable)
    rmse = train_arima_and_evaluate(train['temperature'], test['temperature'], order=(p,d,q))
    print('Optimal p:', p)
    print('Optimal d:', d)
    print('Optimal q:', q)
    print(f'Variable: {variable}'+ f'RMSE: {rmse}')
