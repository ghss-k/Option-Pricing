# -*- coding: utf-8 -*-
"""OptionPricing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SMbPVuSiiCg4jWky5khWKCVrD7MWgC57
"""

import pandas as pd
import numpy as np

"""# Data Generation



"""

# Define the number of options to generate
num_options = 2000

# Define the stock price range
stock_price_range = (10, 500)

# Define the maturity range
maturity_range = (1, 365*3)

# Define the risk-free rate range
risk_free_rate_range = (0.01, 0.03)

# Define the volatility range
volatility_range = (0.05, 0.9)

# Moneyeness : (0.97, 1.03)--> OTM, (1.04, 66)--> ATM, (0.02, 0.96)--> ITM
s_k_ranges = [(0.97, 1.03), (1.04, 66), (0.02, 0.96)]

# Generate the option data
df = []

for s_k_range in s_k_ranges:
 for i in range(num_options):
    underlying_price = np.random.uniform(*stock_price_range)
    sk = np.random.uniform(*s_k_range)
    strike = underlying_price/sk
    maturity_days = np.random.randint(*maturity_range)
    expiration = pd.Timestamp('today') + pd.Timedelta(days=maturity_days)
    risk_free_rate = np.random.uniform(*risk_free_rate_range)
    volatility = np.random.uniform(*volatility_range)
    option_type = np.random.choice(['Call', 'Put'])
    df.append({
        'Option Type': option_type,
        'Underlying Price': underlying_price,
        'Strike Price': strike,
        'Expiration': expiration,
        'Time to Expiration': maturity_days,
        'Risk-Free Rate': risk_free_rate,
        'Volatility': volatility,
        'S/K': sk
    })
data = pd.DataFrame(df)
data

"""Then the price of each option sample is calculated using four methods:
1. BS: Black-Scholes Merton

2. MC: Monte Carlo simulation

3. BT3: Binomial Tree with three periods

4. BT4: Binomial Tree with four periods
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def bsm_option_price(s, k, t, r, sigma, option_type):
    d1 = (np.log(s / k) + (r + 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s / k) + (r - 0.5 * sigma ** 2) * t) / (sigma * np.sqrt(t))
    if option_type == 'call':
        option_price = (s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    else:
        option_price = (k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))
    return option_price


def monte_carlo_option_price(s, k, t, r, sigma, option_type, n_simulations):
    dt = t / 252
    sim_price_paths = np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_simulations, int(t / dt)))
    sim_price_paths = np.cumprod(np.insert(sim_price_paths, 0, s, axis=1), axis=1)
    sim_price_paths = sim_price_paths[:, -1]
    if option_type == 'call':
        option_price = np.mean(np.maximum(sim_price_paths - k, 0)) * np.exp(-r * t)
    else:
        option_price = np.mean(np.maximum(k - sim_price_paths, 0)) * np.exp(-r * t)
    return option_price


def  binomial_tree_option_price(S, K, T, r, sigma, option_type,number_of_time_steps): 
        """Calculates price for call option according to the Binomial formula."""
        # Delta t, up and down factors
        dT = T / number_of_time_steps                             
        u = np.exp(sigma * np.sqrt(dT))                 
        d = 1.0 / u                                    

        # Price vector initialization
        V = np.zeros(number_of_time_steps + 1)                       

        # Underlying asset prices at different time points
        S_T = np.array( [(S * u**j * d**(number_of_time_steps - j)) for j in range(number_of_time_steps + 1)])

        a = np.exp(r * dT)      # risk free compounded return
        p = (a - d) / (u - d)        # risk neutral up probability
        q = 1.0 - p                  # risk neutral down probability   
        if option_type == 'call':
          V[:] = np.maximum(S_T - K, 0.0)
        else:
          V[:] = np.maximum(K - S_T, 0.0)
    
        # Overriding option price 
        for i in range(number_of_time_steps - 1, -1, -1):
            V[:-1] = np.exp(-r * dT) * (p * V[1:] + q * V[:-1]) 

        return V[0]
 


def add_prices(df, n_simulations=1000):
    df['BS'] = df.apply(lambda x: bsm_option_price(x['Underlying Price'],	x['Strike Price'], x['Time to Expiration'],x['Risk-Free Rate'],	x['Volatility'], x["Option Type"]), axis=1)
    df['MC'] = df.apply(lambda x: monte_carlo_option_price(x['Underlying Price'],	x['Strike Price'], x['Time to Expiration'],x['Risk-Free Rate'],	x['Volatility'], x["Option Type"], n_simulations), axis=1)
    df['BT3'] = df.apply(lambda x: binomial_tree_option_price(x['Underlying Price'],	x['Strike Price'], x['Time to Expiration'],x['Risk-Free Rate'],	x['Volatility'], x["Option Type"],3), axis=1)
    df['BT4'] = df.apply(lambda x:  binomial_tree_option_price(x['Underlying Price'],	x['Strike Price'], x['Time to Expiration'],x['Risk-Free Rate'],	x['Volatility'], x["Option Type"],4), axis=1)
    return df

data = add_prices(data)

data

# Calculate the MSE, RMSE, and MAE between the prices calculated using classical methods

def cal(model1,model2):
  mse = np.mean((data[model1] - data[model2])**2)
  rmse = np.sqrt(mse)
  mae = np.mean(np.abs(data[model1]  - data[model2]))
  df=pd.DataFrame(columns=[model1+ '-'+ model2])
  return df

"""The combinations that presented the lowest errors are respectively (in this order): BS vs BT4,
BS vs BT3 and BS vs MC. This lead as to choose Black-Scholes for price computation.

# Data preparation
"""

data=data.drop(columns=["MC", "BT3",'BT4'])

data.isnull().sum()

import matplotlib.pyplot as plt
# Plot the boxplot
plt.boxplot(data['BS'])

# Add labels and title
plt.xlabel("BS")
plt.ylabel("Price")
plt.title("Option Prices Boxplot")

# Show the plot
plt.show()

#The interquartile range
def remove_outliers(df, target_variable):
    q1, q3 = np.percentile(target_variable, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered_target_variable = [x for x in target_variable if x >= lower_bound and x <= upper_bound]
    filtered_indices = [i for i, x in enumerate(target_variable) if x >= lower_bound and x <= upper_bound]
    df_filtered = df.loc[filtered_indices, :]
    return df_filtered

data=remove_outliers(data,data["BS"])

import seaborn as sns
# Count the number of 'Call' and 'Put' options for each option class
option_counts = data.groupby([ 'Option Type']).size().reset_index(name='counts')
# Plot the number of options for each option type and option class
sns.barplot(x=option_counts['Option Type'], y=option_counts['counts'])
plt.title("Number of options for each option type and option class")
plt.xlabel("Option Class")
plt.ylabel("Counts")
plt.show()
option_counts

"""Puts are only 100 samples away from calls. Which is only 2.24% of our dataset. We can say
that our data is fairly balanced.

# Data Normalisation
"""

data['BS'] = data['BS'] / data['Strike Price']

data=data.drop(columns=['Strike Price','Underlying Price','Expiration'])

data

"""# Categorical data

"""

# Convert categorical column to numerical values
data['Option Type'] = data['Option Type'].astype('category').cat.codes
data

"""# Correlation"""

# Compute correlation matrix
corr = data.corr()

# Plot correlation graph using heatmap
sns.heatmap(corr, annot=True)

# Show plot
plt.show()

"""The figure above shows that there is low correlation between the variables and therefore no
problem for the modeling.

# Modeling
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split

# Split your dataset into features (X) and target variable (y)
X = data.drop('BS', axis=1)
y = data['BS']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

"""Random Forest


"""

# Loop over different number of estimators
for n_estimators in [10, 50, 100, 500,1000]:
    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=n_estimators)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Use the model to make predictions on the testing data
    y_pred = model.predict(X_test)
    
    # Calculate the MAE, MSE, and RMSE on the testing data
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print the results
    print(f'Number of estimators: {n_estimators}')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

"""The best results were obtained with 1000 estimators

Ridge regression
"""

from sklearn.linear_model import Ridge
# Loop over different values of alpha
for alpha in [0.1, 1, 10, 100]:
    # Initialize the Ridge Regression model
    model = Ridge(alpha=alpha)
    
    # Fit the model to the training data
    model.fit(X_train, y_train)
    
    # Use the model to make predictions on the testing data
    y_pred = model.predict(X_test)
    
    # Calculate the MAE, MSE, and RMSE on the testing data
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # Print the results
    print(f'Alpha: {alpha}')
    print(f'MAE: {mae}')
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')

"""The best results were obtained with Alpha 100

LSTM
"""

X_train

X_train.shape

from keras.models import Sequential
from keras.layers import LSTM, Dense
# Initialize the LSTM network
       
# Convert the x_train and y_train to numpy arrays 
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape the data
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (X_train.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Use the model to make predictions on the testing data
y_pred = model.predict(X_test)

# Convert the predicted values back to their original scale
y_pred = y_pred.flatten()

# Calculate the MAE, MSE, and RMSE on the testing data
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Print the results
print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')