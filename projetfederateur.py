import datetime

import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import train_test_split

import streamlit as st
import seaborn as sns
import scipy.stats as sp
import streamlit.components.v1 as components


def blackScholes(S, K, r, T, sigma, type="c"):
    "Calculate Black Scholes option price for a call/put"
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            price = S * norm.cdf(d1, 0, 1) - K * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            price = K * np.exp(-r * T) * norm.cdf(-d2, 0, 1) - S * norm.cdf(-d1, 0, 1)

        return price
    except:
        st.sidebar.error("Please confirm all option parameters!")


def monte_carlo(S, K, r, T, sigma, type="c"):
    # precompute constants
    N = 10  # number of time steps
    M = 1000  # number of simulations
    dt = T / N
    nudt = (r - 0.5 * sigma ** 2) * dt
    volsdt = sigma * np.sqrt(dt)
    lnS = np.log(S)

    # Monte Carlo Method
    Z = np.random.normal(size=(N, M))
    delta_lnSt = nudt + volsdt * Z
    lnSt = lnS + np.cumsum(delta_lnSt, axis=0)
    lnSt = np.concatenate((np.full(shape=(1, M), fill_value=lnS), lnSt))
    ST = np.exp(lnSt)

    try:
        if type == "c":
            CT = np.maximum(0, ST - K)
        elif type == "p":
            CT = np.maximum(0, K - ST)
        C0 = np.exp(-r * T) * np.sum(CT[-1]) / M
        price = np.round(C0, 2)

        return price
    except:
        st.sidebar.error("Please confirm all option parameters!")

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


def delta(strike, expiry, spot, riskfree, dividend, volatility):
    cdf = sp.norm(0, 1).cdf
    t = days_to_expiry(expiry) / 365
    if (type == "c"):
        delta = np.exp(-dividend * t) * cdf(d1(strike, expiry, spot, riskfree, dividend, volatility))
    elif (type == "p"):
        delta = np.exp(-dividend * t) * (cdf(d1(strike, expiry, spot, riskfree, dividend, volatility)) - 1)
    return delta


def gamma(strike, expiry, spot, riskfree, dividend, volatility):
    t = days_to_expiry(expiry) / 365
    gamma = (np.exp(-dividend * t) / (spot * volatility * np.sqrt(t))) * snpdf(strike, expiry, spot, riskfree, dividend,
                                                                               volatility)
    return gamma


def vega(strike, expiry, spot, riskfree, dividend, volatility):
    t = days_to_expiry(expiry) / 365
    vega = (0.01 * spot * np.exp(-dividend * t) * np.sqrt(t)) * snpdf(strike, expiry, spot, riskfree, dividend,
                                                                      volatility)
    return vega


def theta(strike, expiry, spot, riskfree, dividend, volatility, type):
    cdf = sp.norm(0, 1).cdf
    t = days_to_expiry(expiry) / 365
    if (type == "c"):
        theta = (-(((spot * volatility * np.exp(-dividend * t)) / (2 * np.sqrt(t))) * snpdf(strike, expiry, spot,
                                                                                            riskfree, dividend,
                                                                                            volatility)) - (
                             riskfree * strike * np.exp(-riskfree * t) * cdf(
                         d2(strike, expiry, spot, riskfree, dividend, volatility))) + (
                             dividend * spot * np.exp(-dividend * t) * cdf(
                         d2(strike, expiry, spot, riskfree, dividend, volatility)))) / 365
    elif (type == "p"):
        theta = (-(((spot * volatility * np.exp(-dividend * t)) / (2 * np.sqrt(t))) * snpdf(strike, expiry, spot,
                                                                                            riskfree, dividend,
                                                                                            volatility)) + (
                             riskfree * strike * np.exp(-riskfree * t) * cdf(
                         -d2(strike, expiry, spot, riskfree, dividend, volatility))) - (
                             dividend * spot * np.exp(-dividend * t) * cdf(
                         -d2(strike, expiry, spot, riskfree, dividend, volatility)))) / 365
    return theta


def rho(strike, expiry, spot, riskfree, dividend, volatility, type):
    cdf = sp.norm(0, 1).cdf
    t = days_to_expiry(expiry) / 365
    if (type == "c"):
        rho = 0.01 * strike * t * np.exp(-riskfree * t) * cdf(d2(strike, expiry, spot, riskfree, dividend, volatility))
    elif (type == "p"):
        rho = -0.01 * strike * t * np.exp(-riskfree * t) * cdf(
            -d2(strike, expiry, spot, riskfree, dividend, volatility))
    return rho

def optionDelta(S, K, r, T, sigma, type="c"):
    "Calculates option delta"
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            delta = norm.cdf(d1, 0, 1)
        elif type == "p":
            delta = -norm.cdf(-d1, 0, 1)

        return delta
    except:
        st.sidebar.error("Please confirm all option parameters!")


def optionGamma(S, K, r, T, sigma):
    "Calculates option gamma"
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        gamma = norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
        return gamma
    except:
        st.sidebar.error("Please confirm all option parameters!")


def optionTheta(S, K, r, T, sigma, type="c"):
    "Calculates option theta"
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d2, 0,
                                                                                                                1)

        elif type == "p":
            theta = - ((S * norm.pdf(d1, 0, 1) * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d2, 0,
                                                                                                                1)
        return theta / 365
    except:
        st.sidebar.error("Please confirm all option parameters!")


def optionVega(S, K, r, T, sigma):
    "Calculates option vega"
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        vega = S * np.sqrt(T) * norm.pdf(d1, 0, 1) * 0.01
        return vega
    except:
        st.sidebar.error("Please confirm all option parameters!")


def optionRho(S, K, r, T, sigma, type="c"):
    "Calculates option rho"
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    try:
        if type == "c":
            rho = 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2, 0, 1)
        elif type == "p":
            rho = 0.01 * -K * T * np.exp(-r * T) * norm.cdf(-d2, 0, 1)
        return rho
    except:
        st.sidebar.error("Please confirm all option parameters!")
option = st.selectbox(
    'Pick a Model please!',
    ('Black-Scholes', 'Monte Carlo', 'Binomial tree'))

if option == 'Black-Scholes':

    sidebar_title = st.sidebar.header("Black-Scholes Parameters")

    space = st.sidebar.header("")
    r = st.sidebar.number_input("Risk-Free Rate", min_value=0.000, max_value=1.000, step=0.001, value=0.030)
    S = st.sidebar.number_input("Underlying Asset Price", min_value=1.00, step=0.10, value=30.00)
    K = st.sidebar.number_input("Strike Price", min_value=1.00, step=0.10, value=50.00)
    days_to_expiry = st.sidebar.number_input("Time to Expiry Date (in days)", min_value=1, step=1, value=250)
    sigma = st.sidebar.number_input("Volatility", min_value=0.000, max_value=1.000, step=0.01, value=0.30)
    type_input = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    type = ""
    if type_input == "Call":
        type = "c"
    elif type_input == "Put":
        type = "p"

    T = days_to_expiry / 365

    spot_prices = [i for i in range(0, int(S) + 50 + 1)]

    prices = [blackScholes(i, K, r, T, sigma, type) for i in spot_prices]
    deltas = [optionDelta(i, K, r, T, sigma, type) for i in spot_prices]
    gammas = [optionGamma(i, K, r, T, sigma) for i in spot_prices]
    thetas = [optionTheta(i, K, r, T, sigma, type) for i in spot_prices]
    vegas = [optionVega(i, K, r, T, sigma) for i in spot_prices]
    rhos = [optionRho(i, K, r, T, sigma, type) for i in spot_prices]

    sns.set_style("whitegrid")

    fig1, ax1 = plt.subplots()
    sns.lineplot(prices)
    ax1.set_ylabel('Option Price')
    ax1.set_xlabel("Underlying Asset Price")
    ax1.set_title("Option Price")

    fig2, ax2 = plt.subplots()

    sns.lineplot(deltas)
    ax2.set_ylabel('Delta')
    ax2.set_xlabel("Underlying Asset Price")
    ax2.set_title("Delta")

    fig3, ax3 = plt.subplots()
    sns.lineplot(gammas)
    ax3.set_ylabel('Gamma')
    ax3.set_xlabel("Underlying Asset Price")
    ax3.set_title("Gamma")

    fig4, ax4 = plt.subplots()
    sns.lineplot(thetas)
    ax4.set_ylabel('Theta')
    ax4.set_xlabel("Underlying Asset Price")
    ax4.set_title("Theta")

    fig5, ax5 = plt.subplots()
    sns.lineplot(vegas)
    ax5.set_ylabel('Vega')
    ax5.set_xlabel("Underlying Asset Price")
    ax5.set_title("Vega")

    fig6, ax6 = plt.subplots()
    sns.lineplot(rhos)
    ax6.set_ylabel('Rho')
    ax6.set_xlabel("Underlying Asset Price")
    ax6.set_title("Rho")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig6.tight_layout()

    st.markdown("<h2 align='center'>Black-Scholes Option Price Calculator</h2>", unsafe_allow_html=True)

    st.header("")

    st.header("")
    st.markdown("<h3 align='center'>Option Prices and Greeks</h3>", unsafe_allow_html=True)
    st.header("")
    col1, col2, col3, col4, col5 = st.columns(5)
    col2.metric("Call Price", str(round(blackScholes(S, K, r, T, sigma, type="c"), 3)))
    col4.metric("Put Price", str(round(blackScholes(S, K, r, T, sigma, type="p"), 3)))

    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
    bcol1.metric("Delta", str(round(blackScholes(S, K, r, T, sigma, type="c"), 3)))
    bcol2.metric("Gamma", str(round(optionGamma(S, K, r, T, sigma), 3)))
    bcol3.metric("Theta", str(round(optionTheta(S, K, r, T, sigma, type="c"), 3)))
    bcol4.metric("Vega", str(round(optionVega(S, K, r, T, sigma), 3)))
    bcol5.metric("Rho", str(round(optionRho(S, K, r, T, sigma, type="c"), 3)))

    st.header("")
    st.markdown("<h3 align='center'>Visualization of the Greeks</h3>", unsafe_allow_html=True)
    st.header("")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig6)
elif option == 'Monte Carlo':
    sidebar_title = st.sidebar.header("Monte Carlo Parameters")

    space = st.sidebar.header("")
    r = st.sidebar.number_input("Risk-Free Rate", min_value=0.000, max_value=1.000, step=0.001, value=0.030)
    S = st.sidebar.number_input("Underlying Asset Price", min_value=1.00, step=0.10, value=30.00)
    K = st.sidebar.number_input("Strike Price", min_value=1.00, step=0.10, value=50.00)
    days_to_expiry = st.sidebar.number_input("Time to Expiry Date (in days)", min_value=1, step=1, value=250)
    sigma = st.sidebar.number_input("Volatility", min_value=0.000, max_value=1.000, step=0.01, value=0.30)
    type_input = st.sidebar.selectbox("Option Type", ["Call", "Put"])

    type = ""
    if type_input == "Call":
        type = "c"
    elif type_input == "Put":
        type = "p"

    T = days_to_expiry / 365

    spot_prices = [i for i in range(0, int(S) + 50 + 1)]

    prices = [monte_carlo(i, K, r, T, sigma, type) for i in spot_prices]
    deltas = [optionDelta(i, K, r, T, sigma, type) for i in spot_prices]
    gammas = [optionGamma(i, K, r, T, sigma) for i in spot_prices]
    thetas = [optionTheta(i, K, r, T, sigma, type) for i in spot_prices]
    vegas = [optionVega(i, K, r, T, sigma) for i in spot_prices]
    rhos = [optionRho(i, K, r, T, sigma, type) for i in spot_prices]

    sns.set_style("whitegrid")

    fig1, ax1 = plt.subplots()
    sns.lineplot(prices)
    ax1.set_ylabel('Option Price')
    ax1.set_xlabel("Underlying Asset Price")
    ax1.set_title("Option Price")

    fig2, ax2 = plt.subplots()

    sns.lineplot(deltas)
    ax2.set_ylabel('Delta')
    ax2.set_xlabel("Underlying Asset Price")
    ax2.set_title("Delta")

    fig3, ax3 = plt.subplots()
    sns.lineplot(gammas)
    ax3.set_ylabel('Gamma')
    ax3.set_xlabel("Underlying Asset Price")
    ax3.set_title("Gamma")

    fig4, ax4 = plt.subplots()
    sns.lineplot(thetas)
    ax4.set_ylabel('Theta')
    ax4.set_xlabel("Underlying Asset Price")
    ax4.set_title("Theta")

    fig5, ax5 = plt.subplots()
    sns.lineplot(vegas)
    ax5.set_ylabel('Vega')
    ax5.set_xlabel("Underlying Asset Price")
    ax5.set_title("Vega")

    fig6, ax6 = plt.subplots()
    sns.lineplot(rhos)
    ax6.set_ylabel('Rho')
    ax6.set_xlabel("Underlying Asset Price")
    ax6.set_title("Rho")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig6.tight_layout()

    st.markdown("<h2 align='center'>Monte Carlo Option Price Calculator</h2>", unsafe_allow_html=True)

    st.header("")

    st.header("")
    st.markdown("<h3 align='center'>Option Prices and Greeks</h3>", unsafe_allow_html=True)
    st.header("")
    col1, col2, col3, col4, col5 = st.columns(5)
    col2.metric("Call Price", str(round(monte_carlo(S, K, r, T, sigma, type="c"), 3)))
    col4.metric("Put Price", str(round(monte_carlo(S, K, r, T, sigma, type="p"), 3)))

    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
    bcol1.metric("Delta", str(round(monte_carlo(S, K, r, T, sigma, type="c"), 3)))
    bcol2.metric("Gamma", str(round(optionGamma(S, K, r, T, sigma), 3)))
    bcol3.metric("Theta", str(round(optionTheta(S, K, r, T, sigma, type="c"), 3)))
    bcol4.metric("Vega", str(round(optionVega(S, K, r, T, sigma), 3)))
    bcol5.metric("Rho", str(round(optionRho(S, K, r, T, sigma, type="c"), 3)))

    st.header("")
    st.markdown("<h3 align='center'>Visualization of the Greeks</h3>", unsafe_allow_html=True)
    st.header("")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig6)
elif option=='Binomial tree':
    sidebar_title = st.sidebar.header("Binomial Parameters")

    steps = st.sidebar.number_input(label="Steps", min_value=1, value=4)
    simulations = 0
    type_input = st.sidebar.selectbox(label="Type", options=["Call", "Put"])
    spot = st.sidebar.number_input(label="Spot", min_value=0.0, value=1.1850, step=0.0005, format="%.4f")
    strike = st.sidebar.number_input(label="Strike", min_value=0.0, value=1.1650, step=0.0005, format="%.4f")
    expiry = st.sidebar.date_input(label="Expiry", value=datetime.date(2022, 12, 31))
    style = st.sidebar.selectbox(label="Style", options=["EU", "US"], index=0)
    print(expiry)
    st.sidebar.header("Market")
    riskfree = st.sidebar.slider(label="Riskfree Rate", min_value=0.0, max_value=1.0, value=.05, step=0.01)
    dividend = st.sidebar.number_input(label="Dividend", min_value=0.0, value=0.0)
    volatility = st.sidebar.slider(label="Implied Volatility", min_value=0.0, max_value=1.0, value=0.2, step=0.0005)

    type = ""
    if type_input == "Call":
        type = "c"
    elif type_input == "Put":
        type = "p"
    spot_prices = [i for i in range(0, int(spot) + 50 + 1)]
    prices = [price(strike, expiry, i, riskfree, dividend, volatility, style, type, steps=4) for i in spot_prices]
    deltas = [delta(strike, expiry, i, riskfree, dividend, volatility) for i in spot_prices]
    gammas = [gamma(strike, expiry, i, riskfree, dividend, volatility) for i in spot_prices]
    thetas = [theta(strike, expiry, i, riskfree, dividend, volatility, type) for i in spot_prices]
    vegas = [vega(strike, expiry, i, riskfree, dividend, volatility) for i in spot_prices]
    rhos = [rho(strike, expiry, i, riskfree, dividend, volatility, type) for i in spot_prices]
    # print(prices)
    sns.set_style("whitegrid")

    fig1, ax1 = plt.subplots()
    sns.lineplot(prices)
    ax1.set_ylabel('Option Price')
    ax1.set_xlabel("Underlying Asset Price")
    ax1.set_title("Option Price")

    fig2, ax2 = plt.subplots()
    sns.lineplot(deltas)
    ax2.set_ylabel('Delta')
    ax2.set_xlabel("Underlying Asset Price")
    ax2.set_title("Delta")

    fig3, ax3 = plt.subplots()
    sns.lineplot(gammas)
    ax3.set_ylabel('Gamma')
    ax3.set_xlabel("Underlying Asset Price")
    ax3.set_title("Gamma")

    fig4, ax4 = plt.subplots()
    sns.lineplot(thetas)
    ax4.set_ylabel('Theta')
    ax4.set_xlabel("Underlying Asset Price")
    ax4.set_title("Theta")

    fig5, ax5 = plt.subplots()
    sns.lineplot(vegas)
    ax5.set_ylabel('Vega')
    ax5.set_xlabel("Underlying Asset Price")
    ax5.set_title("Vega")

    fig6, ax6 = plt.subplots()
    sns.lineplot(rhos)
    ax6.set_ylabel('Rho')
    ax6.set_xlabel("Underlying Asset Price")
    ax6.set_title("Rho")

    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    fig5.tight_layout()
    fig6.tight_layout()

    st.markdown("<h2 align='center'>Binomial Option Price Calculator</h2>", unsafe_allow_html=True)

    st.header("")

    st.header("")
    st.markdown("<h3 align='center'>Option Prices and Greeks</h3>", unsafe_allow_html=True)
    st.header("")
    col1, col2, col3, col4, col5 = st.columns(5)
    col2.metric("Price",
                str(round(price(strike, expiry, spot, riskfree, dividend, volatility, style, type="c", steps=4), 3)))
    col4.metric("Payoff", str(round(get_payoff(strike, spot), 3)))

    bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns(5)
    bcol1.metric("Delta",
                 str(round(price(strike, expiry, spot, riskfree, dividend, volatility, style, type="c", steps=4), 3)))
    bcol2.metric("Gamma", str(round(gamma(strike, expiry, spot, riskfree, dividend, volatility), 3)))
    bcol3.metric("Theta", str(round(theta(strike, expiry, spot, riskfree, dividend, volatility, type="c"), 3)))
    bcol4.metric("Vega", str(round(vega(strike, expiry, spot, riskfree, dividend, volatility), 3)))
    bcol5.metric("Rho", str(round(rho(strike, expiry, spot, riskfree, dividend, volatility, type="c"), 3)))

    st.header("")
    st.markdown("<h3 align='center'>Visualization of the Greeks</h3>", unsafe_allow_html=True)
    st.header("")
    st.pyplot(fig1)
    st.pyplot(fig2)
    st.pyplot(fig3)
    st.pyplot(fig4)
    st.pyplot(fig5)
    st.pyplot(fig6)

def RF()
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

elif option=='Machine Learning':
    sidebar_title = st.sidebar.header("Random forest")
    st.markdown("<h2 align='center'>Random forest Option Price Calculator</h2>", unsafe_allow_html=True)

    st.header("")

    st.header("")
    st.markdown("<h3 align='center'>Option Price</h3>", unsafe_allow_html=True)
    st.header("")
    col1, col2 = st.columns(2)
    col2.metric("Option Price", str(round(RF()

