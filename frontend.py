import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

df = pd.read_csv('cleaned_dataset_tk.csv', index_col=0, parse_dates=True)

def run_model():
    state = state_var.get()
    model_type = model_var.get()
    
    if state not in df.columns:
        messagebox.showerror("Error", "Invalid state selected")
        return
    
    if model_type == "ARIMA":
        run_arima(state)
    elif model_type == "Polynomial Regression":
        run_polynomial_regression(state)
    elif model_type == "Random Forest":
        run_random_forest(state)
    else:
        messagebox.showerror("Error", "Invalid model selected")

def run_arima(state):
    train_size = int(len(df) * 0.8)
    train, test = df[state][:train_size], df[state][train_size:]

    model = ARIMA(train, order=(5, 1, 0))
    model_fit = model.fit()

    predictions = model_fit.forecast(steps=len(test))
    test.index = predictions.index

   
    plt.figure(figsize=(10, 6))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('ARIMA: Predicted vs Actual')
    plt.legend()
    plt.show()
def run_polynomial_regression(state):
    train_size = int(len(df) * 0.8)
    train, test = df[state][:train_size], df[state][train_size:]

    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train.values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    y_test = test.values

    poly = PolynomialFeatures(degree=2)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    predictions = model.predict(X_test_poly)

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, y_test, label='Actual')
    plt.plot(test.index, predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Polynomial Regression: Predicted vs Actual')
    plt.legend()
    plt.show()

def run_random_forest(state):
    train_size = int(len(df) * 0.8)
    train, test = df[state][:train_size], df[state][train_size:]

    X_train = np.arange(len(train)).reshape(-1, 1)
    y_train = train.values
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)
    y_test = test.values

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    plt.figure(figsize=(10, 6))
    plt.plot(test.index, y_test, label='Actual')
    plt.plot(test.index, predictions, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.title('Random Forest: Predicted vs Actual')
    plt.legend()
    plt.show()


def search_date():
    date = date_entry.get()
    try:
        date_data = df.loc[date]
        messagebox.showinfo("Date Search", f"Data for {date}:\n{date_data}")
    except KeyError:
        messagebox.showerror("Error", "Date not found in the dataset")


def predict_date():
    date = date_entry.get()
    state = state_var.get()
    model_type = model_var.get()
    
    if state not in df.columns:
        messagebox.showerror("Error", "Invalid state selected")
        return
    
    try:
        date_index = pd.to_datetime(date)
    except ValueError:
        messagebox.showerror("Error", "Invalid date format")
        return
    
    if model_type == "ARIMA":
        train_size = int(len(df) * 0.8)
        train = df[state][:train_size]

        model = ARIMA(train, order=(5, 1, 0))
        model_fit = model.fit()

        predictions = model_fit.forecast(steps=(date_index - train.index[-1]).days)
        prediction = predictions[-1]
    elif model_type == "Polynomial Regression":
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[state].values

        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        date_index = (date_index - df.index[0]).days
        prediction = model.predict(poly.transform([[date_index]]))[0]
    elif model_type == "Random Forest":
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[state].values

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        date_index = (date_index - df.index[0]).days
        prediction = model.predict([[date_index]])[0]
    else:
        messagebox.showerror("Error", "Invalid model selected")
        return
    
    messagebox.showinfo("Prediction", f"Predicted electricity usage for {state} on {date}: {prediction}")

root = tk.Tk()
root.title("Model Runner")

state_var = tk.StringVar()
model_var = tk.StringVar()

ttk.Label(root, text="State:").grid(row=0, column=0)
state_combobox = ttk.Combobox(root, textvariable=state_var, values=list(df.columns))
state_combobox.grid(row=0, column=1)

ttk.Label(root, text="Model:").grid(row=1, column=0)
model_combobox = ttk.Combobox(root, textvariable=model_var, values=["ARIMA", "Polynomial Regression", "Random Forest"])
model_combobox.grid(row=1, column=1)

run_button = ttk.Button(root, text="Run Model", command=run_model)
run_button.grid(row=2, column=0, columnspan=2)

ttk.Label(root, text="Date (YYYY-MM-DD):").grid(row=3, column=0)
date_entry = ttk.Entry(root)
date_entry.grid(row=3, column=1)

search_button = ttk.Button(root, text="Search Date", command=search_date)
search_button.grid(row=4, column=0, columnspan=2)

predict_button = ttk.Button(root, text="Predict Usage", command=predict_date)
predict_button.grid(row=5, column=0, columnspan=2)

root.mainloop()