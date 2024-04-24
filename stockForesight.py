import streamlit as st # type: ignore
import yfinance as yf # type: ignore
import pandas as pd # type: ignore
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split # type: ignore
import plotly.graph_objects as go # type: ignore
import re
import time

st.set_page_config(
    page_title="stockForesight | TDS",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded" 
)
def get_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        history = stock.history(period="Max")
        if history.empty:
            print(f"Error: Invalid ticker {ticker}")
            return False
        st.session_state["df"] = pd.DataFrame(history)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def clean_data(df):
    try:
        print(df)
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)

        x_date_unix = np.array([date.timestamp() for date in df.index])
        x_open = np.array(df['Open'])

        st.session_state["x_open"] = x_open
        st.session_state["x_date_unix"] = x_date_unix
        st.session_state["y_high"] = np.array(df['High'])
        st.session_state["y_low"] = np.array(df['Low'])

        st.session_state["x"] = np.column_stack((x_date_unix, x_open))
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False
    
def plot3d():
    x_date_unix = st.session_state.x_date_unix
    x_open = st.session_state.x_open
    y_high = st.session_state.y_high
    y_low = st.session_state.y_low

    fig_high = go.Figure(data=[go.Scatter3d(
        x=x_date_unix,
        y=x_open,
        z=y_high,
        mode='lines',
        line=dict(
            color='green',
        ),
        name='High'
    )])

    fig_low = go.Figure(data=[go.Scatter3d(
        x=x_date_unix,
        y=x_open,
        z=y_low,
        mode='lines',
        line=dict(
            color='red',
        ),
        name='Low'
    )])

    layout = dict(
        scene=dict(
            xaxis=dict(title='Unix Timestamp'),
            yaxis=dict(title='Open'),
            zaxis=dict(title='High'),
            camera=dict(
                eye=dict(x=1.2, y=-3, z=1)
            )
        ),
        
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    layout1 = dict(
        scene=dict(
            xaxis=dict(title='Unix Timestamp'),
            yaxis=dict(title='Open'),
            zaxis=dict(title='Low'),
            camera=dict(
                eye=dict(x=1.2, y=-3, z=1)
            )
        ),
        
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig_high.update_layout(title='High Prices', **layout)
    fig_low.update_layout(title='Low Prices', **layout1)
    with st.chat_message("assistant"):
        st.write("Inclusive influence of date and opening price on the high and low stock prices.")
        st.plotly_chart(fig_high, use_container_width=True)
        st.plotly_chart(fig_low, use_container_width=True)

    
def train_model():
    x = st.session_state["x"]
    y_high = st.session_state["y_high"]
    y_low = st.session_state["y_low"]
    
    class LinearRegression:

        def __init__(self):
            self.coef_ = None
            self.intercept_ = None

        def fit(self,X_train,y_train):
            X_train = np.insert(X_train,0,1,axis=1)

            betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
            self.intercept_ = betas[0]
            self.coef_ = betas[1:]

        def predict(self,X_test):
            y_pred = np.dot(X_test,self.coef_) + self.intercept_
            return y_pred

        def score(self, X_test, y_test):
            y_pred = self.predict(X_test)
            mean_y = np.mean(y_test)
            ss_res = np.sum((y_test - y_pred)**2)
            ss_tot = np.sum((y_test - mean_y)**2)
            r_squared = 1 - (ss_res / ss_tot)
            return r_squared

    x_train, st.session_state["x_test"], y_high_train, st.session_state["y_high_test"], y_low_train, st.session_state["y_low_test"] = train_test_split(x, y_high, y_low, test_size=0.2)
    st.session_state["model_high"] = LinearRegression()
    st.session_state["model_low"] = LinearRegression()
    st.session_state.model_high.fit(x_train, y_high_train)
    st.session_state.model_low.fit(x_train, y_low_train)

def model_evaluation(): 
    score_high = st.session_state.model_high.score(st.session_state.x_test, st.session_state.y_high_test)
    score_low = st.session_state.model_low.score(st.session_state.x_test, st.session_state.y_low_test)

    return score_high, score_low
    

def predict(date, open):
    model_high = st.session_state["model_high"]
    model_low = st.session_state["model_low"]
    dd_mm_yyyy = date
    open_price = open
    date = datetime.strptime(dd_mm_yyyy, "%d_%m_%Y").timestamp()
    test_data = np.array([date, open_price]).reshape(1, -1)
    high = model_high.predict(test_data)
    low = model_low.predict(test_data)
    return f"Predicted high: {high[0]}\nPredicted low: {low[0]}"


if __name__ == "__main__":
    st.title("StockForesight")
    st.write("Introducing stockForesight: a cutting-edge project utilizing multiple linear regression models to predict high and low stock prices.\n")


    with st.sidebar:
        st.title("Other Projects:")
        st.write("Groot:")
        st.write("An enhanced AI marvel by Team Vision. Expanding beyond GPT-3's limits, Groot leverages RAG to dynamically learn from external files, revolutionizing personalized learning experiences.")
        st.link_button("View Project", "https://github.com/thedevyashsaini/Groot")

    with st.spinner("Loading..."):
        time.sleep(2)

    st.markdown("#")

    if "ticker" not in st.session_state:
        st.session_state["ticker"] = "META"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if ticker := st.text_input(label="Stock Symbol"):
        st.session_state.ticker = ticker
        if get_data(ticker):
            with st.chat_message("assistant"):
                st.write(f"Got it! Here the running trends of {ticker}.")
                st.write(f"Date Vs high:")
                st.line_chart(st.session_state.df["High"], color="#00ff00")
                st.write(f"Date Vs high:")
                st.line_chart(st.session_state.df["Low"], color="#ff0000")

            if clean_data(st.session_state.df):
                try:
                    plot3d()
                    train_model()
                    with st.chat_message("assistant"):
                        st.write(f"Regression model trained!")
                        st.write(f"Model evaluation initiated...")
                        high, low = model_evaluation() 
                        st.write(f"High prediction score: {high}")
                        st.write(f"Low prediction score: {low}")
                    with st.chat_message("assistant"):
                        st.write(f"Now you can start prediction stock price, just give input for date and stock opening price.")
                        st.write(f"Format: dd_mm_yyyy openingprice.")
                        st.write(f"Example: 24_04_2024 100.2")

                    if prompt := st.chat_input():
                        with st.chat_message("user"):
                            st.markdown(prompt)

                        flag = False
                        date = None
                        open = None

                        try:
                            a = prompt.split()
                            if len(a) != 2:
                                raise Exception("Invalid input format. (dd_mm_yyyy price)")
                            date, open = a
                            open = float(open)
                            pattern = r"^(0[1-9]|[1-2][0-9]|3[0-1])_(0[1-9]|1[0-2])_(\d{4})$"
                            if not re.match(pattern, date):
                                raise Exception("Invalid date format. (dd_mm_yyyy)")
                            
                        except Exception as e:
                            flag = True
                            print(f"Error: {e}")
                            with st.chat_message("assistant"):
                                st.markdown(f"{e}")

                        if not flag:
                            response = predict(date, open)
                            with st.chat_message("assistant"):
                                st.markdown(response)

                except Exception as e:
                    print("Error:", str(e))
        else:
            with st.chat_message("assistant"):
                st.write("Invalid stock symbol. Please try again.")