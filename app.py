import os
import sys
import base64
import requests # type: ignore
import io
from PIL import Image # type: ignore
import ccxt # type: ignore
import pandas as pd # type: ignore
import plotly.graph_objects as go # type: ignore
from plotly.subplots import make_subplots # type: ignore
import streamlit as st # type: ignore
import streamlit.components.v1 as components # type: ignore
import ta # type: ignore
from datetime import datetime
import pytz # type: ignore
from tzlocal import get_localzone # type: ignore
from openai import OpenAI # type: ignore
import logging
import json
from fastapi.responses import JSONResponse # type: ignore
import time

# Load environment variables
openai_api_key = st.secrets["openai_api_key"]
assistant_id = st.secrets["assistant_id"]

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OpenAI client
client = OpenAI(api_key=openai_api_key)

# Initialize the exchange
exchange = ccxt.kraken()  # Replace with the options to pick exchange
last_timestamp = None

# Get the local timezone
local_tz = get_localzone()

# Initialize the assistant configuration
nite_trader_gpt = {
    "model": "gpt-4o",  # Updated to vision model
    "assistant_id": assistant_id,
    "tools": [
        {"type": "file_search"},
        {"type": "code_interpreter"}
    ]
}

# Clear Streamlit's redirect-related query parameters
if st.experimental_get_query_params().get("redirect_uri"):
    st.experimental_set_query_params()  # Clear query params
    #st.experimental_rerun()  # Restart the app flow without the redirect

# App config
st.set_page_config(
   page_title="NiteTraderAI",
   page_icon="favicon.ico",
   layout="centered",
   initial_sidebar_state="expanded",
)


# Authentication function
def verify_sub_id(sub_id):
    url = "https://nitetrader.io/verify_sub_id"
    logging.info(f"Verifying sub_id={sub_id} with URL={url}")
    try:
        response = requests.get(url, params={"sub_id": sub_id})
        response.raise_for_status()
        logging.info(f"Verification response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Verification failed: {e}")
        st.error(f"Verification failed: {e}")
        return None

# Core functions
def fetch_data(symbol: str, timeframe: str) -> pd.DataFrame:
    global last_timestamp
    if last_timestamp:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=last_timestamp)
    else:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['timestamp'] = df['timestamp'].dt.tz_convert(local_tz)
    if not df.empty:
        last_timestamp = int(df['timestamp'].iloc[-1].timestamp() * 1000)
    return df

def add_indicators(df):
    df['MA50'] = ta.trend.sma_indicator(df['close'], window=50)
    df['MA200'] = ta.trend.sma_indicator(df['close'], window=200)
    df['RSI'] = ta.momentum.rsi(df['close'])
    return df

def check_signals(row: pd.Series, buy_threshold: float, sell_threshold: float) -> str:
    if row['close'] >= sell_threshold:
        return 'Sell Signal'
    elif row['close'] <= buy_threshold:
        return 'Buy Signal'
    return 'Hold'

def get_initial_instruction(mode):
    # Read instructions from secrets.toml
    priority_instructions = st.secrets["instructions"]["priority"]
    instructions = f"""
    {priority_instructions}
    Current Mode: {mode}
    """
    return instructions

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_response_with_image(nite_trader_gpt, conversation, base64_image=None):
    try:
        messages = conversation[:-1]
        last_user_message = conversation[-1]['content']

        if base64_image:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": last_user_message},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            })
        else:
            messages.append({"role": "user", "content": last_user_message})

        log_messages = [
            {**msg, 'content': msg['content'] if isinstance(msg['content'], str) else '[Image data]'}
            for msg in messages
        ]
        logging.info(f"Sending messages to API: {json.dumps(log_messages, indent=2)}")

        response = client.chat.completions.create(
            model=nite_trader_gpt['model'],
            messages=messages,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.exception(f"An error occurred while getting response: {e}")
        st.error(f"An error occurred: {e}")
        return "An error occurred while processing your request."

def logout():
    st.session_state.authenticated = False
    js = f"""
    <script>
        window.location.href = 'https://nitetrader.io';
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)


# Main app logic
def main():
    logging.info("Starting the main function.")

    # Initialize session state variables
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        logging.debug("Initialized 'authenticated' in session state.")

    if "is_subscribed" not in st.session_state:
        st.session_state.is_subscribed = False
        logging.debug("Initialized 'is_subscribed' in session state.")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        logging.debug("Initialized 'conversation' in session state.")

    if "mode" not in st.session_state:
        st.session_state.mode = "Normie"
        logging.debug("Initialized 'mode' in session state.")

    # Add custom CSS for background image
    st.image("https://nitetrader.io/nitetrader-1.png", width=250, use_column_width=True)

    # Check for sub_id in URL parameters
    query_params = st.experimental_get_query_params()
    logging.debug(f"Query Parameters: {query_params}")
    sub_id_values = query_params.get("sub_id")

    if sub_id_values and not st.session_state.authenticated:
        sub_id = sub_id_values[0]
        logging.info(f"Received sub_id: {sub_id}")
        verification = verify_sub_id(sub_id)
        if verification and verification["status"] == "valid":
            if verification["is_subscribed"] == 1:
                st.session_state.authenticated = True
                st.session_state.is_subscribed = True
                st.session_state.user_email = verification["email"]
                st.experimental_set_query_params()  # Clear query params
                logging.info(f"Authenticated user: {verification['email']}")
                
            else:
                st.error("Subscription required. Please return to https://nitetrader.io to purchase access.")
        else:
            st.error("Verification failed. Please try again.")

    # Main content (only show if authenticated)
    if st.session_state.authenticated:
        st.sidebar.header("Navigation")
        st.sidebar.button("Logout", on_click=logout)

        # Select application mode
        app_mode = st.sidebar.selectbox("Choose App Mode", ["Strategies", "Trading Assistant"])

        # Dashboard mode
        if app_mode == "Strategies":
            st.title("Strategies")
            current_time = datetime.now(local_tz)
            symbol = st.text_input("Enter Trading Symbol (e.g., ADA/USD)", "ADA/USD")
            timeframe = st.selectbox("Select Timeframe", options=['1m', '5m', '15m', '30m', '1h', '4h'], index=2)
            buy_threshold = st.number_input("Enter Buy Threshold ($)")
            sell_threshold = st.number_input("Enter Sell Threshold ($)")

            if st.button("Fetch Data"):
                all_data = fetch_data(symbol, timeframe)
                all_data = add_indicators(all_data)
                all_data['Signal'] = all_data.apply(lambda row: check_signals(row, buy_threshold, sell_threshold), axis=1)
                st.write(all_data[['timestamp', 'close', 'Signal']])

                # Create figure with secondary y-axis
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])

                # Add candlestick
                fig.add_trace(go.Candlestick(
                    x=all_data['timestamp'],
                    open=all_data['open'],
                    high=all_data['high'],
                    low=all_data['low'],
                    close=all_data['close'],
                    name=symbol
                ), row=1, col=1)

                st.plotly_chart(fig)

            # Add CSV file upload section
            st.header("CSV Data Analysis")
            uploaded_csv = st.file_uploader("Upload a CSV file for analysis:", type=["csv"], key="csv_uploader")

            if uploaded_csv is not None:
                try:
                    csv_data = pd.read_csv(uploaded_csv)
                    st.subheader("Data Preview")
                    st.write(csv_data.head())
                    columns = csv_data.columns.tolist()
                    x_column = st.selectbox("Select X-axis column", columns)
                    y_column = st.selectbox("Select Y-axis column", columns)

                    # Create bar chart
                    fig_csv = go.Figure()
                    fig_csv.add_trace(go.Bar(x=csv_data[x_column], y=csv_data[y_column]))
                    st.plotly_chart(fig_csv)
                except Exception as e:
                    st.error(f"Error processing CSV file: {str(e)}")

        # Trading Assistant mode
        elif app_mode == "Trading Assistant":
            st.title("Trading Assistant")
            mode = st.sidebar.radio("Select Trading Mode:", ["Normie", "DEGEN!"])

            if "conversation" not in st.session_state or st.session_state.get("mode") != mode:
                initial_instruction = get_initial_instruction(mode)
                st.session_state.conversation = [{"role": "system", "content": initial_instruction}]
                st.session_state.mode = mode

            for msg in st.session_state.conversation[1:]:
                if msg["role"] == "user":
                    st.markdown(f"**You:** {msg['content']}")
                else:
                    st.markdown(f"**NiteTraderAI:** {msg['content']}")

            uploaded_file = st.file_uploader("Upload a TradingView chart image to get started!", type=["png", "jpg", "jpeg"])

            base64_image = None
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Chart", use_column_width=True)
                base64_image = encode_image(image)

            with st.form(key="chat_form", clear_on_submit=True):
                query = st.text_area("Enter your request or observations:", key="chat_input")
                submitted = st.form_submit_button("Send")

                if submitted and query:
                    st.session_state.conversation.append({"role": "user", "content": query})
                    assistant_response = get_response_with_image(nite_trader_gpt, st.session_state.conversation, base64_image)
                    st.session_state.conversation.append({"role": "assistant", "content": assistant_response})
                    st.markdown(f"**NiteTraderAI:** {assistant_response}")

    else:
        st.error("Please go to https://nitetrader.io to login.")

if __name__ == "__main__":
    main()
