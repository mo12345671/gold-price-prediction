import streamlit as st
import numpy as np
import pickle
import io

# Load the pre-trained model once and cache it
@st.cache_resource
def load_model():
    try:
        with open(r"gld.sav", "rb") as file:
            loaded_model = pickle.load(file)
        return loaded_model
    except FileNotFoundError:
        st.error("Model file 'gld.sav' not found. Please ensure the file exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

model = load_model()

# Prediction function
def gold_price_prediction(input_data):
    try:
        input_array = np.array(input_data, dtype=float).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction[0]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="Gold Price Prediction", layout="wide")

    st.title("💰 Gold Price Prediction Web App")
    st.write("Predict gold prices based on key financial indicators using a trained machine learning model.")

    # Sidebar info
    with st.sidebar:
        st.header("How to Use")
        st.markdown("""
        - Enter financial indicators in the fields.
        - Click **Predict Gold Price**.
        - View the predicted price below.
        - Optionally, download the result.
        """)
        st.markdown("---")
        st.header("About")
        st.write("This app uses an ML model trained on historical data to estimate gold prices based on S&P 500, Oil, Silver prices, and EUR/USD rate.")

    # Input form in columns
    st.subheader("Enter Financial Data")

    col1, col2 = st.columns(2)
    with col1:
        spx = st.number_input("S&P 500 (SPX)", min_value=0.0, value=4000.0, step=1.0, format="%.2f")
        silver = st.number_input("Silver Price (SLV)", min_value=0.0, value=25.0, step=0.1, format="%.2f")
    with col2:
        oil = st.number_input("Oil Price (USO)", min_value=0.0, value=70.0, step=0.1, format="%.2f")
        eurusd = st.number_input("EUR/USD Exchange Rate", min_value=0.0, value=1.1, step=0.001, format="%.4f")

    input_data = [spx, oil, silver, eurusd]

    # Predict button with spinner
    if st.button("Predict Gold Price"):
        with st.spinner("Predicting..."):
            prediction = gold_price_prediction(input_data)
            if prediction is not None:
                st.success(f"Predicted Gold Price (GLD): ${prediction:.2f}")

                # Download prediction result
                result_text = f"Gold Price Prediction Result\n\n" \
                              f"S&P 500 (SPX): {spx}\n" \
                              f"Oil Price (USO): {oil}\n" \
                              f"Silver Price (SLV): {silver}\n" \
                              f"EUR/USD Exchange Rate: {eurusd}\n\n" \
                              f"Predicted Gold Price: ${prediction:.2f}\n"

                buffer = io.StringIO()
                buffer.write(result_text)
                buffer.seek(0)

                st.download_button(
                    label="Download Prediction Result",
                    data=buffer,
                    file_name="gold_price_prediction.txt",
                    mime="text/plain"
                )

    # Optional: Show input data for debugging
    if st.checkbox("Show input data"):
        st.write("Input Data:", input_data)

    # Optional: Show model feature importance (if available)
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importance")
        import pandas as pd
        import matplotlib.pyplot as plt

        fi = model.feature_importances_
        features = ["SPX", "Oil", "Silver", "EUR/USD"]
        df_fi = pd.DataFrame({"Feature": features, "Importance": fi}).sort_values(by="Importance", ascending=False)

        fig, ax = plt.subplots()
        ax.barh(df_fi["Feature"], df_fi["Importance"], color="gold")
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")

        st.pyplot(fig)

if __name__ == "__main__":
    main()
