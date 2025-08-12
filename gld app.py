import numpy as np
import streamlit as st
import pickle

# Load the pre-trained model
try:
    model = pickle.load(open(r"gld.sav", "rb"))
except FileNotFoundError:
    st.error("Model file 'gld.sav' not found. Please ensure the file exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Prediction function
def gold_price_prediction(input_data):
    try:
        input_data_as_numpy_array = np.asarray(input_data, dtype=float)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        return f"Predicted Gold Price (GLD): ${prediction[0]:.2f}"
    except ValueError as e:
        return f"Error: Please ensure all inputs are valid numbers. Details: {str(e)}"
    except Exception as e:
        return f"Prediction error: {str(e)}"

def main():
    st.title('Gold Price Prediction Web App')

    # Input fields using number_input for safer numeric input
    st.subheader("Enter Financial Data")
    SPX = st.number_input('S&P 500 (SPX)', value=4000.0, step=1.0, format="%.2f")
    USO = st.number_input('Oil Price (USO)', value=70.0, step=0.1, format="%.2f")
    SLV = st.number_input('Silver Price (SLV)', value=25.0, step=0.1, format="%.2f")
    EURUSD = st.number_input('EUR/USD Exchange Rate', value=1.1, step=0.001, format="%.4f")
    
    input_data = [SPX, USO, SLV, EURUSD]

    # Prediction button
    if st.button('Predict Gold Price'):
        prediction = gold_price_prediction(input_data)
        st.success(prediction)

    # Optional: Display input data for debugging
    if st.checkbox("Show input data"):
        st.write("Input Data:", input_data)

if __name__ == '__main__':
    main()
