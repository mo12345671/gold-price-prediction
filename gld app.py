import numpy as np
import streamlit as st
import pickle

# Load the pre-trained model
try:
  

 model = pickle.load(open(r"gld.sav", "rb"))


except FileNotFoundError:
    st.error("Model file 'gold_price_model.sav' not found. Please ensure the file exists.")
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

    # Input fields
    st.subheader("Enter Financial Data")
    SPX = st.text_input('S&P 500 (SPX)', '')
    USO = st.text_input('Oil Price (USO)', '')
    SLV = st.text_input('Silver Price (SLV)', '')
    EURUSD = st.text_input('EUR/USD Exchange Rate', '')
    input_data = [SPX, USO, SLV, EURUSD]

    # Prediction button
    if st.button('Predict Gold Price'):
        try:
            if any(x == '' for x in input_data):
                st.error("Please fill all fields with valid numbers.")
            else:
                prediction = gold_price_prediction(input_data)
                st.success(prediction)
        except Exception as e:
            st.error(f"Input error: {str(e)}")

    # Optional: Display input data for debugging
    if st.checkbox("Show input data"):
        st.write("Input Data:", input_data)

if __name__ == '__main__':
    main()

