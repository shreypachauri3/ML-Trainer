import streamlit as st
import pandas as pd
from pycaret.classification import *
from pycaret.regression import *

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            # Use pandas to read the file
            df = pd.read_csv(uploaded_file)  # Assuming it's a CSV file, adjust if necessary
            return df
        except Exception as e:
            st.error(f"Error: {e}")
    return None

def main():
    st.title("Train your model in Few Clicks")

    # Upload dataset
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv", "xlsx", "txt"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        st.success("Dataset loaded successfully!")

        # Choose target variable
        target_variable = st.selectbox("Select the target variable", df.columns)

        # Train the model
        if st.button("Train Model"):
            st.info("Training the model... Please wait.")

            setup_data = setup(df, target=target_variable)

            model = compare_models()

            # Tune the model
            tuned_model = tune_model(model)

            st.success("Model training and tuning completed!")

            # Save the best model
            saved_model_path = "pycaret_model.pkl"
            save_model(tuned_model, saved_model_path)

            st.info(f"Best model saved as '{saved_model_path}'")

            # Provide a link to download the model file
            with open(saved_model_path + ".pkl", "rb") as model_file:
                model_content = model_file.read()

            st.download_button(
                label="Download Trained Model",
                data=model_content,
                key="download_button",
                help="Click to download the trained model",
            )
            st.info(f"To use model chnage the extension to pkl type...")

if __name__ == "__main__":
    main()
