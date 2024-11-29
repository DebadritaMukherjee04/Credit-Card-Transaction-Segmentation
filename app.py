from sklearn import preprocessing 
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Add your credentials (for demonstration purposes)
username = "user123"
password = "password123"

# Function to check login credentials
def check_login(user, pwd):
    return user == username and pwd == password

# Show login form first
def login_page():
    st.title("Login Page")
    user_input = st.text_input("Username")
    pwd_input = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        if check_login(user_input, pwd_input):
            st.success("Login successful!")
            return True
        else:
            st.error("Invalid credentials. Please try again.")
            return False

# If user is logged in, display the rest of the app
def main_app():
    # Load model and dataset
    filename = 'final_model.pkl'
    try:
        loaded_model = pickle.load(open(filename, 'rb'))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return
    
    try:
        df = pd.read_csv("Clustered_Customer_Data.csv")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return
    
    st.title("Customer Segmentation Prediction")

    # Form for user input
    with st.form("prediction_form"):
        st.header("Enter Customer Details")
        balance = st.number_input(label='Balance', step=0.001, format="%.6f")
        balance_frequency = st.number_input(label='Balance Frequency', step=0.001, format="%.6f")
        purchases = st.number_input(label='Purchases', step=0.01, format="%.2f")
        oneoff_purchases = st.number_input(label='OneOff Purchases', step=0.01, format="%.2f")
        installments_purchases = st.number_input(label='Installments Purchases', step=0.01, format="%.2f")
        cash_advance = st.number_input(label='Cash Advance', step=0.01, format="%.6f")
        purchases_frequency = st.number_input(label='Purchases Frequency', step=0.01, format="%.6f")
        oneoff_purchases_frequency = st.number_input(label='OneOff Purchases Frequency', step=0.1, format="%.6f")
        purchases_installment_frequency = st.number_input(label='Purchases Installments Frequency', step=0.1, format="%.6f")
        cash_advance_frequency = st.number_input(label='Cash Advance Frequency', step=0.1, format="%.6f")
        cash_advance_trx = st.number_input(label='Cash Advance Transactions', step=1)
        purchases_trx = st.number_input(label='Purchases Transactions', step=1)
        credit_limit = st.number_input(label='Credit Limit', step=0.1, format="%.1f")
        payments = st.number_input(label='Payments', step=0.01, format="%.6f")
        minimum_payments = st.number_input(label='Minimum Payments', step=0.01, format="%.6f")
        prc_full_payment = st.number_input(label='PRC Full Payment', step=0.01, format="%.6f")
        tenure = st.number_input(label='Tenure', step=1)

        # Organize data as a 2D array
        data = [[balance, balance_frequency, purchases, oneoff_purchases, installments_purchases, cash_advance,
                 purchases_frequency, oneoff_purchases_frequency, purchases_installment_frequency, cash_advance_frequency,
                 cash_advance_trx, purchases_trx, credit_limit, payments, minimum_payments, prc_full_payment, tenure]]

        # Submit button
        submitted = st.form_submit_button("Submit")

    # Process form submission
    if submitted:
        # Predict cluster
        try:
            cluster = loaded_model.predict(data)[0]
            st.success(f"The input data belongs to Cluster {cluster}")
            
            # Filter data for the cluster
            cluster_df = df[df['Cluster'] == cluster]

            st.header(f"Feature Distributions for Cluster {cluster}")
            st.write(f"Cluster {cluster} contains {len(cluster_df)} data points.")

            # Plot feature distributions for the cluster
            plt.rcParams["figure.figsize"] = (8, 4)
            for column in cluster_df.drop(columns=["Cluster"]).columns:
                fig, ax = plt.subplots()
                sns.histplot(cluster_df[column], kde=True, ax=ax)
                ax.set_title(f'Distribution of {column}')
                st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Main flow
if login_page():
    main_app()
