from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session management

# Load the model and dataset
MODEL_FILE = "final_model.pkl"
DATA_FILE = "Clustered_Customer_Data.csv"

try:
    loaded_model = pickle.load(open(MODEL_FILE, "rb"))
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    print(f"Error loading model or dataset: {e}")
    loaded_model = None
    df = None

# Route: Login Page
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username == "user123" and password == "password123":
            session["logged_in"] = True
            return redirect(url_for("predict"))
        else:
            flash("Invalid credentials, please try again.")
    
    return render_template("login.html")

# Route: Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    
    if request.method == "POST":
        # Get user input
        try:
            user_input = [
                float(request.form.get("balance", 0)),
                float(request.form.get("balance_frequency", 0)),
                float(request.form.get("purchases", 0)),
                float(request.form.get("oneoff_purchases", 0)),
                float(request.form.get("installments_purchases", 0)),
                float(request.form.get("cash_advance", 0)),
                float(request.form.get("purchases_frequency", 0)),
                float(request.form.get("oneoff_purchases_frequency", 0)),
                float(request.form.get("purchases_installment_frequency", 0)),
                float(request.form.get("cash_advance_frequency", 0)),
                int(request.form.get("cash_advance_trx", 0)),
                int(request.form.get("purchases_trx", 0)),
                float(request.form.get("credit_limit", 0)),
                float(request.form.get("payments", 0)),
                float(request.form.get("minimum_payments", 0)),
                float(request.form.get("prc_full_payment", 0)),
                int(request.form.get("tenure", 0)),
            ]
            
            # Predict cluster
            cluster = loaded_model.predict([user_input])[0]
            
            # Filter cluster data
            cluster_df = df[df["Cluster"] == cluster]
            
            # Plot feature distributions
            plot_dir = "static/plots"
            os.makedirs(plot_dir, exist_ok=True)
            plot_paths = []

            for column in cluster_df.drop(columns=["Cluster"]).columns:
                plt.figure()
                sns.histplot(cluster_df[column], kde=True)
                plt.title(f"Distribution of {column}")
                plt.tight_layout()

                plot_path = os.path.join(plot_dir, f"{column}.png")
                plt.savefig(plot_path)
                plt.close()
                plot_paths.append(f"/{plot_path}")
            
            return render_template("result.html", cluster=cluster, plots=plot_paths)
        
        except Exception as e:
            flash(f"Error in prediction: {e}")
            return redirect(url_for("predict"))
    
    return render_template("predict.html")

# Route: Logout
@app.route("/logout")
def logout():
    session.pop("logged_in", None)
    flash("You have been logged out.")
    return redirect(url_for("login"))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
