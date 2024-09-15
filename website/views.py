from flask import Blueprint, render_template, request
from flask_login import login_required, current_user
from ml_model.majority_vote_classifier import MajorityVoteClassifier
from ml_model.preporocessing import OHEncoding, FeatureScaler

import pickle
import warnings
import pandas as pd


warnings.filterwarnings("ignore")

views = Blueprint("views", __name__)
loaded_model = pickle.load(open("website\\ml_model\\majority_vote_classifier.pkl", "rb"))


def process_data(input_data: list):
    column_names = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    dataframe_ = pd.DataFrame([input_data], columns=column_names)

    return dataframe_


@views.route("/", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        transaction_type = request.form.get("transaction_type")
        name_origin = request.form.get("name_origin")
        amount = request.form.get("amount")
        old_balance_origin = request.form.get("old_balance_origin")
        new_balance_origin = request.form.get("new_balance_origin")
        name_dest = request.form.get("name_dest")
        old_balance_dest = request.form.get("old_balance_dest")
        new_balnace_dest = request.form.get("new_balance_dest")

        data = [transaction_type, amount, old_balance_origin, new_balance_origin, old_balance_dest, new_balnace_dest]

        pred_result = loaded_model.predict(process_data(data))
        model_results_list = loaded_model.predictions

        print(pred_result, model_results_list)

    return render_template("home.html", user=current_user)
