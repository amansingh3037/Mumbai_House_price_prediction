
# A Complete Machine Learning Model use to predict the House price for Mumbai !!!

import os
from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


app = Flask(__name__)

# defining file name for model and pipeline
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Defining the function for the Pipeline 

def building_pipeline_structure(numerical_dataset,categorical_dataset):
    # for Numerical Pipeline
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("feature_scaling", StandardScaler())
    ])

    # for Categorical pipeline
    cat_pipeline = Pipeline([
        ("Encoding",OneHotEncoder(handle_unknown="ignore"))
    ])

    # for Full Pipeline Structure
    full_pipeline = ColumnTransformer([                # Always remember "ColumnTransformer" !!
        ("num",num_pipeline,numerical_dataset),
        ("cat",cat_pipeline,categorical_dataset)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    # Train the MODEL
    df = pd.read_csv("Housing Dataset.csv")

    df["rating"] = pd.cut(df["median_income"], bins = [0,1.5,3.0,4.5,6.0,np.inf], labels=[1,2,3,4,5])

    # split the dataset into training and testing segments
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
    for train_index, test_index in split.split(df,df["rating"]):
        strat_train_set = df.loc[train_index].drop("rating",axis=1)
        strat_test_set = df.loc[test_index].drop("rating",axis=1)
        strat_test_set.to_csv("input.csv", index=False)
    
    new_df = strat_train_set.copy()  # This is now the training dataset.

    # Separating the features and labels
    dataset_labels = strat_train_set["median_house_value"].copy()
    dataset_features = new_df.drop("median_house_value",axis=1)

    # Separating dataset_features into numerical and categorical datasets
    numerical_dataset = dataset_features.drop("ocean_proximity",axis=1).columns.tolist() # This is List 
    categorical_dataset = ["ocean_proximity"]             # This is List 

    # Calling Pipeline and Preparing features !!
    pipeline = building_pipeline_structure(numerical_dataset,categorical_dataset)
    Prepared_features = pipeline.fit_transform(dataset_features)

    # Training the ML model 
    model = RandomForestRegressor(random_state=42)
    model.fit(Prepared_features,dataset_labels)

    # Now, dumping the model and pipeline into their resp. Files 
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)

    print("Congratulations!! Files got dumped Successfully.")

else:
    # WE will go for the Inference (Extra data for predictions)

    # Now, Load the files
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    # Read the Input File
    input_dataset = pd.read_csv("input.csv")

    # Define feature order
    FEATURE_COLUMNS = [
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity"
    ]


    @app.route("/")
    def home():
        return render_template("index.html")
    
    # GET RANDOM ROW (RAW DATA)
    # -----------------------------
    @app.route("/get-data")
    def get_data():

        random_row = input_dataset.sample(n=1)

        # Extract features
        features = random_row[FEATURE_COLUMNS].iloc[0]

        # Extract target
        target_value = random_row["median_house_value"].iloc[0]

        return jsonify({
            "longitude": float(features["longitude"]),
            "latitude": float(features["latitude"]),
            "housing_median_age": int(features["housing_median_age"]),
            "total_rooms": float(features["total_rooms"]),
            "total_bedrooms": float(features["total_bedrooms"]),
            "population": float(features["population"]),
            "households": float(features["households"]),
            "median_income": float(features["median_income"]),
            "ocean_proximity": str(features["ocean_proximity"]),
            "median_house_value": float(target_value)
        })
    
    # Before jsonify the data, first convert it into original python datatypes(int, float, etc.)


    # -----------------------------
    # PREDICTION 
    # -----------------------------
    @app.route("/predict", methods=["POST"])
    def predict():

        data = request.json

        # Convert incoming data to DataFrame
        input_df = pd.DataFrame([{
            "longitude": data["longitude"],
            "latitude": data["latitude"],
            "housing_median_age": data["housing_median_age"],
            "total_rooms": data["total_rooms"],
            "total_bedrooms": data["total_bedrooms"],
            "population": data["population"],
            "households": data["households"],
            "median_income": data["median_income"],
            "ocean_proximity": data["ocean_proximity"]
        }])

        # Apply preprocessing pipeline
        Transformed_input_dataset = pipeline.transform(input_df)

        # Make prediction
        prediction = model.predict(Transformed_input_dataset)

        return jsonify({
            "prediction": float(prediction[0])
        })

if __name__ == "__main__":
    app.run(debug=True)

    # Transformed_input_dataset = pipeline.fit_transform(input_dataset)
    # Prediction = model.predict(Transformed_input_dataset)

    # input_dataset["Predicted_price"] = Prediction

    # # save this dataset as a Output.csv file 
    # input_dataset.to_csv("Output.csv", index=False)

    # print("Prediction has been done Successfully. Saved in the Output.csv file !!")
