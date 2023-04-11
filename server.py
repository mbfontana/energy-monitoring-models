from flask import Flask, jsonify, request
import pickle
import pandas as pd
import re

# Load the trained random forest model from a file
with open("./random-forest/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Create a Flask web server
app = Flask(__name__)

# Define a function to preprocess the input data
def preprocess_data(data):
    index = range(len(data))
    # Convert the input data to a pandas dataframe
    df = pd.DataFrame(data, index=index, columns=["fft01", "fft03", "fft05", "fft07", "fft09", "fft11",
                      "fft13", "fft15", "fft17", "fft19", "fft21", "fft23", "fft25", "fft27", "fft29", "fft31"])
    return df

# Define a function to postprocess the predicted labels
def postprocess_labels(predicted_labels):
    labels_str = str(predicted_labels[0])
    labels_lst = [x == 'True' for x in labels_str.strip('[]').split()]
    appliances_id = [i+1 for i in range(len(labels_lst)) if labels_lst[i] == True]
    return appliances_id


# Define a route for making predictions with the trained model
@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request body
    input_data = request.get_json()

    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data)

    # Use the trained model to make predictions on the preprocessed data
    predicted_labels = rf_model.predict(preprocessed_data)

    postprocessed_labels = postprocess_labels(predicted_labels)

    # Return the true indexes as a JSON response
    return jsonify(postprocessed_labels)


# Start the Flask web server
if __name__ == "__main__":
    app.run(debug=True)
