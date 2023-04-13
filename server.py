from flask import Flask, jsonify, request
from flask_cors import CORS
import pickle
import pandas as pd

# Load the trained random forest model from a file
with open("./random-forest/rf_model.pkl", "rb") as f:
    rf_model = pickle.load(f)

# Create a Flask web server
app = Flask(__name__)
CORS(app)

# Global variable to store previous sample
previous_sample = pd.DataFrame([{
    "fft01": 0.00,
    "fft03": 0.00,
    "fft05": 0.00,
    "fft07": 0.00,
    "fft09": 0.00,
    "fft11": 0.00,
    "fft13": 0.00,
    "fft15": 0.00,
    "fft17": 0.00,
    "fft19": 0.00,
    "fft21": 0.00,
    "fft23": 0.00,
    "fft25": 0.00,
    "fft27": 0.00,
    "fft29": 0.00,
    "fft31": 0.00
}], columns=["fft01", "fft03", "fft05", "fft07", "fft09", "fft11",
             "fft13", "fft15", "fft17", "fft19", "fft21", "fft23", "fft25", "fft27", "fft29", "fft31"])
# Global variable to store the current connected appliances
connected_appliances = []
# Global variable to identify if a appliance was connected or disconnected
connected_new_appliance = True

# Define a function to map the index with its appliance name
def map_appliance(index):
    match(index):
        case 1:
            return "Appliance 1"
        case 2:
            return "Appliance 2"
        case 3:
            return "Appliance 3"
        case 4:
            return "Appliance 4"
        case 5:
            return "Appliance 5"
        case 6:
            return "Appliance 6"
        case 7:
            return "Appliance 7"
        case 8:
            return "Appliance 8"

# Define a function to preprocess the input data
def preprocess_data(data):
    global previous_sample
    global connected_new_appliance

    # Convert the input data to a pandas dataframe
    current_sample = pd.DataFrame([data], columns=["fft01", "fft03", "fft05", "fft07", "fft09", "fft11",
                                                   "fft13", "fft15", "fft17", "fft19", "fft21", "fft23", "fft25", "fft27", "fft29", "fft31"])
    # Get mean values
    current_sample_mean = current_sample.sum().sum() / 16
    previous_sample_mean = previous_sample.sum().sum() / 16
    print(current_sample_mean)
    print(previous_sample_mean)

    if current_sample_mean > previous_sample_mean:
        # Appliance connected
        connected_new_appliance = True
        df = current_sample - previous_sample
        previous_sample = current_sample
        return df
    elif current_sample_mean < previous_sample_mean:
        # Appliance disconnected
        connected_new_appliance = False
        df = previous_sample - current_sample
        previous_sample = current_sample
        return df

# Define a function to postprocess the predicted labels
def postprocess_data(prediction):
    labels_str = str(prediction)
    labels_lst = [x == 'True' for x in labels_str.strip('[]').split()]
    appliance = [map_appliance(
        i+1) for i in range(len(labels_lst)) if labels_lst[i] == True]
    return appliance[0]

# Define a route for making predictions with the trained model
@app.route("/predict", methods=["POST"])
def predict():
    global connected_appliances
    global connected_new_appliance

    # Get the input data from the request body
    input_data = request.get_json()

    # Preprocess the input data
    preprocessed_data = preprocess_data(input_data)

    # Use the trained model to make predictions on the preprocessed data
    prediction = rf_model.predict(preprocessed_data)

    appliance = postprocess_data(prediction)

    if connected_new_appliance:
        connected_appliances.append(appliance)
    else:
        connected_appliances.remove(appliance)

    # Return the list with all connected appliances
    return jsonify(connected_appliances)


# Start the Flask web server
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3002, debug=True)
