from flask import Flask, request, render_template
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the trained model and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Creating Flask app
app = Flask(__name__)

# Route to render the index.html template
@app.route('/')
def index():
    return render_template("index.html")

# Route to handle form submission and make predictions
@app.route("/predict", methods=['POST'])
def predict():
    # Get form data
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    # Prepare features for prediction
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # Scale the features
    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)

    # Make prediction using the model
    prediction = model.predict(final_features)

    # Mapping of crop labels to names
    crop_dict = {1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                 19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"}

    # Get the predicted crop name
    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = f"{crop} is the best crop to be cultivated here."
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."

    # Render the result on index.html
    return render_template('index.html', result=result)

# Main function to run the Flask app
if __name__ == "__main__":
    # Run the app on 0.0.0.0:10000 (Render's expected port)
    app.run(host='0.0.0.0', port=10000)
