from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('../backend/model/heart_disease_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = [
        data['age'],
        data['gender'],
        data['cigsPerDay'],
        data['totChol'],
        data['sysBP'],
        data['glucose']
    ]
    
    return jsonify({'prediction': predict_heart_disease(input_data)})

def predict_heart_disease(user_data):

    logreg = joblib.load('../backend/model/logistic_regression_model.pkl')
    scaler = joblib.load('../backend/model/scaler.pkl')

    """
    Predicts heart disease risk for a user based on input data.
    user_data: list or array containing [age, Sex_male, cigsPerDay, totChol, sysBP, glucose]
    """

    # Convert the input to a numpy array and reshape if necessary
    user_data = np.array(user_data).reshape(1, -1)

    # Standardize the input data using the trained scaler
    user_data_scaled = scaler.transform(user_data)

    # Make the prediction
    prediction = logreg.predict(user_data_scaled)[0]  # Predicted class (0 or 1)
    probability = logreg.predict_proba(user_data_scaled)[0][1]  # Probability of heart disease (class 1)

    # Interpret and display the result
    if prediction == 1:
        print("High risk of heart disease.")
        print(f"Probability of heart disease: {probability * 100:.2f}%")

        return "High risk of heart disease. Probability of heart disease: {:.2f}%".format(probability * 100);
    else:
        print("Low risk of heart disease.")
        print(f"Probability of heart disease: {probability * 100:.2f}%")

        return "Low risk of heart disease. Probability of heart disease: {:.2f}%".format(probability * 100);

if __name__ == '__main__':
    app.run()
