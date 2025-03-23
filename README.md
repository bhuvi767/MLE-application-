# MLE-application-
# 🏡 House Price Prediction - Machine Learning API

This project is a **Machine Learning-based House Price Prediction System** that estimates house prices based on various features like location, size, number of bedrooms, and more. It is built using **Python, Flask/FastAPI, Scikit-Learn, and Docker**.

---

## 🚀 Features
- 📊 **Data Preprocessing**: Cleans and prepares housing datasets.
- 🤖 **Model Training**: Uses **Linear Regression** and other ML models.
- ⚙️ **Hyperparameter Tuning**: Optimizes model performance.
- 🌐 **REST API**: Built with Flask/FastAPI for easy integration.
- 🐳 **Dockerized**: Deployable as a containerized service.

---

## 🛠️ Tech Stack
- **Python** 
- **Flask / FastAPI** 
- **Scikit-Learn** 
- **Pandas & NumPy** 
- **Docker** 

---

Create a Virtual Environment & Install Dependencies

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt

🔹 3. Run the API Locally

python app.py

Visit http://127.0.0.1:5000/predict to test.
🐳 Running with Docker
🔹 1. Build the Docker Image

docker build -t house-price-api .

🔹 2. Run the Container

docker run -p 5000:5000 house-price-api

Now, the API is accessible at http://localhost:5000/predict.
📜 API Usage (Example)

Endpoint: POST /predict
Request (JSON):

{
  "square_feet": 1500,
  "num_bedrooms": 3,
  "num_bathrooms": 2,
  "location": "New York"
}

Response (JSON):

{
  "predicted_price": 250000
}
