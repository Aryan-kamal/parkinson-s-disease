# Parkinsons-Disease-Detection

This is a project which predicts whether a person have Parkinson's Disease or not.

Parkinson's disease is a neurodegenerative disorder that primarily affects the central nervous system, particularly the motor system. It's characterized by a gradual loss of dopamine-producing cells in a region of the brain called the substantia nigra. Parkinson's disease is chronic and progressive, meaning its symptoms tend to worsen over time. The exact cause of Parkinson's disease is not fully understood, but a combination of genetic and environmental factors is believed to play a role. While there is no cure for Parkinson's disease, there are various treatment options available to manage its symptoms. These treatments include medication to increase dopamine levels in the brain, deep brain stimulation (a surgical procedure that involves implanting electrodes to stimulate specific brain areas), physical therapy, and lifestyle modifications.

What this project basically does is it uses Support Vector Classifier (SVC) to predict whether a person have Parkinson's Disease or not. The machine learning model is connected with a website built on top of React with the help of FastAPI and with the help of this website, a person can send real time input to the machine learning model to predict whether he/she have Parkinson's disease or not.

#### Checkout the website: https://parkinson-disease-predictor.vercel.app/

#### Deployed backend API URL of the machine learning model: https://parkinson-disease-pred.onrender.com/

#### Swagger Documentation of the machine learning model API: https://parkinson-disease-pred.onrender.com/docs

## Overview of Project

https://github.com/vishal815/Parkinsons-Disease-Detection/assets/83393190/ee343fcf-320e-4ae3-a9b6-490e0fe5b68f


Local host - 
as it is using fastAPI so use uvicorn app:app --reload to start the backend server (do basic setup of .venv before )
ans for frontend server as normal - npm install -> npm start



⚙️ 1️⃣ Initial Setup of backend(One Time Only) 

In your backend folder (e.g. D:\Projects\Parkinsons-Disease-Detection-main\backend):

# Create a virtual environment (remove already existing .venv file)
python -m venv .venv

# Activate the environment
.venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip


Then install all your backend dependencies (if you have a requirements.txt):

pip install -r requirements.txt


If you don’t have it yet, install the key packages manually:

pip install fastapi uvicorn scikit-learn joblib numpy pandas





also fix .env file if required during starting the project 