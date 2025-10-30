# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import numpy as np
# import pickle


# app = FastAPI()


# origins = ["*"]


# model = pickle.load(open('parkinsondisease_model.pkl', 'rb'))

# scaler = pickle.load(open('scaler_file.pkl', 'rb'))


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# class ParkinsonModel(BaseModel):
#     mdvpFo: float
#     mdvpFhi: float
#     mdvpFlo: float
#     mdvpJitterPercent: float
#     mdvpJitterAbsolute: float
#     mdvpRap: float
#     mdvpPPQ: float
#     jitterDDP: float
#     mdvpJimmer: float
#     mdvpJimmerDB: float
#     shimmerAPQ3: float
#     shimmerAPQ5: float
#     mdvpAPQ: float
#     shimmerDDA: float
#     NHR: float
#     HNR: float
#     RPDE: float
#     DFA: float
#     spread1: float
#     spread2: float
#     d2: float
#     PPE: float


# @app.get('/')
# def welcome():
#     return {
#         'success': True,
#         'message': 'server of parkinson disease detection is up and running successfully'
#     }


# @app.post('/pred-parkinson-disease')
# async def park_disease(parkDiseaseParameter: ParkinsonModel):
#     mdvpFo = parkDiseaseParameter.mdvpFo
#     mdvpFhi = parkDiseaseParameter.mdvpFhi
#     mdvpFlo = parkDiseaseParameter.mdvpFlo
#     mdvpJitterPercent = parkDiseaseParameter.mdvpJitterPercent
#     mdvpJitterAbsolute = parkDiseaseParameter.mdvpJitterAbsolute
#     mdvpRap = parkDiseaseParameter.mdvpRap
#     mdvpPPQ = parkDiseaseParameter.mdvpPPQ
#     jitterDDP = parkDiseaseParameter.jitterDDP
#     mdvpJimmer = parkDiseaseParameter.mdvpJimmer
#     mdvpJimmerDB = parkDiseaseParameter.mdvpJimmerDB
#     shimmerAPQ3 = parkDiseaseParameter.shimmerAPQ3
#     shimmerAPQ5 = parkDiseaseParameter.shimmerAPQ5
#     mdvpAPQ = parkDiseaseParameter.mdvpAPQ
#     shimmerDDA = parkDiseaseParameter.shimmerDDA
#     NHR = parkDiseaseParameter.NHR
#     HNR = parkDiseaseParameter.HNR
#     RPDE = parkDiseaseParameter.RPDE
#     DFA = parkDiseaseParameter.DFA
#     spread1 = parkDiseaseParameter.spread1
#     spread2 = parkDiseaseParameter.spread2
#     d2 = parkDiseaseParameter.d2
#     PPE = parkDiseaseParameter.PPE

#     pred_data = (mdvpFo, mdvpFhi, mdvpFlo, mdvpJitterPercent, mdvpJitterAbsolute, mdvpRap, mdvpPPQ, jitterDDP, mdvpJimmer,
#                  mdvpJimmerDB, shimmerAPQ3, shimmerAPQ5, mdvpAPQ, shimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, d2, PPE)

#     pred_data_as_numpy_array = np.asarray(pred_data)

#     pred_data_as_numpy_array_reshaped = pred_data_as_numpy_array.reshape(1, -1)

#     standard_pred_data = scaler.transform(pred_data_as_numpy_array_reshaped)

#     prediction = model.predict(standard_pred_data)

#     prediction_msg = ''

#     if prediction[0] == 0:
#         prediction_msg = 'the person does not have parkinson disease'
#     elif prediction[0] == 1:
#         prediction_msg = 'the person is having parkinson disease'

#     return {
#         'success': True,
#         'pred_value': float(prediction[0]),
#         'pred_msg': prediction_msg
#     }



import os
import logging
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ===============================
# Setup Logging
# ===============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

# ===============================
# Allowed Origins (Frontend URLs)
# ===============================
origins = [
    "https://parkinson-s-disease-inky.vercel.app",  # deployed frontend
    "http://localhost:3000",  # for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===============================
# Load Model and Scaler
# ===============================
try:
    logging.info("Loading model and scaler...")
    model = pickle.load(open('parkinsondisease_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler_file.pkl', 'rb'))
    logging.info("✅ Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"❌ Error loading model/scaler: {e}")
    raise e


# ===============================
# Input Schema
# ===============================
class ParkinsonModel(BaseModel):
    mdvpFo: float
    mdvpFhi: float
    mdvpFlo: float
    mdvpJitterPercent: float
    mdvpJitterAbsolute: float
    mdvpRap: float
    mdvpPPQ: float
    jitterDDP: float
    mdvpJimmer: float
    mdvpJimmerDB: float
    shimmerAPQ3: float
    shimmerAPQ5: float
    mdvpAPQ: float
    shimmerDDA: float
    NHR: float
    HNR: float
    RPDE: float
    DFA: float
    spread1: float
    spread2: float
    d2: float
    PPE: float


# ===============================
# Routes
# ===============================
@app.get("/")
async def welcome():
    logging.info("GET / request received.")
    return {
        "success": True,
        "message": "✅ Parkinson Disease Detection API is up and running!"
    }


@app.post("/pred-parkinson-disease")
async def park_disease(parkDiseaseParameter: ParkinsonModel, request: Request):
    logging.info("POST /pred-parkinson-disease request received.")
    logging.info(f"Request origin: {request.client.host}")

    try:
        pred_data = (
            parkDiseaseParameter.mdvpFo,
            parkDiseaseParameter.mdvpFhi,
            parkDiseaseParameter.mdvpFlo,
            parkDiseaseParameter.mdvpJitterPercent,
            parkDiseaseParameter.mdvpJitterAbsolute,
            parkDiseaseParameter.mdvpRap,
            parkDiseaseParameter.mdvpPPQ,
            parkDiseaseParameter.jitterDDP,
            parkDiseaseParameter.mdvpJimmer,
            parkDiseaseParameter.mdvpJimmerDB,
            parkDiseaseParameter.shimmerAPQ3,
            parkDiseaseParameter.shimmerAPQ5,
            parkDiseaseParameter.mdvpAPQ,
            parkDiseaseParameter.shimmerDDA,
            parkDiseaseParameter.NHR,
            parkDiseaseParameter.HNR,
            parkDiseaseParameter.RPDE,
            parkDiseaseParameter.DFA,
            parkDiseaseParameter.spread1,
            parkDiseaseParameter.spread2,
            parkDiseaseParameter.d2,
            parkDiseaseParameter.PPE
        )

        logging.info("✅ Input data received successfully.")
        pred_data_as_numpy_array = np.asarray(pred_data).reshape(1, -1)
        logging.info("✅ Reshaped input for model prediction.")

        standard_pred_data = scaler.transform(pred_data_as_numpy_array)
        logging.info("✅ Data scaled successfully.")

        prediction = model.predict(standard_pred_data)
        logging.info(f"✅ Prediction result: {prediction}")

        prediction_msg = (
            "✅ The person does not have Parkinson's disease."
            if prediction[0] == 0
            else "⚠️ The person is likely to have Parkinson's disease."
        )

        return {
            "success": True,
            "pred_value": float(prediction[0]),
            "pred_msg": prediction_msg
        }

    except Exception as e:
        logging.error(f"❌ Error during prediction: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Internal Server Error while processing prediction."
        }


# ===============================
# Run the App (for Railway)
# ===============================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"🚀 Starting server on port {port}")
    print(f"🚀 Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port)
