from typing import Union
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image
import requests
import pickle
from sklearn.preprocessing import LabelEncoder
from joblib import load
import urllib3
from xgboost import XGBClassifier
from fastapi.middleware.cors import CORSMiddleware
from keras.models import load_model
import keras.utils as image
from io import BytesIO


le = load('label_encoder/label_encoder.joblib')
symptoms_list = ['itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough', 'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration', 'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea', 'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid',
                 'brittle_nails', 'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload.1', 'blood_in_sputum', 'prominent_veins_on_calf', 'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze']
xgbmodel = load('model/model.joblib')
app = FastAPI()
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def loadImage(URL):
    with urllib3.request.urlopen(URL) as url:
        img = image.load_img(BytesIO(url.read()), target_size=(125, 125))

    return image.img_to_array(img)


class Item(BaseModel):
    Age: int | None = None
    Gender: str | None = None
    Severity: str | None = None
    Symptoms: list[str] = []


class img(BaseModel):
    image: str | None = None


class response(BaseModel):
    Age: int | None = None
    Gender: str | None = None
    Severity: str | None = None
    Symptoms: list[str] = []
    Disease: list[str] = []
    Probability: list[float] = []


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict", response_model=response,)
def predict(item: Item):
    print(item.Age)
    final_list = item.Symptoms
    prediction_value = [0 for i in range(131)]
    for sym in final_list:
        index = symptoms_list.index(sym)
        prediction_value[index] = 1
    prediction_value = pd.DataFrame(prediction_value).T
    predicted_proba = xgbmodel.predict_proba(prediction_value)
    top_three = []
    class_prob = [(le.classes_[i], prob)
                  for i, prob in enumerate(predicted_proba[0])]
    sorted_classes = sorted(class_prob, key=lambda x: x[1], reverse=True)[:3]
    print(sorted_classes)
    # top_three.append(sorted_classes)
    transpose_list = list(zip(*sorted_classes))
    result = [list(i) for i in transpose_list]
    Disease = result[0]
    Probability = result[1]

    return {"Age": item.Age, "Gender": item.Gender, "Severity": item.Severity, "Symptoms": item.Symptoms, "Disease": Disease, "Probability": Probability}


@app.post("/predict_image",)
def predict_image(item: img):
    print(item.image)

    # Load the model and label encoder
    model = load_model("model/model_herb.h5")
    labels = load("label_encoder/labels_herb.joblib")

    # Download and open the image
    response = requests.get(item.image, stream=True, verify=False)
    img = Image.open(BytesIO(response.content))
    img = img.resize((150, 150))

    # Display the image (optional)
    img.show()

    # Convert the image to a NumPy array
    test_image = image.img_to_array(img)
    test_image = np.expand_dims(test_image, axis=0)

    # Make the prediction
    result = model.predict(test_image)

    # Get the predicted label
    prediction = labels[np.argmax(result)]
    print(prediction)

    return {"image": prediction}
