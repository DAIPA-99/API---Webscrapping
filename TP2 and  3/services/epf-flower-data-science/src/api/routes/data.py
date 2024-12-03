from fastapi import APIRouter
import opendatasets as od
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import json
import joblib


router = APIRouter()

os.environ['KAGGLE_CONFIG_DIR'] = os.path.expanduser("~/.kaggle")

# Endpoint to download the Iris dataset
@router.get("/download-dataset")
async def download_dataset():
    """
    Downloads the Iris dataset from Kaggle and saves only the CSV file to the src/data folder.
    """
    try:
        api = KaggleApi()
        api.authenticate() 
        
        api.dataset_download_files('uciml/iris', path='src/data/', unzip=True)
        
        return {"message": "CSV file(s) downloaded and moved to src/data/"}
    except Exception as e:
        return {"error": str(e)}

# Endpoint to load the Iris dataset and return it as a DataFrame in JSON format
@router.get("/load-dataset")
async def load_dataset():
    """
    Loads the Iris dataset from the src/data folder, converts it to a DataFrame, and returns it as JSON.
    """
    try:
        dataset_path = "src/data/iris/Iris.csv" 

        if not os.path.exists(dataset_path):
            return {"error": f"Dataset not found at {dataset_path}"}
        
        df = pd.read_csv(dataset_path)

        data_json = df.to_dict(orient="records")

        return {"data": data_json}

    except Exception as e:
        return {"error": str(e)}

@router.get("/process-dataset")
async def process_dataset():
    """
    Process the Iris dataset by encoding categorical variables,
    scaling features, and saving the processed data to a CSV file.
    """
    try:
        dataset_path = "src/data/iris/Iris.csv"
        df = pd.read_csv(dataset_path)

        if 'Species' not in df.columns:
            return {"error": "La colonne 'Species' n'existe pas dans le dataset."}

        if df.isnull().sum().any():
            return {"error": "Le dataset contient des valeurs manquantes."}

        label_encoder = LabelEncoder()
        df['Species'] = label_encoder.fit_transform(df['Species'])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.drop('Species', axis=1))

        processed_df = pd.DataFrame(X_scaled, columns=df.columns[:-1])
        processed_df['Species'] = df['Species'] 
     
        processed_file_path = "src/data/processed_iris.csv"
        processed_df.to_csv(processed_file_path, index=False)

        example_data = processed_df.head(5).to_dict(orient="records")  

        return {
            "message": "Dataset processed successfully.",
            "example_data": example_data  
        }

    except Exception as e:
        return {"error": str(e)}

@router.get("/split-dataset")
async def split_dataset():
    """
    Split the Iris dataset into training and testing sets.
    """
    try:
        dataset_path = "src/data/iris/Iris.csv"
        df = pd.read_csv(dataset_path)
        
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        
        return {
            "train": train.to_dict(orient="records"),
            "test": test.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
    
def load_model_parameters():
    with open("src/config/model_parameters.json") as f:
        return json.load(f)

@router.post("/train-model")
async def train_model():
    """
    Train the classification model using the Iris dataset and save it to a file.
    """
    try:
        # Charger le dataset prétraité
        dataset_path = "src/data/processed_iris.csv"
        df = pd.read_csv(dataset_path)

        # Vérifier si le DataFrame est vide
        if df.empty:
            return {"error": "Le dataset est vide."}

        # Séparer les caractéristiques et la cible
        X = df.drop('Species', axis=1)
        y = df['Species']

        # Charger les paramètres du modèle
        params = load_model_parameters()

        # Créer et entraîner le modèle
        model = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=params['random_state']
        )
        
        # Vérifier les formes de X et y avant l'entraînement
        if X.shape[0] == 0 or y.shape[0] == 0:
            return {"error": "Les données d'entrée sont vides."}

        model.fit(X, y)

        # Sauvegarder le modèle entraîné dans un fichier
        model_file_path = "src/data/random_forest_model.pkl"
        joblib.dump(model, model_file_path)

        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        return {"error": str(e)}


@router.post("/predict")
async def predict(species_data: dict):
    """
    Make predictions using the trained model.
    """
    try:
        # Charger le modèle depuis le fichier sauvegardé
        model_file_path = "src/data/random_forest_model.pkl"
        
        if not os.path.exists(model_file_path):
            return {"error": "Le modèle n'a pas été trouvé."}

        model = joblib.load(model_file_path)

        # Convertir les données d'entrée en DataFrame
        input_data = pd.DataFrame([species_data])

        if input_data.empty or input_data.shape[1] != 4:  
            return {"error": "Les données d'entrée ne sont pas valides."}

        # Faire la prédiction
        prediction = model.predict(input_data)

        return {"predicted_species": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
    
    import joblib