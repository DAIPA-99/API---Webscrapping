from fastapi import APIRouter
import opendatasets as od
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi.responses import FileResponse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

router = APIRouter()
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

        # Vérifier si la colonne 'Species' existe
        if 'Species' not in df.columns:
            return {"error": "La colonne 'Species' n'existe pas dans le dataset."}

        if df.isnull().sum().any():
            return {"error": "Le dataset contient des valeurs manquantes."}

        label_encoder = LabelEncoder()
        df['Species'] = label_encoder.fit_transform(df['Species'])

        scaler = StandardScaler()
        
        df_features = df.drop(columns=['Id', 'Species'])  

        X_scaled = scaler.fit_transform(df_features)

        processed_df = pd.DataFrame(X_scaled, columns=df_features.columns)

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
        dataset_path = "src/data/processed_iris.csv"
        df = pd.read_csv(dataset_path)
        
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        
        return {
            "train": train.to_dict(orient="records"),
            "test": test.to_dict(orient="records")
        }
    except Exception as e:
        return {"error": str(e)}
    

@router.post("/train-model")
async def train_model():
    """
    Train the classification model using the Iris dataset and save it to a file.
    """
    try:
        # Charger le dataset d'origine
        original_dataset_path = "src/data/iris/Iris.csv"
        df = pd.read_csv(original_dataset_path)

        if df.empty:
            return {"error": "Le dataset est vide."}

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(df['Species'])

        processed_dataset_path = "src/data/processed_iris.csv"
        processed_df = pd.read_csv(processed_dataset_path)

        if processed_df.empty:
            return {"error": "Le dataset prétraité est vide."}

        X = processed_df 
        y = y_encoded 

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        model_file_path = "src/data/random_forest_model.pkl"
        joblib.dump(model, model_file_path)

        encoder_file_path = "src/data/label_encoder.pkl"
        joblib.dump(label_encoder, encoder_file_path)

        return {"message": "Model trained and saved successfully."}
    except Exception as e:
        return {"error": str(e)}
    
@router.post("/predict")
async def predict(species_data: dict):
    """
    Make predictions using the trained model and return the species name.
    """
    try:

        model_file_path = "src/data/random_forest_model.pkl"
        encoder_file_path = "src/data/label_encoder.pkl"

        if not os.path.exists(model_file_path) or not os.path.exists(encoder_file_path):
            return {"error": "Le modèle ou le label encoder n'a pas été trouvé."}

        model = joblib.load(model_file_path)
        label_encoder = joblib.load(encoder_file_path)

        required_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
        
        for column in required_columns:
            if column not in species_data:
                return {"error": f"La colonne '{column}' est manquante dans les données d'entrée."}

        input_data = pd.DataFrame([species_data])

        prediction_encoded = model.predict(input_data)

        predicted_species = label_encoder.inverse_transform(prediction_encoded)

        return {"predicted_species": predicted_species[0]}
    except Exception as e:
        return {"error": str(e)}