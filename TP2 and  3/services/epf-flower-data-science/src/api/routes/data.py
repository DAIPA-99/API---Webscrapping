from fastapi import APIRouter
import opendatasets as od
import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from fastapi import APIRouter
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from fastapi.responses import FileResponse
import os
from sklearn.model_selection import train_test_split

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
        api.authenticate()  # Authentification avec le fichier kaggle.json
        
        # Télécharger le dataset
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
        # Path to the downloaded CSV file
        dataset_path = "src/data/iris/Iris.csv"  # Adjusted path if necessary

        # Check if the dataset file exists
        if not os.path.exists(dataset_path):
            return {"error": f"Dataset not found at {dataset_path}"}

        # Load the dataset into a pandas DataFrame
        df = pd.read_csv(dataset_path)

        # Convert the DataFrame to a dictionary and return it as JSON
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
        # Charger le dataset
        dataset_path = "src/data/iris/Iris.csv"
        df = pd.read_csv(dataset_path)

        # Vérifier si la colonne 'Species' existe (notez la majuscule)
        if 'Species' not in df.columns:
            return {"error": "La colonne 'Species' n'existe pas dans le dataset."}

        # Vérifier les valeurs manquantes
        if df.isnull().sum().any():
            return {"error": "Le dataset contient des valeurs manquantes."}

        # Encoder la variable cible (espèce) avec le bon nom de colonne
        label_encoder = LabelEncoder()
        df['Species'] = label_encoder.fit_transform(df['Species'])

        # Normaliser les caractéristiques
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df.drop('Species', axis=1))

        # Créer un DataFrame avec les données traitées
        processed_df = pd.DataFrame(X_scaled, columns=df.columns[:-1])
        processed_df['Species'] = df['Species']  # Ajouter la colonne cible

        # Sauvegarder le DataFrame traité dans un fichier CSV
        processed_file_path = "src/data/processed_iris.csv"
        processed_df.to_csv(processed_file_path, index=False)

        # Renvoyer le fichier traité
        return FileResponse(processed_file_path, media_type='text/csv', filename='processed_iris.csv')

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