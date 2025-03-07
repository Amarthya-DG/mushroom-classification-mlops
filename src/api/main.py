import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

import joblib
import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.data_processing import DataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for the FastAPI application.
    """
    # Startup: Load model and preprocessor
    load_model()
    yield
    # Shutdown: Clean up if needed
    pass


app = FastAPI(
    title="Mushroom Classification API",
    description="API for predicting whether a mushroom is poisonous or not",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response models
class MushoomFeatures(BaseModel):
    cap_shape: str = Field(..., description="Shape of the mushroom cap")
    cap_surface: str = Field(..., description="Surface texture of the mushroom cap")
    cap_color: str = Field(..., description="Color of the mushroom cap")
    has_bruises: bool = Field(..., description="Whether the mushroom has bruises")
    odor: str = Field(..., description="Odor of the mushroom")
    gill_attachment: str = Field(..., description="How gills are attached to the stalk")
    gill_spacing: str = Field(..., description="Spacing between gills")
    gill_size: str = Field(..., description="Size of gills")
    gill_color: str = Field(..., description="Color of gills")
    stalk_shape: str = Field(..., description="Shape of the stalk")
    stalk_surface_above_ring: str = Field(
        ..., description="Surface texture of stalk above ring"
    )
    stalk_surface_belows_ring: str = Field(
        ..., description="Surface texture of stalk below ring"
    )
    stalk_color_above_ring: str = Field(..., description="Color of stalk above ring")
    stalk_color_below_ring: str = Field(..., description="Color of stalk below ring")
    veil_type: str = Field(..., description="Type of veil")
    veil_color: str = Field(..., description="Color of veil")
    number_of_rings: str = Field(..., description="Number of rings on stalk")
    ring_type: str = Field(..., description="Type of ring")
    spore_print_color: str = Field(..., description="Color of spore print")
    population: str = Field(..., description="Population density where found")
    habitat: str = Field(..., description="Habitat where found")

    model_config = {
        "json_schema_extra": {
            "example": {
                "cap_shape": "x",  # convex
                "cap_surface": "s",  # smooth
                "cap_color": "n",  # brown
                "has_bruises": True,
                "odor": "p",  # pungent
                "gill_attachment": "f",  # free
                "gill_spacing": "c",  # close
                "gill_size": "n",  # narrow
                "gill_color": "k",  # black
                "stalk_shape": "e",  # enlarging
                "stalk_surface_above_ring": "s",  # smooth
                "stalk_surface_belows_ring": "s",  # smooth
                "stalk_color_above_ring": "w",  # white
                "stalk_color_below_ring": "w",  # white
                "veil_type": "p",  # partial
                "veil_color": "w",  # white
                "number_of_rings": "o",  # one
                "ring_type": "p",  # pendant
                "spore_print_color": "k",  # black
                "population": "s",  # scattered
                "habitat": "u",  # urban
            }
        },
        "protected_namespaces": (),
    }


class PredictionResponse(BaseModel):
    prediction: int = Field(
        ..., description="Prediction (1: poisonous, 0: not poisonous)"
    )
    probability: float = Field(..., description="Probability of being poisonous")
    prediction_text: str = Field(..., description="Human-readable prediction")
    model_version: str = Field(
        ..., description="Version of the model used for prediction"
    )
    timestamp: str = Field(..., description="Timestamp of the prediction")

    model_config = {"protected_namespaces": ()}


class ModelInfo(BaseModel):
    name: str = Field(..., description="Name of the model")
    version: str = Field(..., description="Version of the model")
    created_at: str = Field(..., description="When the model was created")
    metrics: Dict[str, float] = Field(
        ..., description="Performance metrics of the model"
    )
    description: str = Field(..., description="Description of the model")

    model_config = {"protected_namespaces": ()}


# Global variables for model and preprocessor
model = None
data_processor = None
model_metadata = None


def load_model():
    """
    Load the production model and preprocessor.
    """
    global model, data_processor, model_metadata

    try:
        # Load model metadata
        metadata_path = os.path.join("models", "production", "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                "model_name": "default_model",
                "timestamp": datetime.now().isoformat(),
                "metrics": {"accuracy": 0.0, "f1_score": 0.0},
                "description": "Default model",
            }

        # Load the model
        model_path = os.path.join("models", "production", "model.joblib")
        if not os.path.exists(model_path):
            # If production model doesn't exist, try to find any model
            model_files = [f for f in os.listdir("models") if f.endswith(".joblib")]
            if model_files:
                model_path = os.path.join("models", model_files[0])
            else:
                raise FileNotFoundError("No model file found")

        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load the preprocessor
        data_processor = DataProcessor()
        data_processor.load_preprocessor()
        logger.info("Preprocessor loaded")

        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


# Dependency to ensure model is loaded
async def get_model():
    if model is None:
        success = load_model()
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load model")
    return model


@app.on_event("startup")
async def startup_event():
    """
    Load model and preprocessor on startup.
    """
    load_model()


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    """
    return {
        "message": "Welcome to the Mushroom Classification API",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model/info",
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    """
    if model is None or data_processor is None:
        return {"status": "error", "message": "Model or preprocessor not loaded"}
    return {"status": "ok", "message": "API is healthy"}


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(_=Depends(get_model)):
    """
    Get information about the current model.
    """
    return {
        "model_name": model_metadata.get("model_name", "unknown"),
        "model_version": "1.0.0",
        "created_at": model_metadata.get("timestamp", datetime.now().isoformat()),
        "metrics": model_metadata.get("metrics", {"accuracy": 0.0, "f1_score": 0.0}),
        "description": model_metadata.get(
            "description", "Mushroom classification model"
        ),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(features: MushoomFeatures, model=Depends(get_model)):
    """
    Make a prediction for a mushroom sample.
    """
    try:
        # Convert input to dictionary
        input_data = features.dict()

        # Preprocess the input
        processed_input = data_processor.prepare_inference_input(input_data)

        # Make prediction
        prediction = model.predict(processed_input)[0]

        # Get probability if available
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(processed_input)[0][1])

        # Create response
        prediction_text = "Poisonous" if prediction == 1 else "Not Poisonous"

        return {
            "prediction": int(prediction),
            "probability": probability,
            "prediction_text": prediction_text,
            "model_version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
