from fastapi import HTTPException, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import uvicorn
import json
import os
import time
import torch
import shutil
from datetime import datetime
from processors.model_processing import ModelProcessor
from app import app

model_processor = ModelProcessor()
job_history = []  # Store API request history

ACTIVE_MODELS_DIR = "./models"
ARCHIVED_MODELS_DIR = "./models/archive"

# Ensure archive directory exists
os.makedirs(ARCHIVED_MODELS_DIR, exist_ok=True)

def log_request(endpoint: str, request_data: dict, response_data: dict):
    """
    Logs request and response details into job history.
    """
    job_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data
    }
    job_history.append(job_entry)


@app.get("/jobs/history", summary="Retrieve job execution history", tags=["Jobs"])
async def get_job_history():
    """
    Returns a list of all previously executed API calls with timestamps.
    """
    return JSONResponse(content={"job_history": job_history})


@app.get("/models/metadata/{model_name}", summary="Get metadata of a specific model", tags=["Models"])
async def get_model_metadata(model_name: str):
    try:
        model_details = model_processor.get_model_details(model_name)
        log_request("/models/metadata", {"model_name": model_name}, model_details)
        return JSONResponse(content=model_details)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/performance", summary="Get model performance metrics", tags=["Performance"])
async def get_performance_metrics():
    metrics = model_processor.get_performance_metrics()
    log_request("/performance", {}, metrics)
    return JSONResponse(content=metrics)


@app.post("/models/upload", summary="Upload a new model", tags=["Models"])
async def upload_model(uploaded_file: UploadFile = File(...)):
    """
    Uploads a new `.pth` model and automatically detects the architecture.
    """
    try:
        response = model_processor.upload_model(uploaded_file)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/models/enable/{model_name}", summary="Enable a model", tags=["Models"])
async def enable_model(model_name: str):
    """
    Moves a model from the archive to the active models directory using ModelProcessor.
    """
    try:
        response = model_processor.enable_model(model_name)
        log_request("/models/enable", {"model_name": model_name}, response)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/disable/{model_name}", summary="Disable a model", tags=["Models"])
async def disable_model(model_name: str):
    """
    Moves a model from the active models directory to the archive using ModelProcessor.
    """
    try:
        response = model_processor.disable_model(model_name)
        log_request("/models/disable", {"model_name": model_name}, response)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", summary="List available models", tags=["Models"])
async def list_models():
    """
    Lists all active models in the `./models/` directory.
    """
    try:
        models = [file.replace(".pth", "") for file in os.listdir(ACTIVE_MODELS_DIR) if file.endswith(".pth")]
        log_request("/models", {}, {"models": models})
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/archived", summary="List archived models", tags=["Models"])
async def list_archived_models():
    """
    Lists all archived models in the `./models/archive/` directory.
    """
    try:
        models = [file.replace(".pth", "") for file in os.listdir(ARCHIVED_MODELS_DIR) if file.endswith(".pth")]
        log_request("/models/archived", {}, {"archived_models": models})
        return {"archived_models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/load/{model_name}", summary="Load a model for inference", tags=["Models"])
async def load_model(model_name: str):
    try:
        model_processor.load_model(model_name)
        response = {"message": f"Model '{model_name}' loaded successfully"}
        log_request("/models/load", {"model_name": model_name}, response)
        return response
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/inference/execution", summary="Run model inference and update JSON", tags=["Inference"])
async def inference_execution(
    json_string: str = Form(...),
    image: UploadFile = File(...)
):
    try:
        if model_processor.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

        try:
            json_dict = json.loads(json_string)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        image_bytes = await image.read()
        for entry in json_dict.get("data", []):
            try:
                result = model_processor.run_inference(image_bytes)
                entry["geoai_analysis"] = {
                    "prediction": result["prediction"],
                    "confidence": result["confidence"]
                }
            except Exception as e:
                entry["geoai_analysis"] = {"error": str(e)}

        log_request("/inference/execution", {"json_string": json_dict}, json_dict)
        return JSONResponse(content=json_dict)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/inference/batch_execution", summary="Run model inference on multiple images", tags=["Inference"])
async def batch_inference(json_string: str = Form(...), images: list[UploadFile] = File(...)):
    try:
        if model_processor.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Load a model first.")

        try:
            json_data = json.loads(json_string)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        if not images:
            raise HTTPException(status_code=400, detail="No images uploaded")

        results = []
        for img in images:
            image_bytes = await img.read()
            if not image_bytes:
                continue  

            result = model_processor.run_inference(image_bytes)
            results.append({
                "id": img.filename,
                "prediction": result["prediction"],
                "confidence": result["confidence"]
            })

        log_request("/inference/batch_execution", {"json_string": json_data, "num_images": len(images)}, results)
        return JSONResponse(content={"results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/history", summary="Retrieve job execution history", tags=["Jobs"])
async def get_job_history():
    return JSONResponse(content={"job_history": job_history})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
