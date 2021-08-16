from logging import getLogger
from typing import Any, Dict, List
import time 

from fastapi import APIRouter, HTTPException 
from src.ml.prediction import Data, classifier
from src.ml.outlier_detection import outlier_detector
from src.utils.profiler import log_decorator

logger = getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {
        "health": "ok",
    }


@router.get("/metadata")
def metadata() -> Dict[str, Any]:
    return {
        "data_type": "float32",
        "data_structure": (1,4),
        "data_sample": [[5.1, 3.5, 1.4, 0.2]],
        "prediction_type": "float32",
        "prediction_structure": (1,3),
        "prediction_sample": [0.97093159, 0.01558308, 0.01348537],
    }


##########################################################################
@router.get("/label")
def label() -> Dict[int, str]:
    return classifier.label


##########################################################################
@log_decorator(endpoint="/predict/test", logger=logger)
def _predict_test(data: Data, job_id: str) -> Dict[str, Any]:
    start_class = time.time()
    pred: List[float] = classifier.predict(data.data)
    end_elapsed = 1000 * (time.time() - start_class)

    start_outlier = time.time()
    is_outlier, outlier_scoer = outlier_detector.predict(data.data)
    outlier_elapsed = 1000 * (time.time() - start_outlier)

    return {
        "job_id": job_id,
        "data": data.data,
        "prediction": pred,
        "prediction_elapsed": end_elapsed,
        "outlier_score": outlier_scoer,
        "is_outlier": is_outlier,
        "outlier_elapsed": outlier_elapsed
    }

@router.get("/predict/test")
def predict_test() -> Dict[str, Any]:
    result = _predict_test(Data(), "TEST")
    return result 


##########################################################################
@log_decorator(endpoint="/predict/test/label", logger=logger)
def _predict_test_label(data: Data, job_id: str) -> Dict[str, Any]:
    start_class = time.time()
    pred: List[float] = classifier.predict_label(data.data)
    end_elapsed = 1000 * (time.time() - start_class)

    start_outlier = time.time()
    is_outlier, outlier_scoer = outlier_detector.predict(data.data)
    outlier_elapsed = 1000 * (time.time() - start_outlier)

    return {
        "job_id": job_id,
        "data": data.data,
        "prediction": pred,
        "prediction_elapsed": end_elapsed,
        "outlier_score": outlier_scoer,
        "is_outlier": is_outlier,
        "outlier_elapsed": outlier_elapsed
    }

@router.get("/predict/test/label")
def predict_test_label() -> Dict[str, Any]:
    result = _predict_test_label(Data(), "TEST")
    return result 


##########################################################################
@log_decorator(endpoint="/predict", logger=logger)
def _predict(data: Data, job_id: str) -> Dict[str, Any]:
    if len(data.data) != 1 or len(data.data[0]) != 4:
        raise HTTPException(status_code=404, detail="invalid input data")

    start_class = time.time()
    pred: List[float] = classifier.predict(data.data)
    end_elapsed = 1000 * (time.time() - start_class)

    start_outlier = time.time()
    is_outlier, outlier_scoer = outlier_detector.predict(data.data)
    outlier_elapsed = 1000 * (time.time() - start_outlier)

    return {
        "job_id": job_id,
        "data": data.data,
        "prediction": pred,
        "prediction_elapsed": end_elapsed,
        "outlier_score": outlier_scoer,
        "is_outlier": is_outlier,
        "outlier_elapsed": outlier_elapsed
    }

@router.post("/predict")
def predict(data: Data, job_id: str) -> Dict[str, Any]:
    result = _predict(data, job_id)
    return result 


##########################################################################
@log_decorator(endpoint="/predict/label", logger=logger)
def _predict_label(data: Data, job_id: str) -> Dict[str, Any]:
    if len(data.data) != 1 or len(data.data[0]) != 4:
        raise HTTPException(status_code=404, detail="invalid input data")
        
    start_class = time.time()
    pred: str = classifier.predict_label(data.data)
    end_elapsed = 1000 * (time.time() - start_class)

    start_outlier = time.time()
    is_outlier, outlier_scoer = outlier_detector.predict(data.data)
    outlier_elapsed = 1000 * (time.time() - start_outlier)

    return {
        "job_id": job_id,
        "data": data.data,
        "prediction": pred,
        "prediction_elapsed": end_elapsed,
        "outlier_score": outlier_scoer,
        "is_outlier": is_outlier,
        "outlier_elapsed": outlier_elapsed
    }
    

@router.post("/predict/label")
def predict_label(data: Data, job_id: str) -> Dict[str, Any]:
    result = _predict_label(data, job_id)
    return result 

##########################################################################
    
