import uuid
from logging import getLogger
from typing import Any, Dict, List
import time 

from fastapi import APIRouter
from src.ml.prediction import Data, classifier
from src.ml.outlier_detection import outlier_detector

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


@router.get("/label")
def label() -> Dict[int, str]:
    return classifier.label


@router.get("/predict/test")
def predict_test() -> Dict[str, Any]:
    result = _predict(Data(), "TEST", True)
    return result 

@router.get("/predict/test/label")
def predict_test_label() -> Dict[str, Any]:
    result = _predict(Data(), "TEST", False)
    return result 

@router.post("/predict")
def predict(data: Data, job_id: str) -> Dict[str, Any]:
    result = _predict(data, job_id, True)
    return result 

@router.post("/predict/label")
def predict_label(data: Data, job_id: str) -> Dict[str, Any]:
    result = _predict(data, job_id, False)
    return result 


def _predict(data: Data, job_id: str, flg: bool) -> Dict[str, Any]:
    start_class = time.time()
    if flg:
        pred = classifier.predict(data.data)
    else:
        pred = classifier.predict_label(data.data)
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
    