from io import BytesIO
from logging import getLogger
from typing import Any, Tuple, List
from io import BytesIO

import numpy as np
from src.configurations import ModelConfigurations
import onnxruntime as rt

logger = getLogger(__name__)

class OutlierDetector(object):
    def __init__(self, model_path: str):
       self.outlier_filepath: str = model_path
       self.outlier = None 
       self.input_name = ""
       self.output_name = "" 

       self.load_model()

    def load_model(self):
       logger.info(f"load model in {self.outlier_filepath}")
       with open(self.outlier_filepath, "rb") as f:
          model_data = BytesIO(f.read())

       self.outlier = rt.InferenceSession(
          model_data.read()
       )
       self.input_name = self.outlier.get_inputs()[0].name 
       self.output_name = self.outlier.get_outputs()[0].name 
       logger.info("intialized model")

    def predict(self, data: List[List[float]]) -> Tuple[bool, float]:
        np_data = np.array(data).reshape(1, -1).astype(np.float32)
        pred = self.outlier.run(
           None, {self.input_name: np_data}
        )
        outlier_score: float = float(pred[1][0][0])
        is_outlier: bool = outlier_score < 0.0
        return is_outlier, outlier_score

outlier_detector = OutlierDetector(
   model_path=ModelConfigurations().outlier_filepath
)