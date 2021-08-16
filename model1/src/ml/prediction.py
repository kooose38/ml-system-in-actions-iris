import json
from logging import getLogger
from typing import Dict, List, Sequence
from io import BytesIO
import joblib 

import numpy as np
import onnxruntime as rt
from pydantic import BaseModel
from src.configurations import ModelConfigurations
from sklearn.preprocessing import StandardScaler

logger = getLogger(__name__)


class Data(BaseModel):
    data: List[List[float]] = [[5.1, 3.5, 1.4, 0.2]]


class Classifier(object):
    def __init__(
        self,
        model_filepath: str,
        label_filepath: str,
        data_filepath: str,
    ):
        self.model_filepath: str = model_filepath
        self.label_filepath: str = label_filepath
        self.data_filepath = data_filepath
        self.classifier = None
        self.label: Dict[str, str] = {}
        self.input_name: str = ""
        self.output_name: str = ""

        self.load_model()
        self.load_label()

    def load_model(self):
        logger.info(f"load model in {self.model_filepath}")
        with open(self.model_filepath, "rb") as f:
            model_data = BytesIO(f.read())
        self.classifier = rt.InferenceSession(
            model_data.read()
        )
        # inputの数に応じて追加する
        self.input_name = self.classifier.get_inputs()[0].name
        self.output_name = self.classifier.get_outputs()[0].name
        logger.info(f"initialized model")

    def load_label(self):
        logger.info(
            f"load label in {self.label_filepath}",
        )
        with open(self.label_filepath, "r") as f:
            self.label = json.load(f)
        logger.info(f"label: {self.label}")

    def transform(self, data: np.ndarray) -> np.ndarray:
        scaler = StandardScaler()
        scaler.fit(np.load(self.data_filepath))
        data = scaler.transform(data)
        return data 
        

    def predict(self, data: List[List[float]]) -> list:
        data = np.array(data).reshape(1, -1).astype(np.float32)
        data = self.transform(data)
        pred = self.classifier.run(None, {self.input_name: data})

        pred_proba = pred[1][0].tolist()
        return pred_proba 

    def predict_label(self, data: List[List[float]]) -> str:
        pred_proba = self.predict(data=data)
        argmax = int(np.array(pred_proba).reshape(1, -1).argmax(-1)[0])
        return self.label[str(argmax)]


classifier = Classifier(
    model_filepath=ModelConfigurations().model_filepath,
    label_filepath=ModelConfigurations().label_filepath,
    data_filepath=ModelConfigurations().data_filepath
)