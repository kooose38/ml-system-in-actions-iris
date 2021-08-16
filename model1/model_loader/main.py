
import os
from logging import DEBUG, Formatter, StreamHandler, getLogger

import requests 
import click 

logger = getLogger(__name__)
logger.setLevel(DEBUG)
strhd = StreamHandler()
strhd.setFormatter(Formatter("%(asctime)s %(levelname)8s %(message)s"))
logger.addHandler(strhd)

@click.command(name="model loader")
@click.option("--model_filepath", type=str, required=True, help="Local model file path")
@click.option("--outlier_filepath", type=str, required=True, help="Local outlier model file path")
def main(model_filepath: str, outlier_filepath: str):
    url = os.getenv("MODEL_FILE", "https://test-models-iris.s3.ap-northeast-1.amazonaws.com/models/iris_svm.onnx")
    dirname = os.path.dirname(model_filepath)
    os.makedirs(dirname, exist_ok=True)
    logger.info(f"start dowmload from {url}")

    res = requests.get(url)

    with open(model_filepath, "wb") as f:
       f.write(res.content)

    logger.info(f"complete loading tasks form {url} to {dirname}")

    url_ = os.getenv("OUTLIER_FILE", "https://test-models-iris.s3.ap-northeast-1.amazonaws.com/models/outlier.onnx")
    dirname_ = os.path.dirname(outlier_filepath)
    os.makedirs(dirname_, exist_ok=True)
    logger.info("start download outlier models from {url}")

    res_ = requests.get(url_)

    with open(outlier_filepath, "wb") as f_:
        f_.write(res_.content)
    logger.info(f"complete loading tasks from {url} to {dirname}")

if __name__ == "__main__":
    main()