FROM python:3.8-slim

ENV PROJECT_DIR model1
WORKDIR /${PROJECT_DIR}
COPY ./model_loader/requirements.txt /${PROJECT_DIR}/
COPY ./model_loader/main.py /${PROJECT_DIR}/src/main.py
RUN apt-get -y update && \
    apt-get -y install apt-utils gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt && \
    touch src/__init__.py

ENV MODEL_FILE https://test-models-iris.s3.ap-northeast-1.amazonaws.com/models/iris_svm.onnx
ENV OUTLIER_FILE https://test-models-iris.s3.ap-northeast-1.amazonaws.com/models/outlier.onnx