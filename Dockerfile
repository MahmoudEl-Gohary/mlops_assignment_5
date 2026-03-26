FROM python:3.10-slim

ARG RUN_ID

WORKDIR /app

RUN echo "Downloading model weights from MLflow for Run ID: ${RUN_ID}" > download_log.txt

CMD ["cat", "download_log.txt"]