FROM python:3.10-slim

# Define the build argument to accept the Run ID from GitHub Actions
ARG RUN_ID

# Set the working directory inside the container
WORKDIR /app

# Simulate the model download process during the image build
RUN echo "Downloading model weights from MLflow for Run ID: ${RUN_ID}" > download_log.txt

# Keep the container running briefly to inspect it if needed
CMD ["cat", "download_log.txt"]