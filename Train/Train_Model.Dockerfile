# Dockerfile
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy main Python script
COPY Train_Process.py .

# Install Python packages
RUN pip install ultralytics torch onnxruntime

# Define volume mount points
VOLUME ["/Data/Output"]

ENV Input_Path=/Data/Output/Yaml_Split.txt Kf=2

# Run the main Python script
CMD ["python", "Train_Process.py"]
