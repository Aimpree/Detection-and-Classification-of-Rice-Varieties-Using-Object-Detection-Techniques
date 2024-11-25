# Dockerfile
FROM python:3.9

# Set the working directory inside the container
WORKDIR /code

# Copy main Python script
COPY Kfold.py .

# Install Python packages
RUN pip install -U scikit-learn pandas pyyaml

# Define volume mount points || Add bind mount to Data/Rice_Data and get kfold and yamllist output from Data/Output
VOLUME ["/Data/Rice_Data", "/Data/Output"] 

ENV Input_Path=/Data/Rice_Data Yaml_Path=/Data/Rice_Data/data.yaml Output_Path=/Data/Output Kf=2

# Run the main Python script
CMD ["python", "Kfold.py"]
