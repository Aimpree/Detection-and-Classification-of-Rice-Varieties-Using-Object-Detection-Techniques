from ultralytics import YOLO
import os

def Train_Model(Yaml_List: list, ksplit: int):

    file = open(Yaml_List, "r")
    content = file.read()

    X = content.split(',')

    # Create a new model
    model = YOLO('yolov8n.yaml')

    # Load pretrained model to avoid training from scratch
    model = YOLO('yolov8n.pt', task="detect")

    results = {}

    # Define your parameters
    batch = 16
    project = "kfold_3Class_rice"
    epochs = 100

    # Assume ksplit and ds_yamls are defined
    for k in range(ksplit):
        dataset_yaml = X[k]
        
        # Train the model on the k-th split
        model.train(data=dataset_yaml, epochs=epochs, batch=batch, project=project)  # include any train arguments

        # Validate after training
        val_metrics = model.val(data=dataset_yaml, batch=batch)

        # Store both training metrics and validation results
        results[k] = {
            "train_metrics": model.metrics,  # save training metrics
            "val_metrics": val_metrics  # save validation metrics
        }

    
if __name__ == "__main__":

    Input_Path = os.getenv("Input_Path")
    Kfold = int(os.getenv("Kf"))

    Train_Model(Input_Path, Kfold)

