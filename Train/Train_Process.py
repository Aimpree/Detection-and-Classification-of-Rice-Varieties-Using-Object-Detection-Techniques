from ultralytics import YOLO
from pathlib import Path
import torch
import os
import csv


def Train_Model(Yaml_List: list, Output_Path: str ,ksplit: int):

    Out_Path = Path(Output_Path)
    # In_Path = Path(Input_Path)

    Out_Path_Train = Path(Out_Path / "Train")
    Out_Path_Validate = Path(Out_Path / "Validate")
    Out_Path_Test = Path(Out_Path / "Test")
    
    Out_Path_Train.mkdir(parents=True, exist_ok=True)
    Out_Path_Validate.mkdir(parents=True, exist_ok=True)
    Out_Path_Test.mkdir(parents=True, exist_ok=True)

    file = open(Yaml_List, "r")
    content = file.read()

    X = content.split(',')

    Weights_Path = "./yolov8n.pt"
    Use_Model = YOLO(Weights_Path, task="detect")

    results = {}

    batch = 32
    epochs = 400
    patience = 50
    imgsz = 640

    for k in range(ksplit):

        dataset_yaml = X[k]
        
        
        Use_Model.train(data=dataset_yaml, epochs=epochs, batch=batch, project=Out_Path_Train, 
                        patience=patience, imgsz=imgsz, name=f"fold_{k}_train") 

        
        val_metrics = Use_Model.val(data=dataset_yaml, batch=batch, project=Out_Path_Validate, 
                                    imgsz=imgsz, name=f"fold_{k}_val")


        test_metrics = Use_Model.val(split="test", data=dataset_yaml, batch=batch, 
                                    project=Out_Path_Test, imgsz=imgsz, name=f"fold_{k}_test")


        results[k] = {
            "train_metrics": Use_Model.metrics,  
            "val_metrics": val_metrics, 
            "test_metrics": test_metrics
        }



    
if __name__ == "__main__":

    Ymal_List = '/home/s6410301038/_workspace/Data-Pipe-Line/kfold_file/Yaml_Split.txt'
    Out_path = '/home/s6410301038/_workspace/Data-Pipe-Line/result_all/result_12-4-2024:11:35'
    Kfold = 5

    Train_Model(Ymal_List, Out_path, Kfold)

