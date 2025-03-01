from ultralytics import YOLO
from pathlib import Path
import torch
import os
import csv


def Train_Model(Yaml_List: list, Output_Path: str ,ksplit = 1):

    Out_Path = Path(Output_Path)

    Out_Path_Train = Path(Out_Path / "Train")
    Out_Path_Validate = Path(Out_Path / "Validate")
    
    Out_Path_Train.mkdir(parents=True, exist_ok=True)
    Out_Path_Validate.mkdir(parents=True, exist_ok=True)

    file = open(Yaml_List, "r")
    content = file.read()

    X = content.split(',')

    Weights_Path = "./yolo11m.pt"
    Use_Model = YOLO(Weights_Path, task="detect")

    results = {}

    batch = 8
    epochs = 300
    patience = 50
    imgsz = 640
    conf = 0.5
    iou = 0.6

    for k in range(ksplit):

        dataset_yaml = X[k]
        
        
        Use_Model.train(data=dataset_yaml, epochs=epochs, batch=batch, project=Out_Path_Train, 
                        patience=patience, imgsz=imgsz, name=f"fold_{k}_train") 

        
        val_metrics = Use_Model.val(data=dataset_yaml, batch=batch, project=Out_Path_Validate, 
                                    imgsz=imgsz, name=f"fold_{k}_val", conf=conf, iou=iou)


        results[k] = {
            "train_metrics": Use_Model.metrics,  
            "val_metrics": val_metrics 
        }



    
if __name__ == "__main__":

    Ymal_List = 'Yaml_Split.txt'
    Out_path = 'Output_Path'

    Train_Model(Ymal_List, Out_path)

