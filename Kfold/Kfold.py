from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import datetime
import shutil
import os

def Kfold_Process(Path_Input: str, Yaml_Input: str, Path_Output: str ,ksplit: int):

    pd.set_option('future.no_silent_downcasting', True)

    dataset_path = Path(Path_Input)  # replace with 'path/to/dataset' for your custom data | Path_Input
    Out_path = Path(Path_Output)
    labels = sorted(dataset_path.rglob("*labels/*.txt"))  # all data in 'labels'

    yaml_file = Yaml_Input  # your data YAML with data directories and names dictionary | Yaml_Input
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    indx = [label.stem for label in labels]  # uses base filename as ID (no extension)
    # global labels_df
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in labels:
        lbl_counter = Counter()

        with open(label, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            # classes for YOLO label uses integer at first position of each line
            lbl_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)  # replace `nan` values with `0.0`

    # print(labels_df)
    # global kfolds

    labels_df["Marker"] = ""

    labels_df["Marker"] = labels_df.apply(
    lambda row: 0 if row[0] == 9 else (1 if row[1] == 9 else (2 if row[2] == 9 else None)),
    axis=1
    )

    kf = StratifiedKFold(n_splits=ksplit)
    kfolds = list(kf.split(labels_df, labels_df['Marker']))

    folds = [f"split_{n}" for n in range(1, ksplit + 1)]
    folds_df = pd.DataFrame(index=indx, columns=folds)

    for idx, (train, val) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.iloc[train].index, f"split_{idx}"] = "train"
        folds_df.loc[labels_df.iloc[val].index, f"split_{idx}"] = "val"

    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    # print(fold_lbl_distrb)

    for n, (train_indices, val_indices) in enumerate(kfolds, start=1):
        train_totals = labels_df.iloc[train_indices].sum()
        val_totals = labels_df.iloc[val_indices].sum()

        # To avoid division by zero, we add a small value (1E-7) to the denominator
        ratio = val_totals / (train_totals + 1e-7)
        fold_lbl_distrb.loc[f"split_{n}"] = ratio

    supported_extensions = [".jpg", ".jpeg", ".png"]

    # Initialize an empty list to store image file paths
    images = []

    # Loop through supported extensions and gather image files
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    #print(images)

    # Create the necessary directories and dataset YAML files (unchanged)
    save_path = Path(Out_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val") #Change path
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []


    for split in folds_df.columns:
        # Create directories
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (split_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)

        # Create dataset YAML files
        dataset_yaml = split_dir / f"{split}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": split_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "names": classes,
                },
                ds_y,
            )
  
    for image, label in zip(images, labels):
        for split, k_split in folds_df.loc[image.stem].items():
            # Destination directory
            img_to_path = save_path / split / k_split / "images"
            lbl_to_path = save_path / split / k_split / "labels"

            # Copy image and label files to new directory (SamefileError if file already exists)
            shutil.copy(image, img_to_path / image.name)
            shutil.copy(label, lbl_to_path / label.name)

    with open(os.path.join(Out_path, "Yaml_Split.txt"), "w") as file1:
        for write in ds_yamls:
            toFile = (str(write) + "," + '\n')
            file1.write(toFile)

    print(f'Compleat Create Kfold :{ksplit} fold')
    

if __name__ == "__main__":

    Input_Path = '/home/s6410301038/_workspace/Data-Pipe-Line/data_dont_have_test'
    Yaml_Path = '/home/s6410301038/_workspace/Data-Pipe-Line/data_dont_have_test/data.yaml'
    Output_Path = '/home/s6410301038/_workspace/Data-Pipe-Line/kfold_file'
    Kfold = 5

    Kfold_Process(Input_Path, Yaml_Path, Output_Path, Kfold)
