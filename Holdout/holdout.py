from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import datetime
import shutil
import os



def Holdout_Process(Path_Input: str, Yaml_Input: str, Path_Output: str, Seed: int):
    pd.set_option('future.no_silent_downcasting', True)

    dataset_path = Path(Path_Input)
    Out_path = Path(Path_Output)
    labels = sorted(dataset_path.rglob("*labels/*.txt"))

    yaml_file = Yaml_Input
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]

    indx = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=classes.keys(), index=indx)

    for label in labels:
        lbl_counter = Counter()
        with open(label, "r") as lf:
            lines = lf.readlines()
        for line in lines:
            lbl_counter[int(line.split(" ")[0])] += 1
        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)
    labels_df["Marker"] = labels_df.apply(
        lambda row: 0 if row[0] == 9 else (1 if row[1] == 9 else (2 if row[2] == 9 else None)), axis=1
    )

    feature = labels_df.drop("Marker", axis=1)
    X_train_v, X_test, y_train_v, y_test = train_test_split(
        feature, labels_df["Marker"], test_size=0.2, stratify=labels_df["Marker"], random_state=Seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_v, y_train_v, test_size=0.25, stratify=y_train_v, random_state=Seed
    )

    save_path = Path(Out_path / f"{datetime.date.today().isoformat()}-Holdout")
    save_path.mkdir(parents=True, exist_ok=True)

    splits = {"train": (X_train, save_path / "train"), 
              "val": (X_val, save_path / "val"), 
              "test": (X_test, save_path / "test")}

    for split, (data, split_path) in splits.items():
        (split_path / "images").mkdir(parents=True, exist_ok=True)
        (split_path / "labels").mkdir(parents=True, exist_ok=True)

        for idx in data.index:
            img_path = dataset_path / "images" / f"{idx}.jpg"  # Adjust extension as needed
            lbl_path = dataset_path / "labels" / f"{idx}.txt"

            if img_path.exists() and lbl_path.exists():
                shutil.copy(img_path, split_path / "images" / img_path.name)
                shutil.copy(lbl_path, split_path / "labels" / lbl_path.name)
            else:
                print(f"Missing file: {img_path if not img_path.exists() else lbl_path}")

    dataset_yaml = save_path / f"dataset.yaml"
    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump(
            {
                "path": save_path.as_posix(),
                "train": "train",
                "val": "val",
                "test": "test",
                "names": classes,
            },
            ds_y,
        )
