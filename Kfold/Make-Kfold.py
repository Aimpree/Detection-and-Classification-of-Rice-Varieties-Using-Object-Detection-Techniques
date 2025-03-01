from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import StratifiedKFold
import datetime
import shutil
import os


def Kfold_Process(Path_Input: str, Yaml_Input: str, Path_Output: str, ksplit: int):
    pd.set_option('future.no_silent_downcasting', True)

    dataset_path = Path(Path_Input)
    Out_path = Path(Path_Output)
    labels = sorted(dataset_path.rglob("*labels/*.txt"))

    yaml_file = Yaml_Input
    with open(yaml_file, "r", encoding="utf8") as y:
        classes = yaml.safe_load(y)["names"]
    cls_idx = sorted(classes.keys())

    indx = [label.stem for label in labels]
    labels_df = pd.DataFrame([], columns=cls_idx, index=indx)

    for label in labels:
        lbl_counter = Counter()

        with open(label, "r") as lf:
            lines = lf.readlines()

        for line in lines:
            lbl_counter[int(line.split(" ")[0])] += 1

        labels_df.loc[label.stem] = lbl_counter

    labels_df = labels_df.fillna(0.0)
    labels_df["Marker"] = labels_df.apply(
        lambda row: 0 if row[0] == 9 else (1 if row[1] == 9 else (2 if row[2] == 9 else None)),
        axis=1
    )

    kf = StratifiedKFold(n_splits=ksplit, shuffle=True, random_state=42)
    supported_extensions = [".jpg", ".jpeg", ".png"]
    images = []
    for ext in supported_extensions:
        images.extend(sorted((dataset_path / "images").rglob(f"*{ext}")))

    save_path = Path(Out_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val")
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = []

    for fold_idx, (train_val_indices, test_indices) in enumerate(kf.split(labels_df, labels_df['Marker']), start=1):
        fold_dir = save_path / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        test_dir = fold_dir / "test"
        (test_dir / "images").mkdir(parents=True, exist_ok=True)
        (test_dir / "labels").mkdir(parents=True, exist_ok=True)

        # Split train/val (144 images into train (108) and val (36))
        train_val_df = labels_df.iloc[train_val_indices]
        test_df = labels_df.iloc[test_indices]

        inner_kf = StratifiedKFold(n_splits=ksplit - 1, shuffle=True, random_state=2)
        train_indices, val_indices = next(inner_kf.split(train_val_df, train_val_df['Marker']))

        train_df = train_val_df.iloc[train_indices]
        val_df = train_val_df.iloc[val_indices]

        # Create directories for train, val
        train_dir = fold_dir / "train"
        val_dir = fold_dir / "val"
        (train_dir / "images").mkdir(parents=True, exist_ok=True)
        (train_dir / "labels").mkdir(parents=True, exist_ok=True)
        (val_dir / "images").mkdir(parents=True, exist_ok=True)
        (val_dir / "labels").mkdir(parents=True, exist_ok=True)

        # YAML for the fold
        dataset_yaml = fold_dir / f"fold_{fold_idx}_dataset.yaml"
        ds_yamls.append(dataset_yaml)

        with open(dataset_yaml, "w") as ds_y:
            yaml.safe_dump(
                {
                    "path": fold_dir.as_posix(),
                    "train": "train",
                    "val": "val",
                    "test": "test",
                    "names": classes,
                },
                ds_y,
            )

        # Copy train data
        for idx in train_df.index:
            image = next((img for img in images if img.stem == idx), None)
            label = next((lbl for lbl in labels if lbl.stem == idx), None)
            if image and label:
                shutil.copy(image, train_dir / "images" / image.name)
                shutil.copy(label, train_dir / "labels" / label.name)

        # Copy val data
        for idx in val_df.index:
            image = next((img for img in images if img.stem == idx), None)
            label = next((lbl for lbl in labels if lbl.stem == idx), None)
            if image and label:
                shutil.copy(image, val_dir / "images" / image.name)
                shutil.copy(label, val_dir / "labels" / label.name)

        # Copy test data
        for idx in test_df.index:
            image = next((img for img in images if img.stem == idx), None)
            label = next((lbl for lbl in labels if lbl.stem == idx), None)
            if image and label:
                shutil.copy(image, test_dir / "images" / image.name)
                shutil.copy(label, test_dir / "labels" / label.name)

    # Write YAML paths to file
    with open(os.path.join(Out_path, "Yaml_Split.txt"), "w") as file1:
        for write in ds_yamls:
            toFile = (str(write) + "," + '\n')
            file1.write(toFile)

    print(f'Complete Kfold creation: {ksplit} folds with train/val/test splits')

if __name__ == "__main__":
    Input_Path = 'C:/Users/Aimpr/OneDrive/Desktop/Test_dataset'
    Yaml_Path = 'C:/Users/Aimpr/OneDrive/Desktop/Test_dataset/data.yaml'
    Output_Path = 'C:/Users/Aimpr/OneDrive/Desktop/Test_dataset'
    Kfold = 5

    Kfold_Process(Input_Path, Yaml_Path, Output_Path, Kfold)
