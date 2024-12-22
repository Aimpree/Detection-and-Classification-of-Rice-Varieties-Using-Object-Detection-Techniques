import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
from PIL import Image

def Augmentation(Input_Path:str, Rotate:int, Start_Number:int, 
                 Image_Name:str, images_width:int = 5100, images_Height:int = 3750):

    Input = Path(Input_Path)

    boxs = sorted(Input.rglob("*labels/*.txt"))
    images = sorted(Input.rglob("*images/*.jpg"))
    print(images)

    file_paht_txt = {'File_Paht_txt': [str(box) for box in boxs]}
    file_paht_jpg = {'File_Paht_jpg': [str(image) for image in images]}

    file_paht_store_txt = pd.DataFrame(data=file_paht_txt)
    file_paht_store_jpg = pd.DataFrame(data=file_paht_jpg)

    All_File_Path = pd.concat((file_paht_store_txt, file_paht_store_jpg), axis=1)

    container = []
    sub_images = []

    for txt_file , jpg_file in zip(All_File_Path['File_Paht_txt'].values, All_File_Path['File_Paht_jpg'].values):

        with open(txt_file, "r") as file:
            content = file.read()
    
        lines = content.splitlines()
        
        for line in lines:
            values = line.split(' ')
            values.append(jpg_file)  
            container.append(values)

    data = pd.DataFrame(container, columns=['Class', 'X', 'Y', 'W', 'H', 'IMAGES'])

    data['W'] = pd.to_numeric(data['W']) * images_width
    data['H'] = pd.to_numeric(data['H']) * images_Height
    
    data['X'] = (pd.to_numeric(data['X']) * images_width) - (data['W'] / 2)
    data['Y'] = (pd.to_numeric(data['Y']) * images_Height) - (data['H'] / 2)

    current_image = None
    dark_image = None
    sub_images = []
    count = Start_Number

    for idx, row in data.iterrows():
        image_path = row['IMAGES']

        # Detect if the path has changed
        if current_image is None or current_image['path'] != image_path:
            # Save the previously processed dark image, if it exists
            if dark_image is not None:
                dark_image.save(f'output_image_{count}.jpg')
                print(f"Saved: output_image_{count}.jpg")
                count += 1

            # Create a new black background
            dark_image = Image.new('RGB', (images_width, images_Height), color=(0, 0, 0))

            # Open the new image
            img = Image.open(image_path)
            current_image = {'path': image_path, 'image': img}

        # Process the current row
        crop_box = (row['X'], row['Y'], row['X'] + row['W'], row['Y'] + row['H'])
        sub_image = current_image['image'].crop(crop_box).rotate(Rotate, expand=True)
        sub_images.append(sub_image)

        # Paste onto the dark background
        rotated_box = (int(row['X']), int(row['Y']))
        dark_image.paste(sub_image, box=rotated_box)

    # Save the last processed dark image
    if dark_image is not None:
        dark_image.save(f'{Image_Name}_{count}.jpg')
        print(f"Saved: {Image_Name}_{count}.jpg")

    
    


