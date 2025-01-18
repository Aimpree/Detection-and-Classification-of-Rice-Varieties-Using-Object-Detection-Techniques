import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import pandas as pd
import random
from PIL import Image, ImageDraw

def Find_Regtangle_Area(Input_Path: str):
    
    Input = Path(Input_Path)

    Image = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)

    Contours, hierarchy = cv2.findContours(image=Image, 
                                           mode=cv2.RETR_TREE, 
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    Min_Contour_Area = 5000 

    Label_list = []

    for Contour in Contours:
        
        Contour_area = cv2.contourArea(Contour)
        
        if Contour_area > Min_Contour_Area:
            
            x, y, w, h = cv2.boundingRect(Contour)
    
            Label_list.append([x, y, w, h])

    return Label_list


def Calculate_Data(Input_List_Of_Data: list, Class: int, Image_W: int = 5100, Image_H: int = 3750):

    images_width = Image_W
    images_Height = Image_H
    
    Data1 = pd.DataFrame(Input_List_Of_Data, columns=['X', 'Y', 'W', 'H'])
    Data2 = pd.DataFrame(columns=['Class', 'X', 'Y', 'W', 'H'])

    m = np.shape(Data1['X'])
    Data2['Class'] = np.full((m), Class)

    Data2['Class'] = Data2['Class'].astype(str)

    Data2['X'] = (pd.to_numeric(Data1['X']) + pd.to_numeric(Data1['W'] / 2)) / images_width
    Data2['Y'] = (pd.to_numeric(Data1['Y']) + pd.to_numeric(Data1['H'] / 2)) / images_Height

    Data2['W'] = pd.to_numeric(Data1['W']) / images_width
    Data2['H'] = pd.to_numeric(Data1['H']) / images_Height

    return Data2.values


def Create_TXT(Input_List_Of_Data: list, Output_Folder: str, 
               Output_Name: str):

    Output = Path(Output_Folder)
    os.makedirs(Output, exist_ok=True)

    TXT_File_Path = Path(f'{Output}/{Output_Name}')

    with open(TXT_File_Path, 'w') as file:
        for row in Input_List_Of_Data:
            file.write(' '.join(map(str, row)) + '\n')



def Augmentation(Input_Path: str, Output_Path: str):

    Input = Path(Input_Path)
    Output = Path(Output_Path)

    # boxs = sorted(Input.glob("labels/*.txt"))
    images = sorted(Input.glob(f"images/*.jpg"))

    # file_paht_txt = {'File_Paht_txt': [str(box) for box in boxs]}
    file_paht_jpg = {'File_Paht_jpg': [str(image) for image in images]}
    name_jpg = {'Name_jpg': [str(image.name) for image in images]}
    
    # file_paht_store_txt = pd.DataFrame(data=file_paht_txt)
    file_paht_store_jpg = pd.DataFrame(data=file_paht_jpg)
    name_store_jpg = pd.DataFrame(name_jpg)

    All_File_Path = pd.concat((file_paht_store_jpg, name_store_jpg), axis=1)

    container1 = []
    container2 = []

    for jpg, name in zip(All_File_Path['File_Paht_jpg'].values, All_File_Path['Name_jpg'].values):

        datas = Find_Regtangle_Area(jpg)

        for data in datas:
            container1 = [data[0], data[1], data[2], 
                          data[3], jpg, name]
            
            container2.append(container1)

    data = pd.DataFrame(container2, columns=['X', 'Y', 'W', 'H', 'IMAGES', 'NAME'])

    current_image = None
    processed_image = None
    random.seed(5)

    for idx, row in data.iterrows():
        image_path = row['IMAGES']
        image_name = row['NAME']
        
        # Only open the image if the current image is different
        if current_image is None or current_image['path'] != image_path:

            if processed_image is not None:
                # Generate the save name based on original image name
                base_name = current_image['name'].split('.')[0]  # Ensure it only uses the base name, no extra suffix
                save_name = f'{base_name}.jpg'  # Format: a90_RD41_0001.jpg

                Output_Path = os.path.join(Output, save_name)
                processed_image.save(Output_Path)
                # print(f"Saved: {save_name}")

            # Open new image and prepare it for processing
            img = Image.open(image_path)
            current_image = {'path': image_path, 'image': img, 'name': image_name}
            processed_image = img.copy()

        ##########################

        padding_x = 100
        padding_y = 150

        circle_radius = int(max(row['W'] + padding_x, row['H'] + padding_y) // 2)
        center_x = int(row['X'] + row['W'] // 2)
        center_y = int(row['Y'] + row['H'] // 2)

        crop_box = (
            max(0, center_x - circle_radius),
            max(0, center_y - circle_radius),
            min(current_image['image'].width, center_x + circle_radius),
            min(current_image['image'].height, center_y + circle_radius),
        )

        sub_image = current_image['image'].crop(crop_box)

        mask = Image.new("L", sub_image.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, sub_image.size[0], sub_image.size[1]), fill=255)

        circular_cropped = Image.new("RGBA", sub_image.size)
        circular_cropped.paste(sub_image, (0, 0), mask)

        
        a = random.randrange(1, 359)
        rotated_sub_image = circular_cropped.rotate(a, expand=1)

        rotated_box = (
            max(0, center_x - rotated_sub_image.size[0] // 2),
            max(0, center_y - rotated_sub_image.size[1] // 2),
        )
        processed_image.paste(rotated_sub_image, box=rotated_box, mask=rotated_sub_image)

    ##########################

    # Save the last processed image
    if processed_image is not None:
        base_name = current_image['name'].split('.')[0]  # Ensure no extra numbers or suffix
        save_name = f'{base_name}.jpg'

        Output_Path = os.path.join(Output, save_name)
        processed_image.save(Output_Path)
        # print(f"Saved: {save_name}")


def Process(Input_Path: str, Class: int, Randoms: list):

    Input = Path(Input_Path)
    Output_Folder = Input / 'Augmentation File'

    for Random in range(Randoms):

        os.makedirs(Path(f'{Output_Folder}/{Random}/images'), exist_ok=True)
        os.makedirs(Path(f'{Output_Folder}/{Random}/labels'), exist_ok=True)

        Augmentation(
            Input_Path=str(Input),
            Output_Path=Path(f'{Output_Folder}/{Random}/images')
        )

        Images = sorted(Output_Folder.glob(f"{Random}/images/*.jpg"))

        for image in Images:
            Data = Find_Regtangle_Area(image)
            anotation = Calculate_Data(Data, Class)
            Create_TXT(anotation, Path(f'{Output_Folder}/{Random}/labels'), f'{image.name[0:-4]}.txt')


if __name__ == "__main__":


    Input_Path = r'D:\Workflow_project\PREPROCESS FILE JM105'
    Randoms = 1

    Process(Input_Path, 0, Randoms)
