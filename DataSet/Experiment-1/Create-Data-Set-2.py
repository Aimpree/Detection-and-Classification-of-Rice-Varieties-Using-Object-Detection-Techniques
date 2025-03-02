from pathlib import Path
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
import time

def Create_Sup_Images(Original_Image ,Image_X, Image_Y, Image_Width, Image_Height, padding_x: int = 150, padding_y: int = 100):

    Circle_Radius = max(Image_Width + padding_x, Image_Height + padding_y) // 2
    Center_X = Image_X + Image_Width // 2
    Center_Y = Image_Y + Image_Height // 2

    SPX = int(Center_X - Circle_Radius)
    SPY = int(Center_Y - Circle_Radius)
    EX = int(Center_X + Circle_Radius)
    EY = int(Center_Y + Circle_Radius)

    crop_box1 = (
        max(0, SPX),
        max(0, SPY),
        min(5100, EX),
        min(3750, EY),
    )

    Sub_Image = Original_Image.crop(crop_box1)

    return Sub_Image, (max(0, SPX), max(0, SPY))

def Calculate_Data(Input_List_Of_Data: list, Image_W: int = 5100, Image_H: int = 3750):

    images_width = Image_W
    images_Height = Image_H
    
    Data1 = pd.DataFrame(Input_List_Of_Data, columns=('Class', 'X', 'Y', 'W', 'H'))
    Data2 = pd.DataFrame(columns=('Class', 'X', 'Y', 'W', 'H'))

    Data2['Class'] = Data1['Class']

    Data2['X'] = pd.to_numeric((Data1['X']) * images_width) - (pd.to_numeric(Data1['W']) * images_width / 2)
    Data2['Y'] = pd.to_numeric((Data1['Y']) * images_Height) - (pd.to_numeric(Data1['H']) * images_Height / 2)

    Data2['W'] = pd.to_numeric(Data1['W']) * images_width
    Data2['H'] = pd.to_numeric(Data1['H']) * images_Height

    return Data2

def Create_Mix_Class(Input_Path, Output_Path):

    start = time.time()

    Input = Path(Input_Path)
    Output = Path(Output_Path)

    Output_Folder = (Output / 'Mix Data-set')
    
    Output_Folder.mkdir(exist_ok=True)
    (Output / 'Mix Data-set' / 'labels').mkdir(exist_ok=True)
    (Output / 'Mix Data-set' / 'images').mkdir(exist_ok=True)

    labels_Class_0 = sorted(Input.rglob("*labels/*JM105*.txt"))
    labels_Class_1 = sorted(Input.rglob("*labels/*RD49*.txt"))
    labels_Class_2 = sorted(Input.rglob("*labels/*RD61*.txt"))

    images_Class_0 = sorted(Input.rglob("*images/*JM105*.jpg"))
    images_Class_1 = sorted(Input.rglob("*images/*RD49*.jpg"))
    images_Class_2 = sorted(Input.rglob("*images/*RD61*.jpg"))

    max_images = len(images_Class_0)
    Name = [f"Mix_image" for i in range(max_images)]

    try:
        All_Path = ({
        'Images Class 0': images_Class_0,
        'Labels Class 0': labels_Class_0,
        
        'Images Class 1': images_Class_1,
        'Labels Class 1': labels_Class_1,

        'Images Class 2': images_Class_2,
        'Labels Class 2': labels_Class_2,

        'Names': Name
                })
    except:
        print("Error each class is have differacne length")

    data = pd.DataFrame(All_Path)
    print(data.drop(['Names'], axis=1))
    print(data['Images Class 0'][0][0:-4])
    count = 1

    for idx, row in data.iterrows():

        Full_Class_0 = []
        Full_Class_1 = []
        Full_Class_2 = []

        Label_Out_1 = []
        Label_Out_2 = []
        Label_Out_3 = []

        Label_C0 = row['Labels Class 0']
        Label_C1 = row['Labels Class 1']
        Label_C2 = row['Labels Class 2']

        Image_C0 = row['Images Class 0']
        Image_C1 = row['Images Class 1']
        Image_C2 = row['Images Class 2']

        Name = row['Names']
    
        with open(Label_C0) as f1, open(Label_C1) as f2, open(Label_C2) as f3:

            lines1 = f1.readlines()
            lines2 = f2.readlines()
            lines3 = f3.readlines()

            Label_Out_1.append(lines1[0:3])
            Label_Out_1.append(lines2[3:6])
            Label_Out_1.append(lines3[6:9])

            Label_Out_2.append(lines3[0:3])
            Label_Out_2.append(lines1[3:6])
            Label_Out_2.append(lines2[6:9])

            Label_Out_3.append(lines2[0:3])
            Label_Out_3.append(lines3[3:6])
            Label_Out_3.append(lines1[6:9])

            Full_Class_0.append(lines1)
            Full_Class_1.append(lines2)
            Full_Class_2.append(lines3)

        names_and_labels = [
            (Output_Folder / 'labels' / f'{Name}_{str(count).zfill(4)}.txt', Label_Out_1),
            (Output_Folder / 'labels' / f'{Name}_{str(count + 1).zfill(4)}.txt', Label_Out_2),
            (Output_Folder / 'labels' / f'{Name}_{str(count + 2).zfill(4)}.txt', Label_Out_3)
        ]

        for name, label_out in names_and_labels:
            with open(name, 'w') as f:
                for data in label_out:
                    f.write(data[0])
                    f.write(data[1])
                    f.write(data[2])

        ####
        New_Image_1 = []
        New_Image_2 = []
        New_Image_3 = []

        for data_image_1, data_image_2, data_image_3 in zip(Label_Out_1, Label_Out_2, Label_Out_3):
            for split1, split2, split3 in zip(data_image_1, data_image_2, data_image_3):

                img_C_0 = split1.split()[0]
                img_X_0 = split1.split()[1]
                img_Y_0 = split1.split()[2]
                img_W_0 = split1.split()[3]
                img_H_0 = split1.split()[4]

                img_C_1 = split2.split()[0]
                img_X_1 = split2.split()[1]
                img_Y_1 = split2.split()[2]
                img_W_1 = split2.split()[3]
                img_H_1 = split2.split()[4]

                img_C_2 = split3.split()[0]
                img_X_2 = split3.split()[1]
                img_Y_2 = split3.split()[2]
                img_W_2 = split3.split()[3]
                img_H_2 = split3.split()[4]

                store_image_1 = [img_C_0, float(img_X_0), float(img_Y_0), float(img_W_0), float(img_H_0)]
                New_Image_1.append(store_image_1)

                store_image_2 = [img_C_1, float(img_X_1), float(img_Y_1), float(img_W_1), float(img_H_1)]
                New_Image_2.append(store_image_2)

                store_image_3 = [img_C_2, float(img_X_2), float(img_Y_2), float(img_W_2), float(img_H_2)]
                New_Image_3.append(store_image_3)

        infomations_image_1 = Calculate_Data(New_Image_1)
        infomations_image_2 = Calculate_Data(New_Image_2)
        infomations_image_3 = Calculate_Data(New_Image_3)

        infomations_image_1['Image Path'] = infomations_image_1['Class'].apply(
            lambda x: Image_C0 if x == '0' else Image_C1 if x == '1' else Image_C2 if x == '2' else Image_C2
        )
        infomations_image_2['Image Path'] = infomations_image_2['Class'].apply(
            lambda x: Image_C0 if x == '0' else Image_C1 if x == '1' else Image_C2 if x == '2' else Image_C2
        )
        infomations_image_3['Image Path'] = infomations_image_3['Class'].apply(
            lambda x: Image_C0 if x == '0' else Image_C1 if x == '1' else Image_C2 if x == '2' else Image_C2
        )

        Sub_Images1 =[]
        Positions1 = []

        Sub_Images2 =[]
        Positions2 = []
        
        Sub_Images3 =[]
        Positions3 = []

        for (_, row1), (_, row2), (_, row3) in zip(infomations_image_1.iterrows(), infomations_image_2.iterrows(), infomations_image_3.iterrows()):

            width, height = 5100, 3750

            Image_X_1 = row1['X']
            Image_Y_1 = row1['Y']
            Image_W_1 = row1['W']
            Image_H_1 = row1['H']
            image_1 = row1['Image Path']

            Image_X_2 = row2['X']
            Image_Y_2 = row2['Y']
            Image_W_2 = row2['W']
            Image_H_2 = row2['H']
            image_2 = row2['Image Path']

            Image_X_3 = row3['X']
            Image_Y_3 = row3['Y']
            Image_W_3 = row3['W']
            Image_H_3 = row3['H']
            image_3 = row3['Image Path']

            read_image_1 = Image.open(image_1)
            read_image_2 = Image.open(image_2)
            read_image_3 = Image.open(image_3)
            
            black_image1 = Image.new("RGB", (width, height), color=(0, 0, 0))
            black_image2 = Image.new("RGB", (width, height), color=(0, 0, 0))
            black_image3 = Image.new("RGB", (width, height), color=(0, 0, 0))

            Sub_Image1, Position1 = Create_Sup_Images(read_image_1, Image_X_1, Image_Y_1, Image_W_1, Image_H_1)
            Sub_Image2, Position2 = Create_Sup_Images(read_image_2, Image_X_2, Image_Y_2, Image_W_2, Image_H_2)
            Sub_Image3, Position3 = Create_Sup_Images(read_image_3, Image_X_3, Image_Y_3, Image_W_3, Image_H_3)

            Sub_Images1.append(Sub_Image1)
            Positions1.append(Position1)

            Sub_Images2.append(Sub_Image2)
            Positions2.append(Position2)

            Sub_Images3.append(Sub_Image3)
            Positions3.append(Position3)

        for image1, P1, image2, P2, image3, P3, in zip(Sub_Images1, Positions1, Sub_Images2, Positions2, Sub_Images3, Positions3):

            black_image1.paste(image1, P1)
            black_image2.paste(image2, P2)
            black_image3.paste(image3, P3)
        
        black_image1.save(Output_Folder / 'images' / f'{Name}_{str(count).zfill(4)}.jpg')
        black_image2.save(Output_Folder / 'images' / f'{Name}_{str(count + 1).zfill(4)}.jpg')
        black_image3.save(Output_Folder / 'images' / f'{Name}_{str(count + 2).zfill(4)}.jpg')

        count += 3  

    end = time.time()
    execution_time_in_seconds = end - start
    minutes = (execution_time_in_seconds % 3600) // 60
    seconds = execution_time_in_seconds % 60
    print(f'Success: {minutes} minutes {seconds} sec')

Create_Mix_Class('Input_Path', 'Output_Path')