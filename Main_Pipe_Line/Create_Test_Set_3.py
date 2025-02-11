from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
import random
import time

    

def Alpha_Add(Sub_Images):
    
    Output = []

    for pil_image in Sub_Images:

        img_np = np.array(pil_image)
        # img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)

        r, g, b = cv2.split(img_np)
        rgba = cv2.merge((r, g, b, thresh)) 
        
        rgba_pil = Image.fromarray(rgba)
        Output.append(rgba_pil)

    return Output

def Random_Rice_9_Position(Item1, Item2, Item3):
    random.seed(14)

    Add_Key1 = {index + 1: value.strip() for index, value in enumerate(Item1)}
    Add_Key2 = {index + 1: value.strip() for index, value in enumerate(Item2)}
    Add_Key3 = {index + 1: value.strip() for index, value in enumerate(Item3)}
    Data_With_Keys = {1: Add_Key1, 2: Add_Key2, 3: Add_Key3}

    # print(Data_With_Keys)
    Pre_data1 = {}
    Pre_data2 = {}
    Pre_data3 = {}
    pre_data_list = [Pre_data1, Pre_data2, Pre_data3]

    
    for i in pre_data_list:
        while len(i) < 9:
            random_key_out = random.choice(list(Data_With_Keys.keys()))
            if Data_With_Keys[random_key_out]:  ## == len(Data_With_Keys) != 0
                random_key_in = random.choice(list(Data_With_Keys[random_key_out].keys()))
                if random_key_in not in i:
                    data = Data_With_Keys[random_key_out].pop(random_key_in)
                    i[random_key_in] = data  + '\n'

    return Pre_data1.values(), Pre_data2.values(), Pre_data3.values()

def Create_Sup_Images(Original_Image ,Image_X, Image_Y, Image_Width, Image_Height, padding_x: int = 10, padding_y: int = 10):


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

    

    count = 1

    for idx, row in data.iterrows():

        # Full_Class_0 = []
        # Full_Class_1 = []
        # Full_Class_2 = []

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

            Label_Out_1, Label_Out_2, Label_Out_3 = Random_Rice_9_Position(lines1, lines2, lines3)

        names_and_labels = [
            (Output_Folder / 'labels' / f'{Name}_{str(count).zfill(4)}.txt', Label_Out_1),
            (Output_Folder / 'labels' / f'{Name}_{str(count + 1).zfill(4)}.txt', Label_Out_2),
            (Output_Folder / 'labels' / f'{Name}_{str(count + 2).zfill(4)}.txt', Label_Out_3)
        ]

        for name, label_out in names_and_labels:
            with open(name, 'w') as f:
                for data in label_out:
                    f.write(data)

        ####
        New_Image_1 = []
        New_Image_2 = []
        New_Image_3 = []

        for data_image_1, data_image_2, data_image_3 in zip(Label_Out_1, Label_Out_2, Label_Out_3):
            data_1 = data_image_1.replace(' ', ',').split(',')
            data_2 = data_image_2.replace(' ', ',').split(',')
            data_3 = data_image_3.replace(' ', ',').split(',')
          
            store_image_1 = [data_1[0], float(data_1[1]), float(data_1[2]), float(data_1[3]), float(data_1[4])]
            New_Image_1.append(store_image_1)

            store_image_2 = [data_2[0], float(data_2[1]), float(data_2[2]), float(data_2[3]), float(data_2[4])]
            New_Image_2.append(store_image_2)

            store_image_3 = [data_3[0], float(data_3[1]), float(data_3[2]), float(data_3[3]), float(data_3[4])]
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

        Sub_Images_Alpha1 = []
        Sub_Images_Alpha2 = []
        Sub_Images_Alpha3 = []

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

            Sub_Images_Alpha1 = Alpha_Add(Sub_Images1)
            Sub_Images_Alpha2 = Alpha_Add(Sub_Images2)
            Sub_Images_Alpha3 = Alpha_Add(Sub_Images3)

        for image1, P1, image2, P2, image3, P3, in zip(Sub_Images_Alpha1, Positions1, Sub_Images_Alpha2, Positions2, Sub_Images_Alpha3, Positions3):

            black_image1.paste(image1, P1, mask=image1)
            black_image2.paste(image2, P2, mask=image2)
            black_image3.paste(image3, P3, mask=image3)
        
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