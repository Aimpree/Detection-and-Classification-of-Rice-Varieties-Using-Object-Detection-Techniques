from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import random
import time
import math

def Alpha_Add(Sub_Images):
    
    img_np = np.array(Sub_Images)
    img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(img_gray, 5, 255, cv2.THRESH_BINARY)

    r, g, b = cv2.split(img_np)
    rgba = cv2.merge((r, g, b, thresh)) 
    
    rgba_pil = Image.fromarray(rgba)

    return rgba_pil

def Create_Images(Input_DataFrame_1, Input_DataFrame_2, Output_Path):

    Concat_Data = []

    width, height = 5100, 3750
    

    for range_list in range(len(Input_DataFrame_1)):

        Input_DataFrame_2[range_list].reverse()

        for Data1, Data2 in  zip(Input_DataFrame_1[range_list], Input_DataFrame_2[range_list]):

            Concat_Data.append(pd.concat((Data1, Data2), axis=0))
    
        for count, Data in enumerate(Concat_Data):
            
            Save_Lables = Data.drop(columns=['Images', 'Position'])
            Save_Lables.to_csv(Output_Path / 'labels' / f'Imbalance_Mix_{str(count).zfill(4)}.txt', index=False, header=False, sep=' ')
            black_image = Image.new("RGB", (width, height), color=(0, 0, 0))
            
            for _, row in Data.iterrows():

                image = row['Images']
                position = row['Position']

                images_add_alpha = Alpha_Add(image)
                black_image.paste(images_add_alpha, position, mask=images_add_alpha)

            black_image.save(Output_Path / 'images' / f'Imbalance_Mix_{str(count).zfill(4)}.png')
            

##ใส่ ทั้งสองอันเข้ามา ทั้งสอง DataFrame

def Mix_Instances_With_Differance_Ration(Input_DataFrame, max_instances, Ration=[10, 20, 30, 40, 50, 60, 70, 80, 90]):

    max_row = len(Input_DataFrame)
    Result_Final = []  # Stores the final structured output
    index_after_split = [0]

    while index_after_split[-1] < max_row:
        temp_indices = [index_after_split[-1]]
        result_9 = []  # Temporary list to hold values for a single round

        for i in Ration:
            # if i is not None:
            next_index = temp_indices[-1] + int((max_instances * i) / 100)
            if next_index >= max_row:
                next_index = max_row
                temp_indices.append(next_index)
                break  

            temp_indices.append(next_index)

        for start_index, stop_index in zip(temp_indices[:-1], temp_indices[1:]):
            result_9.append(Input_DataFrame.iloc[start_index:stop_index])  # Append each slice to the temp list
        
        Result_Final.append(result_9)  # Append each round's result as a separate list
        index_after_split = [temp_indices[-1]]

    return Result_Final


def Random_Position(Input_DataFrame):

    Data1 = pd.DataFrame(Input_DataFrame, columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    Data2 = pd.DataFrame(columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))

    X = [random.random() for _ in range(len(Data1))]
    Y = [random.random() for _ in range(len(Data1))]
    
    Data2['Class'] = Data1['Class']
    Data2['Images_Path'] = Data1['Images_Path']

    Data2['X'] = X
    Data2['Y'] = Y

    Data2['W'] = Data1['W']
    Data2['H'] = Data1['H']

    return Data2

def Calcultae_Data(Input_DataFrame: list, Image_W: int = 5100, Image_H: int = 3750):

    images_width = Image_W
    images_Height = Image_H
    
    Data1 = pd.DataFrame(Input_DataFrame, columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    Data2 = pd.DataFrame(columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))

    Data2['Class'] = Data1['Class']
    Data2['Images_Path'] = Data1['Images_Path']

    Data2['X'] = pd.to_numeric((Data1['X']) * images_width) - (pd.to_numeric(Data1['W']) * images_width / 2)
    Data2['Y'] = pd.to_numeric((Data1['Y']) * images_Height) - (pd.to_numeric(Data1['H']) * images_Height / 2)

    Data2['W'] = pd.to_numeric(Data1['W']) * images_width
    Data2['H'] = pd.to_numeric(Data1['H']) * images_Height

    return Data2

def Stored_Small_Images(Input_DataFrame):

    Info = Input_DataFrame
    Info_DataFrame = pd.DataFrame(Info, columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    Stored_All_Images = []

    New_Position_DataFrame = Random_Position(Info_DataFrame) ##ยังไม่ได้เปลี่ยนอะไร

    Calculate_to_Create_Sup_Images = Calcultae_Data(Info_DataFrame)
    Calculate_New_Position = Calcultae_Data(New_Position_DataFrame)

    def Create_Small_Images(Original_Image ,Image_X, Image_Y, Image_Width, Image_Height, padding_x: int = 10, padding_y: int = 10):


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
    
    for (index1, row1), (index2, row2) in zip(Calculate_to_Create_Sup_Images.iterrows(), Calculate_New_Position.iterrows()):

        image_path1 = row1['Images_Path']

        read_image = Image.open(image_path1)

        image_x1, image_x2 = row1['X'], row2['X']
        image_y1, image_y2 = row1['Y'], row2['Y']
        image_width1, image_width2 = row1['W'], row2['W']
        image_height1, image_hidth2 = row1['H'], row2['H']
      
        Images, _ = Create_Small_Images(read_image, image_x1, image_y1, image_width1, image_height1)
        _, Position = Create_Small_Images(read_image, image_x2, image_y2, image_width2, image_hidth2)

        Pack_Images = {"Images": Images, "Position": Position}
        Stored_All_Images.append(Pack_Images)

    Images_Obj_and_Position = pd.DataFrame(Stored_All_Images)
    New_Position_DataFrame_drop = New_Position_DataFrame.drop('Images_Path', axis=1)
    Images_and_Labels = pd.concat((New_Position_DataFrame_drop, Images_Obj_and_Position), axis=1)

    return Images_and_Labels


def Extract_Labels(Input_DataFrame, Instance = 'defult'):

    Info = Input_DataFrame
    Info_DataFrame = pd.DataFrame(Info, columns=("Images_Path", "Lables_Path"))

    Stored = []

    def read_Instance(Label_Path, Images_Path):

        all_instance_info = []

        match Instance:

            case 'defult':

                with open(Label_Path) as f:

                    lines = f.readlines()  

                    for line in lines:

                        split_info = line.replace(' ', ',').split(',')
                        instance_info = split_info[0], float(split_info[1]), float(split_info[2]), float(split_info[3]), float(split_info[4]), Images_Path
                        all_instance_info.append(instance_info)

                return all_instance_info
        
            case 'type 1':

                with open(Label_Path) as f:

                    lines = f.readlines()  

                    for line in lines:

                        split_info = line.replace(' ', ',').split(',')
                        instance_info = 1, float(split_info[1]), float(split_info[2]), float(split_info[3]), float(split_info[4]), Images_Path
                        all_instance_info.append(instance_info)

                while len(all_instance_info) > 3:
                
                    random.seed(Random_Seed)
                    index_to_remove = random.randrange(0, len(all_instance_info))
                    all_instance_info.pop(index_to_remove)

                return all_instance_info
    
    for index, row in Info_DataFrame.iterrows():

        image = row['Images_Path']
        label = row['Lables_Path']

        Stored.extend(read_Instance(label, image))
        global Random_Seed
        Random_Seed += 1

    return Stored


def Check_Labels(Input_Path, Output_Path):

    Input = Path(Input_Path)
    Output = Path(Output_Path)
    Output.mkdir(parents=True, exist_ok=True)  

    Labels = sorted(Input.rglob('labels/*.txt'))
    Score = {'Class JM105': 0, 'Class Other': 0}  

    for label_path in Labels:
        with open(label_path, 'r') as file:
            lines = file.readlines()

            for line in lines:
                split_info = line.split()  
                class_name = split_info[0]

                if class_name == '0':
                    Score['Class JM105'] += 1
                else:
                    Score['Class Other'] += 1
    
   
    output_label_path = Output / 'Class_Count.txt'

    with open(output_label_path, 'w') as file:
        file.write(f"Class JM105: {Score['Class JM105']}\n")
        file.write(f"Class Other: {Score['Class Other']}\n")



def Create_Data_Set(Input_Path, Output_Path):

    Input = Path(Input_Path)
    Output = Path(Output_Path)

    Output_Folder = (Output / 'Data-set')
    
    Output_Folder.mkdir(exist_ok=True)
    (Output_Folder / 'labels').mkdir(exist_ok=True)
    (Output_Folder / 'images').mkdir(exist_ok=True)

    labels_Class_JM105 = sorted(Input.rglob("*labels/*JM105*.txt"))
    images_Class_JM105 = sorted(Input.rglob("*images/*JM105*.jpg"))

    ###
    labels_Class_other1 = sorted(Input.rglob("*labels/*RD41*.txt"))
    images_Class_other1 = sorted(Input.rglob("*images/*RD41*.jpg"))

    labels_Class_other2 = sorted(Input.rglob("*labels/*RD49*.txt"))
    images_Class_other2 = sorted(Input.rglob("*images/*RD49*.jpg"))

    labels_Class_other3 = sorted(Input.rglob("*labels/*RD61*.txt"))
    images_Class_other3 = sorted(Input.rglob("*images/*RD61*.jpg"))
    ###

    Data_JM105 = ({"Images_Path": images_Class_JM105,
                    "Lables_Path": labels_Class_JM105})
    
    Data_RD41 = ({"Images_Path": images_Class_other1,
                    "Lables_Path": labels_Class_other1})
    
    Data_RD49 = ({"Images_Path": images_Class_other2,
                    "Lables_Path": labels_Class_other2})
    
    Data_RD61 = ({"Images_Path": images_Class_other3,
                    "Lables_Path": labels_Class_other3})

    ###
    Other_Class1 = pd.DataFrame(Extract_Labels(Data_RD41, 'type 1'), columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    Other_Class2 = pd.DataFrame(Extract_Labels(Data_RD49, 'type 1'), columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    Other_Class3 = pd.DataFrame(Extract_Labels(Data_RD61, 'type 1'), columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    ###

    Focus_Class = pd.DataFrame(Extract_Labels(Data_JM105, 'defult'), columns=('Class', 'X', 'Y', 'W', 'H', 'Images_Path'))
    Other_Class = pd.concat((Other_Class1, Other_Class2, Other_Class3), axis=0)
    Other_Class = Other_Class.sample(frac = 1, random_state=5, ignore_index=True)

    print("JM105")
    print(Focus_Class)
    print("Other")
    print(Other_Class)

    Focus_Class_images = Stored_Small_Images(Focus_Class)
    Other_Class_images = Stored_Small_Images(Other_Class)
    
    Focus = Mix_Instances_With_Differance_Ration(Focus_Class_images, max_instances= 20)
    Other = Mix_Instances_With_Differance_Ration(Other_Class_images, max_instances= 20)

    Create_Images(Focus, Other, Output_Folder)
    Check_Labels(Output_Folder, Output)


if __name__ == "__main__":

    global Random_Seed
    Random_Seed = 0
    Create_Data_Set(r'D:\Workflow_project\test', r'D:\Workflow_project\runs')