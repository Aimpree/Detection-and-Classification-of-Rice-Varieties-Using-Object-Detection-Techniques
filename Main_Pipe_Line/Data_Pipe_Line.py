import cv2
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

def RGB2BINARY_Transform(Input_Path: str, Output_Folder1: str, Output_Folder2: str,
                          Save_As_Name1: str , Save_As_Name2: str):
    
    Input = Path(Input_Path)
    Output1 = Path(Output_Folder1)
    Output2 = Path(Output_Folder2)
    # Output3 = Path(Output_Folder3)
    
    os.makedirs(Output1, exist_ok=True)
    os.makedirs(Output2, exist_ok=True)
    # os.makedirs(Output3, exist_ok=True)
    
    Img = cv2.imread(Input)

    Gray_Image = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY)

    # Tran_Image1 = cv2.adaptiveThreshold(Gray_Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 4)
    Tran_Image1 = cv2.adaptiveThreshold(Gray_Image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 5, 8)
    _, Tran_Image2 = cv2.threshold(Gray_Image, 48, 255, cv2.THRESH_BINARY) 
    
    union_image = cv2.bitwise_or(Tran_Image1, Tran_Image2)

    Output_Path1 = os.path.join(Output1, Save_As_Name1)
    Output_Path2 = os.path.join(Output2, Save_As_Name2)
    # Output_Path3 = os.path.join(Output3, Save_As_Name3)

    cv2.imwrite(Output_Path1, union_image)
    cv2.imwrite(Output_Path2, Gray_Image)
    # cv2.imwrite(Output_Path3, HSV_Image)



def Label_Extract(Input_Path: str, Class_Name: str, Image_W: int = 5100, Image_H: int = 3750):
    
    Input = Path(Input_Path)
    # Output = Path(Output_Folder)
    
    # os.makedirs(Output, exist_ok=True)

    Image = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)

    Contours, hierarchy = cv2.findContours(image=Image, 
                                           mode=cv2.RETR_TREE, 
                                           method=cv2.CHAIN_APPROX_SIMPLE)

    Image_copy = cv2.cvtColor(Image, cv2.COLOR_GRAY2BGR)

    Min_Contour_Area = 5000 

    Label_list = []

    for Contour in Contours:

        W = Image_W
        H = Image_H
        
        Contour_area = cv2.contourArea(Contour)
        
        
        if Contour_area > Min_Contour_Area:
            
            x, y, w, h = cv2.boundingRect(Contour)
    
            ## YOLOv8 PyTorch TXT

            center_x = x + (w / 2)
            center_y = y + (h / 2)

            Nor_center_x = center_x / W
            Nor_center_y = center_y / H

            Nor_W = w/W
            Nor_h = h/H

            Label_list.append(Class_Name)

            Label_list.append(Nor_center_x)
            Label_list.append(Nor_center_y)

            Label_list.append(Nor_W)
            Label_list.append(Nor_h)

            cv2.rectangle(Image_copy, (x,y), (x + w, y +h), (0, 255, 0), 2)

    # Output_Path = os.path.join(Output, Save_As_Name)

    # cv2.imwrite(Output_Path, Image_copy)

    return Label_list


def Create_TXT(Input_List_Of_Data: list, Output_Folder: str, 
               Output_Name: str, Number_Of_Chunk = 5):

    Output = Path(Output_Folder)
    os.makedirs(Output, exist_ok=True)
   
    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    TXT_File_Path = Path(f'{Output}/{Output_Name}')

    chunked_data = chunk_list(Input_List_Of_Data, Number_Of_Chunk)

    with open(TXT_File_Path, 'w') as file:
        for row in chunked_data:
            file.write(' '.join(map(str, row)) + '\n')


def De_Noisse(Input_Path: str, Output_Folder: str, 
              Save_As_Name: str, H: int = 70):

    Input = Path(Input_Path)
    Output = Path(Output_Folder)
    os.makedirs(Output, exist_ok=True)

    image_bw = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)
    
    noiseless_image_bw = cv2.fastNlMeansDenoising(image_bw, None, H, templateWindowSize=7, searchWindowSize=21)

    Output_Path = os.path.join(Output, Save_As_Name)

    cv2.imwrite(Output_Path, noiseless_image_bw)


def apply_opening(Input_Path: str, Output_Folder: str, Save_As_Name: str,
                  kernel_size: int = 5, iterations: int = 2):

    # Convert paths to Path objects for better path management
    Input = Path(Input_Path)
    Output = Path(Output_Folder)
    os.makedirs(Output, exist_ok=True)

    # Read the input image in grayscale
    image = cv2.imread(str(Input), cv2.IMREAD_GRAYSCALE)

    # Define the structuring element (kernel)
    kernel1 = np.ones((kernel_size, kernel_size), np.uint8)
    kernel2 = np.ones((kernel_size, kernel_size), np.uint8)

    # Perform morphological opening: erosion followed by dilation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel1, iterations=iterations)
    # closed_image = cv2.morphologyEx(opened_image, cv2.MORPH_CLOSE, kernel2, iterations=iterations)

    # Construct the output file path
    Output_Path = os.path.join(Output, Save_As_Name)

    # Save the result
    cv2.imwrite(Output_Path, opened_image)

def Merge_Images(Input_Origin: str, Input_Mask: str, 
                 Output_Folder1: str, Output_Folder2: str,
                 Save_As_Name1: str, Save_As_Name2: str):

    Input_Ori = Path(Input_Origin)
    Input_Mas = Path(Input_Mask)

    Output1 = Path(Output_Folder1)
    Output2 = Path(Output_Folder2)
    os.makedirs(Output1, exist_ok=True)
    os.makedirs(Output2, exist_ok=True)
    
    img_org  = cv2.imread(Input_Ori)
    HSV_Image = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV) 
    img_mask = cv2.imread(Input_Mas, cv2.IMREAD_GRAYSCALE)
    

    _, binary_mask = cv2.threshold(img_mask, 48, 255, cv2.THRESH_BINARY)

    black_background = np.zeros_like(img_org)

    img_output1 = np.where(binary_mask[:, :, None] == 255, img_org, black_background)
    img_output2 = np.where(binary_mask[:, :, None] == 255, HSV_Image, black_background)

    Output_Path1 = os.path.join(Output1, Save_As_Name1)
    Output_Path2 = os.path.join(Output2, Save_As_Name2)

    cv2.imwrite(Output_Path1, img_output1)
    cv2.imwrite(Output_Path2, img_output2)


def Preprocess(Input_Path, Class_Number, Name):

    Input_Images_Name = os.listdir(Input_Path)
    Input_Labels_Name = []

    for i in Input_Images_Name:

        Change_Type = i.replace('.jpg', '.txt')
        Input_Labels_Name.append(Change_Type)

    Output_folder = Path(f'./PREPROCESS FILE {Name}')
    GRAY_Images = Path(f'{Output_folder}/SOURCE/GRAY IMAGES')
    # HSV_Images = Path(f'{Output_folder}/SOURCE/HSV IMAGES')
    BINARY_Images = Path(f'{Output_folder}/SOURCE/BINARY Images')
    DENOISSES_Images = Path(f'{Output_folder}/SOURCE/DENOISSES Images')
    DILATION_Images = Path(f'{Output_folder}/SOURCE/DILATION Images')
    PREPROCESS_RGB_Images = Path(f'{Output_folder}/images RGB')
    PREPROCESS_HSV_Images = Path(f'{Output_folder}/images HSV')
    LABELS = Path(f'{Output_folder}/labels')

    os.makedirs(Output_folder, exist_ok=True)
    os.makedirs(GRAY_Images, exist_ok=True)
    os.makedirs(BINARY_Images, exist_ok=True)
    # os.makedirs(HSV_Images, exist_ok=True)
    os.makedirs(DENOISSES_Images, exist_ok=True)
    os.makedirs(DILATION_Images, exist_ok=True)
    os.makedirs(PREPROCESS_RGB_Images, exist_ok=True)
    os.makedirs(PREPROCESS_HSV_Images, exist_ok=True)
    os.makedirs(LABELS, exist_ok=True)

    Input = Path(Input_Path)
    Images = sorted(Input.rglob("*.jpg"))

    Images_Name = {'Name' :[str(Name) for Name in Input_Images_Name]}
    Labels_Name = {'Label' :[str(Label) for Label in Input_Labels_Name]}

    Images_File_Path = {'Original Images' :[str(image) for image in Images]}
    Images_B_File_Path = {'Binary Images' :[f'{BINARY_Images}\B_{str(Name)}' for Name in Input_Images_Name]}
    Images_G_File_Path = {'Gray Images' :[f'{GRAY_Images}\G_{str(Name)}' for Name in Input_Images_Name]}
    # Images_H_File_Path = {'HSV Images' :[f'{HSV_Images}\H_{str(Name)}' for Name in Input_Images_Name]}

    Images_DE_File_Path = {'Denoisses Images' :[f'{DENOISSES_Images}\DE_{str(Name)}' for Name in Input_Images_Name]}
    Images_DI_File_Path = {'Dilation Images' :[f'{DILATION_Images}\DI_{str(Name)}' for Name in Input_Images_Name]}
    Images_PRE_File_Path = {'RGB Preprocess Imagess' :[f'{PREPROCESS_RGB_Images}\{str(Name)}' for Name in Input_Images_Name]}
    Images_PRE_File_Path = {'HSV Preprocess Imagess' :[f'{PREPROCESS_HSV_Images}\{str(Name)}' for Name in Input_Images_Name]}

    Images_Name_Store = pd.DataFrame(data=Images_Name)
    Labels_Name_Store = pd.DataFrame(data=Labels_Name)
    Images_File_Path_Store = pd.DataFrame(data=Images_File_Path)
    Images_B_File_Path_Store = pd.DataFrame(data=Images_B_File_Path)
    Images_G_File_Path_Store = pd.DataFrame(data=Images_G_File_Path)
    Images_DE_File_Path_Store = pd.DataFrame(data=Images_DE_File_Path)
    Images_DI_File_Path_Store = pd.DataFrame(data=Images_DI_File_Path)
    Images_PRE_File_Path_Store = pd.DataFrame(data=Images_PRE_File_Path)

    All_Images_File_Path = pd.concat((Images_Name_Store,
                                    Labels_Name_Store,
                                    Images_File_Path_Store,
                                    Images_B_File_Path_Store,
                                    Images_G_File_Path_Store,
                                    Images_DE_File_Path_Store,
                                    Images_DI_File_Path_Store,
                                    Images_PRE_File_Path_Store), axis=1)
    
    

    for idx, row in All_Images_File_Path.iterrows():

        Name = row['Name']
        Labels = row['Label']
        Ori_Path = row['Original Images']

        ##

        B_Path = row['Binary Images']
        G_Path = row['Gray Images']
        DE_Path = row['Denoisses Images']
        DI_Path = row['Dilation Images']
        # PRE_Path = row['RGB Preprocess Imagess']
    
        Images_Name_B = f'B_{Name}'
        Images_Name_G = f'G_{Name}'
        # Images_Name_H = f'H_{Name}'
        Images_Name_DE = f'DE_{Name}'
        Images_Name_DI = f'DI_{Name}'
       
        RGB2BINARY_Transform(Ori_Path, BINARY_Images, GRAY_Images, Images_Name_B, Images_Name_G)
        De_Noisse(B_Path, DENOISSES_Images, Images_Name_DE)
        apply_opening(DE_Path, DILATION_Images, Images_Name_DI)
        Merge_Images(Ori_Path, DI_Path, PREPROCESS_RGB_Images, PREPROCESS_HSV_Images, Name, Name)

        List_of_Data = Label_Extract(B_Path, Class_Number)
        Create_TXT(List_of_Data, LABELS, Labels)

    print('Compleat')

if __name__ == "__main__":

    Class_type = 'Name'
    Input = 'Input_Path'
    Name = Class_type
    Class = "Number_Of_Class"

    Preprocess(Input, Class, Name)