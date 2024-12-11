import cv2
import os
import time
import numpy as np

def RGB2BINARY_Transform(Input: str, Output_Folder1: str, Output_Folder2: str, 
                         Save_As_Name1: str , Save_As_Name2: str):

    os.makedirs(Output_Folder1, exist_ok=True)
    os.makedirs(Output_Folder2, exist_ok=True)
    
    Gray_Image = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)

    _, Tran_Image = cv2.threshold(Gray_Image, 48, 255, cv2.THRESH_BINARY) ## + cv2.THRESH_OTSU
    
    Output_Path1 = os.path.join(Output_Folder1, Save_As_Name1)
    Output_Path2 = os.path.join(Output_Folder2, Save_As_Name2)

    cv2.imwrite(Output_Path1, Tran_Image)
    cv2.imwrite(Output_Path2, Gray_Image)


def Label_Extract(Input: str, Output_Folder: str, Save_As_Name: str, 
                  Class_Name: str, Image_W: int = 5100, Image_H: int = 3750):
    
    os.makedirs(Output_Folder, exist_ok=True)

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

    Output_Path = os.path.join(Output_Folder, Save_As_Name)

    cv2.imwrite(Output_Path, Image_copy)

    return Label_list


def Create_TXT(Input_List_Of_Data: list, Number_Of_Chunk: 
               int, Output_Name: str, Output_Folder: str):

    os.makedirs(Output_Folder, exist_ok=True)
   
    def chunk_list(lst, chunk_size):
        return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

    TXT_File_Path = f'{Output_Folder}/{Output_Name}'

    chunked_data = chunk_list(Input_List_Of_Data, Number_Of_Chunk)

    with open(TXT_File_Path, 'w') as file:
        for row in chunked_data:
            file.write(' '.join(map(str, row)) + '\n')


def De_Noisse(Input: str, Output_Folder: str, 
              Save_As_Name: str, H: int = 70):

    os.makedirs(Output_Folder, exist_ok=True)

    image_bw = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)
    
    noiseless_image_bw = cv2.fastNlMeansDenoising(image_bw, None, H, templateWindowSize=7, searchWindowSize=21)

    Output_Path = os.path.join(Output_Folder, Save_As_Name)

    cv2.imwrite(Output_Path, noiseless_image_bw)


def apply_dilation(Input: str, Output_Folder: str, Save_As_Name: str,
                   kernel_size: int = 4, iterations: int = 1):

    os.makedirs(Output_Folder, exist_ok=True)

    image = cv2.imread(Input, cv2.IMREAD_GRAYSCALE)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(image, kernel, iterations=iterations)

    Output_Path = os.path.join(Output_Folder, Save_As_Name)

    cv2.imwrite(Output_Path, dilated_image)

def Merge_Images(Input_Origin: str, Input_Mask: str, 
                 Output_Folder: str, Save_As_Name: str):

    os.makedirs(Output_Folder, exist_ok=True)
    
    img_org  = cv2.imread(Input_Origin)
    img_mask = cv2.imread(Input_Mask, cv2.IMREAD_GRAYSCALE)

    _, binary_mask = cv2.threshold(img_mask, 48, 255, cv2.THRESH_BINARY)

    black_background = np.zeros_like(img_org)

    img_output = np.where(binary_mask[:, :, None] == 255, img_org, black_background)

    Output_Path = os.path.join(Output_Folder, Save_As_Name)

    cv2.imwrite(Output_Path, img_output)



def main(Name: str, Input: str, Class_Number: int, Number_of_Chunk: int):

    Starttime = time.time()
    
    Final_Output = f'Process Images {Name}'
    os.makedirs(Final_Output, exist_ok=True)

    Count_Origin_Images = os.listdir(Input)

    New_file_Type = []

    Output_Binary_Images = f'{Final_Output}/Binary Images'

    Output_Gray_Images = f'{Final_Output}/Gray Images'

    Output_De_Noiss_Images = f'{Final_Output}/Denoiss Images'

    Output_Label_File = f'{Final_Output}/Label Text File'

    Output_Dilation_Images = f'{Final_Output}/Dilation Images'

    Output_Train_Images = f'{Final_Output}/Train Images'

    for i in Count_Origin_Images:

        Change_Type = i.replace('.jpg', '.txt')
        New_file_Type.append(Change_Type)
    
    print(f'Starting Process RGB to BINARY')

    for i in range(len(Count_Origin_Images)):

        RGB2BINARY_Transform(f'{Input}/{Count_Origin_Images[i]}', Output_Binary_Images, 
                             Output_Gray_Images, f'BR_{Count_Origin_Images[i]}', 
                             f'GR_{Count_Origin_Images[i]}')
        
        print(f'Save.... BR_{Count_Origin_Images[i]}')

    print(f'Starting Process Denoisse')
    
    for i in range(len(Count_Origin_Images)):

        De_Noisse(f'./Process Images {Name}/Binary Images/BR_{Count_Origin_Images[i]}', 
                  Output_De_Noiss_Images, 
                  f'BR_DE_{Count_Origin_Images[i]}')

        print(f'Save.... BR_DE_{Count_Origin_Images[i]}')


    print(f'Starting Process Apply Dilation Images')
    
    for i in range(len(Count_Origin_Images)):

        apply_dilation(f'./Process Images {Name}/Denoiss Images/BR_DE_{Count_Origin_Images[i]}', 
                       Output_Dilation_Images, 
                       f'BR_DE_DI{Count_Origin_Images[i]}')

        print(f'Save.... BR_DE_DI{Count_Origin_Images[i]}')

    ##

    print(f'Starting Create Train Images')
    
    for i in range(len(Count_Origin_Images)):

        Merge_Images(f'./Process Images {Name}/Gray Images/GR_{Count_Origin_Images[i]}',
                     f'./Process Images {Name}/Dilation Images/BR_DE_DI{Count_Origin_Images[i]}',
                     Output_Train_Images, 
                     Count_Origin_Images[i])

        print(f'Save.... {Count_Origin_Images[i]}')


    Count_Binary_Images = os.listdir(Output_Binary_Images)

    print(f'Starting Process Create txt file')

    for i in range(len(Count_Binary_Images)):

        List_of_Data = Label_Extract(f'{Output_Binary_Images}/{Count_Binary_Images[i]}', 
                                     f'{Final_Output}/Display Boundary Box Images', 
                                     f'DP_{Count_Binary_Images[i]}', Class_Number)
        
        Create_TXT(List_of_Data, Number_of_Chunk, 
                   New_file_Type[i], Output_Label_File)
        
        print(f'Save.... {New_file_Type[i]}')
    
    print("Compleat All")
    


if __name__ == "__main__":

    Name = "RD61"
    Input_Path = 'C:/Users/Lenovo/Desktop/Rice dataset2/RD61'
    Number_of_Chunk = 5
    Class_Number = 2

    main(Name, Input_Path, Class_Number, Number_of_Chunk)
