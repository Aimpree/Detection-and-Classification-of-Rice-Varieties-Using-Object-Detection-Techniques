from ultralytics import YOLO
import pandas as pd
import cv2  
from pathlib import Path
from PIL import Image
from tabulate import tabulate

class Create_Report:

    def __init__(self, input_path, output_path, true_label):

        self.input_path =  input_path
        self.output_path =  output_path
        self.true_label = true_label
        
        
    def Calculate_Performace(self, Input_Data, Name, True_instance):

        Output = Path(self.output_path)
        Df_Distribution = pd.DataFrame.from_dict(Input_Data, orient='index')

        Sum_All_Predict = Df_Distribution['All Predict'].sum()
        Sum_True = Df_Distribution['True Predict'].sum()
        Sum_False = Df_Distribution['False Predict'].sum()


        Df_Sum_Scores = pd.DataFrame({
            'Metrix': ['Actual Instance', 'All Predict', 'True Predict', 'False Predict'],
            'SUM': [True_instance, Sum_All_Predict, Sum_True, Sum_False]
        })
        

        Recall = Sum_True / True_instance
        Precision = Sum_True / (Sum_True + Sum_False)


        Df_Performance = pd.DataFrame({
            'Metrix': ['Recall', 'Precision'],
            'Preformance': [Recall, Precision]
        })
        
        
        with open(f"{Output}/{Name}_Report.txt", "w", encoding="utf-8") as file:
            file.write(f'#########-{Name}-#########\n\n')
            
            file.write(tabulate(Df_Distribution, headers='keys', tablefmt='rounded_grid'))
            file.write("\n\n")  
            
            file.write(tabulate(Df_Sum_Scores, headers='keys', tablefmt='rounded_grid'))
            file.write("\n\n")
            
            file.write(tabulate(Df_Performance, headers='keys', tablefmt='rounded_grid'))
            file.write("\n")
        

    def Create_Data_Distribution(self, Input_Path):

        confidence_ranges = [
        "0.1 - 0.19", "0.2 - 0.29", "0.3 - 0.39", "0.4 - 0.49",
        "0.5 - 0.59", "0.6 - 0.69", "0.7 - 0.79", "0.8 - 0.89", "0.9 - 0.99"
        ]

        confidence_data = {}

        for conf_range in confidence_ranges:
            true_path = sorted((Input_Path / f'Confident_{conf_range}/True').glob('*.jpg'))
            false_path = sorted((Input_Path / f'Confident_{conf_range}/False').glob('*.jpg'))

            confidence_data[f"Confident_{conf_range}"] = {
                "All Predict": len(true_path) + len(false_path),
                "True Predict": len(true_path),
                "False Predict": len(false_path)
            }

        return confidence_data

    def Show_Performance(self):

        Input = Path(self.input_path)
        Input_JM105 = Input / 'JM105'
        Input_RD49 = Input / 'RD49'
        Input_RD61 = Input / 'RD61'

        True_label_JM105 = sum(self.true_label['JM105'])
        True_label_RD49 = sum(self.true_label['RD49'])
        True_label_RD61 = sum(self.true_label['RD61'])

        Result_JM105 = self.Create_Data_Distribution(Input_JM105)
        self.Calculate_Performace(Result_JM105, "JM105", True_label_JM105)

        Result_RD49 = self.Create_Data_Distribution(Input_RD49)
        self.Calculate_Performace(Result_RD49, "RD49", True_label_RD49)

        Result_RD61 = self.Create_Data_Distribution(Input_RD61)
        self.Calculate_Performace(Result_RD61, "RD61", True_label_RD61)


def Create_Conf_Folder(Output_Path):
    confidence_ranges = [
        (0.1, 0.19), (0.2, 0.29), (0.3, 0.39), (0.4, 0.49),
        (0.5, 0.59), (0.6, 0.69), (0.7, 0.79), (0.8, 0.89), (0.9, 0.99)
    ]
    
    for conf_min, conf_max in confidence_ranges:

        conf_folder_JM105 = Output_Path / 'JM105' / f'Confident_{conf_min:.1f} - {conf_max:.2f}'
        (conf_folder_JM105 / 'True').mkdir(parents=True, exist_ok=True)
        (conf_folder_JM105 / 'False').mkdir(parents=True, exist_ok=True)

        conf_folder_RD49 = Output_Path / 'RD49' / f'Confident_{conf_min:.1f} - {conf_max:.2f}'
        (conf_folder_RD49 / 'True').mkdir(parents=True, exist_ok=True)
        (conf_folder_RD49 / 'False').mkdir(parents=True, exist_ok=True)

        conf_folder_RD61 = Output_Path / 'RD61' / f'Confident_{conf_min:.1f} - {conf_max:.2f}'
        (conf_folder_RD61 / 'True').mkdir(parents=True, exist_ok=True)
        (conf_folder_RD61 / 'False').mkdir(parents=True, exist_ok=True)



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


    return Sub_Image

def Calculate_Data(Input_List_Of_Data: list, Image_W: int = 5100, Image_H: int = 3750):

    images_width = Image_W
    images_Height = Image_H
    
    Data1 = pd.DataFrame(Input_List_Of_Data, columns=('Class', 'X', 'Y', 'W', 'H', 'Conf', 'Images Path', 'Predict Result'))
    Data2 = pd.DataFrame(columns=('Class', 'X', 'Y', 'W', 'H', 'Conf', 'Images Path', 'Predict Result'))

    Data2['Class'] = Data1['Class']
    Data2['Conf'] = Data1['Conf']
    Data2['Images Path'] = Data1['Images Path']
    Data2['Predict Result'] = Data1['Predict Result']

    Data2['X'] = pd.to_numeric((Data1['X']) * images_width) - (pd.to_numeric(Data1['W']) * images_width / 2)
    Data2['Y'] = pd.to_numeric((Data1['Y']) * images_Height) - (pd.to_numeric(Data1['H']) * images_Height / 2)

    Data2['W'] = pd.to_numeric(Data1['W']) * images_width
    Data2['H'] = pd.to_numeric(Data1['H']) * images_Height

    return Data2


def calculate_iou(box1, box2):
    
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Compute intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Compute areas of each box
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Compute union
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def Matcher_Labels(Predict_Labels, True_Labels, IOU_Threshold):

    Index_Result = []

    for (index_labels_pred, row_labels_pred) in Predict_Labels.iterrows():

        X_Pred, Y_Pred, W_Pred, H_Pred = row_labels_pred['X'], row_labels_pred['Y'], row_labels_pred['W'], row_labels_pred['H']
        target_box = (X_Pred, Y_Pred, W_Pred, H_Pred)
        
        for (_, row_labels_ac) in True_Labels.iterrows():

            X_True, Y_True, W_True, H_True = row_labels_ac['X'], row_labels_ac['Y'], row_labels_ac['W'], row_labels_ac['H']
            True_box = (X_True, Y_True, W_True, H_True)

            IOU_score = calculate_iou(target_box, True_box)

            if IOU_score >= IOU_Threshold:
                Index_Result.append(index_labels_pred)

    return Index_Result

def Sorted_Images(Output_Paht, Labels_Predict, Class, Number):
    
    for index, row in Calculate_Data(Labels_Predict).iterrows():
        
        Image_path = row['Images Path']
        Confident = row['Conf']
        Marker = row['Predict Result']

        X_image, Y_image, W_image, H_image, = row['X'], row['Y'], row['W'], row['H']

        read_image = Image.open(Image_path)

        image = Create_Sup_Images(read_image, X_image, Y_image, W_image, H_image)

        if 0.1 <= Confident <= 0.19:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.1 - 0.19' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.1 - 0.19' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

        elif 0.2 <= Confident <= 0.29:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.2 - 0.29' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.2 - 0.29' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')
        
        elif 0.3 <= Confident <= 0.39:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.3 - 0.39' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.3 - 0.39' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            
        elif 0.4 <= Confident <= 0.49:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.4 - 0.49' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.4 - 0.49' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

        elif 0.5 <= Confident <= 0.59:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.5 - 0.59' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.5 - 0.59' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

        elif 0.6 <= Confident <= 0.69:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.6 - 0.69' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.6 - 0.69' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

        elif 0.7 <= Confident <= 0.79:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.7 - 0.79' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.7 - 0.79' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

        elif 0.8 <= Confident <= 0.89:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.8 - 0.89' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.8 - 0.89' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

        elif 0.9 <= Confident <= 0.99:
            if Marker == 'True':
                image.save(Output_Paht / Class / 'Confident_0.9 - 0.99' / 'True' / f'image_{Number}_{str(index).zfill(4)}.jpg')
            else:
                image.save(Output_Paht / Class / 'Confident_0.9 - 0.99' / 'False' / f'image_{Number}_{str(index).zfill(4)}.jpg')

def Extrac_Predict_Result(Input_Predict, Images_Path):

    image = cv2.imread(Images_Path)
    H, W, _ = image.shape  
    
    Element = []

    for pred in Input_Predict:
 
        for box in pred.boxes:

            cls = int(box.cls.item())  
            conf = float(box.conf.item()) 

            xyxy = box.xyxy.cpu().numpy()[0] 

            x_min, y_min, x_max, y_max = xyxy
            x_center = ((x_min + x_max) / 2) / W  
            y_center = ((y_min + y_max) / 2) / H 
            width = (x_max - x_min) / W  
            height = (y_max - y_min) / H  

            Element.append((cls, x_center, y_center, width, height, conf, Images_Path))

    return Element

def Split_Confident(Model, Input_Path, Output_Path, IOU = 0.6):

    Input = Path(Input_Path)
    Output = Path(Output_Path)
    Input_Model = Path(Model)

    Create_Conf_Folder(Output)

    Labels = sorted(Input.glob('labels/*.txt'))
    Images = sorted(Input.glob('images/*.jpg'))

    ALl_Path = pd.DataFrame({'Images Path': Images, 'Labels Path': Labels})

    model = YOLO(Input_Model)

    # display(ALl_Path)

    Number = 1
    Count_True_Labels = {'JM105': [], 'RD49': [], 'RD61': []}

    for index, row in ALl_Path.iterrows():

        image_path = row['Images Path']
        label_path = row['Labels Path']
    
        results_Class_0 = model.predict(image_path, conf=0.3, classes=[0])
        results_Class_1 = model.predict(image_path, conf=0.3, classes=[1])
        results_Class_2 = model.predict(image_path, conf=0.3, classes=[2])

        pred_results_Class_0 = []
        pred_results_Class_1 = []
        pred_results_Class_2 = []

        pred_results_Class_0 = Extrac_Predict_Result(results_Class_0, image_path)
        pred_results_Class_1 = Extrac_Predict_Result(results_Class_1, image_path)
        pred_results_Class_2 = Extrac_Predict_Result(results_Class_2, image_path)

        with open(label_path, 'r') as file:
            contents = file.readlines()

        True_Labels = [content.split() for content in contents]

        Labels_Pred_Class_0 = pd.DataFrame(pred_results_Class_0, columns=['Class', 'X', 'Y', 'W', 'H', 'Conf', 'Images Path'])
        Labels_Pred_Class_1 = pd.DataFrame(pred_results_Class_1, columns=['Class', 'X', 'Y', 'W', 'H', 'Conf', 'Images Path'])
        Labels_Pred_Class_2 = pd.DataFrame(pred_results_Class_2, columns=['Class', 'X', 'Y', 'W', 'H', 'Conf', 'Images Path'])

        Labels_Checker  = pd.DataFrame(True_Labels, columns=['Class', 'X', 'Y', 'W', 'H'])
        Labels_Checker = Labels_Checker[['Class', 'X', 'Y', 'W', 'H']].astype(float)

        Labels_Checker_Class_0 = Labels_Checker[Labels_Checker['Class'] == 0.0]
        Count_True_Labels['JM105'].append(Labels_Checker_Class_0['Class'].count())

        Labels_Checker_Class_1 = Labels_Checker[Labels_Checker['Class'] == 1.0]
        Count_True_Labels['RD49'].append(Labels_Checker_Class_1['Class'].count())

        Labels_Checker_Class_2 = Labels_Checker[Labels_Checker['Class'] == 2.0]
        Count_True_Labels['RD61'].append(Labels_Checker_Class_2['Class'].count())

        Index_Result_Class_0 = Matcher_Labels(Labels_Pred_Class_0, Labels_Checker_Class_0, IOU)
        Index_Result_Class_1 = Matcher_Labels(Labels_Pred_Class_1, Labels_Checker_Class_1, IOU)
        Index_Result_Class_2 = Matcher_Labels(Labels_Pred_Class_2, Labels_Checker_Class_2, IOU)

        Labels_Pred_Class_0["Predict Result"] = "False"
        Labels_Pred_Class_0.loc[Index_Result_Class_0, "Predict Result"] = "True"

        Labels_Pred_Class_1["Predict Result"] = "False"
        Labels_Pred_Class_1.loc[Index_Result_Class_1, "Predict Result"] = "True"

        Labels_Pred_Class_2["Predict Result"] = "False"
        Labels_Pred_Class_2.loc[Index_Result_Class_2, "Predict Result"] = "True"

        Sorted_Images(Output, Labels_Pred_Class_0, 'JM105', Number)
        Sorted_Images(Output, Labels_Pred_Class_1, 'RD49', Number)
        Sorted_Images(Output, Labels_Pred_Class_2, 'RD61', Number)

        Number += 1
        
    return Count_True_Labels

   


if __name__ == "__main__" :

    Input_Model = Path('Model Path')
    Input = Path('Input Path')
    Output = Path('Output Path')

    True_labels = Split_Confident(Input_Model, Input, Output)
    
    report = Create_Report('Input Path', 'Output_Path_Report_Result', True_labels)
    report.Show_Performance()
