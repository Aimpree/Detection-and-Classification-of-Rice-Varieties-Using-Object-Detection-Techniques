# 🧪 Research on Detection and Classification of Rice Varieties Using Object Detection Techniques

## 🌾 Rice Varieties

This project focuses on detecting and classifying the following rice varieties:

* **Jasmine Rice 105**
* **RD Rice 49**
* **RD Rice 61**

---

## 📸 Original Rice Dataset

These are the original images used for dataset creation:

### Jasmine Rice 105

<img src="https://github.com/user-attachments/assets/faf71491-2cf6-4b00-bceb-0e2a93b56408" alt="Jasmine Rice 105" style="width:30%; height:auto;">  


### RD Rice 49

<img src="https://github.com/user-attachments/assets/228f5c3b-5896-41c1-8680-b04d55009be1" alt="RD Rice 49" style="width:30%; height:auto;">  


### RD Rice 61

<img src="https://github.com/user-attachments/assets/8ffbfb47-b4fa-488b-8c28-f8394f8e0e1c" alt="RD Rice 61" style="width:30%; height:auto;">  


---

## ⚙️ Preprocessing Workflow

Steps used to preprocess and prepare the rice grain images for training:

1. Load raw input images.
2. Apply preprocessing filters (e.g., grayscale, thresholding, noise removal).
3. Extract bounding boxes for individual rice grains.
4. Output image-mask pairs and YOLO-compatible labels.

---

### 🖼️ Preprocessed Images and YOLO Labels

#### Jasmine Rice 105

<img src="https://github.com/user-attachments/assets/2c854e4b-2b26-41ca-9f25-1163f3e611bf" alt="Jasmine Rice 105 Processed" style="width:30%; height:auto;">  
<img src="https://github.com/user-attachments/assets/3c9350c1-3d0c-4182-9332-5935b0583e47" alt="Jasmine Rice 105 Labels" style="width:60%; height:auto;">  
Preprocessed image and corresponding YOLO labels for Jasmine Rice 105

#### RD Rice 49

<img src="https://github.com/user-attachments/assets/4f7da1b5-3b8e-4c12-a67d-44ead483aa51" alt="RD Rice 49 Processed" style="width:30%; height:auto;">  
<img src="https://github.com/user-attachments/assets/f6698405-3f31-4179-ab25-6339b7eb7827" alt="RD Rice 49 Labels" style="width:60%; height:auto;">  
Preprocessed image and YOLO labels for RD Rice 49

#### RD Rice 61

<img src="https://github.com/user-attachments/assets/be09e438-865e-44d6-8fd3-fb83fb163ef3" alt="RD Rice 61 Processed" style="width:30%; height:auto;">  
<img src="https://github.com/user-attachments/assets/ff43eff7-f2ad-431e-8ba8-45ace640a231" alt="RD Rice 61 Labels" style="width:60%; height:auto;">  
Preprocessed image and YOLO labels for RD Rice 61

---

## 🧪 Experimental 1: Synthetic Dataset (Multi-Rice Patterns)

To increase dataset diversity and model robustness, new patterns were generated through data augmentation and mixing rice varieties.

### 🔄 Type 1: Rotational Augmentation

<img src="https://github.com/user-attachments/assets/3d8cca8f-ad52-48d4-89c4-1043d637f4fd" alt="Type 1" style="width:30%; height:auto;">  
Rice grains rotated every 22.5 degrees to create diverse perspectives

### 🔀 Type 2: Mixed, Same Position (3 Grains)

<img src="https://github.com/user-attachments/assets/f85efbd6-1c30-4961-959d-cfef9cfba974" alt="Type 2" style="width:30%; height:auto;">  
Three different rice varieties placed in the same location in a single image

### 🔄 Type 3: Mixed, Different Positions (3 Grains)

<img src="https://github.com/user-attachments/assets/2a0fe1f4-83d8-4266-94d1-eb12300e5a06" alt="Type 3" style="width:30%; height:auto;">  
Three rice grains from different varieties, each placed in a different position

### 🧩 Type 4: Mixed, Different Positions (9 Grains)

<img src="https://github.com/user-attachments/assets/08b31c48-9572-4d3f-b95b-ae1a2f71c5e9" alt="Type 4" style="width:30%; height:auto;">  
Image contains nine grains from different rice varieties, randomly positioned

---

## ⚖️ Experimental 2: Imbalanced Dataset Creation

To test how class imbalance affects model performance, datasets were constructed with varying class distribution between **Jasmine Rice 105** and **RD Rice 61**.

### ⚠️ Ratio 10% : 90%

<img src="https://github.com/user-attachments/assets/5791e522-5e1c-446b-855b-f2fdfc0f1b39" alt="10:90 Ratio" style="width:30%; height:auto;">  
Only 10% Jasmine Rice 105 and 90% RD Rice 61

### ⚠️ Ratio 80% : 20%

<img src="https://github.com/user-attachments/assets/39e23e4d-3627-4e9f-9fae-a8249515ad36" alt="80:20 Ratio" style="width:30%; height:auto;">  
80% Jasmine Rice 105 and 20% RD Rice 61

> 📌 **Note:** Imbalance ratios range from **10:90 to 90:10**, incremented by 10%, to evaluate model sensitivity to imbalance.

---

## 🤖 Model Training and Comparison

Each dataset—both balanced and imbalanced, original and synthetic—was used to train object detection models (YOLOv10).

### ✅ Objective:

To **compare performance** across all experiments and determine:

* The **most accurate model**
* The **best data composition** (e.g., balanced vs. imbalanced, synthetic vs. original)
* The **impact of data augmentation and mixing strategies**

### 📊 Evaluation Metrics:

* **mAP (mean Average Precision)**
* **mAR (mean Average Recall)**

> The final results help identify the **optimal training strategy** for rice variety classification using object detection techniques.

---
[Detection and Classification of Rice Varieties Using Object Detection Techniques.pdf](https://github.com/user-attachments/files/22427592/_._._.Object.Detection.pdf)
