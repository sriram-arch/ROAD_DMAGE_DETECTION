# Automated Road Damage Detection for Infrastructure Maintenance

## Overview
This project aims to develop an automated system for real-time road damage detection using computer vision techniques. Road infrastructure faces degradation over time due to factors like weather and traffic, leading to various types of damage such as cracks and potholes. Detecting these issues early is crucial for safety and cost-effectiveness, but traditional inspection methods are slow and labor-intensive. To address this challenge, we've created a simple, automated system capable of quickly and accurately detecting road damage in real-time.

## Project Workflow
1. **Data Collection:** We collected road images containing different types of cracks (Longitudinal, Transverse, Alligator) and potholes. These images were organized into folders based on their respective classes (e.g., D00, D10, D20, D40).

2. **Annotation:** Using Roboflow, we annotated the images by drawing bounding boxes around the cracks and potholes. Each bounding box was assigned a class label corresponding to the type of defect it represents.

3. **Training:** We utilized the YOLOv8 object detection algorithm to train a custom model on the annotated dataset. The training process was performed in a Google Colab notebook, with the dataset uploaded to Google Drive for easy access. The notebook for data training can be found [here](https://colab.research.google.com/drive/1jr7OKO4Kr53jkHsJTaNcwG4iRKR_g86s?usp=sharing).

4. **Evaluation:** After training, we evaluated the model's performance on a separate validation dataset to assess its accuracy.

5. **Deployment:** Upon obtaining the trained model (`best.pt`), we deployed it for real-time road damage detection using a live webcam feed. A Python script was used to load the model and run inference on webcam frames, detecting cracks and potholes in real-time.


## Sample Output Screenshots
- You can view sample output screenshots in the given link
- ([https://drive.google.com/drive/folders/1uWkqgsZrs3rXm5esrtl9Ft07RIZhir5g?usp=sharing](https://drive.google.com/drive/folders/1mCoX16DeJ4kDb_SO8u1nap2hc697KnCG?usp=sharing)).


## Usage
1. **Installation:** Clone this repository to your local machine.
   ```
   git clone https://github.com/Rajalakshmi21IT/TRINIT_-Techno4-_-Trackchoosen-ML-.git
   ```
   
2. **Dependencies:** Install the required dependencies using pip.
   ```
   pip install -r requirements.txt
   ```

3. **Execution:** Run the Python script for real-time road damage detection.
   ```
   python cv.py
   ```

4. **Usage Notes:** Press 'q' to exit the webcam feed.

## Future Improvements
- Improve model accuracy through data augmentation and hyperparameter tuning.
- Enhance user interface for better usability.
- Explore additional features such as GPS tagging for precise location-based reporting.

