Road Sign Detection with YOLOv8

This project implements a road sign object detection pipeline using YOLOv8 and compares two different training approaches:

YOLOv8n trained from scratch

YOLOv8n trained using transfer learning

The project covers dataset preparation, annotation conversion, model training, inference, visualization, and evaluation.

Project Overview

The goal of this project is to detect and classify road signs in images using YOLOv8.

The original dataset uses Pascal VOC XML annotations. The notebook automatically converts these annotations into the YOLO format required by Ultralytics, creates training and validation sets, trains two YOLOv8 models, and compares their predictions and validation performance.

Dataset

The project uses the Road Sign Detection dataset from Kaggle.

Dataset handle:

andrewmvd/road-sign-detection

The dataset is downloaded automatically using kagglehub.

The original annotations are stored in Pascal VOC XML format and contain:

Object class

Image dimensions

Bounding-box coordinates

xmin

ymin

xmax

ymax

The notebook automatically extracts all road sign class names from the XML annotation files.

Dataset Preparation

The notebook searches the downloaded dataset for:

.jpg, .jpeg, and .png images

.xml annotation files

Images and XML files are matched using their filenames.

The matched samples are randomly shuffled using:

random.seed(42)

The dataset is then divided into:

80% training data

20% validation data

Pascal VOC to YOLO Conversion

The original bounding boxes use Pascal VOC coordinates:

xmin, ymin, xmax, ymax

These coordinates are converted into normalized YOLO format:

class_id x_center y_center width height

The conversion uses the image width and height so that all bounding-box values are normalized.

The resulting dataset structure is:

road_sign_yolo/
|
├── images/
│   ├── train/
│   └── val/
|
├── labels/
│   ├── train/
│   └── val/
|
└── data.yaml

The notebook also generates the data.yaml file automatically using the class names discovered from the XML annotations.

Models

Two YOLOv8 Nano models are trained and compared.

YOLOv8 From Scratch

The first model is initialized using the YOLOv8n architecture without pretrained weights:

scratch_model = YOLO("yolov8n.yaml")

Training configuration:

scratch_model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=416,
    batch=8,
    device="cpu",
    name="road_sign_from_scratch",
    pretrained=False
)

Because the model starts with randomly initialized weights, it must learn visual features directly from the road sign dataset.

YOLOv8 Transfer Learning

The second model starts from pretrained YOLOv8n weights:

transfer_model = YOLO("yolov8n.pt")

Training configuration:

transfer_model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=416,
    batch=8,
    device="cpu",
    name="road_sign_transfer_learning"
)

Transfer learning allows the model to reuse previously learned visual features and adapt them to road sign detection.

Training Configuration

Parameter

Value

Architecture

YOLOv8n

Epochs

10

Image Size

416 × 416

Batch Size

8

Device

CPU

Training Split

80%

Validation Split

20%

Random Seed

42

The same dataset and training configuration are used for both models so their results can be compared fairly.

Model Comparison

After training, the notebook loads the best weights produced by each model:

scratch_best = YOLO(
    "/content/runs/detect/road_sign_from_scratch/weights/best.pt"
)

transfer_best = YOLO(
    "/content/runs/detect/road_sign_transfer_learning/weights/best.pt"
)

Both models are then tested on the same validation image.

This makes it possible to compare:

Detected road sign classes

Bounding-box locations

Confidence scores

Visual prediction quality

Inference

Both models perform inference on the same image using a confidence threshold of 0.25.

scratch_pred = scratch_best.predict(
    source=TEST_IMAGE,
    conf=0.25,
    save=True,
    name="scratch_prediction"
)

transfer_pred = transfer_best.predict(
    source=TEST_IMAGE,
    conf=0.25,
    save=True,
    name="transfer_prediction"
)

The output images are displayed side by side for visual comparison.

Bounding-Box Comparison

For every detected object, the notebook prints:

Predicted class

Confidence score

Bounding-box coordinates

Example output format:

Class: <road-sign-class>
Confidence: <confidence-score>
Box: [x1, y1, x2, y2]

This allows the predictions of the two models to be compared in more detail.

Evaluation Metrics

Both trained models are evaluated on the validation dataset.

scratch_metrics = scratch_best.val(
    data=str(DATA_YAML),
    device="cpu"
)

transfer_metrics = transfer_best.val(
    data=str(DATA_YAML),
    device="cpu"
)

The project compares the following object-detection metrics:

Precision

Precision measures how many predicted road signs are actually correct.

Recall

Recall measures how many real road signs in the dataset were successfully detected.

mAP@50

Mean Average Precision calculated using an Intersection over Union threshold of 0.50.

mAP@50-95

Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

This is a stricter measurement of both detection accuracy and bounding-box localization quality.

The notebook prints the final metrics using:

print("mAP50:", scratch_metrics.box.map50)
print("mAP50-95:", scratch_metrics.box.map)
print("Precision:", scratch_metrics.box.mp)
print("Recall:", scratch_metrics.box.mr)

and the equivalent values for the transfer-learning model.

The uploaded notebook was saved without executed outputs, so final numerical validation results are not included in this README. Run the notebook to generate the actual metrics.

From Scratch vs Transfer Learning

The main difference between the two models is their initialization.

From Scratch

The from-scratch model starts with random weights.

It must learn:

Edges

Colors

Shapes

Object features

Road sign categories

directly from the road sign dataset.

Transfer Learning

The transfer-learning model starts with pretrained YOLOv8 weights.

These weights already contain useful general visual features learned from a large dataset. The model then fine-tunes these features for road sign detection.

Transfer learning can often provide faster convergence and stronger performance when the available training dataset is limited.

Technologies Used

Python

YOLOv8

Ultralytics

KaggleHub

Pillow

Matplotlib

XML ElementTree

Google Colab

Installation

Install the required packages:

pip install ultralytics kagglehub

Running the Project

1. Clone the Repository

git clone <YOUR-REPOSITORY-URL>
cd <YOUR-REPOSITORY-NAME>

2. Open the Notebook

Open the project notebook in Google Colab or Jupyter Notebook.

Recommended notebook name:

road_sign_detection_yolov8.ipynb

3. Install Dependencies

Run:

pip install ultralytics kagglehub

4. Download the Dataset

The notebook automatically downloads the dataset using:

path = kagglehub.dataset_download(
    "andrewmvd/road-sign-detection"
)

5. Prepare the Dataset

Run the preprocessing cells to:

Locate images and XML annotation files.

Extract class names.

Match images with annotations.

Split the data into training and validation sets.

Convert Pascal VOC annotations into YOLO format.

Generate data.yaml.

6. Train Both Models

Run the training cells for:

YOLOv8n from scratch

YOLOv8n transfer learning

7. Run Inference

The notebook selects the same validation image for both trained models and generates annotated prediction images.

8. Evaluate the Models

Run the validation cells to obtain:

Precision

Recall

mAP@50

mAP@50-95

Then compare the two training approaches.

Project Workflow

Kaggle Road Sign Dataset
          |
          v
Pascal VOC XML Annotations
          |
          v
Extract Classes and Match Images
          |
          v
Convert XML to YOLO Format
          |
          v
80% Train / 20% Validation
          |
          v
     +------------------+
     |                  |
     v                  v
 YOLOv8n             YOLOv8n
 From Scratch     Transfer Learning
     |                  |
     +--------+---------+
              |
              v
       Model Validation
              |
              v
   Same-Image Inference
              |
              v
Bounding Box and Confidence
          Comparison

Future Improvements

Possible extensions include:

Increasing the number of training epochs

Training with GPU acceleration

Comparing YOLOv8n with YOLOv8s or YOLOv8m

Adding data augmentation

Performing hyperparameter optimization

Creating a separate untouched test set

Evaluating per-class performance

Adding confusion-matrix visualization

Testing on external road images

Performing real-time road sign detection using a webcam or video

Deploying the trained model as a web application

Conclusion

This project demonstrates a complete object-detection workflow using YOLOv8 for road sign detection.

It includes:

Kaggle dataset acquisition

Pascal VOC annotation processing

Automatic class extraction

YOLO annotation conversion

Train-validation splitting

YOLOv8 training from scratch

YOLOv8 transfer learning

Object detection inference

Bounding-box and confidence comparison

Quantitative model evaluation

The project is designed to demonstrate the practical difference between training a deep-learning object detector from random initialization and adapting a pretrained model through transfer learning.
