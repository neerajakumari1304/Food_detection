# Multi_Model_Food_Detection
This project is a comprehensive Deep Learning system designed to identify various food items from images and provide real-time nutritional facts. The system compares three different architectures—Custom CNN, VGG16, and ResNet50—to find the most accurate model for food recognition.

.

## 🚀 Features
**Multi-Model Classification:** Support for three different architectures:

**Custom CNN:** Lightweight model optimized for specific food classes (~92% accuracy).

**ResNet50:** Transfer learning approach using deep residual networks (~95% accuracy).

**VGG16:** Baseline model for performance comparison.

**Nutritional Insights:** Real-time lookup of calories, protein, fats, and carbs.

**High-Speed Data Retrieval:** Utilizes Redis as a NoSQL database to store and fetch food_item JSON data instantly.

**Automated Pipeline:** Includes scripts for dataset splitting (Train/Val/Test), automated training, and performance logging.

**Web Interface:** A Flask-based dashboard for image uploads and model selection.

## 📁 Project Structure
Food_detection/
├── app.py                # Main Flask application (Web UI & Inference)

├── main.py               # Central execution script (Data init + Training)

├── split_data.py         # Dataset management (Splits raw data into Train/Val/Test)

├── food_item.py          # Redis data initialization and nutritional logic

├── log_code.py           # Unified logging configuration

├── check_redis.py        # Utility script to verify Redis data integrity

├── food_item.json        # Source nutritional data

├── data_split/           # Organized dataset (created by split_data.py)

│   ├── train/            # 200 images per class

│   ├── valid/            # 50 images per class

│   └── test/             # 10 images per class

├── models/               # Saved .h5 model files

├── logs/                 # Execution logs for debugging

└── metrics/              # JSON files containing model performance results


## 🛠️ Technical Stack
Frameworks: TensorFlow, Keras, Flask

Database: Redis (Key-Value Store)

Languages: Python 3.x

Data Processing: NumPy, Scikit-learn

Logging: Python Logging Module

## 🚀 How to Run
Step 1: Initialize Database & Train Models
Run the central script to populate Redis with nutritional facts and begin the training pipeline:

`python main.py`
Step 2: Start the Web Application
Launch the Flask server:

`python app.py`
Open your browser and navigate to `http://127.0.0.1:5000`.
## 📊 Model Performance
The system evaluates models based on Precision, Recall, and F1-Score. Detailed metrics for each class (e.g., Apple Pie, Burger, Samosa) are exported to the following files:

custom_cnn_metrics.json

resnet50_results.json

vgg16_results.json

## 📝 Logging
All operations (data splitting, training progress, Redis connections) are logged in the logs/ directory. If you encounter issues, check the specific .log file corresponding to the script name.

![App Interface](assets/image.png)
