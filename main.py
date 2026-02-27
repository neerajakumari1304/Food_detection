"""
Train Custom CNN, VGG, and ResNet models for food classification.
Uses the same class splits as app.py. Exports results to JSON.
"""
import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications, callbacks
from sklearn.metrics import classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

from log_code import setup_logging
logger = setup_logging('main')

from food_item import Nutritional_fact
from split_data import split_dataset

# --- Configuration ---
BASE_DIR = r"D:\projects\Food_detection\data_split"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 50

# --- Helper Functions ---

def load_datasets():
    """Loads datasets with safety checks."""
    try:
        if not os.path.exists(BASE_DIR):
            raise FileNotFoundError(f"Directory {BASE_DIR} not found.")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(BASE_DIR, 'train'), image_size=IMG_SIZE, batch_size=BATCH_SIZE)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(BASE_DIR, 'valid'), image_size=IMG_SIZE, batch_size=BATCH_SIZE)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            os.path.join(BASE_DIR, 'test'), image_size=IMG_SIZE, batch_size=BATCH_SIZE, shuffle=False)
        return train_ds, val_ds, test_ds, train_ds.class_names
    except Exception as e:
        logger.error(f"Dataset error: {e}")
        return None, None, None, []


def save_detailed_results(model, test_ds, model_name, class_names):
    """Calculates and saves the TP/FP/FN/TN and Matrix JSON."""
    try:
        logger.info(f"--- Generating Metrics for {model_name} ---")
        y_true, y_pred = [], []

        # Pull labels and predictions from the test set
        for images, labels in test_ds:
            preds = model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=-1))

        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

        detailed_data = {}
        for i, name in enumerate(class_names):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            tn = int(np.sum(cm) - (tp + fp + fn))

            # Formatting exactly as requested for your dashboard
            detailed_data[name] = {
                "model_used": f"{model_name}_model_1",
                "accuracy": float(report[name].get('f1-score', 0)),
                "precision": float(report[name].get('precision', 0)),
                "recall": float(report[name].get('recall', 0)),
                "f1_score": float(report[name].get('f1-score', 0)),
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "confusion_matrix_full": cm.tolist()
            }

        output_path = f"D:/projects/Food_detection/{model_name}_metrics.json"
        with open(output_path, 'w') as f:
            json.dump(detailed_data, f, indent=4)
        logger.info(f"SUCCESS: Metrics saved to {output_path}")
        logger.info(f"Full metrics for {model_name} saved to JSON.")

    except Exception as e:
        print(f"ERROR: Failed to save JSON for {model_name}. {e}")


def train_custom_cnn(train_ds, val_ds, test_ds, class_names):
    """Trains Custom CNN with dropout to prevent overfitting."""
    try:
        logger.info("Training Custom CNN...")
        model = models.Sequential([
            layers.Rescaling(1. / 255, input_shape=(224, 224, 3)),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D(2, 2),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(class_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        model.save("D:/projects/Food_detection/custom_cnn_model.h5")
        save_detailed_results(model, test_ds, "custom_cnn", class_names)
    except Exception as e:
        logger.error(f"Custom CNN failure: {e}")

def train_vgg16(train_ds, val_ds, test_ds, class_names):
    try:
        logger.info("Training vgg16...")
        base_vgg = applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_vgg.trainable = False
        model = models.Sequential([
            base_vgg, layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(len(class_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[callbacks.EarlyStopping(patience=5)])
        model.save(f"D:/projects/Food_detection/vgg16_model.h5")
        save_detailed_results(model, test_ds, "vgg16", class_names)
    except Exception as e:
        logger.error(f"ResNet50 training failed: {e}")

def train_resnet50(train_ds, val_ds, test_ds, class_names):
    try:
        logger.info("Training ResNet50...")
        base_res = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        base_res.trainable = False
        model = models.Sequential([
            base_res,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dense(len(class_names), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
        model.save("D:/projects/Food_detection/resnet50_model.h5")
        save_detailed_results(model, test_ds, history, "resnet50", class_names)
    except Exception as e:
        logger.error(f"ResNet50 training failed: {e}")
# --- Execution ---

if __name__ == "__main__":
    # 1. Store food data in Redis
    obj = Nutritional_fact()
    obj.food_item()

    # 2. Split dataset (Using your existing script)
    split_dataset()

    # 3. Load processed data
    train_ds, val_ds, test_ds, class_names = load_datasets()

    # 4. Run Training for all models
    train_custom_cnn(train_ds, val_ds, test_ds, class_names)
    train_vgg16(train_ds, val_ds, test_ds, class_names)
    train_resnet50(train_ds, val_ds, test_ds, class_names)

    logger.info("All training processes completed successfully.")
