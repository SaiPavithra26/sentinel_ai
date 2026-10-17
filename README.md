# SentinelAI — Continuous Behavioral Authentication for Smartphones

SentinelAI is a mobile security project that explores continuous behavioral authentication using machine learning. It analyzes smartphone interaction patterns to identify whether observed behavior is consistent with the legitimate user, providing an additional layer of protection beyond conventional authentication methods.

## Overview

Traditional smartphone authentication methods, such as PINs and passwords, verify a user primarily at the point of access. SentinelAI explores continuous authentication by learning behavioral patterns from smartphone interaction data and identifying deviations that may indicate unauthorized usage.

The system uses the HMOG (Hand Movement, Orientation, and Grasp) dataset to train a neural-network-based classification model. The trained model is converted to TensorFlow Lite and integrated into a Flutter-based Android application for on-device inference.

## Key Features

- **Behavioral authentication:** Uses smartphone interaction patterns as an additional security signal.
- **Machine learning classification:** Trains a neural network to distinguish the evaluated behavioral classes.
- **On-device inference:** Uses TensorFlow Lite to run the trained model within the mobile application.
- **Security dashboard:** Provides access to the application's security functionality.
- **Emergency contacts:** Supports adding, viewing, editing, and removing trusted contacts.
- **Alert history:** Displays recorded security events and available event details.
- **Recovery mode:** Provides security-status, location, and alert-related information for potential lost-device situations.

> Note: The dashboard, emergency contacts, alert history, and recovery mode are application features. The report's model evaluation does not establish that every feature is connected to a production-grade security or device-recovery service.

## System Architecture

The SentinelAI workflow consists of the following stages:

```text
HMOG Behavioral Dataset
          |
          v
Data Inspection and Preprocessing
          |
          v
Behavioral Feature Preparation
          |
          v
Neural Network Training
     (TensorFlow)
          |
          v
Model Evaluation
          |
          v
TensorFlow Lite Conversion
          |
          v
Flutter Android Application
          |
          v
On-device Behavioral Inference
```

### Machine Learning Pipeline

1. **Data preparation:** Load and inspect the HMOG behavioral dataset, identify relevant features, and prepare the model inputs.
2. **Preprocessing:** Clean the data and prepare separate training, validation, and testing sets.
3. **Model training:** Train a neural-network classifier using TensorFlow.
4. **Evaluation:** Measure model performance using accuracy, precision, recall, F1-score, and AUC.
5. **Mobile deployment:** Convert the trained model to TensorFlow Lite and integrate it into the Flutter Android application.



## Technology Stack

Python 3.12 - Machine learning development 
TensorFlow 2.21 - Neural-network training 
TensorFlow Lite - Mobile model deployment 
Pandas and NumPy - Data processing 
Scikit-learn - Data preparation and evaluation utilities 
Flutter - Android application development 
Dart - Mobile application programming 

The reported confusion matrix contains 124 test samples, all classified correctly.



## Getting Started

### Prerequisites

Install the following development tools:

- [Flutter SDK](https://docs.flutter.dev/get-started/install)
- [Android Studio](https://developer.android.com/studio) or another supported Flutter development environment
- Android SDK and an Android emulator or physical device
- Python 3.12 for machine learning development

### 1. Clone the repository

```bash
git clone https://github.com/SaiPavithra26/sentinel_ai.git
cd sentinel_ai
```

### 2. Install Flutter dependencies

```bash
flutter pub get
```

### 3. Check your Flutter environment

```bash
flutter doctor
```

Resolve any required Android SDK or device configuration issues.

### 4. Run the application

Connect an Android device or start an emulator, then execute:

```bash
flutter run
```

### Machine Learning Model Setup

The trained TensorFlow Lite model and its expected input format must be available to the application for inference.

Before running the behavioral-authentication functionality, verify that:

- The model file is included in the project.
- Its path is correctly configured in the Flutter application.
- The input feature order and preprocessing match those used during training.
- The TensorFlow Lite inference code loads the model successfully.

The exact model filename, asset path, and model-training commands should be documented once confirmed from the repository's implementation.

## Project Structure

The following is a conceptual organization of the project's components. Adapt the paths to match the actual repository structure.

```text
sentinel_ai/
├── lib/                  # Flutter application source
├── android/               # Android platform configuration
├── assets/                # Application assets and model, if included
├── test/                  # Flutter tests, if included
├── pubspec.yaml           # Flutter dependencies and asset configuration
└── README.md
```

The machine-learning training scripts and HMOG dataset may be maintained separately from the Flutter application.

## Limitations

- The evaluation uses a limited test dataset.
- Model performance may vary across users, devices, and behavioral conditions.
- Real-world deployment requires additional testing with diverse users and smartphones.
- Behavioral authentication should supplement, not replace, established smartphone security mechanisms.
- The project is a proof of concept; production-grade anti-theft, remote locking, and recovery capabilities require additional implementation and validation.

## Future Enhancements

- Evaluate the model using larger and more diverse behavioral datasets.
- Collect real-world smartphone interaction data with appropriate user consent.
- Explore continuous personalization as legitimate user behavior changes.
- Incorporate additional behavioral signals, including touch dynamics and motion.
- Improve detection, alerting, and integration with smartphone security mechanisms.


## References

1. HMOG dataset — Hand Movement, Orientation, and Grasp behavioral data.
2. [TensorFlow documentation](https://www.tensorflow.org/)
3. [TensorFlow Lite documentation](https://www.tensorflow.org/lite)
4. [Flutter documentation](https://docs.flutter.dev/)
5. [Firebase documentation](https://firebase.google.com/docs/)

---

*SentinelAI explores how behavioral machine learning can provide an additional layer of smartphone security.*
