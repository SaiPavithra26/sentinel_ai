import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
import joblib

def load_and_prepare_data(data_dir, owner_id):
    """
    Loads train and validation CSV files and converts the multiclass
    user_id column to a binary label (1: Owner, 0: Suspicious User).
    """
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Processed training/validation files not found in {data_dir}. "
                                "Please run preprocess.py first.")
        
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # Separate features and targets
    X_train = df_train.drop(columns=["user_id"]).values
    # Convert targets to string to ensure safe matching
    y_train = (df_train["user_id"].astype(str) == str(owner_id)).astype(int).values
    
    X_val = df_val.drop(columns=["user_id"]).values
    y_val = (df_val["user_id"].astype(str) == str(owner_id)).astype(int).values
    
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features.")
    print(f"Loaded validation data: {X_val.shape[0]} samples.")
    print(f"Training Owner (Class 1) count: {np.sum(y_train)} / {len(y_train)}")
    print(f"Validation Owner (Class 1) count: {np.sum(y_val)} / {len(y_val)}")
    
    return X_train, y_train, X_val, y_val, X_train.shape[1]

def build_mlp_model(input_dim):
    """
    Constructs the Feedforward Neural Network (MLP) architecture:
    - Input Layer
    - Dense (64 nodes) + Batch Normalization + ReLU + Dropout (rate=0.3)
    - Dense (32 nodes) + Batch Normalization + ReLU + Dropout (rate=0.2)
    - Dense (1 node, Sigmoid Output)
    """
    model = tf.keras.Sequential([
        # Input Layer
        tf.keras.layers.Input(shape=(input_dim,)),
        
        # Dense + BN + ReLU + Dropout
        tf.keras.layers.Dense(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.3),
        
        # Dense + BN + ReLU + Dropout
        tf.keras.layers.Dense(32),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dropout(0.2),
        
        # Dense + Sigmoid Output for binary classification
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    
    return model

def calculate_eer_threshold(model, X_val, y_val):
    """
    Computes predictions on the validation set, calculates the ROC curve,
    and returns the threshold that matches the Equal Error Rate (EER).
    """
    try:
        from sklearn.metrics import roc_curve
        y_probs = model.predict(X_val, verbose=0).flatten()
        fpr, tpr, thresholds = roc_curve(y_val, y_probs)
        fnr = 1.0 - tpr
        
        # Find index of threshold where FPR and FNR are closest
        idx = np.nanargmin(np.abs(fpr - fnr))
        optimal_threshold = float(thresholds[idx])
        eer = float(fpr[idx])
        
        # Clamp threshold to reasonable range to prevent extreme predictions
        optimal_threshold = max(min(optimal_threshold, 0.99), 0.01)
    except Exception as e:
        print(f"Warning: could not calculate EER threshold: {e}. Defaulting to 0.5")
        optimal_threshold, eer = 0.5, 0.5
        
    return optimal_threshold, eer

def train_model(data_dir, owner_id, epochs, batch_size, model_dir):
    """
    Compiles, trains, and saves the behavior authentication model.
    """
    # Automatically save every trained model into its own directory
    if not os.path.basename(model_dir).startswith("user_"):
        model_dir = os.path.join(model_dir, f"user_{owner_id}")
    os.makedirs(model_dir, exist_ok=True)
    
    # Load and clean datasets
    X_train, y_train, X_val, y_val, input_dim = load_and_prepare_data(data_dir, owner_id)
    
    # Check if Owner has any samples
    if np.sum(y_train) == 0:
        raise ValueError(f"No training samples found for Owner ID: {owner_id}. "
                         "Please check that this user exists and has valid session features.")
        
    # Calculate class weights to address data imbalance
    # Owner (Class 1) represents 1 user, Suspicious Users (Class 0) represent 116 users
    unique_classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=y_train
    )
    class_weight_dict = {cls: w for cls, w in zip(unique_classes, weights)}
    print(f"Calculated class weights: {class_weight_dict}")
    
    # Build MLP Architecture
    model = build_mlp_model(input_dim)
    model.summary()
    
    # Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )
    
    # Setup Callbacks
    # Early Stopping halts training if validation loss stops improving for 10 consecutive epochs
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    # Learning Rate scheduling to decay learning rates when validation loss plateaus
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    model_save_path = os.path.join(model_dir, "behavior_model.keras")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )
    
    # Train Model
    print(f"\n--- Training Behavioral Model for Owner: {owner_id} ---")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    
    # Fallback: if early stop restored best weights but checkpoint didn't save for some reason,
    # or validation loss didn't trigger checkpoint, we save the final model state
    if not os.path.exists(model_save_path):
        model.save(model_save_path)
        
    print(f"Best model successfully saved to {model_save_path}")
    
    # Calculate EER threshold on validation set for real-time inference
    optimal_threshold, eer = calculate_eer_threshold(model, X_val, y_val)
    print(f"Calculated Optimal EER Threshold for Owner {owner_id}: {optimal_threshold:.4f} (Validation EER: {eer:.4f})")
    
    # Save user calibration config to JSON
    import json
    user_config = {
        "owner_id": str(owner_id),
        "threshold": optimal_threshold,
        "eer": eer
    }
    user_config_path = os.path.join(model_dir, "user_config.json")
    with open(user_config_path, "w") as f:
        json.dump(user_config, f, indent=4)
    print(f"Saved user calibration config to {user_config_path}")
    
    # Save training history dictionary for plotting in evaluation.py
    history_path = os.path.join(model_dir, "training_history.joblib")
    joblib.dump(history.history, history_path)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelAI Module 3: Model Training")
    parser.add_argument("--owner-id", type=str, default="1",
                        help="User ID to treat as the device Owner (Class 1)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train the model")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                        help="Directory containing the train/val datasets")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory to save the trained model")
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        owner_id=args.owner_id,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_dir=args.model_dir
    )
