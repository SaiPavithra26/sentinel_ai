import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)

# Use Agg backend for matplotlib to prevent GUI popup windows during headless execution
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def evaluate_model(data_dir, owner_id, model_dir, output_dir):
    """
    Evaluates the trained model on test data, prints metrics,
    and generates evaluation charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Test Dataset
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
        
    num_train = len(pd.read_csv(train_path)) if os.path.exists(train_path) else 0
    num_val = len(pd.read_csv(val_path)) if os.path.exists(val_path) else 0
    num_test = len(pd.read_csv(test_path)) if os.path.exists(test_path) else 0
    
    print("\n==================================")
    print("        SANITY CHECKS             ")
    print("==================================")
    print(f"Train samples:      {num_train}")
    print(f"Validation samples: {num_val}")
    print(f"Test samples:       {num_test}")
    print("==================================")
    
    df_test = pd.read_csv(test_path)
    X_test = df_test.drop(columns=["user_id"]).values
    # Ensure targets match type of owner_id
    y_test = (df_test["user_id"].astype(str) == str(owner_id)).astype(int).values
    
    print(f"Loaded test data: {X_test.shape[0]} samples")
    
    # 2. Load Trained Model
    model_path = os.path.join(model_dir, "behavior_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}")
        
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded behavior model from {model_path}")
    
    # Load optimal threshold if available
    threshold = 0.5
    user_config_path = os.path.join(model_dir, "user_config.json")
    if os.path.exists(user_config_path):
        import json
        try:
            with open(user_config_path, "r") as f:
                user_cfg = json.load(f)
                threshold = user_cfg.get("threshold", 0.5)
                print(f"Loaded calibrated threshold: {threshold:.4f}")
        except Exception:
            pass
            
    # 3. Predict probabilities and threshold
    y_probs = model.predict(X_test).flatten()
    y_pred = (y_probs >= threshold).astype(int)
    
    total_predictions = len(y_pred)
    owner_samples = int(np.sum(y_test == 1))
    impostor_samples = int(np.sum(y_test == 0))
    
    print("\n--- Evaluation Details ---")
    print(f"Total predictions: {total_predictions}")
    print(f"Owner samples:     {owner_samples}")
    print(f"Impostor samples:  {impostor_samples}")
    
    # 4. Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n==================================")
    print("      EVALUATION METRICS          ")
    print("==================================")
    print(f"Owner ID:         {owner_id}")
    print(f"Accuracy:         {accuracy * 100:.2f}%")
    print(f"Precision:        {precision * 100:.2f}%")
    print(f"Recall:           {recall * 100:.2f}%")
    print(f"F1 Score:         {f1 * 100:.2f}%")
    print("==================================")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Suspicious", "Owner"])
    plt.yticks(tick_marks, ["Suspicious", "Owner"])
    
    # Annotate text inside cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
            
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Confusion Matrix plot saved to {cm_path}")
    
    # 6. ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    print(f"ROC Area Under Curve (AUC): {roc_auc:.4f}")
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()
    print(f"ROC Curve plot saved to {roc_path}")
    
    # 7. Training Loss & Accuracy curves
    history_path = os.path.join(model_dir, "training_history.joblib")
    if os.path.exists(history_path):
        history = joblib.load(history_path)
        
        # Loss Curves plot
        plt.figure(figsize=(12, 5))
        
        # Loss Subplot
        plt.subplot(1, 2, 1)
        plt.plot(history["loss"], label="Train Loss", color="royalblue", lw=2)
        if "val_loss" in history:
            plt.plot(history["val_loss"], label="Val Loss", color="crimson", lw=2)
        plt.title("Model Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        # Accuracy Subplot
        plt.subplot(1, 2, 2)
        plt.plot(history["accuracy"], label="Train Accuracy", color="royalblue", lw=2)
        if "val_accuracy" in history:
            plt.plot(history["val_accuracy"], label="Val Accuracy", color="crimson", lw=2)
        plt.title("Model Training Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.grid(True, linestyle="--", alpha=0.5)
        
        plt.tight_layout()
        metrics_path = os.path.join(output_dir, "training_metrics.png")
        plt.savefig(metrics_path, dpi=300)
        plt.close()
        print(f"Training metrics curves saved to {metrics_path}")
    else:
        print("Training history file not found. Skipping loss/accuracy plots.")

def evaluate_all_users(data_dir, base_model_dir, output_dir):
    """
    Evaluates all trained user models on the test dataset.
    Each user becomes the owner once, and all other users are treated as suspicious.
    Computes average metrics, lists top 10 best and worst performers, and generates reports.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at {test_path}")
        
    num_train = len(pd.read_csv(train_path)) if os.path.exists(train_path) else 0
    num_val = len(pd.read_csv(val_path)) if os.path.exists(val_path) else 0
    num_test = len(pd.read_csv(test_path)) if os.path.exists(test_path) else 0
    
    print("\n==================================")
    print("        SANITY CHECKS             ")
    print("==================================")
    print(f"Train samples:      {num_train}")
    print(f"Validation samples: {num_val}")
    print(f"Test samples:       {num_test}")
    print("==================================")
    
    df_test = pd.read_csv(test_path)
    
    # Find all user directories in base_model_dir
    user_dirs = [d for d in os.listdir(base_model_dir) if d.startswith("user_") and os.path.isdir(os.path.join(base_model_dir, d))]
    # Sort user directories numerically
    user_dirs.sort(key=lambda x: int(x.split("_")[1]))
    
    if not user_dirs:
        print(f"No user model directories found in {base_model_dir}")
        return
        
    results = []
    
    print(f"Starting evaluation of {len(user_dirs)} users...")
    
    for user_dir in user_dirs:
        user_id = user_dir.split("_")[1]
        model_path = os.path.join(base_model_dir, user_dir, "behavior_model.keras")
        
        if not os.path.exists(model_path):
            continue
            
        y_test = (df_test["user_id"].astype(str) == str(user_id)).astype(int).values
        num_samples = len(y_test)
        
        if num_samples == 0:
            print(f"User {user_id} has zero test samples. Skipping.")
            continue
            
        try:
            model = tf.keras.models.load_model(model_path)
            X_test = df_test.drop(columns=["user_id"]).values
            
            # Load user calibrated EER threshold
            threshold = 0.5
            user_config_path = os.path.join(base_model_dir, user_dir, "user_config.json")
            if os.path.exists(user_config_path):
                import json
                try:
                    with open(user_config_path, "r") as f:
                        user_cfg = json.load(f)
                        threshold = user_cfg.get("threshold", 0.5)
                except Exception:
                    pass
            
            y_probs = model.predict(X_test, verbose=0).flatten()
            y_pred = (y_probs >= threshold).astype(int)
            
            # Compute metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
            
            # ROC AUC calculation (safe handling for 0/1 classes)
            roc_auc = np.nan
            num_positives = np.sum(y_test)
            num_negatives = num_samples - num_positives
            
            if num_positives > 0 and num_negatives > 0:
                try:
                    fpr, tpr, _ = roc_curve(y_test, y_probs)
                    roc_auc = auc(fpr, tpr)
                except Exception as roc_err:
                    print(f"User {user_id} | Warning: failed to compute ROC curve: {roc_err}")
            else:
                print(f"User {user_id} | Skipping ROC computation (only one class present: positives={num_positives}, negatives={num_negatives})")
            
            results.append({
                "user_id": user_id,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "auc": roc_auc
            })
            
            auc_str = f"{roc_auc:.4f}" if not np.isnan(roc_auc) else "N/A"
            print(f"User {user_id:3s} | Accuracy: {accuracy*100:5.2f}% | Precision: {precision*100:5.2f}% | Recall: {recall*100:5.2f}% | F1: {f1*100:5.2f}% | AUC: {auc_str} | Preds: {len(y_pred)} (Owner: {int(num_positives)}, Impostor: {int(num_negatives)})")
            
        except Exception as e:
            print(f"Error evaluating User {user_id}: {e}")
            
    if not results:
        print("No users were successfully evaluated.")
        return
        
    df_results = pd.DataFrame(results)
    
    # Save detailed per-user results
    results_csv_path = os.path.join(output_dir, "all_users_metrics.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\nDetailed per-user metrics saved to {results_csv_path}")
    
    # Compute averages
    avg_accuracy = df_results["accuracy"].mean()
    avg_precision = df_results["precision"].mean()
    avg_recall = df_results["recall"].mean()
    avg_f1 = df_results["f1_score"].mean()
    avg_auc = df_results["auc"].dropna().mean()
    
    # Sort users by performance (F1-score descending, with AUC and Accuracy as tiebreakers)
    df_sorted = df_results.sort_values(by=["f1_score", "auc", "accuracy"], ascending=False)
    
    top_10_best = df_sorted.head(10)
    top_10_worst = df_sorted.tail(10).iloc[::-1]  # reverse to show the absolute worst at the end
    
    # Print the averages
    print("\n==================================================")
    print("        AVERAGE METRICS ACROSS ALL USERS")
    print("==================================================")
    print(f"Average Accuracy  : {avg_accuracy * 100:.2f}%")
    print(f"Average Precision : {avg_precision * 100:.2f}%")
    print(f"Average Recall    : {avg_recall * 100:.2f}%")
    print(f"Average F1 Score  : {avg_f1 * 100:.2f}%")
    auc_avg_str = f"{avg_auc:.4f}" if not pd.isna(avg_auc) else "N/A"
    print(f"Average ROC AUC   : {auc_avg_str}")
    print("==================================================")
    
    # Print Top 10 Best Performing Users
    print("\nTop 10 Best Performing Users (ranked by F1-score):")
    for _, row in top_10_best.iterrows():
        auc_val = f"{row['auc']:.4f}" if not pd.isna(row['auc']) else "N/A"
        print(f"  User {str(row['user_id']):3s} | F1: {row['f1_score']*100:5.2f}% | AUC: {auc_val:6s} | Accuracy: {row['accuracy']*100:5.2f}%")
        
    # Print Top 10 Worst Performing Users
    print("\nTop 10 Worst Performing Users (ranked by F1-score):")
    for _, row in top_10_worst.iterrows():
        auc_val = f"{row['auc']:.4f}" if not pd.isna(row['auc']) else "N/A"
        print(f"  User {str(row['user_id']):3s} | F1: {row['f1_score']*100:5.2f}% | AUC: {auc_val:6s} | Accuracy: {row['accuracy']*100:5.2f}%")
        
    # Save summary.txt
    best_user = df_sorted.iloc[0]["user_id"]
    best_f1 = df_sorted.iloc[0]["f1_score"]
    worst_user = df_sorted.iloc[-1]["user_id"]
    worst_f1 = df_sorted.iloc[-1]["f1_score"]
    
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("==================================================\n")
        f.write("             EVALUATION SUMMARY REPORT            \n")
        f.write("==================================================\n")
        f.write(f"Number of evaluated users : {len(df_results)}\n")
        f.write(f"Average Accuracy          : {avg_accuracy * 100:.2f}%\n")
        f.write(f"Average Precision         : {avg_precision * 100:.2f}%\n")
        f.write(f"Average Recall            : {avg_recall * 100:.2f}%\n")
        f.write(f"Average F1 Score          : {avg_f1 * 100:.2f}%\n")
        f.write(f"Average ROC AUC           : {auc_avg_str}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Best User (by F1-score)   : User {best_user} (F1: {best_f1 * 100:.2f}%)\n")
        f.write(f"Worst User (by F1-score)  : User {worst_user} (F1: {worst_f1 * 100:.2f}%)\n")
        f.write("==================================================\n")
        
    print(f"\nEvaluation summary report saved to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelAI Module 3: Model Evaluation")
    parser.add_argument("--owner-id", type=str, default="1",
                        help="User ID corresponding to the targeted device Owner")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                        help="Directory containing the test dataset")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Directory containing the saved model(s)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation plot images and files")
    parser.add_argument("--all-users", action="store_true",
                        help="Evaluate all user models found in model-dir and compute averages")
    
    args = parser.parse_args()
    
    if args.all_users or args.owner_id.lower() == "all":
        evaluate_all_users(
            data_dir=args.data_dir,
            base_model_dir=args.model_dir,
            output_dir=args.output_dir
        )
    else:
        # Check if the model directory needs to be adjusted.
        # If --owner-id is set but the model is inside a subdirectory user_{owner_id},
        # check if models/user_{owner_id}/behavior_model.keras exists.
        model_dir = args.model_dir
        user_specific_dir = os.path.join(model_dir, f"user_{args.owner_id}")
        if os.path.exists(os.path.join(user_specific_dir, "behavior_model.keras")):
            model_dir = user_specific_dir
            
        evaluate_model(
            data_dir=args.data_dir,
            owner_id=args.owner_id,
            model_dir=model_dir,
            output_dir=args.output_dir
        )
