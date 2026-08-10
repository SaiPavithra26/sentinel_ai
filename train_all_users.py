import os
import time
import argparse
import pandas as pd
import numpy as np
import traceback
from tqdm import tqdm
from train import train_model

def train_all_users(data_dir: str, base_model_dir: str, epochs: int, batch_size: int) -> None:
    """
    Reads all unique user IDs from the train.csv dataset and trains a behavioral model
    sequentially for each user. Displays progress, elapsed time, and estimated remaining time.
    """
    train_path = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Processed training file not found at {train_path}. Please run preprocess.py first.")
        
    print(f"Reading training dataset from {train_path} to identify users...")
    df_train = pd.read_csv(train_path)
    all_users = sorted(df_train["user_id"].unique())
    num_users = len(all_users)
    print(f"Discovered {num_users} users in the training dataset.")
    
    successful_users = []
    failed_users = []
    
    start_time = time.time()
    
    # tqdm progress bar wrapper
    pbar = tqdm(all_users, desc="Overall Training Progress")
    for idx, user_id in enumerate(pbar):
        user_str = str(user_id)
        current_idx = idx + 1
        
        # Display clear progress message
        print(f"\n==================================================")
        print(f"Training User {user_str} ({current_idx}/{num_users})")
        print(f"==================================================")
        
        # Filter training data for owner to check sample count
        y_train = (df_train["user_id"].astype(str) == user_str).astype(int).values
        num_samples = np.sum(y_train)
        
        if num_samples == 0:
            print(f"User {user_str} has 0 training samples. Skipping training.")
            continue
            
        try:
            # train_model will automatically handle saving to base_model_dir/user_{user_str}/
            train_model(
                data_dir=data_dir,
                owner_id=user_str,
                epochs=epochs,
                batch_size=batch_size,
                model_dir=base_model_dir
            )
            successful_users.append(user_str)
            print(f"\n--> SUCCESS: Model trained for User {user_str}")
        except Exception as e:
            failed_users.append((user_str, str(e)))
            print(f"\n--> ERROR: Failed to train model for User {user_str}: {e}")
            traceback.print_exc()
            
        # Time calculations
        elapsed = time.time() - start_time
        completed_runs = len(successful_users) + len(failed_users)
        
        if completed_runs > 0:
            avg_time = elapsed / completed_runs
            remaining_runs = num_users - completed_runs
            est_remaining = avg_time * remaining_runs
            
            # Format time strings
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            remaining_str = time.strftime("%H:%M:%S", time.gmtime(est_remaining))
            
            print(f"Elapsed Time: {elapsed_str} | Estimated Remaining Time: {remaining_str}")
            
    total_time = time.time() - start_time
    total_time_str = time.strftime("%H:%M:%S", time.gmtime(total_time))
    
    print(f"\n==================================================")
    print(f"               TRAINING COMPLETED                 ")
    print(f"==================================================")
    print(f"Total training time: {total_time_str}")
    print(f"Successfully trained: {len(successful_users)}/{num_users} users")
    print(f"Failed users: {len(failed_users)}")
    if failed_users:
        print("Failed Users:")
        for u, err in failed_users:
            print(f"  - User {u}: {err}")
    print(f"==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelAI: Train All Users sequentially")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train each model")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--data-dir", type=str, default="processed_data",
                        help="Directory containing the train/val/test datasets")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Base directory to save the trained user models")
    args = parser.parse_args()
    
    train_all_users(
        data_dir=args.data_dir,
        base_model_dir=args.model_dir,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
