import os
import glob
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib

# Import extractors and selector from our feature engineering module
from feature_engineering import (
    KeystrokeFeatureExtractor,
    TouchFeatureExtractor,
    MotionFeatureExtractor,
    FeatureSelector
)

def discover_dataset(base_dir):
    """
    Scans the base_dir to find the dataset root directory (starts with 'BB-MAS')
    and returns the paths to all user subfolders.
    """
    print(f"Scanning for dataset directory under: {base_dir}")
    
    # Check if base_dir itself is the dataset root
    if os.path.basename(base_dir.rstrip("\\/")).lower().startswith("bb-mas"):
        dataset_root = base_dir
    else:
        # Search inside base_dir
        matches = [d for d in os.listdir(base_dir) if d.lower().startswith("bb-mas")]
        if not matches:
            # Try recursive search if not immediately found
            matches = glob.glob(os.path.join(base_dir, "**/BB-MAS*"), recursive=True)
            if not matches:
                raise FileNotFoundError(f"Could not find any directory starting with 'BB-MAS' in {base_dir}")
            dataset_root = matches[0]
        else:
            dataset_root = os.path.join(base_dir, matches[0])
            
    print(f"Dataset root discovered: {dataset_root}")
    
    # User folders are subfolders that are numeric names
    user_folders = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path) and item.isdigit():
            user_folders.append(item_path)
            
    # Sort user folders numerically
    user_folders.sort(key=lambda x: int(os.path.basename(x)))
    print(f"Discovered {len(user_folders)} user folders.")
    return dataset_root, user_folders

def parse_timestamps(series):
    """
    Cleans raw timestamp string formats and parses to pandas datetime.
    Formats supported: YYYY-MM-DD HH:MM:SS.mmm and YYYY-MM-DD HH:MM:SS:mmm
    """
    def clean_t(val):
        if pd.isna(val) or not isinstance(val, str):
            return pd.NaT
        val_str = val.strip()
        # TouchEvent format has colon for milliseconds (e.g., 2017-4-14 18:56:47:185)
        # We replace the last colon with a dot if there are 3 colons total in the time string
        if val_str.count(':') == 3:
            parts = val_str.rsplit(':', 1)
            val_str = '.'.join(parts)
        return val_str

    cleaned = series.apply(clean_t)
    return pd.to_datetime(cleaned, errors='coerce')

def clean_and_interpolate_timestamps(df):
    """
    Cleans raw timestamp string formats, parses to pandas datetime,
    and linearly interpolates missing timestamps for intermediate events
    (e.g., intermediate points of swipes).
    """
    if df is None or "time" not in df.columns:
        return df
    
    # 1. Clean and parse timestamps to datetime
    df["time"] = parse_timestamps(df["time"])
    
    # Check if there are any valid timestamps to orient the interpolation
    if df["time"].isna().all():
        return df
        
    # 2. Convert NaT to NaN in a numeric nanoseconds float representation
    time_ns = df["time"].apply(lambda x: float(x.value) if pd.notna(x) else np.nan)
    
    # 3. Linearly interpolate the nanoseconds representation
    time_ns = time_ns.interpolate(method="linear")
    
    # 4. Handle boundary cases (leading/trailing NaNs) using forward/backward fill
    time_ns = time_ns.ffill().bfill()
    
    # 5. Convert back to datetime
    df["time"] = pd.to_datetime(time_ns, unit="ns")
    return df

def load_user_file(user_folder, sensor_name):
    """
    Helper to locate and load a specific sensor file for a user folder.
    Matches files with 'HandPhone' and the sensor name.
    """
    files = os.listdir(user_folder)
    # Search for files containing 'HandPhone' and the target sensor string
    matched_files = [f for f in files if "HandPhone" in f and sensor_name in f]
    if not matched_files:
        return None
    
    filepath = os.path.join(user_folder, matched_files[0])
    try:
        df = pd.read_csv(filepath)
        # Remove duplicate rows
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_user_data(user_folder, session_duration_mins, min_events=2):
    """
    Loads all HandPhone sensor files for a user, standardizes timestamps,
    divides the timeline into non-overlapping sessions, and extracts features.
    """
    user_id = os.path.basename(user_folder)
    
    # Load raw datasets
    acc_df = load_user_file(user_folder, "Accelerometer")
    gyro_df = load_user_file(user_folder, "Gyroscope")
    touch_df = load_user_file(user_folder, "TouchEvent")
    key_df = load_user_file(user_folder, "Keyboard")
    
    # Clean, parse and interpolate timestamps to preserve intermediate events
    acc_df = clean_and_interpolate_timestamps(acc_df)
    gyro_df = clean_and_interpolate_timestamps(gyro_df)
    touch_df = clean_and_interpolate_timestamps(touch_df)
    key_df = clean_and_interpolate_timestamps(key_df)
    
    parsed_timestamps = []
    for df in [acc_df, gyro_df, touch_df, key_df]:
        if df is not None and "time" in df.columns:
            # Drop any remaining NaTs
            df.dropna(subset=["time"], inplace=True)
            # Add non-NaT timestamps to overall timeline range finder
            parsed_timestamps.extend(df["time"].tolist())

    if not parsed_timestamps:
        # No timestamps to orient sessions, skip user
        return []

    # Find the overall session span
    min_time = min(parsed_timestamps)
    max_time = max(parsed_timestamps)
    
    # Instantiate feature extractors
    key_extractor = KeystrokeFeatureExtractor()
    touch_extractor = TouchFeatureExtractor()
    acc_extractor = MotionFeatureExtractor(prefix="acc")
    gyro_extractor = MotionFeatureExtractor(prefix="gyro")
    
    user_sessions = []
    
    # Define session windows
    window_delta = pd.Timedelta(minutes=session_duration_mins)
    current_start = min_time
    session_idx = 0
    
    # Pre-clean coordinates and actionTypes in touch events
    if touch_df is not None and len(touch_df) > 0:
        touch_df["actionType"] = touch_df["actionType"].ffill()
        touch_df.dropna(subset=["Xvalue", "Yvalue"], inplace=True)
        
    while current_start < max_time:
        current_end = current_start + window_delta
        
        # Filter raw event data for this session window
        sess_key = key_df[(key_df["time"] >= current_start) & (key_df["time"] < current_end)] if key_df is not None else None
        sess_touch = touch_df[(touch_df["time"] >= current_start) & (touch_df["time"] < current_end)] if touch_df is not None else None
        sess_acc = acc_df[(acc_df["time"] >= current_start) & (acc_df["time"] < current_end)] if acc_df is not None else None
        sess_gyro = gyro_df[(gyro_df["time"] >= current_start) & (gyro_df["time"] < current_end)] if gyro_df is not None else None
        
        # Check if there is enough interaction in this session
        num_keys = len(sess_key) if sess_key is not None else 0
        num_touches = len(sess_touch) if sess_touch is not None else 0
        
        if num_keys < min_events and num_touches < min_events:
            # Skip idle sessions that don't have enough keystrokes or touches
            current_start = current_end
            continue
            
        # Extract features
        session_duration_sec = session_duration_mins * 60.0
        
        k_feats = key_extractor.extract(sess_key, duration_seconds=session_duration_sec)
        t_feats = touch_extractor.extract(sess_touch)
        a_feats = acc_extractor.extract(sess_acc)
        g_feats = gyro_extractor.extract(sess_gyro)
        
        # Combine all modalities into one feature dictionary
        combined_feats = {}
        combined_feats["user_id"] = user_id
        combined_feats["session_id"] = f"user_{user_id}_sess_{session_idx}"
        
        combined_feats.update(k_feats)
        combined_feats.update(t_feats)
        combined_feats.update(a_feats)
        combined_feats.update(g_feats)
        
        user_sessions.append(combined_feats)
        session_idx += 1
        current_start = current_end

    return user_sessions

def fit_and_apply_outlier_clipping(X_train, X_val, X_test, feature_cols):
    """
    Fits outlier boundaries (IQR) on X_train ONLY to prevent data leakage,
    then applies clipping to X_train, X_val, and X_test.
    Returns the clipped datasets and the dict of bounds for inference.
    """
    X_train_clipped = X_train.copy()
    X_val_clipped = X_val.copy()
    X_test_clipped = X_test.copy()
    
    clipping_bounds = {}
    
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(X_train_clipped[col]):
            q1 = X_train_clipped[col].quantile(0.25)
            q3 = X_train_clipped[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Save bounds
            clipping_bounds[col] = {
                "lower": float(lower_bound),
                "upper": float(upper_bound)
            }
            
            # Clip values
            X_train_clipped[col] = X_train_clipped[col].clip(lower=lower_bound, upper=upper_bound)
            X_val_clipped[col] = X_val_clipped[col].clip(lower=lower_bound, upper=upper_bound)
            X_test_clipped[col] = X_test_clipped[col].clip(lower=lower_bound, upper=upper_bound)
            
    return X_train_clipped, X_val_clipped, X_test_clipped, clipping_bounds

def build_pipeline(dataset_dir, session_duration_mins=5, force_extract=False):
    """
    Orchestrates the preprocessing and feature engineering pipeline.
    Includes caching, missing value imputation, normalization, standardization,
    feature selection, and stratified splitting.
    """
    # Create target directories
    os.makedirs("processed_data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    raw_features_path = os.path.join("processed_data", "raw_session_features.csv")
    
    # 1. Feature Extraction / Ingestion
    if os.path.exists(raw_features_path) and not force_extract:
        print(f"\n--- Loading extracted session features from cache: {raw_features_path} ---")
        features_df = pd.read_csv(raw_features_path)
    else:
        print("\n--- Starting raw dataset ingestion and feature extraction ---")
        dataset_root, user_folders = discover_dataset(dataset_dir)
        
        # Process all users
            
        all_sessions = []
        for idx, u_folder in enumerate(user_folders):
            print(f"[{idx+1}/{len(user_folders)}] Processing user: {os.path.basename(u_folder)}...")
            u_sessions = process_user_data(u_folder, session_duration_mins)
            all_sessions.extend(u_sessions)
            print(f"   Extracted {len(u_sessions)} valid behavioral sessions.")
            
        if not all_sessions:
            raise ValueError("No valid sessions could be extracted. Please check the dataset folders and files.")
            
        features_df = pd.DataFrame(all_sessions)
        
        # Save raw features cache
        features_df.to_csv(raw_features_path, index=False)
        print(f"Saved extracted raw features to cache: {raw_features_path}")
        
    print(f"\nExtracted Feature Matrix Shape: {features_df.shape}")
    
    # Extract identifiers and target labels
    meta_cols = ["user_id", "session_id"]
    feature_cols = [col for col in features_df.columns if col not in meta_cols]
    
    X = features_df[feature_cols].copy()
    y = features_df["user_id"].copy()
    
    # 2. Stratified Data Split (Train, Val, Test)
    # Check minimum class occurrences for safe stratification
    class_counts = y.value_counts()
    min_samples = class_counts.min()
    print(f"Minimum sessions per user: {min_samples}")
    
    # Try stratified split for first split
    try:
        if min_samples >= 2:
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.30, stratify=y, random_state=42
            )
            # Try stratified split for second split
            try:
                temp_class_counts = y_temp.value_counts()
                if temp_class_counts.min() >= 2:
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
                    )
                else:
                    raise ValueError("Insufficient class sizes for second split stratification")
            except Exception:
                print("Warning: Stratification not possible on validation/test split. Falling back to random split for validation and test sets.")
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=0.50, random_state=42
                )
        else:
            raise ValueError("Insufficient class sizes for first split stratification")
    except Exception:
        print("Warning: Stratification not possible. Falling back to non-stratified random split for all datasets.")
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=42
        )
        
    print(f"Dataset split results:")
    print(f"  Train set: {X_train.shape[0]} samples")
    print(f"  Val set:   {X_val.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    # 3. Outlier Removal (Clipping - fitted on Train ONLY to prevent leakage)
    print("\n--- Applying outlier clipping (IQR boundary fitted on Train) ---")
    X_train, X_val, X_test, clipping_bounds = fit_and_apply_outlier_clipping(X_train, X_val, X_test, feature_cols)
    
    # 4. Handle Missing Values (Median Imputation)
    print("\n--- Imputing missing values (median imputation) ---")
    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=feature_cols, index=X_val.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)
    
    # Save the fitted imputer for inference
    joblib.dump(imputer, os.path.join("models", "imputer.joblib"))
    print("Saved Imputer model to models/imputer.joblib")

    # 5. Normalization (MinMax scaling to [0, 1])
    print("\n--- Normalizing numerical values [0, 1] ---")
    minmax = MinMaxScaler()
    X_train_norm = pd.DataFrame(minmax.fit_transform(X_train_imp), columns=feature_cols, index=X_train.index)
    X_val_norm = pd.DataFrame(minmax.transform(X_val_imp), columns=feature_cols, index=X_val.index)
    X_test_norm = pd.DataFrame(minmax.transform(X_test_imp), columns=feature_cols, index=X_test.index)
    
    joblib.dump(minmax, os.path.join("models", "minmax_scaler.joblib"))
    print("Saved MinMaxScaler to models/minmax_scaler.joblib")

    # 6. Feature Selection (on normalized data)
    print("\n--- Performing feature selection on normalized features ---")
    selector = FeatureSelector(variance_threshold=0.01, correlation_threshold=0.95, k_best=50)
    selector.fit(X_train_norm, y_train)
    
    X_train_sel = selector.transform(X_train_norm)
    X_val_sel = selector.transform(X_val_norm)
    X_test_sel = selector.transform(X_test_norm)
    
    selector.save(os.path.join("models", "feature_selector.joblib"))
    
    # 7. Standardization (StandardScaler - on selected features ONLY)
    print("\n--- Standardizing selected features (zero mean, unit variance) ---")
    scaler = StandardScaler()
    selected_cols = selector.selected_features
    X_train_std = pd.DataFrame(scaler.fit_transform(X_train_sel), columns=selected_cols, index=X_train.index)
    X_val_std = pd.DataFrame(scaler.transform(X_val_sel), columns=selected_cols, index=X_val.index)
    X_test_std = pd.DataFrame(scaler.transform(X_test_sel), columns=selected_cols, index=X_test.index)
    
    joblib.dump(scaler, os.path.join("models", "standard_scaler.joblib"))
    print("Saved StandardScaler to models/standard_scaler.joblib")

    # Save Preprocessing Parameters to JSON for Flutter
    import json
    preprocess_params = {
        "feature_cols": feature_cols,
        "clipping_bounds": clipping_bounds,
        "imputer_medians": dict(zip(feature_cols, imputer.statistics_.tolist())),
        "minmax_min": dict(zip(feature_cols, minmax.min_.tolist())),
        "minmax_scale": dict(zip(feature_cols, minmax.scale_.tolist())),
        "selected_features": selected_cols,
        "standard_mean": dict(zip(selected_cols, scaler.mean_.tolist())),
        "standard_scale": dict(zip(selected_cols, scaler.scale_.tolist()))
    }
    
    json_path = os.path.join("models", "preprocess_params.json")
    with open(json_path, "w") as f:
        json.dump(preprocess_params, f, indent=4)
    print(f"Exported preprocessing parameters to {json_path}")
    
    # 8. Save Final Processed Outputs
    # Re-attach user ID label as the first column
    train_out = X_train_std.copy()
    train_out.insert(0, "user_id", y_train)
    
    val_out = X_val_std.copy()
    val_out.insert(0, "user_id", y_val)
    
    test_out = X_test_std.copy()
    test_out.insert(0, "user_id", y_test)
    
    train_out.to_csv(os.path.join("processed_data", "train.csv"), index=False)
    val_out.to_csv(os.path.join("processed_data", "val.csv"), index=False)
    test_out.to_csv(os.path.join("processed_data", "test.csv"), index=False)
    
    print("\n--- Preprocessing Pipeline Complete ---")
    print(f"Final output train shape: {train_out.shape}")
    print(f"Final output val shape:   {val_out.shape}")
    print(f"Final output test shape:  {test_out.shape}")
    print("Processed datasets saved to processed_data/ folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SentinelAI Module 2: Preprocessing and Feature Engineering")
    parser.add_argument("--dataset-dir", type=str, default=r"d:\Documents\sentinel_ai\dataset\BB-MAS_Dataset",
                        help="Path to the BB-MAS dataset root directory")
    parser.add_argument("--session-len", type=int, default=5,
                        help="Duration of behavioral sessions in minutes")
    parser.add_argument("--force", action="store_true",
                        help="Force feature extraction and ignore cached raw features file")
    
    args = parser.parse_args()
    
    # Ensure correct separator handling
    dataset_dir = args.dataset_dir.replace("/", os.sep).replace("\\", os.sep)
    
    build_pipeline(
        dataset_dir=dataset_dir,
        session_duration_mins=args.session_len,
        force_extract=args.force
    )
