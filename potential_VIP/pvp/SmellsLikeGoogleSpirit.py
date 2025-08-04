# [IN] Import Libraries
# Standard libraries
import sys
import os
from pathlib import Path
import pickle
import datetime

# Data processing
import pandas as pd
import numpy as np
import xgboost as xgb

# Cloud storage and database
from google.cloud import storage
from google.cloud import bigquery
from google_auth_oauthlib.flow import InstalledAppFlow

# [IN] Initialize Clients
project_id = "crowncoins-casino"
client = bigquery.Client(project=project_id)


# [IN] Utility Functions
def load_sql(filename):
    """Read and return contents of SQL file"""
    with open(filename, 'r') as file:
        return file.read()


def process_column(df, column_name, values_list):
    """Process categorical columns and create dummy variables"""
    df = df.copy()
    if df[column_name].dtype == 'object':
        df[column_name] = df[column_name].str.lower()

    for value in values_list:
        col_name = f'{column_name}_{value.lower()}'
        df[col_name] = (df[column_name] == value.lower()).astype(int)

    other_col = f'{column_name}_other'
    df[other_col] = 1
    for value in values_list:
        col_name = f'{column_name}_{value.lower()}'
        df.loc[df[col_name] == 1, other_col] = 0
    return df


def log_transform_columns(df, columns):
    """Apply log transformation to specified columns"""
    df = df.copy()
    epsilon = 1e-10

    for column in columns:
        df[f'ln_{column}'] = np.log(df[column] + epsilon)
        df = df.drop(column, axis=1)
    return df


def predict_vip_status_v2(df_model_full, window_models, decision_threshold=0.8):
    """Make predictions using sequential window models"""
    predictions = []
    for idx, row in df_model_full.iterrows():
        current_window = None
        for window in range(5):
            if row[f'window_{window}']:
                current_window = window
                break

        if current_window is None or current_window not in window_models:
            prediction_data = {
                'user_id': row['user_id'],
                'firebase_user_id': row['firebase_user_id'],
                'signup_date': row['signup_date'],
                'first_purchase_date': row['first_purchase_date'],
                'window': None,
                'ln_sum_amt_total': row['ln_sum_amt_total'],
                'ln_sum_redeem_amt': row['ln_sum_redeem_amt'],
                'email': row['email'],
                'full_name': row['full_name'],
                'probability_positive_class': 0.0,
                'potential_vip_ind': 0
            }
            predictions.append(prediction_data)
            continue

        model_info = window_models[current_window]
        features = model_info['features']
        booster = model_info['booster']

        X = pd.DataFrame([row[features]])
        dmatrix = xgb.DMatrix(X)
        prob = booster.predict(dmatrix)[0]
        prediction = int(prob >= decision_threshold)

        prediction_data = {
            'user_id': row['user_id'],
            'firebase_user_id': row['firebase_user_id'],
            'signup_date': row['signup_date'],
            'first_purchase_date': row['first_purchase_date'],
            'window': current_window,
            'ln_sum_amt_total': row['ln_sum_amt_total'],
            'ln_sum_redeem_amt': row['ln_sum_redeem_amt'],
            'email': row['email'],
            'full_name': row['full_name'],
            'probability_positive_class': float(prob),
            'potential_vip_ind': prediction
        }
        predictions.append(prediction_data)

    predictions_df = pd.DataFrame(predictions)
    column_order = [
        'user_id', 'firebase_user_id', 'signup_date', 'first_purchase_date',
        'window', 'email', 'full_name', 'probability_positive_class', 'potential_vip_ind'
    ]
    return predictions_df[column_order]


# [IN] Setup Directory
base_dir = '/tmp/models'
model_dir = os.path.join(base_dir, 'customers_opt')
Path(model_dir).mkdir(parents=True, exist_ok=True)


def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    """Download files from Google Cloud Storage"""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

# Replace the dataset query section with:
try:
    dataset_query = """
    select * from guyb_sandbox.models_datasets_potential_vip_v2_opt_customers 
    """
    print(f"Dataset SQL content length: {len(dataset_query)}")

    dataset_query_job = client.query(dataset_query)
    dataset_rows = dataset_query_job.result()
    df = pd.DataFrame([dict(row.items()) for row in dataset_rows])
    print(f"Dataset query returned {len(df)} rows")

except Exception as e:
    print(f"Error loading/executing SQL: {str(e)}")
    raise

# [IN] Download Files from GCS
gcs_bucket_name = "us-central1-airflow-crownco-67791dcc-bucket"
files_to_download = {
    "existing_users_opt.sql": "models/customers_opt/existing_users_opt.sql",
    "model_features_v2_opt.txt": "models/customers_opt/model_features_v2_opt.txt",
    "pot_vip_v2_opt.pkl": "models/customers_opt/pot_vip_v2_opt.pkl",
    "enp_opt.sql" : "models/customers_opt/enp_opt.sql"
}

for local_name, gcs_blob_name in files_to_download.items():
    local_path = os.path.join(model_dir, local_name)
    try:
        if not os.path.exists(local_path):
            download_from_gcs(gcs_bucket_name, gcs_blob_name, local_path)
        if not os.path.exists(local_path) or os.path.getsize(local_path) == 0:
            raise ValueError(f"Failed to download or empty file: {local_name}")
        print(f"Successfully downloaded: {local_name}")
    except Exception as e:
        print(f"Error downloading {local_name}: {str(e)}")
        raise



# [IN] Load existing users from BigQuery
try:
    existing_users_path = os.path.join(model_dir, 'existing_users_opt.sql')
    existing_users_query = load_sql(existing_users_path)

    if not existing_users_query or len(existing_users_query.strip()) == 0:
        raise ValueError(f"SQL file is empty: {existing_users_path}")
    print(f"Existing users SQL content length: {len(existing_users_query)}")

    existing_users_job = client.query(existing_users_query)
    existing_users_rows = existing_users_job.result()
    df_existing_users = pd.DataFrame([dict(row.items()) for row in existing_users_rows])
    print(f"Existing users query returned {len(df_existing_users)} rows")

except Exception as e:
    print(f"Error loading/executing SQL: {str(e)}")
    raise


# [IN] Load user's enp users from BigQuery
try:
    enp_users_path = os.path.join(model_dir, 'enp_opt.sql')
    enp_users_query = load_sql(enp_users_path)

    if not enp_users_query or len(enp_users_query.strip()) == 0:
        raise ValueError(f"SQL file is empty: {enp_users_path}")
    print(f"Existing users SQL content length: {len(enp_users_query)}")

    enp_users_job = client.query(enp_users_query)
    enp_users_rows = enp_users_job.result()
    df_enp_users = pd.DataFrame([dict(row.items()) for row in enp_users_rows])
    print(f"Existing users query returned {len(df_enp_users)} rows")

except Exception as e:
    print(f"Error loading/executing SQL: {str(e)}")
    raise


# [IN] Load Models
try:
    model_path = os.path.join(model_dir, 'pot_vip_v2_opt.pkl')
    with open(model_path, 'rb') as file:
        old_window_models = pickle.load(file)

    window_models = {}
    for window, model in old_window_models.items():
        # Create new booster directly without scikit-learn wrapper
        booster = model.get_booster()
        features = booster.feature_names
        window_models[window] = {
            'booster': booster,
            'features': features
        }

    print("Models loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# [IN] Data Preprocessing
for column in df.columns:
    if column != 'firebase_user_id':
        df[column] = df[column].apply(lambda x: x.lower() if isinstance(x, str) else x)

print(df.head())

df['channel'] = df['channel'].replace(np.nan, 'other')
df['signup_email_domain'] = df['signup_email_domain'].replace(np.nan, 'other')
df['email'] = df['email'].replace(np.nan, 'other')

df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
df['signup_date'] = pd.to_datetime(df['signup_date'])
df['window_start'] = pd.to_datetime(df['window_start'])
df['window_end'] = pd.to_datetime(df['window_end'])

# [IN] Handle Missing Values
sum_cols = ['total_bet_amt', 'total_win_amt_sc', 'total_win_amt_cc',
            'sum_redeem_amt', 'sum_amt_total', 'cnt_txn_total',
            'sum_sc_coin_total', 'sum_bonus_wheel', 'sum_bonus_plinko',
            'sum_offer_amt', 'sum_bonus_amt', 'sum_daily_mission',
            'sum_leaderboard_reward', 'sum_spin_sc', 'sum_spin_cc',
            'logins', 'total_bet_txn', 'sum_bonus_promotion_bet_amount', 'sum_redeem_txn']

max_min_cols = ['max_purchase_amt_hourly', 'max_redeem_amt_hourly',
                'max_eod_sc_balance', 'max_daily_games_unique',
                'max_win_amt', 'max_bet_amt', 'min_bet_amount']

df[sum_cols + max_min_cols] = df[sum_cols + max_min_cols].fillna(0)

avg_cols = ['avg_spin_amt', 'rtp_ratio', 'avg_rounds_bet', 'avg_bet_amt', 'ggr_amt']
for col in avg_cols:
    df[f"{col}_is_missing"] = df[col].isna().astype(int)
    df[col] = df[col].fillna(0)

# [IN] Create Dummy Variables
email_domains = ['other', 'gmail.com', 'yahoo.com', 'privaterelay.appleid.com', 'icloud.com', 'hotmail.com']
channels = ['other', 'app', 'google', 'invite friend', 'facebook', 'sidelines']
platforms = ['web', 'app']

df_clean_flat = df.copy()
df_clean_flat = process_column(df_clean_flat, "signup_email_domain", email_domains)
df_clean_flat = process_column(df_clean_flat, "channel", channels)
df_clean_flat = process_column(df_clean_flat, "signup_platform", platforms)

# [IN] Transform Columns with Error Handling
columns_to_transform = [
    'sum_amt_total', 'total_bet_amt', 'total_win_amt_sc', 'total_win_amt_cc', 'max_win_amt',
    'sum_redeem_amt', 'max_redeem_amt_hourly', 'sum_sc_coin_total', 'max_eod_sc_balance',
    'sum_bonus_amt', 'sum_offer_amt', 'sum_bonus_promotion_bet_amount'
]

missing_cols = [col for col in columns_to_transform if col not in df_clean_flat.columns]
if missing_cols:
    raise ValueError(f"Missing columns for log transform: {missing_cols}")

try:
    df_clean_log = log_transform_columns(df_clean_flat, columns_to_transform)
    df_clean_log['window_number'] = df_clean_log['window_number'].astype('int')
    df_clean_log['ggr_amt_log'] = np.sign(df_clean_log['ggr_amt']) * np.log1p(np.abs(df_clean_log['ggr_amt']))
    df_clean = df_clean_log.dropna()
except Exception as e:
    print(f"Error during log transformation: {str(e)}")
    raise

# [IN] Create Window Dummies and Final Dataset
window_dummies = pd.get_dummies(df_clean['window_number'], prefix='window')
df_clean = df_clean.drop('window_number', axis=1)
df_model = pd.concat([df_clean, window_dummies], axis=1)
df_model.columns = df_model.columns.str.lower().str.replace(' ', '_')

# [IN] Load Features from File
try:
    with open(os.path.join(model_dir, 'model_features_v2_opt.txt'), 'r') as file:
        features = [line.strip() for line in file]
    if not features:
        raise ValueError("No features loaded from features file")
except Exception as e:
    print(f"Error loading features: {str(e)}")
    raise

# [IN] Prepare Final Model Dataset
windows = df_model.columns[df_model.columns.str.match('window_[0-4]')]
additional_cols = ['user_id', 'signup_date', 'first_purchase_date', 'email', 'full_name', 'firebase_user_id']
df_model_full = df_model[list(features) + list(windows) + additional_cols]

# [IN] Make Predictions
predictions_df = predict_vip_status_v2(df_model_full, window_models)

# [IN] Post-process Predictions
# Create explicit copy after merge
predictions_df_final = predictions_df.merge(
    df[['user_id', 'sum_amt_total', 'sum_redeem_amt', 'total_purchase_amount', 'cnt_txn_total']],
    how='inner',
    on='user_id'
).copy()

predictions_df_final['potential_vip_ind'] = predictions_df_final.apply(
    lambda x: 1 if x['total_purchase_amount'] >= 1500 else x['potential_vip_ind'], axis=1
)

# [IN] Filter VIP - Create explicit copy after filtering
predictions_df_final_vip = predictions_df_final.query("potential_vip_ind == 1").copy()

# Create explicit copy after filtering
df_new_data = predictions_df_final_vip[
    ~predictions_df_final_vip['user_id'].isin(df_existing_users['user_id'])
].copy()

##merge updated enp to split groups

# Replace the entire assign_vip_mentioned function with this simpler version
df_new_data_enp = df_new_data.merge(df_enp_users, on='user_id', how='left')
df_new_data_enp = df_new_data_enp.rename(
    columns={
        'sum_amt_total': 'total_purchase_amt',
        'sum_redeem_amt': 'total_redeem_amt',
        'window': 'window_class',
        'total_purchase_amount': 'total_purchase_amount_orig',
        'enp': 'split_metric'  # Rename to avoid confusion
    }
)

# Define a single function for second_phase assignment
def assign_second_phase(df):
    # Initialize second_phase column
    df['vip_mentioned'] = 0

    # Create the groups based on split_metric
    conditions = [
        (df['split_metric'] < 15),
        (df['split_metric'] >= 15) & (df['split_metric'] < 75),
        (df['split_metric'] >= 75) & (df['split_metric'] < 125),
        (df['split_metric'] >= 125)
    ]

    # Process each group separately
    for condition in conditions:
        # Get indices for this condition group
        group_indices = df[condition].index.tolist()

        if len(group_indices) == 0:
            continue

        # Calculate exactly half (using integer division for exact 50%)
        phase2_needed = len(group_indices) // 2

        if phase2_needed > 0:
            # Shuffle indices to ensure randomness
            np.random.shuffle(group_indices)

            # Take exactly half (or floor of half for odd counts)
            selected_ones = group_indices[:phase2_needed]

            # Assign second_phase = 1 to selected indices
            df.loc[selected_ones, 'vip_mentioned'] = 1

    return df

# Call the function ONCE
df_new_data_split = assign_second_phase(df_new_data_enp)


predictions_df_final = df_new_data_split[[
    'user_id', 'firebase_user_id', 'signup_date', 'first_purchase_date', 'window_class',
    'total_purchase_amt', 'split_metric', 'total_redeem_amt', 'email', 'full_name',
    'probability_positive_class', 'potential_vip_ind', 'vip_mentioned'
]].copy()

# Fix: Use list syntax for isin()
predictions_df_final = predictions_df_final[predictions_df_final['window_class'].isin([0])]

# Filter for same date signup and first purchase
predictions_df_final = predictions_df_final[predictions_df_final['signup_date'].dt.date == predictions_df_final['first_purchase_date'].dt.date]

# Upload new records
destination_table = 'ml_models.customers_opt'

# Format Final Output
predictions_df_final = predictions_df_final.assign(
    signup_date=lambda x: pd.to_datetime(x['signup_date']),
    first_purchase_date=lambda x: pd.to_datetime(x['first_purchase_date']),
    total_purchase_amt=lambda x: x['total_purchase_amt'].fillna(1).round().astype('int'),  # Handle NaN
    total_redeem_amt=lambda x: x['total_redeem_amt'].fillna(0).round().astype('int'),  # Handle NaN
    probability_positive_class=lambda x: x['probability_positive_class'].round(decimals=3),
    split_metric=lambda x: x['split_metric'].fillna(0).round(decimals=2),  # Keep as float, round to 2 decimal places
    version='v2',
    shadow_mode=False,
    updated_at=pd.Timestamp.now()
)

# Add single con_current value for all users in this run
if not predictions_df_final.empty:
    # Query to get the current maximum con_current value from the destination table
    try:
        max_query = f"""
        SELECT COALESCE(MAX(con_current), 1) as max_con_current 
        FROM {destination_table}
        """
        max_result = client.query(max_query).result()
        current_max = list(max_result)[0]['max_con_current']

        # All users in this run get the next number after current max
        next_con_current = current_max + 1

    except Exception as e:
        print(f"Error getting max con_current, starting from 4: {str(e)}")
        next_con_current = 4

    # Assign the same con_current value to all users in this run
    predictions_df_final['con_current'] = next_con_current

    print(f"All {len(predictions_df_final)} records will have con_current = {next_con_current}")

    # Upload to BigQuery
    predictions_df_final.to_gbq(destination_table,
                                project_id="crowncoins-casino",
                                if_exists='append',
                                credentials=None)
else:
    print("No data to upload - predictions_df_final is empty")