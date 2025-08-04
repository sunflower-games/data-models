#!/usr/bin/env python
# coding: utf-8

# Standard libraries
from datetime import datetime

# Data processing and analysis
import numpy as np
import pandas as pd

# Machine Learning
import xgboost as xgb
from xgboost import XGBClassifier

# Database
from google.cloud import bigquery
from google_auth_oauthlib.flow import InstalledAppFlow

# Statistical analysis (if you're using these functions)
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, compare

def load_sql(filename):
    """
    Read and return contents of SQL file
    """
    with open(filename, 'r') as file:
        return file.read()


def process_column(df, column_name, values_list):
    """
    Process categorical columns and create dummy variables with consistent naming
    """
    df = df.copy()

    # Convert column values to lowercase but preserve spaces
    if df[column_name].dtype == 'object':
        df[column_name] = df[column_name].str.lower()

    # Create dummy columns preserving original value formatting
    for value in values_list:
        col_name = f'{column_name}_{value.lower()}'  # Preserve spaces in column names
        df[col_name] = (df[column_name] == value.lower()).astype(int)

    # Create 'other' category
    other_col = f'{column_name}_other'
    df[other_col] = 1
    for value in values_list:
        col_name = f'{column_name}_{value.lower()}'
        df.loc[df[col_name] == 1, other_col] = 0

    return df

def log_transform_columns(df, columns):
    """
    Apply log transformation to specified columns
    """
    df = df.copy()
    epsilon = 1e-10  # Small constant to handle zeros

    for column in columns:
        # Create new column with log transform
        df[f'ln_{column}'] = np.log(df[column] + epsilon)
        # Drop original column
        df = df.drop(column, axis=1)

    return df

def filter_by_time_limit(df, time_limits):
    """
    Filter dataframe by time limits from signup date
    """
    filtered_dfs = {}

    for hours in time_limits:
        time_limit = pd.Timedelta(hours=hours)
        mask = ((df['redeem_time'] - df['signup_date'] <= time_limit) |
                (df['transaction_date_time'] - df['signup_date'] <= time_limit))
        filtered_dfs[f'{hours}h'] = df[mask]

    return filtered_dfs

def trim_outliers_quantile(df, feature_list, lower_quantile=0.01, upper_quantile=0.99):
    """
    Trim outliers based on quantiles
    """
    df_trimmed = df.copy()

    for feature in feature_list:
        lower_bound = df[feature].quantile(lower_quantile)
        upper_bound = df[feature].quantile(upper_quantile)
        df_trimmed[feature] = df_trimmed[feature].clip(lower=lower_bound, upper=upper_bound)

    return df_trimmed

def ensure_required_columns(df, required_columns):
    """
    Ensure all required columns exist in DataFrame, adding missing ones with 0s
    """
    for col in required_columns:
        if col not in df.columns:
            df[col] = 0
    return df


def process_and_group_data(df):
    # Create a copy
    df_copy = df.copy()

    # Find channel_type_con values with 2 distinct traffic_types
    channel_counts = df_copy.groupby('channel_type_con')['traffic_type'].nunique()
    channels_to_modify = channel_counts[channel_counts == 2].index

    # Set traffic_type to 'paid' for identified channels
    df_copy.loc[df_copy['channel_type_con'].isin(channels_to_modify), 'traffic_type'] = 'paid'

    # Group by specified columns and sum the rest
    grouped_df = df_copy.groupby([
        'su_week',
        'channel_type_con',
        'platform',
        'traffic_type'
    ]).sum().reset_index()

    return grouped_df

def filter_columns_by_date_difference(df):
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()

    # Get current date dynamically
    current_date = pd.Timestamp(datetime.now().date())

    # Convert su_week to datetime if it's not already
    result_df['su_week'] = pd.to_datetime(result_df['su_week'])

    # Get value columns (excluding 'group' and 'su_week')
    value_columns = [col for col in result_df.columns
                     if col not in ['group', 'su_week', 'platform', 'traffic_type']]

    # Process each row
    for idx, row in result_df.iterrows():
        # Calculate week difference
        week_diff = int((current_date - row['su_week']).days / 7)

        # Determine how many columns to keep
        if week_diff < 3:
            keep_cols = 2  # Keep 2 columns for recent weeks
        elif 3 <= week_diff < 4:
            keep_cols = 3  # Keep 3 columns for weeks 3-4 weeks ago
        else:
            keep_cols = week_diff  # Keep week_diff columns for older weeks (removed +1)

        # Set remaining columns to NaN
        if keep_cols < len(value_columns):
            result_df.loc[idx, value_columns[keep_cols:]] = np.nan

    return result_df


def trim_group_outliers(df, group_col, k=1.5):
    # Make a copy of the DataFrame
    df_trimmed = df.copy()

    # Get numeric columns excluding the group column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols[numeric_cols != group_col]

    # Dictionary to store outlier statistics
    outlier_stats = {}

    # Process each group
    for group_name, group_df in df.groupby(group_col):
        outlier_stats[group_name] = {}

        for col in numeric_cols:
            # Calculate Q1, Q3, and IQR for the current column in current group
            Q1 = group_df[col].quantile(0.25)
            Q3 = group_df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Calculate bounds
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR

            # Count outliers
            below_lower = group_df[col] < lower_bound
            above_upper = group_df[col] > upper_bound
            outliers_count = below_lower.sum() + above_upper.sum()

            # Trim outliers to the bounds
            df_trimmed.loc[below_lower & (df_trimmed[group_col] == group_name), col] = lower_bound
            df_trimmed.loc[above_upper & (df_trimmed[group_col] == group_name), col] = upper_bound

            # Store statistics
            outlier_stats[group_name][col] = {
                'total_values': len(group_df),
                'outliers_trimmed': outliers_count,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }

    return df_trimmed, outlier_stats


def adjust_row_values(df):
    # Create a copy to avoid modifying the original
    df_adjusted = df.copy()

    # Get all columns except 'group'
    value_columns = df_adjusted.columns.drop('group')

    # Iterate through each row
    for idx in df_adjusted.index:
        # Iterate through columns except the last one
        for i in range(len(value_columns) - 1):
            current_col = value_columns[i]
            next_col = value_columns[i + 1]

            # If current value is less than next value, replace it
            if df_adjusted.loc[idx, current_col] < df_adjusted.loc[idx, next_col]:
                df_adjusted.loc[idx, current_col] = df_adjusted.loc[idx, next_col]

    return df_adjusted


def fill_nan_with_group_multipliers(df_ret_drop_enc, df_combined_mean_adjusted_enc_min_pct):
    # Create a copy to avoid modifying the original
    df_third = df_ret_drop_enc.copy()

    # Get value columns (excluding 'group')
    value_cols = [col for col in df_third.columns if col != 'group']

    # Get mean_overall multipliers
    mean_overall_multipliers = \
    df_combined_mean_adjusted_enc_min_pct[df_combined_mean_adjusted_enc_min_pct['group'] == 'mean_overall'].iloc[0]

    # Process each group separately
    for group_name in df_third['group'].unique():
        # Check if group exists in df_combined_mean_adjusted
        group_data = df_combined_mean_adjusted_enc_min_pct[df_combined_mean_adjusted_enc_min_pct['group'] == group_name]

        # Use group multipliers if available, otherwise use mean_overall
        if group_data.empty:
            print(f"Using mean_overall for group {group_name}")
            group_multipliers = mean_overall_multipliers
        else:
            group_multipliers = group_data.iloc[0]

        # Get rows for this group
        group_mask = df_third['group'] == group_name

        # Process each row in the group
        for idx in df_third[group_mask].index:
            last_valid_value = None

            # Process each column in order
            for col in value_cols:
                if col not in group_multipliers:
                    continue

                current_value = df_third.loc[idx, col]

                if pd.isna(current_value):
                    if last_valid_value is not None:
                        try:
                            # Get multiplier (either from group or mean_overall)
                            multiplier = group_multipliers[col]
                            # Fill NaN with last_valid_value * multiplier
                            df_third.loc[idx, col] = last_valid_value * multiplier
                            # Update last_valid_value to use this new value for next iteration
                            last_valid_value = df_third.loc[idx, col]
                        except (KeyError, ValueError) as e:
                            print(f"Error processing column {col} for group {group_name}: {e}")
                            continue
                else:
                    last_valid_value = current_value

    return df_third


def calculate_interval_targets(df, start_interval=63, end_interval=7, step=7):
    result_df = df.copy()

    for current_interval in range(start_interval - 7, end_interval - 7, -step):
        target_col = f'target_{current_interval}d'
        denominator = 1

        # Calculate cumulative product of all larger intervals
        for interval in range(start_interval, current_interval, -step):
            interval_col = f'interval_{interval}d'
            denominator *= result_df[interval_col]

        result_df[target_col] = result_df['total_costs'] / denominator

    return result_df


def predict_vip_status_v2(df_model_full, window_models, decision_threshold=0.98):
    """
    Make predictions using the sequential window models and return detailed user information
    """
    predictions = []
    for idx, row in df_model_full.iterrows():
        # Determine which window the user is in
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
            
        # Get the model for this window
        model = window_models[current_window]
        
        # Prepare features in correct order
        features = model.get_booster().feature_names
        X = pd.DataFrame([row[features]])
        
        # Make prediction
        prob = model.predict_proba(X)[:, 1][0]
        prediction = int(prob >= decision_threshold)
        
        # Create prediction data dictionary
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
    
    # Convert to DataFrame and set column order
    predictions_df = pd.DataFrame(predictions)
    
    # Define column order
    column_order = [
        'user_id', 'firebase_user_id', 'signup_date', 'first_purchase_date', 
        'window', 'email', 'full_name','probability_positive_class', 'potential_vip_ind'
    ]
    
    return predictions_df[column_order]



