#!/usr/bin/env python
# coding: utf-8

# In[3]:


from google.oauth2 import id_token
import pandas as pd
import numpy as np
import db_dtypes
import matplotlib.pyplot as plt
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.cloud import bigquery
from IPython.core.display import display, HTML
import google.auth
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels.panel import PanelOLS, compare
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# Use the credentials to create a BigQuery client
project_id = "crowncoins-casino"
client = bigquery.Client(credentials=None, project=project_id)


# In[5]:


query = """ 

select * from guyb_sandbox.marketing_projection

"""

df_copy = client.query(query).to_dataframe()


# In[6]:


df_copy['su_week'] = pd.to_datetime(df_copy['su_week'])


# In[11]:


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

df_copy = process_and_group_data(df_copy)


# In[13]:


today = datetime.now().date()

# Get the most recent Monday (start of current week)
current_monday = today - timedelta(days=today.weekday())

# Get two Mondays back (2 weeks ago)
end_date = current_monday - timedelta(days=14)

# Convert to pandas datetime for filtering
cutoff_date = pd.to_datetime(end_date)

# Filter the dataframe
df_copy = df_copy[df_copy['su_week'] <= cutoff_date]


# In[15]:


channel_mapping_include = {
    
    'liftoff_int_app': 'liftoff_int_app', 
    'google_paid search': 'google_paid_search',
    'facebook_app' : 'facebook_app',
    'tiktok_app': 'tiktok_app',
    'facebook_paid social': 'facebook_paid_social',
    'apple_app': 'apple_app',
    'moloco_app': 'moloco_app',
    'tiktok_paid social': 'tiktok_paid_social',
    'snapchat_paid social': 'snapchat_paid_social',
    'applovin_int_app': 'applovin_int_app',
    'snapchat_app': 'snapchat_app',
    'bingsearch_paid search': 'bingsearch_paid_search',
    'smadex_int_app': 'smadex_int_app'
    
}


# In[17]:


import re

def get_group(channel_type):
    
    if channel_type in channel_mapping_include:
        return channel_mapping_include[channel_type]
    
    if (re.search(r'affiliate', channel_type, re.IGNORECASE) or 
        re.search(r'social traffic', channel_type, re.IGNORECASE) or 
        re.search(r'content creator', channel_type, re.IGNORECASE)):
        return 'affiliates'

    if re.search(r'invite friend', channel_type, re.IGNORECASE):
        return 'invite_friend'
    
    
    # Check if channel type contains any key from the mapping
    for key in channel_mapping_include:
        if key in channel_type:
            return channel_mapping_include[key]
    
    # Default case
    return 'other'

# Apply the mapping function to create the group column
df_copy['group'] = df_copy['channel_type_con'].apply(get_group)


# In[19]:


df_copy.loc[df_copy['total_costs'] == 0, 'traffic_type'] = 'organic'
df_copy.loc[df_copy['total_costs'] > 0, 'traffic_type'] = 'paid'


# In[21]:


df_copy.loc[df_copy['group'].str.contains('_app'), 'platform'] = 'app'

mixed_traffic_groups = df_copy.groupby(['su_week', 'group'])['traffic_type'].nunique().reset_index()
mixed_traffic_groups = mixed_traffic_groups[mixed_traffic_groups['traffic_type'] > 1]

mixed_groups_set = set(zip(mixed_traffic_groups['su_week'], mixed_traffic_groups['group']))

for idx, row in df_copy.iterrows():
    if (row['su_week'], row['group']) in mixed_groups_set and row['group'] != 'other':
        df_copy.at[idx, 'traffic_type'] = 'paid'


# In[23]:


df_copy_agg = df_copy.groupby(['su_week','group','platform','traffic_type'])[['total_costs',
       'total_users','total_paid', 'interval_7d',
       'interval_14d', 'interval_21d', 'interval_28d', 'interval_35d',
       'interval_42d', 'interval_49d', 'interval_56d', 'interval_63d',
       'interval_70d', 'interval_77d', 'interval_84d', 'interval_91d',
       'users_7d', 'users_14d', 'users_21d', 'users_28d', 'users_35d',
       'users_42d', 'users_49d', 'users_56d', 'users_63d', 'users_70d',
       'users_77d', 'users_84d', 'users_91d', 'users_cum_7d', 'users_cum_14d',
       'users_cum_21d', 'users_cum_28d', 'users_cum_35d', 'users_cum_42d',
       'users_cum_49d', 'users_cum_56d', 'users_cum_63d', 'users_cum_70d',
       'users_cum_77d', 'users_cum_84d', 'users_cum_91d']].sum().reset_index()


# In[25]:


cum_users = [c for c in df_copy_agg.columns if 'users_cum' in c]

gross_interval = [c for c in df_copy_agg.columns if 'interval' in c]

df_ret = df_copy_agg[['group', 'su_week','platform','traffic_type', 'total_costs'] + gross_interval]

df_users_count = df_copy_agg[['group', 'su_week','platform','traffic_type', 'total_costs'] + cum_users]


# In[27]:


def filter_columns_by_date_difference(df):
    # Create a copy to avoid modifying the original DataFrame
    result_df = df.copy()
    
    # Convert su_week to datetime if it's not already
    result_df['su_week'] = pd.to_datetime(result_df['su_week'])
    
    # Get the most recent week (max) in the dataset
    max_week = result_df['su_week'].max()
    
    # Get value columns (excluding 'group', 'su_week', 'platform', and 'traffic_type')
    value_columns = [col for col in result_df.columns 
                    if col not in ['group', 'su_week', 'platform', 'traffic_type']]
    
    # Process each row
    for idx, row in result_df.iterrows():
        # Calculate week difference from the max week
        week_diff = int((max_week - row['su_week']).days / 7)
        
        keep_cols = 2 + week_diff
        
        # Ensure we don't try to keep more columns than exist
        keep_cols = min(keep_cols, len(value_columns))
        
        # Set remaining columns to NaN
        if keep_cols < len(value_columns):
            result_df.loc[idx, value_columns[keep_cols:]] = np.nan
    
    return result_df


# In[29]:


df_ret_modified = filter_columns_by_date_difference(df_ret)
df_users_count_modified = filter_columns_by_date_difference(df_users_count)


# In[31]:


df_ret_drop = df_ret_modified.sort_values(by=['group','su_week'], ascending=True)
df_users_drop = df_users_count_modified.sort_values(by=['group','su_week'], ascending=True)


# In[33]:


selected_columns = df_ret_drop.iloc[:, 5:]

pct_change_df = selected_columns.pct_change(axis=1) 

original_selected = df_ret_drop[['group', 'su_week']]

df_combined = pd.concat([original_selected, pct_change_df], axis=1).replace(0,np.nan)


# In[35]:


def filter_intervals_by_user_count_complete(df_trimmed, df_users_count_modified, threshold=50):
    result_df = df_trimmed.copy()
    
    # Define all interval and corresponding users_cum columns
    intervals = ['7d', '14d', '21d', '28d', '35d', '42d', '49d', 
                 '56d', '63d', '70d', '77d', '84d', '91d']
    
    # List to keep track of omitted cells
    omitted_cells = []
    
    # Process each row in df_trimmed
    for idx_trimmed, row_trimmed in result_df.iterrows():
        group = row_trimmed['group']
        su_week = row_trimmed['su_week']
        
        # Find matching rows in df_users_count_modified
        matching_rows = df_users_count_modified[
            (df_users_count_modified['group'] == group) & 
            (df_users_count_modified['su_week'] == su_week)
        ]
        
        # Skip if no matching rows found
        if matching_rows.empty:
            continue
        
        # Check each interval
        for interval in intervals:
            interval_col = f'interval_{interval}'
            users_cum_col = f'users_cum_{interval}'
            
            # Skip if either column doesn't exist
            if interval_col not in result_df.columns or users_cum_col not in matching_rows.columns:
                continue
                
            # Check if any matching row passes the threshold check
            if not (matching_rows[users_cum_col] > threshold).any():
                # If no row passes the threshold, set the value to NaN
                result_df.loc[idx_trimmed, interval_col] = np.nan
                
                # Add the omitted cell info to our list
                omitted_cells.append({
                    'group': group,
                    'su_week': su_week,
                    'interval': interval,
                    'original_value': row_trimmed[interval_col],
                    'users_count': matching_rows[users_cum_col].max() if not matching_rows.empty else None
                })
    
    return result_df, omitted_cells

df_combined_enc, omitted_cells = filter_intervals_by_user_count_complete(df_combined, df_users_count_modified, threshold=50)


# In[36]:


def trim_group_outliers_moderate(df, group_col='group', k=1.0):

    df_trimmed = df.copy()
    
    # Get only the interval columns for outlier detection
    interval_cols = [col for col in df.columns if col.startswith('interval_')]
    
    # Dictionary to store outlier statistics
    outlier_stats = {}
    
    for group_name, group_df in df.groupby(group_col):
        group_stats = {}
        
        for col in interval_cols:
            # Skip columns with too few non-NaN values
            non_na_values = group_df[col].dropna()
            if len(non_na_values) < 4:  # Need at least a few values for meaningful quartiles
                continue
                
            # Calculate quantiles, ignoring NaN values
            Q1 = non_na_values.quantile(0.25)
            Q3 = non_na_values.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - k * IQR
            upper_bound = Q3 + k * IQR
            
            # Create masks for the current group
            group_mask = (df_trimmed[group_col] == group_name)
            below_mask = (df_trimmed[col] < lower_bound) & group_mask & ~df_trimmed[col].isna()
            above_mask = (df_trimmed[col] > upper_bound) & group_mask & ~df_trimmed[col].isna()
            
            # Count outliers before adjustment
            below_count = below_mask.sum()
            above_count = above_mask.sum()
            
            # Record original values before trimming (for reference)
            below_original = df_trimmed.loc[below_mask, col].tolist() if below_count > 0 else []
            above_original = df_trimmed.loc[above_mask, col].tolist() if above_count > 0 else []
            
            # Apply trimming
            df_trimmed.loc[below_mask, col] = lower_bound
            df_trimmed.loc[above_mask, col] = upper_bound
            
            # Store statistics for this column
            if below_count > 0 or above_count > 0:
                group_stats[col] = {
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'below_count': below_count,
                    'above_count': above_count,
                    'total_outliers': below_count + above_count,
                    'below_original': below_original,
                    'above_original': above_original
                }
        
        if group_stats:
            outlier_stats[group_name] = group_stats
    
    return df_trimmed, outlier_stats


# In[37]:


df_trimmed, outlier_stats = trim_group_outliers_moderate(df_combined_enc, group_col='group', k=1.0)


# In[41]:


def fill_nan_with_group_mean(df):
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Define our channel groups
    channel_groups = {
        'network': ['applovin_int_app', 'liftoff_int_app', 'smadex_int_app', 'moloco_app'],
        'paid_search': ['google_paid_search', 'bingsearch_paid_search'],
        'paid_social': ['tiktok_paid_social', 'snapchat_paid_social'],
        'social_app': ['tiktok_app', 'snapchat_app', 'facebook_app']
    }
    
    channel_to_group_type = {}
    for group_type, channels in channel_groups.items():
        for channel in channels:
            channel_to_group_type[channel] = group_type
    
    # Find all interval columns
    interval_columns = [col for col in df.columns if col.startswith('interval_')]
    print(f"Found {len(interval_columns)} interval columns: {interval_columns}")
    
    # Create a new column for group type
    result_df['group_type'] = result_df['group'].map(channel_to_group_type)
    
    # Count NaN values before filling
    total_nans_before = result_df[interval_columns].isna().sum().sum()
    print(f"Total NaN values before filling: {total_nans_before}")
    
    # Step 1: Calculate means for each group_type and su_week for each interval
    group_means = {}
    for group_type in channel_groups.keys():
        group_means[group_type] = {}
        
        # Filter dataframe for this group type
        group_df = result_df[result_df['group_type'] == group_type]
        
        # Calculate means for each su_week and interval column
        for su_week in group_df['su_week'].unique():
            group_means[group_type][su_week] = {}
            
            # Filter for this su_week
            week_df = group_df[group_df['su_week'] == su_week]
            
            # Calculate mean for each interval column
            for col in interval_columns:
                mean_value = week_df[col].dropna().mean()
                group_means[group_type][su_week][col] = mean_value
    
    # Step 1b: Calculate overall means for each su_week as fallback
    overall_means = {}
    for su_week in result_df['su_week'].unique():
        overall_means[su_week] = {}
        week_df = result_df[result_df['su_week'] == su_week]
        
        for col in interval_columns:
            overall_means[su_week][col] = week_df[col].dropna().mean()
    
    # Step 2: Fill in NaN values
    filled_count = 0
    filled_with_group_mean = 0
    filled_with_overall_mean = 0
    
    for idx, row in result_df.iterrows():
        # Skip if the channel doesn't belong to a known group
        if row['group'] not in channel_to_group_type:
            continue
            
        group_type = channel_to_group_type[row['group']]
        su_week = row['su_week']
        
        # Check each interval column
        for col in interval_columns:
            # Only process if current value is NaN
            if pd.isna(row[col]):
                # Try to use group type mean first
                group_mean_available = (
                    group_type in group_means and 
                    su_week in group_means[group_type] and 
                    col in group_means[group_type][su_week] and 
                    not pd.isna(group_means[group_type][su_week][col])
                )
                
                if group_mean_available:
                    # Fill with group mean
                    mean_value = group_means[group_type][su_week][col]
                    result_df.loc[idx, col] = mean_value
                    filled_count += 1
                    filled_with_group_mean += 1
                
                # If no group mean available, try overall mean for this su_week
                elif (su_week in overall_means and 
                      col in overall_means[su_week] and 
                      not pd.isna(overall_means[su_week][col])):
                    
                    mean_value = overall_means[su_week][col]
                    result_df.loc[idx, col] = mean_value
                    filled_count += 1
                    filled_with_overall_mean += 1
    
    # Drop the temporary group_type column
    result_df = result_df.drop('group_type', axis=1)
    
    # Count NaN values after filling
    total_nans_after = result_df[interval_columns].isna().sum().sum()
    print(f"Total NaN values before: {total_nans_before}, after: {total_nans_after}")
    print(f"Filled {filled_count} NaN values total")
    print(f" - {filled_with_group_mean} filled with group-specific means")
    print(f" - {filled_with_overall_mean} filled with overall means")
    print(f"Remaining NaN values: {total_nans_after}")
    
    return result_df

df_filled = fill_nan_with_group_mean(df_trimmed)


# In[43]:


def calculate_weighted_group_averages_simple(df, group_column):
    results = {}
    
    # Dictionary mapping interval patterns to number of values to use
    interval_rules = {
        'interval_14d': 9, 'interval_21d': 8,  
        'interval_28d': 7, 'interval_35d': 6,  
        'interval_42d': 5, 'interval_49d': 5,    
        'interval_56d': 4, 'interval_63d': 4    
    }
    
    # Group by the specified column
    for group_name, group_data in df.groupby(group_column):
        group_results = {}
        
        # Get all interval columns
        interval_columns = [col for col in df.columns if col.startswith('interval_')]
        
        # Process each interval column
        for col in interval_columns:
            # Get all non-NaN values for this interval in this group
            values = group_data[col].dropna().values
            
            if len(values) == 0:
                group_results[col] = np.nan
                continue
                
            # Determine how many values to use
            if col in interval_rules:
                n_values = min(interval_rules[col], len(values))
            else:
                # For intervals beyond 63d, use last 3 values
                n_values = min(3, len(values))
                
            # Take the last n_values
            values_to_use = values[-n_values:]
            
            # Create exponential weights (more weight to recent data)
            weights = np.exp(np.linspace(0, 1, n_values))
            weights = weights / weights.sum()
            
            # Calculate weighted average
            weighted_avg = np.sum(values_to_use * weights)
            
            group_results[col] = weighted_avg
            
        results[group_name] = group_results
    
    # Convert results to DataFrame
    return pd.DataFrame.from_dict(results, orient='index')


# In[45]:


df_combined_mean = calculate_weighted_group_averages_simple(df_filled, 'group')

df_combined_mean.reset_index(inplace=True)

df_ret_drop_enc = df_ret_drop.drop(['su_week','total_costs','platform','traffic_type'], axis=1)


# In[47]:


def adjust_row_values(df):
    # Create a copy to avoid modifying the original
    df_adjusted = df.copy()
    
    # Get all columns except 'index'
    value_columns = df_adjusted.columns.drop('index')
    
    # Iterate through each row
    for idx in df_adjusted.index:
        # Iterate through columns except the last one
        # Start from the 3rd column (index 2 if zero-indexed)
        for i in range(2, len(value_columns) - 1):
            current_col = value_columns[i]
            next_col = value_columns[i + 1]
            
            # If next value is greater than current value, replace next value
            if df_adjusted.loc[idx, next_col] > df_adjusted.loc[idx, current_col]:
                df_adjusted.loc[idx, next_col] = df_adjusted.loc[idx, current_col]
    
    return df_adjusted

df_combined_mean_adjusted = adjust_row_values(df_combined_mean)


# In[49]:


# Calculate mean for each numeric column
numeric_cols = df_combined_mean_adjusted.select_dtypes(include=[np.number]).columns
means = df_combined_mean_adjusted[numeric_cols].mean()

# Create the new row
new_row = pd.DataFrame({
    'group': ['mean_overall'],
    **{col: [means[col]] for col in numeric_cols}
})

# Append to the original dataframe
df_combined_mean_adjusted_enc = pd.concat([df_combined_mean_adjusted, new_row], ignore_index=True)


# In[51]:


# Create a copy of the dataframe
df_sec = df_combined_mean_adjusted_enc.copy()

# Get numeric columns
numeric_cols = df_sec.select_dtypes(include=[np.number]).columns

# For each numeric column
for col in numeric_cols:
    # Create a mask for values < 0.03, excluding NaN values
    low_values_mask = (df_sec[col] < 0.03) & (~df_sec[col].isna())
    
    if low_values_mask.any():
        # Calculate mean of other values (not < 0.03 and not NaN) in the column
        valid_values_mask = (~df_sec[col].isna()) & (~low_values_mask)
        column_mean = df_sec.loc[valid_values_mask, col].mean()
        
        # Replace only values < 0.03 with the mean (keeping NaN as NaN)
        df_sec.loc[low_values_mask, col] = column_mean

# Update the original dataframe
df_combined_mean_adjusted_enc = df_sec


# In[53]:


df_combined_mean_adjusted_enc_min_pct = df_combined_mean_adjusted_enc.applymap(lambda x: x + 1 if pd.api.types.is_numeric_dtype(type(x)) else x)


# In[55]:


all_columns = df_combined_mean_adjusted_enc_min_pct.columns
# Keep first two columns unchanged
first_two_cols = all_columns[:3]
columns_to_fill = all_columns[3:]

# Fill NaN values only in columns from third onwards
for col in columns_to_fill:
    # Check if the column has a numeric dtype first
    if pd.api.types.is_numeric_dtype(df_combined_mean_adjusted_enc_min_pct[col].dtype):
        # Calculate mean and fill NaN values only for numeric columns
        col_mean = df_combined_mean_adjusted_enc_min_pct[col].mean()
        df_combined_mean_adjusted_enc_min_pct[col] = df_combined_mean_adjusted_enc_min_pct[col].fillna(col_mean)
    else:
        pass


# In[57]:


df_combined_mean_adjusted_enc_min_pct = df_combined_mean_adjusted_enc_min_pct.drop('group', axis=1).rename(columns= {'index': 'group'})


# In[59]:


def ensure_non_increasing_intervals(df):
   # Get interval columns (3rd onwards)
   interval_columns = df.columns[2:]
   
   # Process each pair of adjacent columns
   for i in range(len(interval_columns) - 1):
       current_col = interval_columns[i]
       next_col = interval_columns[i + 1]
       
       # Where next value > current value, replace with current value
       mask = df[next_col] > df[current_col]
       df.loc[mask, next_col] = df.loc[mask, current_col]
   
   return df

df_combined_mean_adjusted_enc_min_pct = ensure_non_increasing_intervals(df_combined_mean_adjusted_enc_min_pct)


# In[63]:


def fill_nan_with_group_multipliers(df_ret_drop_enc, df_combined_mean_adjusted_enc_min_pct):
    # Create a copy to avoid modifying the original
    df_third = df_ret_drop_enc.copy()
    
    # Get value columns (excluding 'group')
    value_cols = [col for col in df_third.columns if col != 'group']
    
    # Find the row where group is NaN to use as fallback
    fallback_mask = pd.isna(df_combined_mean_adjusted_enc_min_pct['group'])
    
    # If we found a NaN group, use it; otherwise use the last row
    if fallback_mask.any():
        fallback_multipliers = df_combined_mean_adjusted_enc_min_pct[fallback_mask].iloc[0]
        print("Found fallback row with NaN group")
    else:
        fallback_multipliers = df_combined_mean_adjusted_enc_min_pct.iloc[-1]
        print("Using last row as fallback")
    
    # Process each group separately
    for group_name in df_third['group'].unique():
        # Find rows where group matches
        matching_mask = df_combined_mean_adjusted_enc_min_pct['group'] == group_name
        
        # Use group multipliers if available, otherwise use fallback
        if not matching_mask.any():
            print(f"Using fallback for group {group_name}")
            group_multipliers = fallback_multipliers
        else:
            group_multipliers = df_combined_mean_adjusted_enc_min_pct[matching_mask].iloc[0]
            print(f"Found group: {group_name}")
        
        # Get rows for this group in df_third
        group_mask = df_third['group'] == group_name
        
        # Process each row in the group
        for idx in df_third[group_mask].index:
            last_valid_value = None
            
            # Process each column in order
            for col in value_cols:
                # Skip if column doesn't exist in the reference DataFrame
                if col not in df_combined_mean_adjusted_enc_min_pct.columns:
                    continue
                
                current_value = df_third.loc[idx, col]
                
                if pd.isna(current_value):
                    if last_valid_value is not None:
                        try:
                            # Get multiplier
                            multiplier = group_multipliers[col]
                            
                            # Fill NaN with last_valid_value * multiplier
                            df_third.loc[idx, col] = last_valid_value * multiplier
                            
                            # Update last_valid_value for next iteration
                            last_valid_value = df_third.loc[idx, col]
                        except (KeyError, ValueError) as e:
                            print(f"Error processing column {col} for group {group_name}: {e}")
                else:
                    # Update last_valid_value with current non-NaN value
                    last_valid_value = current_value
    
    return df_third

filled_df = fill_nan_with_group_multipliers(df_ret_drop_enc, df_combined_mean_adjusted_enc_min_pct)


# In[65]:


filled_new_df = df_ret_drop[['su_week', 'total_costs','platform','traffic_type']].merge(filled_df, 
                                                             left_index=True, 
                                                              right_index=True)

filled_new_df_loc = filled_new_df.query("interval_7d != 0").iloc[:,:-1]

df_final_value = filled_new_df_loc.dropna()


# In[67]:


# Merge with the specific columns you need
final_df_cal = df_final_value[['su_week','group','total_costs','platform','traffic_type']].merge(
    df_combined_mean_adjusted_enc_min_pct, 
    on='group', 
    how='left'
)

# Drop any unnecessary columns
final_df_cal_drop = final_df_cal.drop('interval_7d', axis=1)


# In[69]:


def calculate_interval_targets(df, start_interval=63, end_interval=7, step=7):
    result_df = df.copy()
    
    for current_interval in range(start_interval-7, end_interval-7, -step):
        target_col = f'target_{current_interval}d'
        denominator = 1
        
        # Calculate cumulative product of all larger intervals
        for interval in range(start_interval, current_interval, -step):
            interval_col = f'interval_{interval}d'
            denominator *= result_df[interval_col]
            
        result_df[target_col] = result_df['total_costs'] / denominator
        
    return result_df

final_df_cal_drop = calculate_interval_targets(final_df_cal_drop)


# In[71]:


target_columns = [c for c in final_df_cal_drop.columns if c.startswith('target')]
selected_columns = ['group', 'su_week','platform', 'traffic_type','total_costs'] + target_columns
final_df_cal_drop_enc = final_df_cal_drop[selected_columns].sort_values(by=['group','su_week','platform','traffic_type'])
final_df_cal_drop_enc.iloc[:, 5:] = final_df_cal_drop_enc.iloc[:, 5:].round().astype('int')


# In[73]:


df_value_actual = df_final_value.copy()
df_value_actual.iloc[:, 5:] = df_value_actual.iloc[:, 5:].round().astype('int')

df_value_actual = df_value_actual.sort_values(by=['group','su_week','platform','traffic_type'])


# In[75]:


df_target = final_df_cal_drop_enc[['group','su_week','platform','traffic_type','total_costs','target_7d','target_14d','target_21d','target_28d','target_35d',
                                   'target_42d','target_49d','target_56d']]

df_pred = df_value_actual[['group','su_week','platform','traffic_type', 'total_costs', 'interval_7d', 'interval_14d', 'interval_21d', 'interval_28d', 'interval_35d',
                             'interval_42d', 'interval_49d', 'interval_56d', 'interval_63d', 'interval_70d', 'interval_77d',
                             'interval_84d']]


df_merged_comp = df_target.merge(df_pred, on =['group','su_week','platform','traffic_type','total_costs'], how='left')


# In[77]:


renamed_columns = {col: col.replace('interval', 'prediction') for col in df_merged_comp.columns if col.startswith('interval')}
df_merged_comp = df_merged_comp.rename(columns=renamed_columns)


# In[79]:


from datetime import datetime, timedelta

def group_weeks(df, date_column='su_week'):
    # Convert dates to datetime if they're not already
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Sort by date
    df = df.sort_values(by=date_column)
    
    # Get unique weeks and sort them
    unique_weeks = np.sort(df[date_column].unique())
    
    # Create a mapping dictionary
    group_mapping = {}
    for i in range(0, len(unique_weeks), 4):
        group_name = unique_weeks[i]  # Format as dd/mm
        # Get up to next 4 weeks (or remaining weeks if less than 4)
        weeks_in_group = unique_weeks[i:i+4]
        for week in weeks_in_group:
            group_mapping[week] = group_name
    
    # Add the group column to the DataFrame
    df['month_group'] = df[date_column].map(group_mapping)
    
    return df

df_deploy = group_weeks(df_merged_comp)


# In[81]:


cols = df_deploy.columns.tolist()

cols.remove('month_group')

# Insert it at the third position (index 2)
cols.insert(2, 'month_group')

# Reorder the dataframe
df_deploy = df_deploy[cols]


# In[83]:


df_deploy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')

df_deploy = df_deploy.rename(columns={'group' : 'channel_group'})


# In[85]:


# Set your destination table
destination_table = 'ml_models.marketing_projection'

# Write the DataFrame to BigQuery
df_deploy.to_gbq(destination_table, project_id=project_id, if_exists='replace', credentials=None)


# In[ ]:




