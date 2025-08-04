#!/usr/bin/env python
# coding: utf-8

# In[29]:


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

# In[31]:


query = """ 

select * from guyb_sandbox.marketing_projection

"""

df = client.query(query).to_dataframe()

# In[33]:


df_copy = df.copy()

df_copy['su_week'] = pd.to_datetime(df_copy['su_week'])


# In[34]:
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

# In[34]:

# Get today's date
today = datetime.now().date()

# Get the most recent Monday (start of current week)
current_monday = today - timedelta(days=today.weekday())

# Get two Mondays back (2 weeks ago)
end_date = current_monday - timedelta(days=14)

# Convert to pandas datetime for filtering
cutoff_date = pd.to_datetime(end_date)

# Filter the dataframe
df_copy = df_copy[df_copy['su_week'] <= cutoff_date]




# In[40]:

# Create a mapping dictionary for channel types to groups
channel_mapping = {

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
    'smadex_int_app': 'smadex_int_app',
    'ironsource_int_app' : 'ironsource_int_app',
    'unityads_int_app' : 'unityads_int_app'
}


# In[41]:
import re


def get_group(channel_type):
    if channel_type in channel_mapping:
        return channel_mapping[channel_type]

    if (re.search(r'affiliate', channel_type, re.IGNORECASE) or
            re.search(r'social traffic', channel_type, re.IGNORECASE) or
            re.search(r'content creator', channel_type, re.IGNORECASE)):
        return 'affiliates'

    if re.search(r'invite friend', channel_type, re.IGNORECASE):
        return 'invite_friend'

    # Check if channel type contains any key from the mapping
    for key in channel_mapping:
        if key in channel_type:
            return channel_mapping[key]

    # Default case
    return 'other'


# Apply the mapping function to create the group column
df_copy['group'] = df_copy['channel_type_con'].apply(get_group)

df_copy.loc[df_copy['total_costs'] == 0, 'traffic_type'] = 'organic'
df_copy.loc[df_copy['total_costs'] > 0, 'traffic_type'] = 'paid'


# In[43]:
df_copy.loc[df_copy['group'].str.contains('_app'), 'platform'] = 'app'

mixed_traffic_groups = df_copy.groupby(['su_week', 'group'])['traffic_type'].nunique().reset_index()
mixed_traffic_groups = mixed_traffic_groups[mixed_traffic_groups['traffic_type'] > 1]

mixed_groups_set = set(zip(mixed_traffic_groups['su_week'], mixed_traffic_groups['group']))

for idx, row in df_copy.iterrows():
    if (row['su_week'], row['group']) in mixed_groups_set and row['group'] != 'other':
        df_copy.at[idx, 'traffic_type'] = 'paid'


# In[44]:
gross_interval = [c for c in df_copy.columns if 'interval' in c]

df_copy['su_week'] = pd.to_datetime(df_copy['su_week'])

df_min = df_copy[df_copy['total_users'] > df_copy['total_users'].mean()]

group_list = df_min['group'].value_counts().head(21).index

df_min_mean = df_min[df_min['group'].isin(group_list)]

df_min_mean_enc = df_min_mean[['group', 'su_week', 'platform', 'traffic_type', 'total_costs'] + gross_interval]

df_ret = df_copy[['group', 'su_week', 'platform', 'traffic_type', 'total_costs'] + gross_interval]

# In[45]:


# For sum (your original data)
df_min_mean_enc_gr = df_min_mean_enc.groupby(['group', 'su_week', 'platform', 'traffic_type'], as_index=False).agg(
    lambda x: x.sum(min_count=1))

df_ret_gr = df_ret.groupby(['group', 'su_week', 'platform', 'traffic_type'], as_index=False).agg(
    lambda x: x.sum(skipna=False))

# In[46]:

cols_to_process = df_ret_gr.columns[6:]

# Process each row in the dataframe
for idx, row in df_ret_gr.iterrows():
    # Get values from columns we're processing
    values = row[cols_to_process].values

    # Track seen values for this row
    seen_values = set()

    # Process each column in this row
    for i, col in enumerate(cols_to_process):
        current_value = row[col]

        # If value is already seen or NaN, replace with NaN
        if pd.isna(current_value) or current_value in seen_values:
            df_ret_gr.at[idx, col] = np.nan
        else:
            # Add this value to the set of seen values
            seen_values.add(current_value)


# In[47]:

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

# In[49]:
df_min_mean_enc_modified = filter_columns_by_date_difference(df_min_mean_enc_gr)
df_ret_drop = filter_columns_by_date_difference(df_ret_gr)

# In[50]:
df_ret_drop = df_ret_drop.sort_values(by=['group','su_week'], ascending=True)
df_min_mean_enc_modified = df_min_mean_enc_modified.sort_values(by=['group','su_week'], ascending=True)

# In[51]:
selected_columns = df_min_mean_enc_modified.iloc[:, 5:]

pct_change_df = selected_columns.pct_change(axis=1)

original_selected = df_min_mean_enc_modified[['group']]

df_combined = pd.concat([original_selected, pct_change_df], axis=1).replace(0, np.nan)


# In[52]:


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

df_trimmed, outlier_stats = trim_group_outliers(df_combined, group_col='group')


# In[54]:
def calculate_weighted_group_averages(df, group_column):
    results = {}

    # Dictionary mapping interval patterns to number of values to use
    interval_rules = {
        'interval_14d': 10, 'interval_21d': 10,  # Last 10 values
        'interval_28d': 8, 'interval_35d': 8,  # Last 8 values
        'interval_42d': 6, 'interval_49d': 6,  # Last 6 values
        'interval_56d': 4, 'interval_63d': 4  # Last 4 values
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

            # Create exponential weights
            weights = np.exp(np.linspace(0, 1, n_values))
            weights = weights / weights.sum()

            # Calculate weighted average
            weighted_avg = np.sum(values_to_use * weights)
            group_results[col] = weighted_avg

        results[group_name] = group_results

    # Convert results to DataFrame
    return pd.DataFrame.from_dict(results, orient='index')


# In[55]:

df_combined_mean = calculate_weighted_group_averages(df_trimmed, 'group')

df_combined_mean.reset_index(inplace=True)

# drop irrelavant columns for prediction in the unrestricted dataset
df_ret_drop_enc = df_ret_drop.drop(['su_week', 'total_costs', 'platform', 'traffic_type'], axis=1)


# In[57]:

def adjust_row_values(df):
    # Create a copy to avoid modifying the original
    df_adjusted = df.copy()

    # Get all columns except 'index'
    value_columns = df_adjusted.columns.drop('index')

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

df_combined_mean_adjusted = adjust_row_values(df_combined_mean)

# In[59]:


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

# In[61]:


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

# In[63]:


df_combined_mean_adjusted_enc_min_pct = df_combined_mean_adjusted_enc.applymap(
    lambda x: x + 1 if pd.api.types.is_numeric_dtype(type(x)) else x)

# In[65]:

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

# In[66]:

df_combined_mean_adjusted_enc_min_pct = df_combined_mean_adjusted_enc_min_pct.drop('group', axis=1).rename(columns= {'index': 'group'})

# In[67]:
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

# In[67]:
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


# In[69]:


filled_df = fill_nan_with_group_multipliers(df_ret_drop_enc, df_combined_mean_adjusted_enc_min_pct)

# In[70]:
filled_new_df = df_ret_drop[['su_week', 'total_costs','platform','traffic_type']].merge(filled_df,
                                                             left_index=True,
                                                              right_index=True)

filled_new_df_loc = filled_new_df.query("interval_7d != 0").iloc[:,:-1]

df_final_value = filled_new_df_loc.dropna()


# In[75]:

# Merge with the specific columns you need
final_df_cal = df_final_value[['su_week','group','total_costs','platform','traffic_type']].merge(
    df_combined_mean_adjusted_enc_min_pct,
    on='group',
    how='left'
)

# Drop any unnecessary columns
final_df_cal_drop = final_df_cal.drop('interval_7d', axis=1)


# In[77]:

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


final_df_cal_drop = calculate_interval_targets(final_df_cal_drop)

# In[79]:


target_columns = [c for c in final_df_cal_drop.columns if c.startswith('target')]
selected_columns = ['group', 'su_week', 'platform', 'traffic_type', 'total_costs'] + target_columns
final_df_cal_drop_enc = final_df_cal_drop[selected_columns].sort_values(
    by=['group', 'su_week', 'platform', 'traffic_type'])
final_df_cal_drop_enc.iloc[:, 5:] = final_df_cal_drop_enc.iloc[:, 5:].round().astype('int')

# In[81]:


df_value_actual = df_final_value.copy()
df_value_actual.iloc[:, 5:] = df_value_actual.iloc[:, 5:].round().astype('int')

df_value_actual = df_value_actual.sort_values(by=['group', 'su_week', 'platform', 'traffic_type'])

# In[83]:


df_target = final_df_cal_drop_enc[
    ['group', 'su_week', 'platform', 'traffic_type', 'total_costs', 'target_7d', 'target_14d', 'target_21d',
     'target_28d', 'target_35d',
     'target_42d', 'target_49d', 'target_56d']]

df_pred = df_value_actual[
    ['group', 'su_week', 'platform', 'traffic_type', 'total_costs', 'interval_7d', 'interval_14d', 'interval_21d',
     'interval_28d', 'interval_35d',
     'interval_42d', 'interval_49d', 'interval_56d', 'interval_63d', 'interval_70d', 'interval_77d',
     'interval_84d', 'interval_91d', 'interval_98d', 'interval_105d', 'interval_112d', 'interval_119d',
     'interval_126d', 'interval_133d']]

df_merged_comp = df_target.merge(df_pred, on=['group', 'su_week', 'platform', 'traffic_type', 'total_costs'],
                                 how='left')

# In[85]:


renamed_columns = {col: col.replace('interval', 'prediction') for col in df_merged_comp.columns if
                   col.startswith('interval')}
df_merged_comp = df_merged_comp.rename(columns=renamed_columns)

# In[87]:

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
        weeks_in_group = unique_weeks[i:i + 4]
        for week in weeks_in_group:
            group_mapping[week] = group_name

    # Add the group column to the DataFrame
    df['month_group'] = df[date_column].map(group_mapping)

    return df


df_deploy = group_weeks(df_merged_comp)

# In[89]:


cols = df_deploy.columns.tolist()

cols.remove('month_group')

# Insert it at the third position (index 2)
cols.insert(2, 'month_group')

# Reorder the dataframe
df_deploy = df_deploy[cols]

# In[91]:


df_deploy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')

# In[91]:
df_deploy = df_deploy.rename(columns={'group': 'channel_group'})

# In[93]:


# Set your destination table
destination_table = 'ml_models.marketing_projection'

# Write the DataFrame to BigQuery
df_deploy.to_gbq(destination_table, project_id=project_id, if_exists='replace', credentials=None)