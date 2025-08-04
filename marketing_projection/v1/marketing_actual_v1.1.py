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

select * from guyb_sandbox.marketing_actual_full 

"""

df = client.query(query).to_dataframe()

# In[33]:

df['su_week'] = pd.to_datetime(df['su_week'])




# In[33]:

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

df = process_and_group_data(df)


# In[37]:

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
    'smadex_int_app': 'smadex_int_app'
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
df['group'] = df['channel_type_con'].apply(get_group)

df.loc[df['total_costs'] == 0, 'traffic_type'] = 'organic'
df.loc[df['total_costs'] > 0, 'traffic_type'] = 'paid'


# In[43]:
df.loc[df['group'].str.contains('_app'), 'platform'] = 'app'

mixed_traffic_groups = df.groupby(['su_week', 'group'])['traffic_type'].nunique().reset_index()
mixed_traffic_groups = mixed_traffic_groups[mixed_traffic_groups['traffic_type'] > 1]

mixed_groups_set = set(zip(mixed_traffic_groups['su_week'], mixed_traffic_groups['group']))

for idx, row in df.iterrows():
    if (row['su_week'], row['group']) in mixed_groups_set and row['group'] != 'other':
        df.at[idx, 'traffic_type'] = 'paid'

# In[34]:

gross_interval = [c for c in df.columns if 'interval' in c]

df['su_week'] = pd.to_datetime(df['su_week'])

df_ret = df[['group', 'su_week', 'platform', 'traffic_type', 'total_costs'] + gross_interval]

# In[38]:

df_ret.iloc[:, 5:] = df_ret.iloc[:, 5:].replace(0, np.nan)

# In[39]:

df_ret_sorted = df_ret.sort_values(by='su_week', ascending=False)

# In[39]:

df_ret_sorted = df_ret_sorted.groupby(['group', 'su_week', 'platform', 'traffic_type']).agg(
    lambda x: x.sum() if not x.isna().all() else np.nan).reset_index()

# In[40]:

df_ret_sorted = df_ret_sorted.rename(columns=lambda x: x.replace('interval', 'actual') if 'interval' in x else x)

# In[40]:
cols_to_process = df_ret_sorted.columns[6:]

# In[41]:

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


df_actual_deploy = group_weeks(df_ret_sorted)


# In[89]:
revenue_cols = df_actual_deploy.columns[6:-1]

# Get the unique weeks, sorted from newest to oldest
unique_weeks = sorted(df_actual_deploy['su_week'].unique(), reverse=True)

# Keep different numbers of columns based on cohort age
for i, week in enumerate(unique_weeks):
    # For each week, determine how many columns to keep
    # The most recent week (i=0) gets 8 columns, next week gets 9, and so on
    cols_to_keep = min(1 + i, len(revenue_cols))

    # Create a mask for this week
    week_mask = df_actual_deploy['su_week'] == week

    # For rows in this week, keep only the specified number of columns
    if cols_to_keep < len(revenue_cols):
        # Get the columns to set to NaN (columns beyond our limit)
        cols_to_null = revenue_cols[cols_to_keep:]

        # Set these columns to NaN for the current week's rows
        for col in cols_to_null:
            df_actual_deploy.loc[week_mask, col] = np.nan


        # In[90]:

cols = df_actual_deploy.columns.tolist()

cols.remove('month_group')

# Insert it at the third position (index 2)
cols.insert(2, 'month_group')

# Reorder the dataframe
df_actual_deploy = df_actual_deploy[cols]


# In[42]:

df_actual_deploy['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M')

# In[43]:

df_actual_deploy = df_actual_deploy.rename(columns={'group': 'channel_group'})


# In[44]:
# Set your destination table
destination_table_actual = 'ml_models.marketing_actual'

# Write the DataFrame to BigQuery
df_actual_deploy.to_gbq(destination_table_actual, project_id=project_id, if_exists='replace')