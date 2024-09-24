#!/usr/bin/env python
# coding: utf-8

# In[53]:


from my_utils import *


# In[55]:


#####import sql datasets######

dataset = load_sql('dataset.sql')


# In[56]:


#upload model pickle
with open('xgb_model_v1.pkl', 'rb') as file:
    xgb_model_loaded = pickle.load(file)


# In[59]:


#upload X files to - debug

with open('debugging_X.txt', 'r') as f:
    column_list = [line.strip() for line in f.readlines()]


# In[61]:


# Convert all string values in the DataFrame to lower case
df = df.applymap(lambda x: x.lower() if isinstance(x, str) else x)


# In[63]:


## define hardcoded_features
top_email_domain = ['gmail.com', 'yahoo.com', 'privaterelay.appleid.com', 'icloud.com']
top_channel = ['google', 'app', 'facebook', 'invite friend', 'sidelines'] 
geo_loc = ['california', 'texas', 'florida', 'illinois', 'georgia']
platform = ['web', 'app']


# In[65]:


#####features apply dummy variables function
df = process_column(df, "signup_email_domain", top_email_domain)
df = process_column(df, "channel", top_channel)
df = process_column(df, "first_geo_region", geo_loc)
df = process_column(df, "signup_platform", platform)


# In[ ]:


##transform the columns into log and drop

columns_to_transform = [
    'total_bet_amt',
    'total_win_amt_sc',
    'total_win_amt_cc',
    'sum_redeem_amt',
    'sum_amt_total',
    'sum_sc_coin_total'
]

df = log_transform_columns(df, columns_to_transform)


# In[ ]:


####feature #### Calculate the difference in minutes
df['time_diff_minutes'] = (df['first_purchase_date'] - df['signup_date']).dt.total_seconds() / 3600


# In[ ]:


##debugging - all columns appear in X

missing_columns = [col for col in column_list if col not in df.columns]

if missing_columns:
    print(f"Error: Missing columns: {', '.join(missing_columns)}")
    exit(1)  # Exit if any column is missing

# Generate new variable X from df including the required columns
X = df[column_list]


# In[ ]:


probabilities = xgb_model_loaded.predict_proba(X)
df['probability_positive_class'] = probabilities[:, 1]


# In[ ]:


df = df[df['probability_positive_class'] >= 0.94]

