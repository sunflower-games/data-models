#!/usr/bin/env python
# coding: utf-8

# In[6]:


#read sql files
def load_sql(filename):
    with open(filename, 'r') as file:
        return file.read()


# In[9]:


#split by hours
def filter_by_time_limit(df, time_limits):    
    filtered_dfs = {}
    
    for hours in time_limits:
        time_limit = pd.Timedelta(hours=hours)
        mask = ((df['redeem_time'] - df['signup_date'] <= time_limit) | 
                (df['transaction_date_time'] - df['signup_date'] <= time_limit))
        filtered_dfs[f'{hours}h'] = df[mask]
    
    return filtered_dfs


# In[ ]:


#trim outliers
def trim_outliers_quantile(df, feature_list, lower_quantile=0.01, upper_quantile=0.99):
    df_trimmed = df.copy()  
    for feature in feature_list:
        lower_bound = df[feature].quantile(lower_quantile)
        upper_bound = df[feature].quantile(upper_quantile)
        
        # Trim values below the lower bound
        df_trimmed[feature] = df_trimmed[feature].clip(lower=lower_bound)
        # Trim values above the upper bound
        df_trimmed[feature] = df_trimmed[feature].clip(upper=upper_bound)
    
    return df_trimmed


# In[ ]:


#create dummy variables

def process_column(df, column, top_values):
    # Replace values based on top_values and "other"
    df[column] = np.where(df[column].isin(top_values),
                          df[column].str.replace("\..*", "", regex=True) if df[column].dtype == 'object' else df[column],
                          "other")
    
    # Create dummy variables for the modified column
    df = pd.get_dummies(data=df, columns=[column], dtype=int)
    
    return df


# In[ ]:


##log transform features and drop old ones

def log_transform_columns(df, columns):
    epsilon = 1e-10
    
    for column in columns:
        df[f'ln_{column}'] = np.log(df[column] + epsilon)
    
    # Drop the original columns
    df.drop(columns, axis=1, inplace=True)
    
    return df

