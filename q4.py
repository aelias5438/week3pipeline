# %%
# College Completion Data Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# %%
# Load the dataset
cc_df = pd.read_csv("Data/cc_institution_details.csv")
cc_df.shape

# %%
cc_df.dtypes

# %%
def cc_pipeline(df):
    df = df.copy()

    # 1.) Convert columns to categorical type
    categorical_cols = [
        "state",
        "level", 
        "control",
        "basic",
        "flagship"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2.) Drop identifier columns that won't help prediction
    id_cols = [
        "index",
        "unitid",
        "chronname",
        "city",
        "nicknames",
        "site",
        "similar",
        "vsa_year",
        "vsa_grad", 
        "vsa_enroll"
    ]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])

    # 3.) Set up target variable - predicting HBCU status
    target_col = "hbcu"
    
    # Convert Yes/No to 1/0
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})
    
    # Drop rows where target is missing
    df = df.dropna(subset=[target_col])

    # 4.) Separate features (X) and target (Y)
    Y = df[target_col]
    X = df.drop(columns=[target_col])

    # 5.) One-hot encode categorical columns
    cat_cols = list(X.select_dtypes(include=["category", "object"]).columns)
    X = pd.get_dummies(X, columns=cat_cols)

    # 6.) Scale numerical features to 0-1 range
    num_cols = list(X.select_dtypes(include="number").columns)
    X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    # 7.) Create train/tune/test splits (60/20/20)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.4, random_state=123
    )
    X_tune, X_test, Y_tune, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=123
    )

    # 8.) Calculate target prevalence
    prevalence = Y_train.mean()
    print(f"HBCU Prevalence in Training Set: {prevalence:.2%}")

    return X_train, X_tune, X_test, Y_train, Y_tune, Y_test

# %%
# Run the pipeline
X_train, X_tune, X_test, y_train, y_tune, y_test = cc_pipeline(cc_df)

# %%
# Check the shapes
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Tune: {X_tune.shape}, {y_tune.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# %%
# Job Placement Data Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# %%
# Load the dataset
jp_df = pd.read_csv("Data/job_placement.csv")
jp_df.shape

# %%
def jp_pipeline(df):
    df = df.copy()
    
    # 1.) Filter to only placed students (they have salary data)
    df = df[df["status"] == "Placed"]
    print(f"Filtered to {len(df)} placed students")

    # 2.) Convert columns to categorical type
    categorical_cols = [
        "gender",
        "ssc_b",
        "hsc_b",
        "hsc_s",
        "degree_t",
        "workex",
        "specialisation"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 3.) Drop identifier and unnecessary columns
    drop_cols = ["sl_no", "status"]  # status is now all "Placed" so not useful
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 4.) Set up target variable - predicting salary
    target_col = "salary"
    
    # Drop rows where salary is missing
    df = df.dropna(subset=[target_col])

    # 5.) Separate features (X) and target (Y)
    Y = df[target_col]
    X = df.drop(columns=[target_col])

    # 6.) One-hot encode categorical columns
    cat_cols = list(X.select_dtypes(include=["category", "object"]).columns)
    X = pd.get_dummies(X, columns=cat_cols)

    # 7.) Scale numerical features to 0-1 range
    num_cols = list(X.select_dtypes(include="number").columns)
    X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    # 8.) Create train/tune/test splits (60/20/20)
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.4, random_state=123
    )
    X_tune, X_test, Y_tune, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=123
    )

    # 9.) Show target stats (for regression we show mean instead of prevalence)
    print(f"Average Salary in Training Set: ${Y_train.mean():,.0f}")

    return X_train, X_tune, X_test, Y_train, Y_tune, Y_test

# %%
# Run the pipeline
X_train, X_tune, X_test, y_train, y_tune, y_test = jp_pipeline(jp_df)

# %%
# Check the shapes
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Tune: {X_tune.shape}, {y_tune.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")