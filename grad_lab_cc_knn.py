#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import random

random.seed(1984) 
cc_df = pd.read_csv("Data/cc_institution_details.csv")


#%%
def prep_cc_data_stratified(df, test_size=0.2, tune_size=0.2, random_state=123):
    df = df.copy()

    # 1) Convert selected columns to categorical
    categorical_cols = ["state", "level", "control", "basic", "flagship"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 2) Drop identifier / non-predictive columns
    id_cols = [
        "index", "unitid", "chronname", "city", "nicknames", "site", "similar",
        "vsa_year", "vsa_grad", "vsa_enroll"
    ]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])

    # 3) Target: hbcu -> 0/1 (your dataset uses "X" for HBCU)
    def hbcu_to_int(v):
        s = str(v).strip().upper()
        return 1 if s in {"YES", "X", "1", "TRUE", "T", "Y"} else 0

    df["hbcu"] = df["hbcu"].apply(hbcu_to_int).astype(int)

    y = df["hbcu"]
    X = df.drop(columns=["hbcu"])

    # 4) One-hot encode categorical/object/string cols
    cat_cols = list(X.select_dtypes(include=["category", "object", "string"]).columns)
    X = pd.get_dummies(X, columns=cat_cols)

    # 5) Stratified splits: Train | Temp then Tune | Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=(test_size + tune_size),
        random_state=random_state,
        stratify=y
    )

    test_share_of_temp = test_size / (test_size + tune_size)
    X_tune, X_test, y_tune, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_share_of_temp,
        random_state=random_state,
        stratify=y_temp
    )

    # 6) Impute missing values (fit on train only)
    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_tune  = pd.DataFrame(imputer.transform(X_tune), columns=X_tune.columns, index=X_tune.index)
    X_test  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 7) Scale numeric features (fit on train only)
    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_tune  = pd.DataFrame(scaler.transform(X_tune), columns=X_tune.columns, index=X_tune.index)
    X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    # 8) Quick check: prevalence should still match closely
    print("HBCU prevalence:",
          f"train={y_train.mean():.4f}, tune={y_tune.mean():.4f}, test={y_test.mean():.4f}")
    print("Shapes:",
          f"X_train={X_train.shape}, X_tune={X_tune.shape}, X_test={X_test.shape}")

    return X_train, X_tune, X_test, y_train, y_tune, y_test


X_train, X_tune, X_test, y_train, y_tune, y_test = prep_cc_data_stratified(cc_df)

#%%
knn3 = KNeighborsClassifier(n_neighbors=3)

knn3.fit(X_train, y_train)

print("Model trained successfully.")

#%%
test_probs = knn3.predict_proba(X_test)[:, 1]

print("First 10 probabilities:", test_probs[:10])
print("Minimum probability:", test_probs.min())
print("Maximum probability:", test_probs.max())

#%%
threshold = 0.5

test_preds = (test_probs >= threshold).astype(int)

print("First 10 predictions:", test_preds[:10])
# %%
# dataframe that includes the test target values, 
# test predicted values, and test probabilities of the positive class.
results_df = pd.DataFrame({
    "actual": y_test.values,
    "predicted": test_preds,
    "probability": test_probs
})

print(results_df.head(10))
# %%
# Question 4: If you adjusted the k hyperparameter what do you think would
# happen to the threshold function? Would the confusion matrix look 
# the same at the same threshold levels or not? Why or why not?

# Answer: a
