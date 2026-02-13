#%%
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
import random
from sklearn.metrics import confusion_matrix


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

# Answer: I think the threshold would be more accurate if it decreases, as k determines the # of points to average out
#  and make a classificaiton decision,so increasing k would allow the model to make more accurate decisions and be able to
# rely on a higher level of confidence.Therefore the confusion matrix would also change, as threshold determines what is a positive
# and what is a negative



#%% 
cm = confusion_matrix(y_test, test_preds)

print("Confusion Matrix:")
print(cm)

accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
print("Accuracy:", accuracy)
# %%
# Confusion Matrix interperation. It appears as if our confusion matrix shows 
# a successful model as it its at 98% accuracy. However, since HBCU's are only 2.5% of the data,
# the model is just picking the majority class almost every time, as the model as 741 true negatives
# (correctly non-HBCU), 0 false positives (hbcu but said non-hbcu), and 4 true positives 
# (correctly HBCU). Therefore, the model does not identify hbcu's incredibly well, but, does do an okay
# job, correctly identifying 4/19 at a 2.5% prevalence rate.



#%%
def prep_cc_data_stratified(df, target_col="hbcu", test_size=0.2, tune_size=0.2, random_state=123):
    df = df.copy()

    categorical_cols = ["state", "level", "control", "basic", "flagship"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    id_cols = [
        "index", "unitid", "chronname", "city", "nicknames", "site", "similar",
        "vsa_year", "vsa_grad", "vsa_enroll"
    ]
    df = df.drop(columns=[c for c in id_cols if c in df.columns])

    def hbcu_to_int(v):
        s = str(v).strip().upper()
        return 1 if s in {"YES", "X", "1", "TRUE", "T", "Y"} else 0

    df[target_col] = df[target_col].apply(hbcu_to_int).astype(int)

    y = df[target_col]
    X = df.drop(columns=[target_col])

    cat_cols = list(X.select_dtypes(include=["category", "object", "string"]).columns)
    X = pd.get_dummies(X, columns=cat_cols)

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

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_tune  = pd.DataFrame(imputer.transform(X_tune), columns=X_tune.columns, index=X_tune.index)
    X_test  = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_tune  = pd.DataFrame(scaler.transform(X_tune), columns=X_tune.columns, index=X_tune.index)
    X_test  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    print("HBCU prevalence:",
          f"train={y_train.mean():.4f}, tune={y_tune.mean():.4f}, test={y_test.mean():.4f}")
    print("Shapes:",
          f"X_train={X_train.shape}, X_tune={X_tune.shape}, X_test={X_test.shape}")

    return X_train, X_tune, X_test, y_train, y_tune, y_test

#%%
def run_knn(X_train, y_train, X_test, y_test, k=3, threshold=0.5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    probs = knn.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    results_df = pd.DataFrame({
        "actual": y_test.values,
        "predicted": preds,
        "probability": probs
    })

    cm = confusion_matrix(y_test, preds)
    acc = (cm[0,0] + cm[1,1]) / cm.sum()

    return results_df, cm, acc


#%%
X_train, X_tune, X_test, y_train, y_tune, y_test = prep_cc_data_stratified(cc_df)

results_df, cm, acc = run_knn(X_train, y_train, X_test, y_test, k=3, threshold=0.5)

print(cm)
print(acc)
print(results_df.head(10))


#%%
k_list = [1, 3, 5, 7, 9, 15]
t_list = [0.1, 0.2, 0.3, 0.4, 0.5]

best = None  # we'll store the best row here

for k in k_list:
    for t in t_list:
        results_df, cm, acc = run_knn(X_train, y_train, X_test, y_test, k=k, threshold=t)

        tn, fp = cm[0,0], cm[0,1]
        fn, tp = cm[1,0], cm[1,1]

        # recall for HBCU (how many HBCUs we successfully catch)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print("k =", k, "threshold =", t,
              "| acc =", round(acc, 4),
              "| tp =", tp, "fn =", fn, "fp =", fp,
              "| recall =", round(recall, 4))

        # pick "best" based on recall first, then accuracy as tie-breaker
        if best is None:
            best = (k, t, acc, tp, fn, fp, recall)
        else:
            _, _, best_acc, best_tp, best_fn, best_fp, best_recall = best

            if recall > best_recall:
                best = (k, t, acc, tp, fn, fp, recall)
            elif recall == best_recall and acc > best_acc:
                best = (k, t, acc, tp, fn, fp, recall)

print("\nBEST (by recall, then accuracy):")
print(best)
# %%
best_k, best_t, best_acc, best_tp, best_fn, best_fp, best_recall = best

best_results, best_cm, best_acc2 = run_knn(
    X_train, y_train, X_test, y_test,
    k=best_k, threshold=best_t
)

print("\nBest k and threshold:", best_k, best_t)
print("Confusion matrix:\n", best_cm)
print("Accuracy:", best_acc2)
print(best_results.head(10))
# %%
#How well does the model perform? Did the interaction of the adjusted 
# thresholds and k values help the model? Why or why not? 

# Adjusting k and threshhold did wonders for improving the model. With the new 
# adjustments, we were able to correctly all but one of the HBCU's, which is 
# a huge improvement from the previous model which only correctly identified 4/19.
# By adjusting based on testing and not intution, we were able to find the best
# combination of the two to provide the best results. As Prevalance is so low
# having a low threshold does better at prediction in a case like this.



#%%
# Step 8: new target variable
X_train, X_tune, X_test, y_train, y_tune, y_test = prep_cc_data_stratified(cc_df, target_col="flagship")

results_df, cm, acc = run_knn(X_train, y_train, X_test, y_test, k=3, threshold=0.5)

print("FLAGSHIP MODEL")
print("Confusion matrix:\n", cm)
print("Accuracy:", acc)
print(results_df.head(10))
# %%
