# Main script for AI Bias Bounty Challenge
# Fill in each phase step by step as per guide.md

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os
from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference, equalized_odds_difference
import warnings
warnings.filterwarnings("ignore", message="pandas.DataFrame with sparse columns found.It will be converted to a dense numpy array.")
warnings.filterwarnings("ignore")
# File paths
TRAIN_PATH = 'datasets/loan_access_dataset.csv'
TEST_PATH = 'datasets/test.csv'

# Protected attributes and target
PROTECTED_ATTRS = ['Gender', 'Race', 'Income', 'Age', 'Zip_Code_Group']
TARGET = 'Loan_Approved'  # Update if actual column name differs

def load_and_inspect():
    print('--- Loading datasets ---')
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print('\n--- Train Data Info ---')
    print(train.info())
    print('\n--- Train Data Sample ---')
    print(train.head())
    print('\n--- Train Data Description ---')
    print(train.describe(include='all'))
    print('\n--- Test Data Info ---')
    print(test.info())
    print('\n--- Test Data Sample ---')
    print(test.head())
    return train, test

def clean_data(df):
    # Standardize column names
    df = df.rename(columns=lambda x: x.strip())
    # Convert target to binary if present
    if 'Loan_Approved' in df.columns:
        df['Loan_Approved'] = df['Loan_Approved'].map({'Approved': 1, 'Denied': 0})
    # Handle missing values (simple strategy: fill numeric with median, categorical with mode)
    for col in df.columns:
        if df[col].dtype == 'O':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

def eda_summary(df):
    print('\n--- Value Counts for Protected Attributes ---')
    for attr in PROTECTED_ATTRS:
        if attr in df.columns:
            print(f'\n{attr} value counts:')
            print(df[attr].value_counts())
    if TARGET in df.columns:
        print(f'\n{TARGET} value counts:')
        print(df[TARGET].value_counts())
    print('\n--- Grouped Approval Rates by Protected Attribute ---')
    for attr in PROTECTED_ATTRS:
        if attr in df.columns and TARGET in df.columns:
            print(f'\nApproval rate by {attr}:')
            print(df.groupby(attr)[TARGET].mean())

def feature_engineering(df):
    # Example: income-to-loan ratio
    if 'Income' in df.columns and 'Loan_Amount' in df.columns:
        df['Income_Loan_Ratio'] = df['Income'] / (df['Loan_Amount'] + 1)
    # Drop redundant or ID columns
    for col in ['ID', 'Age_Group']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def train_and_evaluate(train):
    # Features and target
    X = feature_engineering(train.drop(columns=['Loan_Approved']))
    y = train['Loan_Approved']
    # One-hot encode categorical variables (sparse)
    X = pd.get_dummies(X, drop_first=True, sparse=True)
    # Address class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    # Random Forest with GridSearchCV
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    rf_params = {'n_estimators': [100], 'max_depth': [5, None]}
    rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='f1', n_jobs=-1)
    rf_grid.fit(X_train, y_train)
    y_pred_rf = rf_grid.predict(X_val)
    print('\n--- Random Forest Model Evaluation ---')
    print(f'F1 Score: {f1_score(y_val, y_pred_rf):.3f}')
    print(classification_report(y_val, y_pred_rf))
    return rf_grid.best_estimator_, X.columns

def predict_and_save(model, test, feature_cols):
    # Prepare test features
    X_test = test.drop(columns=['ID'], errors='ignore')
    X_test = pd.get_dummies(X_test, drop_first=True)
    # Align columns with training features
    for col in feature_cols:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_cols]
    # Predict
    preds = model.predict(X_test)
    # Prepare submission
    submission = pd.DataFrame({
        'ID': test['ID'],
        'LoanApproved': preds
    })
    submission['LoanApproved'] = submission['LoanApproved'].map({1: 'Approved', 0: 'Denied'})
    os.makedirs('outputs', exist_ok=True)
    submission.to_csv('outputs/submission.csv', index=False)
    print('\nPredictions saved to outputs/submission.csv')

def bias_fairness_analysis(test, feature_cols):
    # Load predictions
    pred_path = 'outputs/submission.csv'
    if not os.path.exists(pred_path):
        print('No predictions found for bias analysis.')
        return
    preds = pd.read_csv(pred_path)
    # Merge with test set on ID
    merged = test.merge(preds, on='ID', how='left')
    # Map predictions to binary for analysis
    merged['LoanApproved_bin'] = merged['LoanApproved'].map({'Approved': 1, 'Denied': 0})
    print('\n--- Bias/Fairness Analysis ---')
    for attr in PROTECTED_ATTRS:
        if attr in merged.columns:
            print(f'\nApproval rate by {attr}:')
            print(merged.groupby(attr)['LoanApproved_bin'].mean())
            # Show group counts
            print(f'Counts by {attr}:')
            print(merged[attr].value_counts())
    # Example: Disparity between groups (e.g., max-min approval rate)
    print('\n--- Disparity Summary ---')
    for attr in PROTECTED_ATTRS:
        if attr in merged.columns:
            rates = merged.groupby(attr)['LoanApproved_bin'].mean()
            if len(rates) > 1:
                disparity = rates.max() - rates.min()
                print(f'{attr}: max-min approval rate disparity = {disparity:.3f}')

def advanced_fairness_metrics(test, feature_cols):
    pred_path = 'outputs/submission.csv'
    if not os.path.exists(pred_path):
        print('No predictions found for advanced fairness analysis.')
        return
    preds = pd.read_csv(pred_path)
    merged = test.merge(preds, on='ID', how='left')
    merged['LoanApproved_bin'] = merged['LoanApproved'].map({'Approved': 1, 'Denied': 0})
    y_pred = merged['LoanApproved_bin']
    y_true = merged['Loan_Approved'] if 'Loan_Approved' in merged.columns else None
    print('\n--- Advanced Fairness Metrics (fairlearn) ---')
    for attr in PROTECTED_ATTRS:
        if attr in merged.columns:
            print(f'\nAttribute: {attr}')
            mf = MetricFrame(
                metrics={
                    'selection_rate': selection_rate,
                    'accuracy': accuracy_score,
                    'recall': recall_score
                },
                y_true=y_true if y_true is not None else y_pred,  # fallback to y_pred if no true labels
                y_pred=y_pred,
                sensitive_features=merged[attr]
            )
            print('Selection rate by group:')
            print(mf.by_group['selection_rate'])
            print('Accuracy by group:')
            print(mf.by_group['accuracy'])
            print('Recall by group:')
            print(mf.by_group['recall'])
            # Demographic parity difference
            dpd = demographic_parity_difference(y_true if y_true is not None else y_pred, y_pred, sensitive_features=merged[attr])
            print(f'Demographic parity difference: {dpd:.3f}')
            # Equalized odds difference (only if true labels available)
            if y_true is not None:
                eod = equalized_odds_difference(y_true, y_pred, sensitive_features=merged[attr])
                print(f'Equalized odds difference: {eod:.3f}')

def visualize_fairness_metrics(test, feature_cols):
    import matplotlib.pyplot as plt
    import seaborn as sns
    pred_path = 'outputs/submission.csv'
    print(f"Checking for predictions at {pred_path}...")
    if not os.path.exists(pred_path):
        print('No predictions found for fairness visualization.')
        return
    preds = pd.read_csv(pred_path)
    print(f"Predictions loaded: {len(preds)} rows")
    merged = test.merge(preds, on='ID', how='left')
    print(f"Merged test+preds: {len(merged)} rows")
    print("Protected attributes in merged:", [attr for attr in PROTECTED_ATTRS if attr in merged.columns])
    merged['LoanApproved_bin'] = merged['LoanApproved'].map({'Approved': 1, 'Denied': 0})
    print("Unique values in LoanApproved:", merged['LoanApproved'].unique())
    print("Unique values in LoanApproved_bin:", merged['LoanApproved_bin'].unique())
    y_pred = merged['LoanApproved_bin']
    y_true = merged['Loan_Approved'] if 'Loan_Approved' in merged.columns else None
    os.makedirs('charts', exist_ok=True)
    for attr in PROTECTED_ATTRS:
        if attr in merged.columns:
            print(f"Plotting for attribute: {attr}")
            # Selection rate
            rates = merged.groupby(attr)['LoanApproved_bin'].mean()
            print(f"Selection rates for {attr}:\n", rates)
            if len(rates) < 2:
                print(f"Skipping {attr} (not enough groups)")
                continue
            plt.figure(figsize=(7,4))
            sns.barplot(x=rates.index, y=rates.values)
            plt.title(f'Selection Rate by {attr}')
            plt.ylabel('Selection Rate (Approval Rate)')
            plt.xlabel(attr)
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f'charts/selection_rate_by_{attr}.png')
            plt.close()
            # Accuracy and recall by group (if true labels available)
            if y_true is not None:
                from fairlearn.metrics import MetricFrame, selection_rate
                from sklearn.metrics import accuracy_score, recall_score
                mf = MetricFrame(
                    metrics={'accuracy': accuracy_score, 'recall': recall_score},
                    y_true=y_true,
                    y_pred=y_pred,
                    sensitive_features=merged[attr]
                )
                for metric in ['accuracy', 'recall']:
                    print(f"{metric.capitalize()} by group for {attr}:\n", mf.by_group[metric])
                    plt.figure(figsize=(7,4))
                    sns.barplot(x=mf.by_group[metric].index, y=mf.by_group[metric].values)
                    plt.title(f'{metric.capitalize()} by {attr}')
                    plt.ylabel(metric.capitalize())
                    plt.xlabel(attr)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    plt.savefig(f'charts/{metric}_by_{attr}.png')
                    plt.close()
    print('\nFairness metric plots saved in charts/.')

def main():
    train, test = load_and_inspect()
    train = clean_data(train)
    test = clean_data(test)
    eda_summary(train)
    model, feature_cols = train_and_evaluate(train)
    predict_and_save(model, test, feature_cols)
    bias_fairness_analysis(test, feature_cols)
    advanced_fairness_metrics(test, feature_cols)
    visualize_fairness_metrics(test, feature_cols)

if __name__ == "__main__":
    main() 