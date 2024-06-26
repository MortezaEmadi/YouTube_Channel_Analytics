# src/stacking_ensemble_model.py

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from scipy.stats import uniform, randint
from joblib import Parallel, delayed
from scipy.stats import pearsonr
import itertools
import optuna

# Select features and target variable
features = ['view_count', 'comment_count', 'channel_subscribers', 'time_diff',
            'conversation_rate', 'description_length', 'uploads_per_year']


def prepare_data(video_stats_filtered):
    X = video_stats_filtered[features]
    y = video_stats_filtered['engagement_category']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y


def split_data(X, y):
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42,
                                                      stratify=y_train_val)
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    return X_train, X_val, X_test


def random_search_hyperparams(estimator, param_distributions, X_train, y_train):
    random_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions, n_iter=10, cv=3,
                                       scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    return random_search.best_estimator_


def define_models():
    param_distributions_xgb = {
        'n_estimators': randint(150, 200),
        'learning_rate': uniform(0.05, 0.1),
        'max_depth': randint(4, 6)
    }

    xgb = XGBClassifier(tree_method='hist', device='cuda')
    best_xgb = random_search_hyperparams(xgb, param_distributions_xgb, X_train, y_train)

    lgbm = LGBMClassifier()
    gbc = GradientBoostingClassifier()

    models = {
        'xgb': best_xgb,
        'lgbm': lgbm,
        'gbc': gbc
    }

    return models


def create_voting_classifier(models):
    voting_clf = VotingClassifier(estimators=[
        ('xgb', models['xgb']), ('lgbm', models['lgbm']), ('gbc', models['gbc'])], voting='soft')
    return voting_clf


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Accuracy on validation set: {accuracy}")
    print(classification_report(y_val, y_val_pred))
    return accuracy, classification_report(y_val, y_val_pred, output_dict=True)


def cross_validate_model(model, X, y):
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean()}")
    return cv_scores


def advanced_evaluation(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics


def stacking_ensemble_evaluation(base_model_combinations, meta_models, X_train, y_train, X_val, y_val,
                                 passthrough=False):
    results = {}

    for meta_model_name, meta_model in meta_models.items():
        for combination_name, base_models in base_model_combinations.items():
            stacking_clf = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                passthrough=passthrough
            )
            stacking_clf.fit(X_train, y_train)
            y_val_pred = stacking_clf.predict(X_val)
            y_val_pred_proba = stacking_clf.predict_proba(X_val)

            metrics = advanced_evaluation(y_val, y_val_pred, y_val_pred_proba)
            results[(combination_name, meta_model_name)] = metrics

            print(f"Evaluated combination: {combination_name} with meta-model: {meta_model_name}")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

    return results


def select_best_model(results):
    best_combination = max(results, key=lambda k: results[k]['accuracy'])
    return best_combination


def final_evaluation_on_test(best_combination, base_model_combinations, meta_models, X_train_selected, y_train,
                             X_val_selected, y_val, X_test_selected, y_test, use_passthrough):
    final_base_models = base_model_combinations[best_combination[0]]
    final_meta_model = meta_models[best_combination[1]]

    final_stacking_clf = StackingClassifier(
        estimators=final_base_models,
        final_estimator=final_meta_model,
        passthrough=use_passthrough
    )
    final_stacking_clf.fit(np.vstack([X_train_selected, X_val_selected]), np.hstack([y_train, y_val]))

    y_test_pred = final_stacking_clf.predict(X_test_selected)
    y_test_pred_proba = final_stacking_clf.predict_proba(X_test_selected)
    test_metrics = advanced_evaluation(y_test, y_test_pred, y_test_pred_proba)

    print("Final Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value}")

    return test_metrics
