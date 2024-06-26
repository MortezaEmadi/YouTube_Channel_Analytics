# src/evaluate.py

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def evaluate_model(y_true, y_pred, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    roc_auc = roc_auc_score(y_true, y_pred, multi_class='ovr', average=average)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics

def print_evaluation_report(y_true, y_pred):
    report = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(report)
    return report

def cross_validate_model(model, X, y, cv=5):
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean()}")
    return cv_scores

def advanced_evaluation(y_true, y_pred, y_pred_proba, average='weighted'):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average=average)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc
    }

    return metrics

def print_advanced_evaluation(y_true, y_pred, y_pred_proba):
    metrics = advanced_evaluation(y_true, y_pred, y_pred_proba)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
    return metrics
