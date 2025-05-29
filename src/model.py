import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import os

from preprocessing import load_and_prepare_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
)
from sklearn.model_selection import GridSearchCV, cross_validate

# Ustawia seed (powtarzalność wyników)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Tworzy katalogi na wykresy i raporty
os.makedirs("confusion_matrix", exist_ok=True)
os.makedirs("roc_curve", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Ładuje i przygotowuje dane (train/test)
X_train, X_test, y_train, y_test = load_and_prepare_data("../dataset/WineQT.csv")

# Definiuje modele, pipeline ze skalowaniem tam gdzie trzeba
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    ]),
    "Random Forest": RandomForestClassifier(random_state=SEED, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(random_state=SEED, class_weight="balanced"),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier())
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True, class_weight="balanced", random_state=SEED))
    ]),
}

# Parametry do strojenia (GridSearchCV) dla wybranych modeli
param_grids = {
    "Random Forest": {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
    "K-Nearest Neighbors": {
        "model__n_neighbors": [3, 5, 7],
        "model__weights": ["uniform", "distance"],
    },
    "Support Vector Machine": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ['linear', 'rbf'],
    },
    "Logistic Regression": {
        "model__C": [0.1, 1.0, 10.0],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"],
    }
}

results = []

# Trenuje i ocenia każdy model, zapisuje wyniki i wykresy
for name, model in models.items():
    print(f"\n=== {name} ===")
    start_time = time.time()
    try:
        # Jeśli są parametry, używa GridSearchCV do strojenia
        if name in param_grids:
            grid = GridSearchCV(model, param_grids[name], cv=5, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
            print(f"Best params: {grid.best_params_}")
        else:
            # W przeciwnym razie uczy model bez strojenia
            model.fit(X_train, y_train)

        # Walidacja krzyżowa (cross_validate) na train, zbiera metryki
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        cv_scores = cross_validate(model, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1)
        
        # Finalny trening na całości train i predykcja na test
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Pobiera prawdopodobieństwa klasy 1 (jeśli możliwe)
        proba_method = getattr(model, "predict_proba", None)
        probabilities = proba_method(X_test)[:, 1] if proba_method else None

        # Liczy AUC jeśli są prawdopodobieństwa
        auc = roc_auc_score(y_test, probabilities) if probabilities is not None else "N/A"

        # Zapisuje wyniki metryk do listy
        results.append({
            "Model": name,
            "Accuracy (CV Mean)": cv_scores["test_accuracy"].mean(),
            "Precision (CV Mean)": cv_scores["test_precision"].mean(),
            "Recall (CV Mean)": cv_scores["test_recall"].mean(),
            "F1 Score (CV Mean)": cv_scores["test_f1"].mean(),
            "ROC AUC (Test)": auc,
            "Training Time (s)": round(time.time() - start_time, 2),
        })

        # Generuje raport klasyfikacji i wypisuje go
        report = classification_report(y_test, predictions)
        print(report)

        # Zapisuje raport klasyfikacji do pliku
        report_path = f"reports/classification_report_{name.replace(' ', '_')}.txt"
        with open(report_path, "w") as f:
            f.write(report)

        # Rysuje i zapisuje confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"confusion_matrix/confusion_matrix_{name.replace(' ', '_')}.png")
        plt.close()

        # Rysuje i zapisuje krzywą ROC jeśli są prawdopodobieństwa
        if probabilities is not None:
            RocCurveDisplay.from_predictions(y_test, probabilities)
            plt.title(f"ROC Curve: {name}")
            plt.tight_layout()
            plt.savefig(f"roc_curve/roc_curve_{name.replace(' ', '_')}.png")
            plt.close()

    except Exception as e:
        # Obsługuje błędy, żeby pętla działała dalej
        print(f"Error with {name}: {e}")
        continue

# Podsumowuje wyniki wszystkich modeli i wypisuje na ekran
df_results = pd.DataFrame(results)
summary = df_results.sort_values("F1 Score (CV Mean)", ascending=False)

print("\n=== Summary of Models ===")
print(summary)

# Zapisuje podsumowanie do pliku CSV
summary.to_csv("model_summary.csv", index=False)