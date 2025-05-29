import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Wczytuje plik CSV do DataFrame pandas
def load_dataset(path):
    df = pd.read_csv(path)
    return df

# Tworzy kolumnę 'label' na podstawie wartości 'quality'
# Jeśli jakość wina jest większa od tresholdu (domyślnie 7) to daje etykietę 1 (dobre wino) w przeciwnym razie 0 (złe wino)
# Następnie usuwa kolumnę 'quality'
def prepare_labels(df, threshold=7):
    df['label'] = df['quality'].apply(lambda x: 1 if x >= threshold else 0)
    return df.drop('quality', axis=1)

# Usuwa kolumnę 'Id' jeśli istnieje oraz usuwa wiersze z brakującymi wartościami
def clean_data(df):
    if 'Id' in df.columns:
        df = df.drop(columns=['Id'])
    df = df.dropna()
    return df

# Standaryzuje cechy, czyli przekształca je tak, by miały średnią 0 i odchylenie standardowe 1
def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

# Zbalansowuje zbiór danych używając techniki SMOTE, która generuje syntetyczne próbki dla klasy mniejszościowej
def balance_data(X, y):
    smote = SMOTE(random_state=42950)
    X_res, Y_res = smote.fit_resample(X, y)
    return X_res, Y_res

# Łączy wszystkie powyższe kroki: wczytuje dane, czyści, przygotowuje etykiety,
# standaryzuje cechy, balansuje dane i dzieli na zbiór treningowy oraz testowy
def load_and_prepare_data(path):
    df = load_dataset(path)
    df = clean_data(df)
    df = prepare_labels(df)
    
    X = df.drop('label', axis=1)
    y = df['label']

    X_scaled = scale_features(X)
    X_res, y_res = balance_data(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, stratify=y_res, random_state=72940)

    return X_train, X_test, y_train, y_test
