import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from preprocessing import load_and_prepare_data

# Ładuje i przygotowuje dane (train/test)
X_train, X_test, y_train, y_test = load_and_prepare_data("../dataset/WineQT.csv")

# Tworzy DataFrame z cechami wina (nazwy kolumn)
X_train_df = pd.DataFrame(X_train, columns=[
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
])

# Łączy cechy z etykietą jakości
y_train_df = y_train.to_frame(name='quality')
df = pd.concat([X_train_df, y_train_df], axis=1)

# Zostawia tylko dobre wina (quality == 1)
df_good = df[df['quality'] == 1].drop(columns='quality')

# Zamienia każdą cechę na dwie kolumny: czy jest wysoka (> mediana) lub niska (<= mediana)
def binarize_features(df):
    binarized = pd.DataFrame()
    for col in df.columns:
        median = df[col].median()
        binarized[col + "_high"] = df[col] > median
        binarized[col + "_low"] = df[col] <= median
    return binarized.astype(bool)

# Binarne przekształcenie cech
df_bin = binarize_features(df_good)

# Szuka często współwystępujących cech (itemsets) – minimum 10% przypadków
frequent_itemsets = apriori(df_bin, min_support=0.1, use_colnames=True)

# Jeśli są jakieś ciekawe zestawy cech, generuje reguły asocjacyjne i pokazuje top 10
if not frequent_itemsets.empty:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    print(rules.sort_values("lift", ascending=False).head(10))
else:
    print("Nie znaleziono żadnych częstych zestawów przedmiotów (itemsets).")