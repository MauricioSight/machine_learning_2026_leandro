"""Split estratificado 70/30 com seed fixa."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.30


def stratified_split(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int = DEFAULT_SEED,
    test_size: float = DEFAULT_TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split 70/30 estratificado por classe."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    # Cria cópias explícitas para evitar o aviso "SettingWithCopyWarning" do Pandas
    X_train_clean = X_train.copy()
    X_test_clean = X_test.copy()

    # 2. Separa as colunas por tipo
    num_cols = X_train_clean.select_dtypes(include=['number']).columns
    cat_cols = X_train_clean.select_dtypes(include=['object', 'category']).columns

    # 3. Trata variáveis Numéricas (Preenche nulos com a Mediana do treino)
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy='median')
        X_train_clean[num_cols] = imputer_num.fit_transform(X_train_clean[num_cols])
        X_test_clean[num_cols] = imputer_num.transform(X_test_clean[num_cols])

    # 4. Trata variáveis Categóricas/Texto (Preenche nulos com 'Missing')
    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy='constant', fill_value='Missing')
        X_train_clean[cat_cols] = imputer_cat.fit_transform(X_train_clean[cat_cols])
        X_test_clean[cat_cols] = imputer_cat.transform(X_test_clean[cat_cols])

        # 5. Converte todos os textos para códigos numéricos
        for col in cat_cols:
            combined = pd.concat([X_train_clean[col], X_test_clean[col]], axis=0).astype('category')
            X_train_clean[col] = combined.iloc[:len(X_train_clean)].cat.codes
            X_test_clean[col] = combined.iloc[len(X_train_clean):].cat.codes

    return X_train_clean, X_test_clean, y_train, y_test
