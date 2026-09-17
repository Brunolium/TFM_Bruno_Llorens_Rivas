

import os

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import dvgate as dg

COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
        "exang", "oldpeak", "slope", "ca", "thal", "num"]
CAT_COLS = ["cp", "restecg", "slope", "thal"]  
CERO_ES_AUSENTE = ["chol", "trestbps"]
FUENTES = ["cleveland", "hungarian", "switzerland", "va"]
TAM_ESPERADO = [303, 294, 123, 200]
TARGET = "num"


def split_strat(src, y, seed, fr=(0.6, 0.8)):
    """
    Parameters:
        src (ndarray): fuente (0-3) de cada fila.
        y (ndarray): target binario de cada fila.
        seed (int): semilla del sorteo de partición.
        fr (tuple): cortes acumulados (train, train+val) de la proporción.

    Returns:
        ndarray: partición ('tr'/'va'/'te') de cada fila, mismo orden de entrada.
    """
    rng = np.random.default_rng(seed)
    part = np.empty(len(y), dtype="<U2")
    for s in np.unique(src):
        for c in (0, 1):
            m = np.where((src == s) & (y == c))[0]
            rng.shuffle(m)
            n1, n2 = int(fr[0] * len(m)), int(fr[1] * len(m))
            part[m[:n1]], part[m[n1:n2]], part[m[n2:]] = "tr", "va", "te"
    return part


def load(path, seed=99, verbose=True):
    """
    Parameters:
        path (str): carpeta con los cuatro processed.*.data.
        seed (int): semilla, compartida con el resto del proyecto.
        verbose (bool): si True, imprime el diagnóstico de carga y los avisos.

    Returns:
        dict: X, y, src, idx_tr, idx_va, idx_te, players, meta, frac_na,
        más nombres (lista de las 4 fuentes en el orden de 'players').
    """
    dfs = []
    for i, nombre in enumerate(FUENTES):
        f = os.path.join(path, f"processed.{nombre}.data")
        d = pd.read_csv(f, header=None, names=COLS, na_values=["?", "-9", "-9.0"])
        d["fuente"] = i
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CERO_ES_AUSENTE:
        df.loc[df[c] == 0, c] = np.nan

    y = (df[TARGET] > 0).astype(int).values
    src = df.fuente.values.astype(int)
    players = sorted(int(s) for s in np.unique(src))
    cat = [c for c in CAT_COLS]
    num = [c for c in COLS if c not in cat + [TARGET]]
    cols = num + cat

    part = split_strat(src, y, seed)
    idx_tr, idx_va, idx_te = (np.where(part == k)[0] for k in ("tr", "va", "te"))

    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    prep.fit(df.loc[idx_tr, cols])
    Xt = prep.transform(df[cols])
    X = np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt, dtype=np.float64)
    frac_na = df[cols].isna().mean(axis=1).values          # ANTES de imputar
    meta = dg.source_meta(y, src, frac_na, idx_tr, players)

    if verbose:
        t = meta.copy()
        t.index = [FUENTES[i] for i in t.index]
        nv = pd.DataFrame({"val": pd.Series(src[idx_va]).value_counts().reindex(players),
                           "pos": pd.Series(y[idx_va]).groupby(src[idx_va]).sum().reindex(players),
                           "neg": pd.Series(1 - y[idx_va]).groupby(src[idx_va]).sum().reindex(players)})
        nv.index = [FUENTES[i] for i in nv.index]
        tam = list(pd.Series(src).value_counts().sort_index().values)
        print(f"Filas por fuente: {dict(zip(FUENTES, tam))} | total={len(df):,}"
              f"{'' if tam == TAM_ESPERADO else '  <-- NO coincide con ' + str(TAM_ESPERADO)}\n"
              f"{t.round(3).to_string()}\n"
              f"Matriz: {X.shape[0]:,} x {X.shape[1]} ({len(num)} numéricas + "
              f"{len(cat)} categóricas, tras imputar, indicar y codificar)\n"
              f"Particiones: train={len(idx_tr):,} val={len(idx_va):,} test={len(idx_te):,}"
              f" | prevalencia global={y.mean():.3f}\n"
              f"Validación por fuente:\n{nv.to_string()}")
        if nv["neg"].min() < 5 or nv["pos"].min() < 5:
            print(f"AVISO: alguna fuente tiene menos de 5 casos de una clase en validación. "
                  f"Para esa fuente lift_self (y por tanto el TIPO de daño) no es fiable.")
        if len(idx_te) < 4000:
            print(f"AVISO: test de {len(idx_te):,} filas. El MDE del cierre del bucle escala "
                  f"como 1/sqrt(n): con este tamaño la comprobación (d) va a fallar por "
                  f"construcción, y el veredicto será 'sin resolución'. Es el resultado "
                  f"correcto y esperado con 920 filas, no un fallo.")

    return {"X": X, "y": y, "src": src, "idx_tr": idx_tr, "idx_va": idx_va,
            "idx_te": idx_te, "players": players, "meta": meta,
            "frac_na": frac_na, "nombres": FUENTES}
