

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import dvgate as dg

ID_COLS = ["patientunitstayid", "patienthealthsystemstayid", "uniquepid", "hospitalid"]
CAT_COLS = ["gender", "ethnicity", "unittype", "unitadmitsource"]
TARGET = "mortality"


def select_players(counts, n_players, min_enc, n_strata, rng):
    """
    Parameters:
        counts (Series): nº de encuentros por hospitalid
        n_players (int): nº de hospitales a sortear.
        min_enc (int): encuentros mínimos para ser elegible.
        n_strata (int): nº de estratos de volumen
        rng (Generator): generador aleatorio ya inicializado con la semilla

    Returns:
        list[int]: hospitalid de los jugadores seleccionados, ordenados
    """
    elig = counts[counts >= min_enc].sort_values()
    if len(elig) < n_players:
        raise ValueError(f"solo {len(elig)} hospitales con >= {min_enc} encuentros")
    k, r = divmod(n_players, n_strata)
    sel = []
    for i, g in enumerate(np.array_split(elig.index.values, n_strata)):
        sel += rng.choice(g, min(len(g), k + (i < r)), replace=False).tolist()
    rest = [h for h in elig.index if h not in set(sel)]
    if len(sel) < n_players:
        sel += rng.choice(rest, n_players - len(sel), replace=False).tolist()
    return sorted(int(h) for h in sel)


def split_by_patient(hosp, pid, seed, fr=(0.6, 0.8)):
    """
    Parameters:
        hosp (ndarray): hospitalid de cada fila.
        pid (ndarray): uniquepid de cada fila.
        seed (int): semilla del sorteo de partición.
        fr (tuple): cortes acumulados (train, train+val) de la proporción.

    Returns:
        ndarray: partición ('tr'/'va'/'te') de cada fila, mismo orden de entrada.
    """
    rng = np.random.default_rng(seed)
    key = np.char.add(np.char.add(pid.astype(str), "|"), hosp.astype(str))
    part = np.empty(len(hosp), dtype="<U2")
    for h in np.unique(hosp):
        m = np.where(hosp == h)[0]
        u = np.unique(key[m]); rng.shuffle(u)
        n1, n2 = int(fr[0] * len(u)), int(fr[1] * len(u))
        lut = {k: ("tr" if i < n1 else "va" if i < n2 else "te") for i, k in enumerate(u)}
        part[m] = [lut[k] for k in key[m]]
    return part


def load(parquet, n_players=20, min_enc=400, n_strata=4, seed=99, verbose=True):
    """
    Parameters:
        parquet (str): ruta al Parquet de extract_eicu_duckdb.py.
        n_players (int): nº de hospitales a sortear como jugadores.
        min_enc (int): encuentros mínimos para que un hospital sea elegible.
        n_strata (int): nº de estratos de volumen en el sorteo.
        seed (int): semilla, compartida con el resto del proyecto.
        verbose (bool): si True, imprime el diagnóstico de carga y los avisos.

    Returns:
        dict: X, y, src, idx_tr, idx_va, idx_te, players, meta, frac_na,
        más n_num/n_cat (nº de columnas numéricas/categóricas, informativo).
    """
    df = pd.read_parquet(parquet)
    rng = np.random.default_rng(seed)
    counts = df.groupby("hospitalid").size()
    players = select_players(counts, n_players, min_enc, n_strata, rng)
    df = df[df.hospitalid.isin(players)].reset_index(drop=True)

    feats = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    cat = [c for c in CAT_COLS if c in feats]
    num = [c for c in feats if c not in cat and df[c].dtype != object
           and df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    cols = num + cat

    part = split_by_patient(df.hospitalid.values, df.uniquepid.values, seed)
    idx_tr, idx_va, idx_te = (np.where(part == k)[0] for k in ("tr", "va", "te"))

    prep = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                          ("s", StandardScaler())]), num),
        ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                          ("o", OneHotEncoder(handle_unknown="ignore"))]), cat)])
    prep.fit(df.loc[idx_tr, cols])
    Xt = prep.transform(df[cols])
    X = np.asarray(Xt.todense() if hasattr(Xt, "todense") else Xt, dtype=np.float64)

    y = df[TARGET].values.astype(int)
    src = df.hospitalid.values.astype(int)
    frac_na = df[cols].isna().mean(axis=1).values          
    meta = dg.source_meta(y, src, frac_na, idx_tr, players)

    if verbose:
        nh = df.groupby("hospitalid").size()
        print(f"Hospitales: {len(counts)} en el Parquet | {(counts >= min_enc).sum()} "
              f"elegibles (>= {min_enc} encuentros) | {len(players)} sorteados en "
              f"{n_strata} estratos de tamaño\nJugadores: {players}\n"
              f"Encuentros por jugador: min={nh.min():,} mediana={int(nh.median()):,} "
              f"max={nh.max():,} | total={len(df):,}\n"
              f"Matriz: {X.shape[0]:,} x {X.shape[1]} ({len(num)} numéricas + "
              f"{len(cat)} categóricas, tras imputar, indicar y codificar)\n"
              f"Particiones: train={len(idx_tr):,} val={len(idx_va):,} test={len(idx_te):,}"
              f" | hospitales en cada una: {len(set(src[idx_tr]))}/"
              f"{len(set(src[idx_va]))}/{len(set(src[idx_te]))} de {len(players)}\n"
              f"Prevalencia: global={y.mean():.4f} train={y[idx_tr].mean():.4f} "
              f"val={y[idx_va].mean():.4f} test={y[idx_te].mean():.4f}")
        nv = pd.Series(y[idx_va]).groupby(src[idx_va]).agg(["size", "sum"])
        print(f"Por jugador en validación: filas min={int(nv['size'].min())} "
              f"mediana={int(nv['size'].median())} | positivos min={int(nv['sum'].min())} "
              f"mediana={int(nv['sum'].median())}")
        if nv["sum"].min() < 15:
            print(f"AVISO: {int((nv['sum'] < 15).sum())} jugador(es) con menos de 15 "
                  f"positivos en validación. La comprobación (a) decide con lift_out, que "
                  f"se evalúa sobre train+val y aguanta, pero lift_self se evalúa solo en "
                  f"validación: para esas fuentes el TIPO ('ruido' vs 'concepto') no es "
                  f"fiable, aunque la marca sí lo sea.")
        if len(idx_te) < 4000:
            print(f"AVISO: test de {len(idx_te):,} filas. El MDE del cierre del bucle "
                  f"escala como 1/sqrt(n): por debajo de ~4.000 filas la comprobación "
                  f"(d) falla por construcción. Sube min_enc y vuelve a cargar.")
        if len(set(src[idx_va])) < len(players):
            print("AVISO: hay jugadores sin filas en validación; la comprobación (a) "
                  "no puede evaluarlos y saldrán como NaN.")

    return {"X": X, "y": y, "src": src, "idx_tr": idx_tr, "idx_va": idx_va,
            "idx_te": idx_te, "players": players, "meta": meta,
            "frac_na": frac_na, "n_num": len(num), "n_cat": len(cat)}