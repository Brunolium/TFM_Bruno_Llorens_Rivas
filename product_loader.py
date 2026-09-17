
import ast

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import dvgate as dg


SEED = 99


def select_players(counts, n_players, min_enc, n_strata, rng):
    """
    Parameters:
        counts (Series): nº de votos por worker.
        n_players (int): nº de trabajadores a sortear.
        min_enc (int): votos mínimos para ser elegible.
        n_strata (int): nº de estratos de volumen.
        rng (Generator): generador aleatorio ya inicializado con la semilla.

    Returns:
        list: worker ids seleccionados, ordenados.
    """
    elig = counts[counts >= min_enc].sort_values()
    if len(elig) < n_players:
        raise ValueError(f"solo {len(elig)} trabajadores con >= {min_enc} votos")
    k, r = divmod(n_players, n_strata)
    sel = []
    for i, g in enumerate(np.array_split(elig.index.values, n_strata)):
        sel += rng.choice(g, min(len(g), k + (i < r)), replace=False).tolist()
    rest = [w for w in elig.index if w not in set(sel)]
    if len(sel) < n_players:
        sel += rng.choice(rest, n_players - len(sel), replace=False).tolist()
    return sorted(sel)


def _catalogo(path):

    recs = []
    with open(f"{path}/data_product") as f:
        header = ast.literal_eval(next(f).strip())
        for line in f:
            line = line.strip()
            if line:
                recs.append(ast.literal_eval(line))
    P = pd.DataFrame(recs, columns=header).set_index("id")
    P["texto"] = (P.name.fillna("") + " " + P.description.fillna("")).str.lower()
    P["precio"] = pd.to_numeric(
        P.price.astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    return P


def _features(A, Pcat):

    import difflib
    import re
    TOK = re.compile(r"[a-z0-9]+")
    MODELO = re.compile(r"\b[a-z]*\d+[a-z0-9\-]*\b")

    ids1 = A.question.str.split("_").str[0].astype(int)
    ids2 = A.question.str.split("_").str[1].astype(int)
    t1, t2 = Pcat.texto.reindex(ids1).values, Pcat.texto.reindex(ids2).values
    n1 = Pcat.name.fillna("").str.lower().reindex(ids1).values
    n2 = Pcat.name.fillna("").str.lower().reindex(ids2).values
    pr1, pr2 = Pcat.precio.reindex(ids1).values, Pcat.precio.reindex(ids2).values

    filas, falta_precio = [], []
    for a, b, na, nb, pa, pb in zip(t1, t2, n1, n2, pr1, pr2):
        ta = set(TOK.findall(a))
        tb = set(TOK.findall(b))
        jac = len(ta & tb) / max(len(ta | tb), 1)
        sim = difflib.SequenceMatcher(None, na, nb).ratio()
        ma, mb = set(MODELO.findall(na)), set(MODELO.findall(nb))
        modelo_match = float(len(ma & mb) > 0)
        falta = not (pd.notna(pa) and pd.notna(pb))
        dprecio = np.nan if falta else abs(pa - pb) / max(pa, pb, 1)
        dlen = abs(len(ta) - len(tb))
        filas.append([jac, sim, modelo_match, dprecio, dlen])
        falta_precio.append(falta)

    F = pd.DataFrame(filas, columns=["jaccard_tok", "sim_nombre", "modelo_match",
                                     "dif_precio", "dif_len"])
    F["dif_precio"] = F.dif_precio.fillna(F.dif_precio.median())
    return F, np.array(falta_precio, dtype=float)


def split_by_task(task_ids, target_por_tarea, seed, fr=(0.6, 0.8)):
    """
    Parameters:
        task_ids (ndarray): id de tarea (question) de cada fila.
        target_por_tarea (ndarray): mayoría de answer.csv por tarea, alineada
            fila a fila con task_ids.
        seed (int): semilla del sorteo de partición.
        fr (tuple): cortes acumulados (train, train+val) de la proporción.

    Returns:
        ndarray: partición ('tr'/'va'/'te') de cada fila, mismo orden de entrada.
    """
    rng = np.random.default_rng(seed)
    u_tar, first = np.unique(task_ids, return_index=True)
    u_t = target_por_tarea[first]
    parte = np.empty(len(u_tar), dtype="<U2")
    for c in np.unique(u_t):
        m = np.where(u_t == c)[0]
        rng.shuffle(m)
        n1, n2 = int(fr[0] * len(m)), int(fr[1] * len(m))
        parte[m[:n1]], parte[m[n1:n2]], parte[m[n2:]] = "tr", "va", "te"
    lut = dict(zip(u_tar, parte))
    return np.array([lut[t] for t in task_ids])


def load(path, n_players=16, min_enc=100, n_strata=4, seed=SEED, verbose=True):
    """
    Parameters:
        path (str): carpeta con answer.csv, truth.csv y data_product.
        n_players (int): nº de trabajadores a sortear como jugadores.
        min_enc (int): votos mínimos para que un trabajador sea elegible.
        n_strata (int): nº de estratos de volumen en el sorteo.
        seed (int): semilla, compartida con el resto del proyecto.
        verbose (bool): si True, imprime el diagnóstico de carga y los avisos.

    Returns:
        dict: X, y, src, idx_tr, idx_va, idx_te, players, meta, frac_na,
        más calidad_real y votos_particion (auditoría externa, fuera del
        contrato de dvgate).
    """
    A = pd.read_csv(f"{path}/answer.csv")
    Tr = pd.read_csv(f"{path}/truth.csv")
    Pcat = _catalogo(path)

    counts = A.worker.value_counts()
    rng = np.random.default_rng(seed)
    players = select_players(counts, n_players, min_enc, n_strata, rng)

    Fall, falta_precio_all = _features(A, Pcat)
    T = A[A.worker.isin(players)].reset_index(drop=True)
    mask = A.worker.isin(players).values
    Xraw = Fall.values[mask]
    y = T.answer.values.astype(int)
    src = T.worker.values
    frac_na = falta_precio_all[mask] / 5.0         

    mayoria_tarea = A.groupby("question").answer.mean().round().astype(int)
    part = split_by_task(T.question.values, mayoria_tarea.reindex(T.question.values).values, seed)
    idx_tr, idx_va, idx_te = (np.where(part == k)[0] for k in ("tr", "va", "te"))

    X = StandardScaler().fit(Xraw[idx_tr]).transform(Xraw)
    meta = dg.source_meta(y, src, frac_na, idx_tr, players)

    
    M = A.merge(Tr, on="question", how="left")
    M["acierto"] = M.answer == M.truth
    calidad = M[M.worker.isin(players)].groupby("worker").agg(
        n_total=("acierto", "size"), precision_real=("acierto", "mean"))

    nv = pd.DataFrame({
        "tr": pd.Series(src[idx_tr]).value_counts().reindex(players).fillna(0).astype(int),
        "va": pd.Series(src[idx_va]).value_counts().reindex(players).fillna(0).astype(int),
        "te": pd.Series(src[idx_te]).value_counts().reindex(players).fillna(0).astype(int)})

    if verbose:
        print(f"Trabajadores: {len(counts)} en total | {(counts >= min_enc).sum()} "
              f"elegibles (>= {min_enc} votos) | {len(players)} sorteados en "
              f"{n_strata} estratos de tamaño\nJugadores: {players}\n"
              f"Matriz: {X.shape[0]:,} x {X.shape[1]} (jaccard_tok, sim_nombre, "
              f"modelo_match, dif_precio, dif_len)\n"
              f"Particiones: train={len(idx_tr):,} val={len(idx_va):,} test={len(idx_te):,}"
              f" | tareas únicas: {T.question.nunique():,}\n"
              f"Prevalencia: global={y.mean():.3f} train={y[idx_tr].mean():.3f} "
              f"val={y[idx_va].mean():.3f} test={y[idx_te].mean():.3f}\n"
              f"Votos en validación por jugador: min={nv.va.min()} mediana={nv.va.median():.0f} "
              f"max={nv.va.max()}")
        dup = T.groupby("question").worker.nunique()
        if (dup > 1).any():
            print(f"AVISO: {(dup > 1).sum()} tareas tienen más de un jugador entre "
                  f"sus 3 anotadores; sus filas viajan juntas a la misma partición.")
        if nv.va.min() < 15:
            print(f"AVISO: el jugador {nv.va.idxmin()} tiene solo {nv.va.min()} votos "
                  f"en validación (<15). Ver advertencia de semilla junto a SEED arriba.")

    return {"X": X, "y": y, "src": src, "idx_tr": idx_tr, "idx_va": idx_va,
            "idx_te": idx_te, "players": players, "meta": meta,
            "frac_na": frac_na, "calidad_real": calidad, "votos_particion": nv}
