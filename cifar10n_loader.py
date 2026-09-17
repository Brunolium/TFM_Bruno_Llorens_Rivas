
import pickle
import zipfile

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

import dvgate as dg

NOMBRES_CLASE = ["airplane", "automobile", "bird", "cat", "deer", "dog",
                 "frog", "horse", "ship", "truck"]
CAT, DOG = 3, 5


def _cargar_imagenes(path):


    datos, labels = [], []
    for i in range(1, 6):
        with open(f"{path}/data_batch_{i}", "rb") as f:
            b = pickle.load(f, encoding="bytes")
        datos.append(b[b"data"])
        labels.append(np.asarray(b[b"labels"]))
    X = np.concatenate(datos).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return X, np.concatenate(labels)


def _cargar_etiquetas_humanas(path):

    try:
        import torch
        return {k: np.asarray(v) for k, v in
                torch.load(f"{path}/CIFAR-10_human.pt", map_location="cpu",
                          weights_only=False).items()}
    except Exception:
        z = zipfile.ZipFile(f"{path}/CIFAR-10_human.pt")
        nombre = [n for n in z.namelist() if n.endswith("data.pkl")][0]
        with z.open(nombre) as f:
            return {k: np.asarray(v) for k, v in pickle.load(f).items()}


def _anotaciones_gato_perro(path, clean_orig, dado_por_ronda):


    df = pd.read_csv(f"{path}/side_info_cifar10N.csv")
    filas = []
    for _, r in df.iterrows():
        lo = int(r["Image-batch"].split("--")[0])
        img = np.arange(lo, lo + 10)
        for ronda, w in enumerate(("Worker1-id", "Worker2-id", "Worker3-id")):
            filas.append(pd.DataFrame({"img": img, "worker": int(r[w]),
                                       "dado": dado_por_ronda[ronda][img]}))
    A = pd.concat(filas, ignore_index=True)
    A["limpio"] = clean_orig[A.img.values]
    base = A[A.limpio.isin([CAT, DOG])]
    fuera = ~base.dado.isin([CAT, DOG])
    return base[~fuera].reset_index(drop=True), int(fuera.sum())


def _features(X):


    gris = X.mean(axis=3)                                    # (N,32,32)
    bloques = gris.reshape(-1, 8, 4, 8, 4).mean(axis=(2, 4))  # (N,8,8)
    color = np.stack([X.mean(axis=(1, 2)), X.std(axis=(1, 2))], axis=1)
    return np.concatenate([bloques.reshape(len(X), -1),
                           color.reshape(len(X), -1)], axis=1).astype(np.float64)


def select_players(counts, n_players, min_enc, n_strata, rng):
    """
    Parameters:
        counts (Series): nº de anotaciones gato/perro por worker.
        n_players (int): nº de trabajadores a sortear.
        min_enc (int): anotaciones mínimas para ser elegible.
        n_strata (int): nº de estratos de volumen.
        rng (Generator): generador aleatorio ya inicializado con la semilla.

    Returns:
        list[int]: worker ids seleccionados, ordenados.
    """
    elig = counts[counts >= min_enc].sort_values()
    if len(elig) < n_players:
        raise ValueError(f"solo {len(elig)} trabajadores con >= {min_enc} anotaciones")
    k, r = divmod(n_players, n_strata)
    sel = []
    for i, g in enumerate(np.array_split(elig.index.values, n_strata)):
        sel += rng.choice(g, min(len(g), k + (i < r)), replace=False).tolist()
    rest = [w for w in elig.index if w not in set(sel)]
    if len(sel) < n_players:
        sel += rng.choice(rest, n_players - len(sel), replace=False).tolist()
    return sorted(int(w) for w in sel)


def split_by_image(img_ids, target, seed, fr=(0.6, 0.8)):
    """
    Parameters:
        img_ids (ndarray): id de imagen de cada fila.
        target (ndarray): etiqueta DADA (0/1) de cada fila.
        seed (int): semilla del sorteo de partición.
        fr (tuple): cortes acumulados (train, train+val) de la proporción.

    Returns:
        ndarray: partición ('tr'/'va'/'te') de cada fila, mismo orden de entrada.
    """
    rng = np.random.default_rng(seed)
    u_img, first = np.unique(img_ids, return_index=True)
    u_t = target[first]
    parte = np.empty(len(u_img), dtype="<U2")
    for c in np.unique(u_t):
        m = np.where(u_t == c)[0]
        rng.shuffle(m)
        n1, n2 = int(fr[0] * len(m)), int(fr[1] * len(m))
        parte[m[:n1]], parte[m[n1:n2]], parte[m[n2:]] = "tr", "va", "te"
    lut = dict(zip(u_img, parte))
    return np.array([lut[i] for i in img_ids])


def load(path_cifar, path_n, n_players=20, min_enc=100, n_strata=4, cap=400,
        seed=99, verbose=True):
    """
    Parameters:
        path_cifar (str): carpeta con data_batch_1..5.
        path_n (str): carpeta con side_info_cifar10N.csv y CIFAR-10_human.pt.
        n_players (int): nº de trabajadores a sortear como jugadores.
        min_enc (int): anotaciones gato/perro mínimas para ser elegible.
        n_strata (int): nº de estratos de volumen en el sorteo.
        cap (int): tope de filas por jugador (estratificado por target).
        seed (int): semilla, compartida con el resto del proyecto.
        verbose (bool): si True, imprime el diagnóstico de carga y los avisos.

    Returns:
        dict: X, y, src, idx_tr, idx_va, idx_te, players, meta, frac_na,
        más error_real (auditoría externa contra clean_label, fuera del
        contrato de dvgate).
    """
    X_img, clean_orig = _cargar_imagenes(path_cifar)
    hum = _cargar_etiquetas_humanas(path_n)
    if not np.array_equal(clean_orig, hum["clean_label"]):
        raise ValueError("Los data_batch_* no están alineados con CIFAR-10_human.pt. "
                         "Revisa el orden de los 5 lotes antes de continuar.")
    dado_por_ronda = [hum["random_label1"], hum["random_label2"], hum["random_label3"]]
    anota, n_descartadas = _anotaciones_gato_perro(path_n, clean_orig, dado_por_ronda)

    counts = anota.worker.value_counts()
    rng = np.random.default_rng(seed)
    players = select_players(counts, n_players, min_enc, n_strata, rng)


    filas = []
    for p in players:
        sub = anota[anota.worker == p]
        if cap and len(sub) > cap:
            y_bin = (sub.dado == DOG).astype(int)
            pos, neg = sub.index[y_bin == 1], sub.index[y_bin == 0]
            k1 = int(min(len(pos), max(1, round(cap * len(pos) / len(sub)))))
            k0 = int(min(len(neg), cap - k1))
            sub = sub.loc[np.concatenate([rng.choice(pos, k1, replace=False),
                                          rng.choice(neg, k0, replace=False)])]
        filas.append(sub)
    T = pd.concat(filas, ignore_index=True)

    Ximg = _features(X_img)                      
    Xraw = Ximg[T.img.values]
    y = (T.dado == DOG).astype(int).values                    
    src = T.worker.values.astype(int)
    frac_na = np.zeros(len(T))                                

    part = split_by_image(T.img.values, y, seed)
    idx_tr, idx_va, idx_te = (np.where(part == k)[0] for k in ("tr", "va", "te"))


    X = StandardScaler().fit(Xraw[idx_tr]).transform(Xraw)
    meta = dg.source_meta(y, src, frac_na, idx_tr, players)


    T["error_real"] = T.dado != T.limpio
    real = anota[anota.worker.isin(players)].assign(
        error_real=lambda d: d.dado != d.limpio).groupby("worker").agg(
        n_total=("error_real", "size"), tasa_error_real=("error_real", "mean"))

    if verbose:
        print(f"Universo gato/perro: {anota.img.nunique():,} imágenes | "
              f"{n_descartadas:,} anotaciones descartadas por dar una clase "
              f"distinta de gato/perro ({n_descartadas / (len(anota) + n_descartadas):.1%})\n"
              f"Trabajadores: {len(counts)} en total (tras el filtro) | "
              f"{(counts >= min_enc).sum()} elegibles (>= {min_enc} anotaciones "
              f"gato/perro) | {len(players)} sorteados en {n_strata} estratos de tamaño\n"
              f"Jugadores: {players}\n"
              f"Anotaciones por jugador (tras tope={cap}): "
              f"{T.worker.value_counts().reindex(players).to_dict()}\n"
              f"Matriz: {X.shape[0]:,} x {X.shape[1]} (8x8 gris + 6 estad. de color)\n"
              f"Particiones: train={len(idx_tr):,} val={len(idx_va):,} test={len(idx_te):,}"
              f" | imágenes únicas: {T.img.nunique():,}\n"
              f"Prevalencia (animal=1): global={y.mean():.3f} train={y[idx_tr].mean():.3f} "
              f"val={y[idx_va].mean():.3f} test={y[idx_te].mean():.3f}\n"
              f"Tasa de error REAL por jugador (verdad-terreno externa, NO vista por "
              f"dvgate): min={real.tasa_error_real.min():.3f} "
              f"mediana={real.tasa_error_real.median():.3f} "
              f"max={real.tasa_error_real.max():.3f}")
        dup = T.groupby("img").worker.nunique()
        if (dup > 1).any():
            print(f"AVISO: {(dup > 1).sum()} imágenes tienen más de un jugador entre "
                  f"sus anotadores. Todas sus filas fueron forzadas a la misma "
                  f"partición por split_by_image; es lo esperado, no un fallo.")
        if len(idx_te) < 4000:
            print(f"AVISO: test de {len(idx_te):,} filas. La comprobación (d) puede "
                  f"fallar por construcción con este tamaño; sube n_players o cap.")

    return {"X": X, "y": y, "src": src, "idx_tr": idx_tr, "idx_va": idx_va,
            "idx_te": idx_te, "players": players, "meta": meta,
            "frac_na": frac_na, "error_real": real}
