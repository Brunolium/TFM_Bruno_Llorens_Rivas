""" el dicc que devuleven los loaders:

    X        ndarray (N, p)   matriz de diseño ya preprocesada
    y        ndarray (N,)      etiqueta binaria
    src      ndarray (N,)      identificador de fuente, longitud completa
    idx_tr, idx_va, idx_te     índices de las tres particiones
    players  lista de fuentes que juegan
    meta     DataFrame por fuente: n_filas, prevalencia, frac_ausentes
"""

import time
from math import factorial, sqrt

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


# umbrales
THR = {                   
    "damage_ret":  0.35, 
    "lift_min":    0.05,  
    "mono_asym":  -0.40,  
    "mono_mag":   -0.10,  
    "res_ratio":   2.00,  
    "power_mult":  1.00,  
    "trivial_z":   4.00,  
    "sigma_rule":  2.00,  
    "k_context":   3,     
}


def _fit(X, y, seed):
    # entrena el modelo del proyecto la RegresionLogistic
    """tol=1e-3 en vez del 1e-4 por defect"""
    return LogisticRegression(class_weight="balanced", C=1.0, max_iter=1000,
                              tol=1e-3, random_state=seed).fit(X, y)


def _ap(y, s):
    # da el AP
    return float(average_precision_score(y, s))


def _auc(y, s):
    # da el AUROC
    return float(roc_auc_score(y, s))



def loo(players, v):
    # calcula el valor de dejar uno fuera
    """coste: n+1 evaluaciones

    Parameters:
        players (list): identificadores de las fuentes en juego
        v (callable): utilidad v(S) sobre un subconjunto de players

    Returns:
        dict: {jugador: valor de dejar-uno-fuera}
    """
    players = list(players)
    vN = v(players)
    return {p: vN - v([q for q in players if q != p]) for p in players}


def exact_shapley(players, v):
    # calcula el exacto de Shapley
    """coste 2^n evaluaciones

    Parameters:
        players (list): identificadores de las fuentes en juego
        v (callable): utilidad v(S) sobre un subconjunto de players

    Returns:
        dict: {jugador: valor de Shapley exacto}
    """
    players = list(players)
    n = len(players)
    vals = np.array([v([players[i] for i in range(n) if m >> i & 1])
                     for m in range(1 << n)])
    coef = np.array([factorial(s) * factorial(n - s - 1) / factorial(n)
                     for s in range(n)])
    popc = np.array([bin(m).count("1") for m in range(1 << n)])
    phi = {}
    for i, p in enumerate(players):
        bit = 1 << i
        out = np.array([m for m in range(1 << n) if not m & bit])
        phi[p] = float(np.sum(coef[popc[out]] * (vals[out | bit] - vals[out])))
    return phi


def tmc_shapley(players, v, n_perm, seed, trunc=None):
    # estima Shapley aproximado con permutaciones
    """
    coste n_perm*(n-1) + 2

    Parameters:
        players (list): identificadores de las fuentes en juego.
        v (callable): utilidad v(S) sobre un subconjunto de players.
        n_perm (int): número de permutaciones truncadas.
        seed (int): semilla del muestreo de permutaciones.
        trunc (float, opcional): umbral de truncamiento (no usado por defecto).

    Returns:
        tuple: (phi, se, marg, n_eval) -- valores, error estándar, matriz de
        marginales por permutación, y evaluaciones de v gastadas.
    """
    players = list(players)
    n = len(players)
    rng = np.random.default_rng(seed)
    v_empty, vN = v([]), v(players)
    marg = np.zeros((n_perm, n))

    n_eval = 2
    for t in range(n_perm):
        prev, cur = v_empty, []
        for k in rng.permutation(n):
            if trunc is not None and abs(vN - prev) <= trunc:
                new = prev
            else:
                cur.append(players[k])
                if len(cur) == n:
                    new = vN
                else:
                    new = v(cur)
                    n_eval += 1
            marg[t, k] = new - prev
            prev = new
    phi = {p: float(marg[:, j].mean()) for j, p in enumerate(players)}
    se = {p: float(marg[:, j].std(ddof=1) / sqrt(n_perm)) for j, p in enumerate(players)}
    return phi, se, marg, n_eval





def source_meta(y, src, frac_na, idx, players):
    # da el resumen de cada fuente, en eICU varia s veces
    """
    Parameters:
        y (ndarray): target binario, longitud completa.
        src (ndarray): fuente de cada fila, longitud completa.
        frac_na (ndarray): fracción de ausentes por fila, longitud completa.
        idx (ndarray): índices de la partición sobre la que resumir (train).
        players (list): fuentes a incluir en la tabla.

    Returns:
        DataFrame: n_filas, prevalencia, frac_ausentes por fuente.
    """
    s_ = src[idx]
    return pd.DataFrame({"n_filas": pd.Series(s_).value_counts().reindex(players),
                         "prevalencia": pd.Series(y[idx]).groupby(s_).mean().reindex(players),
                         "frac_ausentes": pd.Series(frac_na[idx]).groupby(s_).mean().reindex(players)})


def corrupt_labels(y, src, targets, frac, seed, mask, mode="balanced"):
    # mete daño en las etiquetas de algunas fuentes
    """Corrupción sintética de etiquetas de la vobjetvio

    Parameters
        y (ndarray): target binario original, longitud completa
        src (ndarray): fuente de cada fila, longitud completa
        targets (list): fuentes a corromper
        frac (float): fracción de la fuente a voltear
        seed (int): semilla del sorteo de filas a voltear
        mask (ndarray): índices donde SÍ puede corromperse (train+al)
        mode (str): "balanced" (voltea igual nº de cada clase) o "random"

    Returns:
        ndarray: copia de y con las etiquetas corrompidas.
    """
    rng, y = np.random.default_rng(seed), y.copy()

    for h in targets:
        idx = mask[src[mask] == h]
        pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
        m = int(min(len(pos), len(neg), round(frac * len(pos))))
        sel = (rng.choice(idx, int(round(frac * len(idx))), replace=False) if mode == "random"
               else np.concatenate([rng.choice(pos, m, replace=False),
                                    rng.choice(neg, m, replace=False)]))
        y[sel] = 1 - y[sel]
    return y


def make_game(X, y, src, idx_tr, idx_va, players, cap, seed, min_rows=30):
    # monta el juego con los datos y prepara v(S)
    """
    Parameters:
        X (ndarray): matriz de features, ya preprocesada
        y (ndarray): target binario, longitud completa
        src (ndarray): fuente de cada fila, longitud completa
        idx_tr (ndarray): índices de entrenamiento.
        idx_va (ndarray): índices de validación (V_fixed)
        players (list): fuentes que entran en el juego.
        cap (int u None): tope de filas de train por fuente
        seed (int): semilla del modelo instrumento y del tope
        min_rows (int): filas mínimas de una coalición para no caer a v(vacío).

    Returns:
        tuple: (v, v_empty, idx_by_p) -- la función de utilidad, su valor en
        el vacío, y el índice de filas de train por jugador.
    """
    rng = np.random.default_rng(seed)
    idx_by_p = {}
    for p in players:
        idx = idx_tr[src[idx_tr] == p]
        if cap and len(idx) > cap:
            pos, neg = idx[y[idx] == 1], idx[y[idx] == 0]
            k1 = int(min(len(pos), max(1, round(cap * len(pos) / len(idx)))))
            k0 = int(min(len(neg), cap - k1))
            idx = np.sort(np.concatenate([rng.choice(pos, k1, replace=False),
                                          rng.choice(neg, k0, replace=False)]))
        idx_by_p[p] = idx
    Xva, yva = X[idx_va], y[idx_va]
    v_empty = _ap(yva, np.full(len(yva), yva.mean()))

    def v(S):
        S = [p for p in S if p in idx_by_p]
        if not S:
            return v_empty
        r = np.concatenate([idx_by_p[p] for p in S])

        if len(r) < min_rows or len(np.unique(y[r])) < 2:
            return v_empty
        return _ap(yva, _fit(X[r], y[r], seed).predict_proba(Xva)[:, 1])

    return v, v_empty, idx_by_p


# kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk

def check_damage(X, y, src, idx_tr, idx_va, players, seed, n_boot=200):
    # mide el daño directo de cada fuente
    """
      lift_out   modelo entrenado SIN h, evaluado en h. El modelo entrenado sin h no ha
                 visto NINGUNA fila de h, ni de train ni de validación, así que
                 se evalúa sobre train+val de h: 4x más filas y la mitad de
                 error estándar, sin ningún ajuste adicional.
      lift_self  modelo entrenado SOLO con h, evaluado en la validación de h.
      lift_cross modelo entrenado SOLO con h, evaluado en la validación del
                 RESTO. Es el MISMO ajuste que lift_self, un predict más.

    Parameters:
        X, y, src, idx_tr, idx_va: mismos arrays que en make_game.
        players (list): fuentes a diagnosticar.
        seed (int): semilla del modelo instrumento y del bootstrap.
        n_boot (int): réplicas de bootstrap para se_lift.

    Returns:
        DataFrame: lift_out, lift_self, lift_cross, ret, se_ret, sin_poder,
        dañina, tipo -- una fila por fuente.
    """
    rng = np.random.default_rng(seed)
    tr = {p: idx_tr[src[idx_tr] == p] for p in players}
    va = {p: idx_va[src[idx_va] == p] for p in players}
    rec = []
    for p in players:
        iv = va[p]
        io = np.sort(np.concatenate([tr[p], iv]))            
        vo = np.concatenate([va[q] for q in players if q != p])
        yo, yv, yc = y[io], y[iv], y[vo]
        row = dict(fuente=p, n_eval=len(io), n_pos_val=int(yv.sum()) if len(iv) else 0,
                   prev=float(yo.mean()) if len(io) else np.nan, lift_out=np.nan,
                   lift_self=np.nan, lift_cross=np.nan, se_lift=np.nan)
        if len(io) >= 30 and len(np.unique(yo)) > 1:
            otras = np.concatenate([tr[q] for q in players if q != p])
            s_out = _fit(X[otras], y[otras], seed).predict_proba(X[io])[:, 1]
            row["lift_out"] = (_ap(yo, s_out) - row["prev"]) / (1 - row["prev"])
            if len(np.unique(y[tr[p]])) > 1:
                m = _fit(X[tr[p]], y[tr[p]], seed)           
                pc = float(yc.mean())
                row["lift_cross"] = (_ap(yc, m.predict_proba(X[vo])[:, 1]) - pc) / (1 - pc)
                if len(np.unique(yv)) > 1:
                    pv = float(yv.mean())
                    row["lift_self"] = (_ap(yv, m.predict_proba(X[iv])[:, 1]) - pv) / (1 - pv)
            b = rng.integers(0, len(io), size=(n_boot, len(io)))
            row["se_lift"] = float(np.std(
                [(_ap(yo[i], s_out[i]) - yo[i].mean()) / (1 - yo[i].mean())
                 for i in b if len(np.unique(yo[i])) > 1], ddof=1))
        rec.append(row)
    d = pd.DataFrame(rec).set_index("fuente")
    med = d.lift_out.median()



    if not med > THR["lift_min"]:
        d["ret"] = d["ret_self"] = d["ret_cross"] = d["se_ret"] = np.nan
        d["sin_poder"] = True
        d["tipo"] = f"(a) inaplicable: lift mediano {med:.3f} <= {THR['lift_min']}"
        d["dañina"] = False
        return d
    d["ret"], d["se_ret"] = d.lift_out / med, d.se_lift / med
    d["ret_self"], d["ret_cross"] = d.lift_self / med, d.lift_cross / med
    bajo_out = (d.ret + 2 * d.se_ret) < THR["damage_ret"]
    bajo_cross = (d.ret_cross < THR["damage_ret"]).fillna(True)
    d["dañina"] = bajo_out & bajo_cross

    d["sin_poder"] = 2 * d.se_lift > THR["damage_ret"] * med
    d["tipo"] = np.where(d.dañina,
                np.where(d.ret_self < THR["damage_ret"], "etiquetas ruidosas",
                         "desplazamiento de concepto"),
                np.where(d.sin_poder, "(a) sin potencia",
                np.where(~bajo_cross & bajo_out, "difícil, no dañina", "sana")))
    if d["dañina"].mean() > 0.4:
        print(f"AVISO: {d['dañina'].mean():.0%} de las fuentes marcadas por (a). Se viola "
              f"el supuesto de mayoría limpia y la mediana deja de ser referencia válida.")
    return d

def check_monotonicity(marg, players, v0, vN):
    # mira si las marginales negativas se repiten
    """
    Parameters:
        marg (ndarray): matriz de marginales del piloto TMC (n_perm x n).
        players (list): fuentes, en el mismo orden que las columnas de marg.
        v0 (float): v(vacío).
        vN (float): v(jugadores completos).

    Returns:
        DataFrame: frac_neg, asimetria, senal -- una fila por fuente.
    """
    share = (vN - v0) / len(list(players))
    return pd.DataFrame({"frac_neg": (marg < 0).mean(0),
                         "asimetria": marg.mean(0) / np.abs(marg).mean(0),
                         "senal": marg.mean(0) / share},
                        index=list(players))


def check_resolution(phi, se, n_perm_pilot, n_perm_full):
    # mira si el ruido permite distinguir la señal
    """
    Parameters:
        phi (dict): valores del piloto TMC.
        se (dict): error estándar del piloto TMC.
        n_perm_pilot (int): permutaciones ya gastadas en el piloto.
        n_perm_full (int): permutaciones previstas para el cálculo completo.

    Returns:
        dict: ratio, rho_esperado, perm_necesarias, pasa.
    """
    v = np.array([phi[p] for p in phi])
    s = np.array([se[p] for p in phi])
    signal = sqrt(max(0.0, v.var(ddof=1) - (s ** 2).mean()))
    se_full = float(s.mean() * sqrt(n_perm_pilot / n_perm_full))
    ratio = signal / se_full if se_full > 0 else np.inf
    req = (n_perm_pilot * (s.mean() * THR["res_ratio"] / signal) ** 2
           if signal > 0 else np.inf)

    return dict(dispersion_corregida=signal, se_previsto=se_full, ratio=ratio,
                rho_esperado=ratio / sqrt(ratio ** 2 + 1),
                perm_necesarias=req, pasa=bool(ratio > THR["res_ratio"]))



def _paired_boot(yt, scores, ref, n_boot, seed):
    #  bootstrap pareado sin reentrenar
    """
    Bootstrap pareado 

    falta lo de parameters y salidas
    """


    rng = np.random.default_rng(seed)
    out = {n: [] for n in scores}
    for _ in range(n_boot):
        i = rng.integers(0, len(yt), len(yt))
        if len(np.unique(yt[i])) < 2:
            continue
        r0 = np.array([_auc(yt[i], scores[ref][i]), _ap(yt[i], scores[ref][i])])
        for n, sc in scores.items():
            out[n].append(np.array([_auc(yt[i], sc[i]), _ap(yt[i], sc[i])]) - r0)
    return {n: np.array(b) for n, b in out.items()}



def check_power(X, y, src, idx_tr, idx_te, players, seed, effect, n_boot=300):
    # mira si hay tamaño de muestra suficiente para ver el efecto
    """
    Parameters:
        X, y, src, idx_tr, idx_te: mismos arrays que en make_game/eval_test.
        players (list): conjunto completo de jugadores.
        seed (int): semilla de los dos ajustes y del bootstrap.
        effect (float): efecto esperado del cierre del bucle (de (a)/(b)).
        n_boot (int): réplicas de bootstrap pareado.

    Returns:
        dict: se_dif_ap, se_hanley_auroc, mde_ap, pasa.
    """
    drop = sorted(players)[len(players) // 2]
  
    m_all, s_all = eval_test(X, y, src, idx_tr, idx_te, players, seed)
    _, s_one = eval_test(X, y, src, idx_tr, idx_te,
                         [p for p in players if p != drop], seed)
    yt = y[idx_te]
    b = _paired_boot(yt, {"todos": s_all, "menos_una": s_one}, "todos", n_boot, seed)
    se_auc, se_ap = b["menos_una"].std(0, ddof=1)
    
    mde = 2.8 * se_ap                      
    return dict(n_test=len(yt), se_dif_ap=float(se_ap), se_dif_auroc=float(se_auc),
                mde_ap=float(mde), efecto_esperado=effect,
                pasa=bool(effect > THR["power_mult"] * mde))

def check_trivial(meta):
    # mira si prevalencia ,  ausencias  explican algo
    """

    Parameters:
        meta (DataFrame): salida de source_meta.

    Returns:
        DataFrame: z_prev, z_falta, trivial -- una fila por fuente.
    """
    def rz(x):                       
        x = np.asarray(x, float)
        m = np.median(x)
        mad = np.median(np.abs(x - m)) * 1.4826
        return np.abs(x - m) / mad if mad > 0 else np.zeros_like(x)
    t = pd.DataFrame({"z_prev": rz(meta.prevalencia.values),
                      "z_falta": rz(meta.frac_ausentes.values)}, index=meta.index)
    t["trivial"] = t[["z_prev", "z_falta"]].max(axis=1) > THR["trivial_z"]
    return t




def gate(dam, mono, res, pw):
    # para panel delas 4 comprobaciones
    """
    Parameters:
        dam (DataFrame): salida de check_damage.
        mono (DataFrame): salida de check_monotonicity.
        res (dict): salida de check_resolution.
        pw (dict): salida de check_power.

    Returns:
        tuple: (veredicto, motivo, tabla) -- PROCEDE/NO PROCEDE, el porqué, y
        la tabla resumen de las cuatro comprobaciones.
    """
    susp = (mono.asimetria < THR["mono_asym"]) & (mono.senal < THR["mono_mag"])
    falta = ([] if res["pasa"] else ["ranking bajo el ruido"]) + \
            ([] if pw["pasa"] else ["test sin potencia"])
    if not (dam["dañina"].any() or susp.any()):
        ver, mot = "NO PROCEDE", ("sin fenómeno: ninguna fuente con déficit de lift "
                                  "ni marginales negativas sistemáticas")
    elif falta:
        ver, mot = "NO PROCEDE", "sin resolución: " + " y ".join(falta)
    else:
        ver, mot = "PROCEDE", (f"{int(dam['dañina'].sum())} fuente(s) con daño directo, "
                               f"{int(susp.sum())} con marginales negativas; resoluble")
    tabla = pd.DataFrame(
        [("a_daño", f"min(ret + 2SE)  [{int(dam.sin_poder.sum())}/{len(dam)} sin potencia]",
          float((dam.ret + 2 * dam.se_ret).min()), THR["damage_ret"],
          bool(dam["dañina"].any())),
         ("b_monotonía", "fuentes con asimetría y magnitud negativas", float(susp.sum()),
          0.0, bool(susp.any())),
         ("c_resolución", "dispersión corregida / SE previsto", res["ratio"],
          THR["res_ratio"], res["pasa"]),
         ("d_potencia", "efecto esperado / MDE (AP)",
          pw["efecto_esperado"] / pw["mde_ap"] if pw["mde_ap"] > 0 else np.inf,
          THR["power_mult"], pw["pasa"])],
        columns=["comprobación", "estadístico", "valor", "umbral", "dispara"])
    return ver, mot, tabla

def run_panel(X, y, src, idx_tr, idx_va, idx_te, players, v, meta, seed,
              n_perm_pilot=25, n_perm_full=180):
    # orquestador
    """
    Parameters:
        X, y, src, idx_tr, idx_va, idx_te: mismos arrays que en make_game.
        players (list): fuentes en juego.
        v (callable): utilidad del juego principal.
        meta (DataFrame): salida de source_meta.
        seed (int): semilla compartida por las cuatro comprobaciones.
        n_perm_pilot (int): permutaciones del piloto TMC.
        n_perm_full (int): permutaciones previstas del cálculo completo.

    Returns:
        dict: daño, monotonia, resolucion, potencia, trivial, veredicto,
        motivo, tabla.
    """
    t0 = time.time()
    dam = check_damage(X, y, src, idx_tr, idx_va, players, seed)
    phi_p, se_p, marg_p, n_eval = tmc_shapley(players, v, n_perm_pilot, seed)
    mono = check_monotonicity(marg_p, players, v([]), v(players))

    res = check_resolution(phi_p, se_p, n_perm_pilot, n_perm_full)


    w = pd.Series({p: float((src[idx_tr] == p).sum()) for p in players})
    w, flag = w / w.sum(), dam.index[dam["dañina"]]
    e_a = float(((1 - dam.loc[flag, "ret"].clip(0, 1)) * w[flag]).sum()
                * dam.lift_out.median()) if len(flag) else 0.0
    susp = (mono.asimetria < THR["mono_asym"]) & (mono.senal < THR["mono_mag"])
    e_b = float(-sum(min(phi_p[p], 0.0) for p in mono.index[susp]))
    effect = max(e_a, e_b)
    pw = check_power(X, y, src, idx_tr, idx_te, players, seed, effect)
    ver, mot, tabla = gate(dam, mono, res, pw)
    return dict(daño=dam, monotonia=mono, resolucion=res, potencia=pw,
                trivial=check_trivial(meta), tabla=tabla, veredicto=ver,
                motivo=mot, marg=marg_p, phi_piloto=phi_p, se_piloto=se_p,
                n_eval=n_eval, minutos=(time.time() - t0) / 60)



def eval_test(X, y, src, idx_tr, idx_te, keep, seed):
    # entrena con algunas delas fuentes y evalua en test
    """
    Parameters:
        X, y, src, idx_tr, idx_te: mismos arrays que en make_game.
        keep (list): fuentes cuyas filas de train se usan para ajustar.
        seed (int): semilla del modelo instrumento.

    Returns:
        tuple: (dict de métricas {auroc, ap}, ndarray de scores en test).
    """
    r = idx_tr[np.isin(src[idx_tr], list(keep))]
    s = _fit(X[r], y[r], seed).predict_proba(X[idx_te])[:, 1]
    return dict(auroc=_auc(y[idx_te], s), ap=_ap(y[idx_te], s), n_train=len(r)), s





def close_loop(X, y, src, idx_tr, idx_te, players, phi, se, meta, seed,
               n_random=5, n_boot=300):
    # cierra el bucle quitando las fuentes marcadas y mide el efecto
    """
    Parameters:
        X, y, src, idx_tr, idx_te: mismos arrays que en make_game.
        players (list): conjunto completo de jugadores.
        phi (dict): valores de Shapley (exacto o TMC).
        se (dict): error estándar de phi.
        meta (DataFrame): salida de source_meta.
        seed (int): semilla de los reentrenamientos y del bootstrap.
        n_random (int): nº de líneas base aleatorias a promediar.
        n_boot (int): réplicas de bootstrap pareado.

    Returns:
        tuple: (DataFrame por estrategia, lista de fuentes que marca la
        regla guiada phi+2·SE<0).
    """
    guiada = sorted([p for p in players if phi[p] + THR["sigma_rule"] * se[p] < 0])


    k, ctx = (len(guiada), False) if guiada else (THR["k_context"], True)
    rng = np.random.default_rng(seed)
    quita = {"completo": [], "guiada": guiada,
             "heur_menor_tamaño": meta.n_filas.nsmallest(k).index.tolist(),
             "heur_mas_ausentes": meta.frac_ausentes.nlargest(k).index.tolist()}
    for j in range(n_random):
        quita[f"_al{j}"] = rng.choice(players, k, replace=False).tolist()
    M, S = {}, {}
    for nm, rem in quita.items():
        M[nm], S[nm] = eval_test(X, y, src, idx_tr, idx_te,
                                 [p for p in players if p not in set(rem)], seed)

    b = _paired_boot(y[idx_te], S, "completo", n_boot, seed + 1)
    rnd = [n for n in quita if n.startswith("_al")]
    b["aleatoria"] = np.mean([b[r] for r in rnd], axis=0)
    M["aleatoria"] = {m: float(np.mean([M[r][m] for r in rnd])) for m in ("auroc", "ap")}
    rows = []
    for nm in [n for n in quita if not n.startswith("_al")] + ["aleatoria"]:
        d = b[nm]
        rows.append(dict(estrategia=nm, auroc=M[nm]["auroc"], ap=M[nm]["ap"],
                         k=0 if nm == "completo" or (nm == "guiada" and ctx) else k,
                         contexto=ctx and nm not in ("completo", "guiada"),
                         **{f"d_{m}{sf}": float(f(d[:, j]))
                            for j, m in enumerate(("auroc", "ap"))
                            for sf, f in (("", np.mean),
                                          ("_lo", lambda a: np.percentile(a, 2.5)),
                                          ("_hi", lambda a: np.percentile(a, 97.5)))}))
    return pd.DataFrame(rows).set_index("estrategia"), guiada




def stability_check(phi_by_cond, se_by_cond, corrupted, ref, escala=None):
    # comprueba que las fuentes limpias no se contaminan cuando corrompemos,,,
    """
    Parameters:
        phi_by_cond (dict): {condición: {jugador: phi}}.
        se_by_cond (dict): {condición: {jugador: se}}.
        corrupted (list): fuentes corrompidas en la escalera.
        ref (str): nombre de la condición limpia de referencia.
        escala (dict, opcional): {condición: v(N)-v(vacío)} para normalizar.

    Returns:
        DataFrame: una fila por condición, con phi_min, n_neg_significativa,
        rho_rangos entre otras.
    """
    esc = escala or {c: 1.0 for c in phi_by_cond}
    limpias = [p for p in phi_by_cond[ref] if p not in set(corrupted)]
    base = np.array([phi_by_cond[ref][p] for p in limpias]) / esc[ref]
    rb, rec = pd.Series(base).rank(), []
    for c, ph in phi_by_cond.items():
        v = np.array([ph[p] for p in limpias]) / esc[c]
        e = np.array([se_by_cond[c][p] for p in limpias]) / esc[c]
        rec.append(dict(condicion=c, n_limpias=len(limpias), phi_min=v.min(),
                        phi_min_en_SE=float((v / e).min()), phi_medio=v.mean(),
                        n_neg_cruda=int((v < 0).sum()),
                        n_neg_significativa=int((v + 2 * e < 0).sum()),
                        max_desv_abs=float(np.abs(v - base).max()),
                        rho_rangos=float(np.corrcoef(pd.Series(v).rank(), rb)[0, 1])))
    return pd.DataFrame(rec).set_index("condicion")




def budget_report(n_players, n_cond, n_perm_pilot, n_perm_full, n_exact,
                  cost_eval, cost_fit):
    #intentar estimar coste computacional
    """
    Parameters:
        n_players (int): nº de jugadores del juego.
        n_cond (int): nº de condiciones de la escalera (1 si no hay escalera).
        n_perm_pilot (int): permutaciones del piloto.
        n_perm_full (int): permutaciones del cálculo completo.
        n_exact (int): tamaño del subconjunto de auditoría exacta.
        cost_eval (float): coste medido de una evaluación de v, en segundos.
        cost_fit (float): coste medido de un ajuste completo, en segundos.

    Returns:
        DataFrame: presupuesto previsto por bloque, en evaluaciones de v.
    """
    m = n_players - 1
    d = pd.DataFrame(
        [("panel: daño por fuente", 0, 2 * n_players, n_cond),
         ("panel: piloto TMC (b y c)", n_perm_pilot * m + 2, 0, n_cond),
         ("panel: potencia", 0, 2, n_cond),
         ("TMC completo (menos el piloto)", (n_perm_full - n_perm_pilot) * m, 0, n_cond),
         ("LOO", n_players + 1, 0, n_cond),
         ("cierre del bucle", 0, 9, n_cond),
         (f"Shapley exacto ({n_exact} jugadores, solo limpio)", 2 ** n_exact, 0, 1)],
        columns=["bloque", "evals_v", "ajustes", "veces"])
    d["min_condicion"] = (d.evals_v * cost_eval + d.ajustes * cost_fit) / 60
    d["min_total"] = d.min_condicion * d.veces
    return d
