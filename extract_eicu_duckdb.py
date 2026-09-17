#!/usr/bin/env python3

import argparse
import os
import sys
import time

import duckdb


WINDOW_MIN = 1440  

VITAL_COLS = ["heartrate", "respiration", "sao2", "temperature", "systemicmean"]

LAB_NAMES = ["glucose", "creatinine", "BUN", "sodium", "potassium",
             "WBC x 1000", "Hgb", "platelets x 1000", "lactate", "total bilirubin"]

APS_KEEP = ["heartrate", "meanbp", "respiratoryrate", "temperature", "sodium",
            "creatinine", "wbc", "hematocrit", "glucose", "bun", "bilirubin",
            "albumin", "pao2", "pco2", "ph", "urine", "eyes", "motor", "verbal",
            "intubated", "vent", "dialysis"]


def find_table(data_path, name):

    for candidate in (name, name.lower()):
        for ext in (".csv.gz", ".csv"):
            p = os.path.join(data_path, candidate + ext)
            if os.path.exists(p):
                return p.replace("\\", "/")
    raise FileNotFoundError(
        f"No encuentro la tabla '{name}' en {data_path}. "
        f"Se esperaba {name}.csv.gz o {name}.csv"
    )


def has_table(data_path, name):

    try:
        find_table(data_path, name)
        return True
    except FileNotFoundError:
        return False


def safe_name(s):

    return s.replace(" ", "_").replace("x_1000", "k")



def sql_cohort(path_patient):

    return f"""
    CREATE OR REPLACE TABLE cohort AS
    SELECT DISTINCT ON (patientunitstayid)
        TRY_CAST(patientunitstayid AS BIGINT)         AS patientunitstayid,
        TRY_CAST(patienthealthsystemstayid AS BIGINT) AS patienthealthsystemstayid,
        uniquepid,
        TRY_CAST(hospitalid AS INTEGER)               AS hospitalid,
        gender,
        CASE
            WHEN age IS NULL OR trim(age) = '' THEN NULL
            WHEN starts_with(trim(age), '>') THEN 90.0
            ELSE TRY_CAST(trim(age) AS DOUBLE)
        END                                         AS age_num,
        ethnicity,
        unittype,
        unitadmitsource,
        TRY_CAST(admissionheight AS DOUBLE)         AS admissionheight,
        TRY_CAST(admissionweight AS DOUBLE)         AS admissionweight,
        CASE WHEN hospitaldischargestatus = 'Expired' THEN 1 ELSE 0 END AS mortality
    FROM read_csv_auto('{path_patient}', header=true, all_varchar=true)
    WHERE hospitaldischargestatus IN ('Expired', 'Alive')
    """


def sql_vitals(path_vitals, cols, window):

    aggs = []
    for c in cols:
        aggs += [
            f"avg(v.{c})                        AS vit_{c}_mean",
            f"min(v.{c})                        AS vit_{c}_min",
            f"max(v.{c})                        AS vit_{c}_max",
            f"count(v.{c})                      AS vit_{c}_count",
            # 'last' = última medición no nula de la ventana. Si hay varias en el
            # mismo minuto (empate en observationoffset) se toma la mayor, para
            # que el resultado sea determinista y reproducible entre ejecuciones.
            f"arg_max(v.{c}, struct_pack(o := v.observationoffset, x := v.{c})) "
            f"AS vit_{c}_last",
        ]
    select_cols = ",\n        ".join(aggs)
    cast_cols = ",\n            ".join(
        [f"TRY_CAST({c} AS DOUBLE) AS {c}" for c in cols]
    )
    return f"""
    CREATE OR REPLACE TABLE vit_feat AS
    SELECT
        v.patientunitstayid,
        {select_cols}
    FROM (
        SELECT
            TRY_CAST(patientunitstayid AS BIGINT)     AS patientunitstayid,
            TRY_CAST(observationoffset AS INTEGER)    AS observationoffset,
            {cast_cols}
        FROM read_csv_auto('{path_vitals}', header=true, all_varchar=true)
    ) v
    JOIN cohort c USING (patientunitstayid)
    WHERE v.observationoffset BETWEEN 0 AND {window}
    GROUP BY v.patientunitstayid
    """


def sql_labs(path_lab, names, window):

    aggs = []
    for nm in names:
        esc = nm.replace("'", "''")
        col = safe_name(nm)
        for agg in ("avg", "min", "max"):
            alias = "mean" if agg == "avg" else agg
            aggs.append(
                f"{agg}(CASE WHEN l.labname = '{esc}' THEN l.labresult END) "
                f"AS lab_{col}_{alias}"
            )
    select_cols = ",\n        ".join(aggs)
    in_list = ", ".join("'" + n.replace("'", "''") + "'" for n in names)
    return f"""
    CREATE OR REPLACE TABLE lab_feat AS
    SELECT
        l.patientunitstayid,
        {select_cols}
    FROM (
        SELECT
            TRY_CAST(patientunitstayid AS BIGINT)  AS patientunitstayid,
            TRY_CAST(labresultoffset AS INTEGER)   AS labresultoffset,
            labname,
            TRY_CAST(labresult AS DOUBLE)          AS labresult
        FROM read_csv_auto('{path_lab}', header=true, all_varchar=true)
    ) l
    JOIN cohort c USING (patientunitstayid)
    WHERE l.labresultoffset BETWEEN 0 AND {window}
      AND l.labname IN ({in_list})
    GROUP BY l.patientunitstayid
    """


def sql_aps(path_aps, keep):

    cols = ",\n        ".join(
        [f"nullif(TRY_CAST({c} AS DOUBLE), -1) AS aps_{c}" for c in keep]
    )
    return f"""
    CREATE OR REPLACE TABLE aps_feat AS
    SELECT DISTINCT ON (patientunitstayid)
        TRY_CAST(patientunitstayid AS BIGINT) AS patientunitstayid,
        {cols}
    FROM read_csv_auto('{path_aps}', header=true, all_varchar=true)
    """


def sql_apache_score(path_apr):

    return f"""
    CREATE OR REPLACE TABLE apache_score AS
    SELECT patientunitstayid, apachescore
    FROM (
        SELECT
            TRY_CAST(patientunitstayid AS BIGINT) AS patientunitstayid,
            TRY_CAST(apachescore AS DOUBLE)       AS apachescore,
            row_number() OVER (
                PARTITION BY TRY_CAST(patientunitstayid AS BIGINT)
                ORDER BY CASE WHEN apacheversion = 'IVa' THEN 0 ELSE 1 END
            ) AS rn
        FROM read_csv_auto('{path_apr}', header=true, all_varchar=true)
        WHERE TRY_CAST(apachescore AS DOUBLE) >= 0
    )
    WHERE rn = 1
    """



def main():
    ap = argparse.ArgumentParser(description="Extracción eICU 24 h con DuckDB")
    ap.add_argument("--data-path", default=os.environ.get("EICU_PATH", "./eicu-crd"),
                    help="Carpeta con patient.csv.gz, vitalPeriodic.csv.gz, ...")
    ap.add_argument("--out", default="eicu_features_24h.parquet",
                    help="Ruta del Parquet de salida")
    ap.add_argument("--window", type=int, default=WINDOW_MIN,
                    help="Ventana en minutos (por defecto 1440 = 24 h)")
    ap.add_argument("--threads", type=int, default=0,
                    help="Hilos de DuckDB (0 = todos los núcleos)")
    ap.add_argument("--memory-limit", default="4GB",
                    help="Límite de memoria de DuckDB; el resto va a disco")
    ap.add_argument("--temp-dir", default=".duckdb_tmp",
                    help="Directorio de volcado a disco de DuckDB")
    args = ap.parse_args()

    dp = args.data_path
    if not os.path.isdir(dp):
        sys.exit(f"ERROR: no existe la carpeta {dp}")

    t_ini = time.time()
    con = duckdb.connect()
    con.execute(f"PRAGMA memory_limit='{args.memory_limit}'")
    con.execute(f"PRAGMA temp_directory='{args.temp_dir}'")
    if args.threads:
        con.execute(f"PRAGMA threads={args.threads}")

    # 1. Cohorte -----------------------------------------------------------
    t0 = time.time()
    con.execute(sql_cohort(find_table(dp, "patient")))
    n_cohort = con.execute("SELECT count(*) FROM cohort").fetchone()[0]
    n_hosp = con.execute("SELECT count(DISTINCT hospitalid) FROM cohort").fetchone()[0]
    prev = con.execute("SELECT avg(mortality) FROM cohort").fetchone()[0]
    print(f"[1/5] Cohorte: {n_cohort:,} estancias | {n_hosp} hospitales | "
          f"prevalencia={prev:.3f}  ({time.time()-t0:.1f}s)")

    tables = ["cohort"]

    # 2. Vitales -----------------------------------------------------------
    if has_table(dp, "vitalPeriodic"):
        t0 = time.time()
        con.execute(sql_vitals(find_table(dp, "vitalPeriodic"), VITAL_COLS, args.window))
        n = con.execute("SELECT count(*) FROM vit_feat").fetchone()[0]
        print(f"[2/5] Vitales 24 h: {n:,} estancias con datos  ({time.time()-t0:.1f}s)")
        tables.append("vit_feat")
    else:
        print("[2/5] AVISO: vitalPeriodic no encontrada; se omite el bloque de vitales.")

    # 3. Labs --------------------------------------------------------------
    if has_table(dp, "lab"):
        t0 = time.time()
        con.execute(sql_labs(find_table(dp, "lab"), LAB_NAMES, args.window))
        n = con.execute("SELECT count(*) FROM lab_feat").fetchone()[0]
        print(f"[3/5] Labs 24 h: {n:,} estancias con datos  ({time.time()-t0:.1f}s)")
        tables.append("lab_feat")
    else:
        print("[3/5] AVISO: lab no encontrada; se omite el bloque de analíticas.")

    # 4. Severidad APACHE --------------------------------------------------
    if has_table(dp, "apacheApsVar"):
        t0 = time.time()
        con.execute(sql_aps(find_table(dp, "apacheApsVar"), APS_KEEP))
        print(f"[4/5] APS: OK  ({time.time()-t0:.1f}s)")
        tables.append("aps_feat")
    else:
        print("[4/5] AVISO: apacheApsVar no encontrada; se omite severidad APS.")

    if has_table(dp, "apachePatientResult"):
        con.execute(sql_apache_score(find_table(dp, "apachePatientResult")))
        tables.append("apache_score")
        print("      apachescore: OK")
    else:
        print("      AVISO: apachePatientResult no encontrada; sin apachescore.")

    # 5. Ensamblado y volcado ---------------------------------------------
    t0 = time.time()
    joins = "\n    ".join(
        f"LEFT JOIN {t} USING (patientunitstayid)" for t in tables[1:]
    )
    con.execute(f"""
    CREATE OR REPLACE TABLE features AS
    SELECT *,
        CASE
            WHEN admissionheight > 0 AND admissionweight > 0
            THEN admissionweight / pow(admissionheight / 100.0, 2)
        END AS bmi
    FROM cohort
    {joins}
    """)
    out = args.out.replace("\\", "/")
    con.execute(f"COPY features TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")

    ncol = con.execute("SELECT count(*) FROM pragma_table_info('features')").fetchone()[0]
    nrow = con.execute("SELECT count(*) FROM features").fetchone()[0]
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"[5/5] Parquet escrito: {args.out}  ({nrow:,} filas x {ncol} columnas, "
          f"{size_mb:.1f} MB)  ({time.time()-t0:.1f}s)")
    print(f"\nTiempo total: {(time.time()-t_ini)/60:.1f} min")
    con.close()


if __name__ == "__main__":
    main()
