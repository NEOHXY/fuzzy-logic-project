# ================================================================
# MALAYSIAN LOAN APPROVAL RISK ASSESSMENT SYSTEM
# GA-Optimised Fuzzy Membership Functions (Arslan & Kaya, 2001)
#
# Based on:
#   Arslan, A. & Kaya, M. (2001). Determination of fuzzy logic
#   membership functions using genetic algorithms.
#   Fuzzy Sets and Systems, 118, 297–306.
#
# Key difference from clustering approach:
#   The GA chromosome DIRECTLY encodes the boundary parameters
#   (base lengths / breakpoints) of each membership function.
#   Fitness = minimise squared error between fuzzy output and
#   actual loan_status labels on a reference dataset sample.
#   This means the MF SHAPES themselves are evolved, not just
#   cluster centers used as proxies for boundaries.
#
# Inputs (5 variables):
#   1. person_income           — GA-optimised MF parameters
#   2. credit_score            — CTOS fixed 5 categories
#   3. loan_percent_income     — GA-optimised MF parameters
#   4. previous_loan_defaults  — Binary: No=0 / Yes=1
#   5. person_emp_exp          — GA-optimised MF parameters
#
# Output: risk_score → LOW / MEDIUM / HIGH → APPROVE / REVIEW / REJECT
# ================================================================

# pip install pandas numpy matplotlib scikit-fuzzy

# ================================================================
# IMPORTS
# ================================================================

import pandas as pd
import numpy as np
import random
import skfuzzy as fuzz
from skfuzzy import control as ctrl
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import matplotlib.gridspec as gridspec
except Exception:
    gridspec = None

# ================================================================
# LOAD & CLEAN DATASET
# ================================================================

DATASET_PATH = "loan_data.csv"
df = pd.read_csv(DATASET_PATH)

print("=" * 60)
print("  MALAYSIAN LOAN APPROVAL — GA MF OPTIMISATION")
print("  Based on: Arslan & Kaya (2001), Fuzzy Sets & Systems")
print("=" * 60)
print(f"\nDataset loaded: {len(df)} rows")

# Cap extreme data-entry errors at 99th percentile
for col in ['person_income', 'person_emp_exp']:
    cap = df[col].quantile(0.99)
    n   = int((df[col] > cap).sum())
    df[col] = df[col].clip(upper=cap)
    print(f"  Capped {col}: {n} rows → max now {df[col].max():.1f}")

# ================================================================
# EXTRACT VARIABLES
# ================================================================

income      = df['person_income'].values.astype(float)
credit_sc   = df['credit_score'].values.astype(float)
loan_ratio  = df['loan_percent_income'].values.astype(float)
default_raw = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1}).values.astype(float)
emp_exp     = df['person_emp_exp'].values.astype(float)
loan_status = df['loan_status'].values.astype(float)   # 0=approved, 1=rejected

# Universe bounds (used in MF construction)
INC_MIN, INC_MAX   = 0.0,  float(income.max())
RAT_MIN, RAT_MAX   = 0.0,  1.0
EMP_MIN, EMP_MAX   = 0.0,  float(emp_exp.max())

# ================================================================
# REFERENCE SAMPLE — FULL DATASET
# ================================================================
# Per Arslan & Kaya (2001): "The optimum selection necessary for
# system gets better as much as the number of reference values
# increases." We use ALL available rows for maximum fitness
# accuracy. The fitness function is fully vectorised (numpy) so
# evaluating large datasets remains fast.

np.random.seed(0)
# ── Exclude the fixed validation holdout from the GA reference sample ──
# The boot/baseline model must never train on the rows used to gate
# promotions. The v0 baseline ADOPTS this boot model, so if the holdout
# leaked in here, every promotion comparison would be unfair. Drop those
# row positions before building the reference (fitness) sample.
import os as _os_h, json as _json_h
_holdout_path = _os_h.path.join(_os_h.path.dirname(_os_h.path.abspath(__file__)),
                                "validation_holdout_index.json")
_holdout_set = set()
try:
    with open(_holdout_path) as _hf:
        _holdout_set = {int(i) for i in _json_h.load(_hf)}
except Exception as _he:
    print(f"  [engine] holdout index unavailable ({_he}); training on FULL data", flush=True)

ref_idx = np.array([i for i in range(len(loan_status)) if i not in _holdout_set], dtype=int)
np.random.shuffle(ref_idx)          # shuffle for consistent ordering

REF_INCOME  = income[ref_idx]
REF_RATIO   = loan_ratio[ref_idx]
REF_EMP     = emp_exp[ref_idx]
REF_DEFAULT = default_raw[ref_idx]
REF_CREDIT  = credit_sc[ref_idx]
REF_STATUS  = loan_status[ref_idx]    # ground truth

n_approved = int((REF_STATUS == 0).sum())
n_rejected = int((REF_STATUS == 1).sum())
print(f"\nReference sample: {len(ref_idx)} rows (holdout-excluded) "
      f"— {n_approved} approved + {n_rejected} rejected")

# ================================================================
# CHROMOSOME DESIGN  (Arslan & Kaya, 2001 §3)
# ================================================================
#
# Each numeric variable needs 2 breakpoints to define 3 MFs:
#   Low  = trapmf(u_min, u_min, a, b)
#   Mid  = trimf(a, b, c)
#   High = trapmf(b, c, u_max, u_max)
#
# So each variable contributes 3 genes: [a, b, c]
# where  u_min < a < b < c < u_max
#
# 3 variables × 3 genes = 9 genes total per chromosome
#
# Chromosome layout:
#   [ inc_a, inc_b, inc_c,        ← income breakpoints
#     rat_a, rat_b, rat_c,        ← ratio breakpoints
#     emp_a, emp_b, emp_c ]       ← emp_exp breakpoints
#
# Each gene is stored as a NORMALISED float in [0, 1].
# This makes crossover and mutation scale-independent.
# The gene is decoded back to the real domain before
# building the membership function (per paper §3 eq).
#
# ================================================================

N_GENES = 9   # 3 variables × 3 breakpoints each

def decode_chromosome(chrom):
    """
    Decode normalised [0,1] genes → real MF breakpoints.
    Enforces strict ordering within each variable's 3 genes
    using sorted() so a <= b <= c always holds.
    Adds eps margin from universe boundaries to satisfy
    skfuzzy trapmf requirement: u_min <= a <= b <= c <= u_max.

    Returns dict with keys: inc, rat, emp
    Each value is [a, b, c] in real units.
    """
    eps = 1e-4

    def to_real(genes_norm, u_min, u_max):
        # Sort the 3 normalised values then map to real range
        g = sorted(float(x) for x in genes_norm)
        span = u_max - u_min
        a = u_min + g[0] * span
        b = u_min + g[1] * span
        c = u_min + g[2] * span
        # Enforce strict ordering with eps gaps
        a = max(u_min + eps,     min(a, u_max - 3*eps))
        b = max(a      + eps,    min(b, u_max - 2*eps))
        c = max(b      + eps,    min(c, u_max - eps))
        return [a, b, c]

    inc = to_real(chrom[0:3], INC_MIN, INC_MAX)
    rat = to_real(chrom[3:6], RAT_MIN, RAT_MAX)
    emp = to_real(chrom[6:9], EMP_MIN, EMP_MAX)

    return {'inc': inc, 'rat': rat, 'emp': emp}


def encode_params(params):
    """Inverse of decode_chromosome: real MF breakpoints -> normalised [0,1] genes.

    Layout matches decode_chromosome:
      [inc_a, inc_b, inc_c, rat_a, rat_b, rat_c, emp_a, emp_b, emp_c]
    Used to warm-start the retraining GA from the active model's chromosome.
    Because active params are themselves decode_chromosome outputs (already
    ordered + eps-clamped), decode_chromosome(encode_params(p)) reproduces p up
    to the eps boundary clamp.
    """
    def enc(vals, u_min, u_max):
        span = (u_max - u_min) or 1.0
        return [float(min(1.0, max(0.0, (float(x) - u_min) / span))) for x in vals]

    return (enc(params['inc'], INC_MIN, INC_MAX)
            + enc(params['rat'], RAT_MIN, RAT_MAX)
            + enc(params['emp'], EMP_MIN, EMP_MAX))


def build_mfs_from_params(params):
    """
    Build skfuzzy membership function arrays from decoded params.
    Returns dict of numpy arrays (not fuzzy Antecedent objects)
    for fast vectorised fitness evaluation.
    """
    inc_u = np.linspace(INC_MIN, INC_MAX, 300)
    rat_u = np.linspace(RAT_MIN, RAT_MAX, 200)
    emp_u = np.linspace(EMP_MIN, EMP_MAX, 100)

    a, b, c = params['inc']
    inc_mfs = {
        'low'   : fuzz.trapmf(inc_u, [INC_MIN, INC_MIN, a, b]),
        'medium': fuzz.trimf( inc_u, [a, b, c]),
        'high'  : fuzz.trapmf(inc_u, [b, c, INC_MAX, INC_MAX]),
    }

    a, b, c = params['rat']
    rat_mfs = {
        'low'   : fuzz.trapmf(rat_u, [RAT_MIN, RAT_MIN, a, b]),
        'medium': fuzz.trimf( rat_u, [a, b, c]),
        'high'  : fuzz.trapmf(rat_u, [b, c, RAT_MAX, RAT_MAX]),
    }

    a, b, c = params['emp']
    emp_mfs = {
        'junior'     : fuzz.trapmf(emp_u, [EMP_MIN, EMP_MIN, a, b]),
        'mid'        : fuzz.trimf( emp_u, [a, b, c]),
        'experienced': fuzz.trapmf(emp_u, [b, c, EMP_MAX, EMP_MAX]),
    }

    return {
        'inc_u': inc_u, 'inc': inc_mfs,
        'rat_u': rat_u, 'rat': rat_mfs,
        'emp_u': emp_u, 'emp': emp_mfs,
    }


def membership_degree(value, universe, mf_array):
    """Interpolate membership degree of a single value (used by GUI)."""
    return float(np.interp(value, universe, mf_array))


# ================================================================
# FITNESS FUNCTION  (Arslan & Kaya, 2001 §3)
# ================================================================
#
# Paper: "total error = Σ (yi - yGAi)²  where yi is actual output
#         and yGAi is output obtained by GA"
#
# Here:
#   yi    = loan_status (0=approved, 1=rejected)
#   yGAi  = normalised risk score [0,1] produced by the fuzzy
#           system using this chromosome's MF parameters.
#
# Risk score is computed via a simplified weighted rule:
#   risk = w_income*(1-income_high) + w_ratio*ratio_high
#        + w_credit*credit_risk + w_default*default
#        + w_emp*(1-emp_exp_high)
# (Each term drives risk up when conditions worsen.)
# Lower total squared error = better chromosome.
# ================================================================

# CTOS credit score risk mapping (fixed, not in GA)
def ctos_risk(score):
    """Returns a 0-1 risk value based on CTOS category (scalar, used by GUI)."""
    if   score <= 449: return 1.00   # very poor
    elif score <= 549: return 0.75   # poor
    elif score <= 649: return 0.50   # fair
    elif score <= 749: return 0.25   # good
    else:              return 0.00   # excellent

def ctos_risk_vec(scores):
    """Vectorised CTOS risk mapping over a numpy array of scores."""
    risk = np.ones(len(scores), dtype=float)        # default: very poor = 1.0
    risk = np.where(scores > 449, 0.75, risk)
    risk = np.where(scores > 549, 0.50, risk)
    risk = np.where(scores > 649, 0.25, risk)
    risk = np.where(scores > 749, 0.00, risk)
    return risk

# Rule weights (expert-defined, consistent with fuzzy rules)
W_INCOME  = 0.20
W_RATIO   = 0.30
W_CREDIT  = 0.25
W_DEFAULT = 0.15
W_EMP     = 0.10


def fuzzy_risk_score(inc_val, rat_val, emp_val,
                     def_val, cred_val, mfs):
    """
    Compute a scalar risk score [0, 1] for one applicant
    using the current chromosome's MF parameters.
    Legacy helper for quick scalar checks; production web / Telegram scoring uses
    the 30-rule ControlSystemSimulation so the served decision stays rule-based.
    """
    inc_high = membership_degree(inc_val,  mfs['inc_u'], mfs['inc']['high'])
    rat_high = membership_degree(rat_val,  mfs['rat_u'], mfs['rat']['high'])
    emp_high = membership_degree(emp_val,  mfs['emp_u'], mfs['emp']['experienced'])
    cred_risk = ctos_risk(cred_val)

    risk = (
        W_INCOME  * (1.0 - inc_high)    # low income → high risk
      + W_RATIO   * rat_high             # high ratio → high risk
      + W_CREDIT  * cred_risk            # poor credit → high risk
      + W_DEFAULT * float(def_val)       # defaulted   → high risk
      + W_EMP     * (1.0 - emp_high)    # junior      → high risk
    )
    return float(np.clip(risk, 0.0, 1.0))


def fuzzy_risk_score_vec(inc_arr, rat_arr, emp_arr,
                         def_arr, cred_arr, mfs):
    """
    改进版向量化的风险分数计算
    - 同时考虑 low / medium / high
    - 更合理的风险逻辑
    """
    # Income: 收入越高越好 → high 降低风险，low 增加风险
    inc_high = np.interp(inc_arr, mfs['inc_u'], mfs['inc']['high'])
    inc_low  = np.interp(inc_arr, mfs['inc_u'], mfs['inc']['low'])
    
    # Ratio: 贷款占比越高越危险
    rat_high = np.interp(rat_arr, mfs['rat_u'], mfs['rat']['high'])
    rat_low  = np.interp(rat_arr, mfs['rat_u'], mfs['rat']['low'])
    
    # Employment: 经验越丰富越好
    emp_high = np.interp(emp_arr, mfs['emp_u'], mfs['emp']['experienced'])
    emp_low  = np.interp(emp_arr, mfs['emp_u'], mfs['emp']['junior'])
    
    cred_risk = ctos_risk_vec(cred_arr)

    # ==================== 改进的风险计算公式 ====================
    risk = (
        W_INCOME  * (0.6 * inc_low + 0.4 * (1 - inc_high)) +   # 低收入惩罚 + 非高收入惩罚
        W_RATIO   * (0.7 * rat_high + 0.3 * (1 - rat_low)) +   # 高比例强惩罚
        W_CREDIT  * cred_risk +                                 # 信用风险
        W_DEFAULT * def_arr +                                   # 违约历史
        W_EMP     * (0.65 * emp_low + 0.35 * (1 - emp_high))   # 低经验惩罚
    )
    
    return np.clip(risk, 0.0, 1.0)


def fitness(chrom):
    """
    改进后的适应度函数
    - 使用更合理的风险计算
    - 增加形状惩罚，防止 MF 塌陷
    """
    params = decode_chromosome(chrom)
    mfs    = build_mfs_from_params(params)
    
    # 主误差
    y_ga = fuzzy_risk_score_vec(
        REF_INCOME, REF_RATIO, REF_EMP,
        REF_DEFAULT, REF_CREDIT, mfs
    )
    squared_error = float(np.sum((REF_STATUS - y_ga) ** 2))
    
    # ==================== 形状惩罚项 ====================
    penalty = 0.0
    penalty_weight = 5000   # 可调，越大惩罚越强
    
    for var_name, breakpoints in params.items():
        a, b, c = breakpoints
        span = 1.0 if var_name == 'rat' else (100000 if var_name == 'inc' else 30)
        
        width1 = b - a
        width2 = c - b
        total_width = c - a
        
        # 惩罚区间过窄（防止塌陷）
        if width1 < 0.08 * span:
            penalty += (0.08 * span - width1) ** 2
        if width2 < 0.08 * span:
            penalty += (0.08 * span - width2) ** 2
        
        # 惩罚整体过窄（三个点挤在一起）
        if total_width < 0.15 * span:
            penalty += (0.15 * span - total_width) ** 2 * 2
    
    final_fitness = squared_error + penalty_weight * penalty
    
    return final_fitness


# ================================================================
# GENETIC ALGORITHM  (Arslan & Kaya, 2001 §2)
# ================================================================
#
# Parameters (research-backed):
#   pop_size       = 30  — Arslan & Kaya used 10–20; we use 30
#                          for better coverage of 9-gene space
#   generations    = 80  — paper converged by gen 17–24 for
#                          simple systems; we allow more for
#                          the larger 9-gene problem
#   crossover_rate = 1.0 — paper: "crossover probability = 1"
#   mutation       = adaptive (paper: fires when avg fitness
#                          of new gen < old gen)
#   encoding       = real-valued normalised [0,1] genes
# ================================================================

GA_POP        = 30
GA_GENS       = 80
GA_CR         = 1.0      # Arslan & Kaya: crossover probability = 1
GA_MR_BASE    = 0.05     # base per-gene mutation rate
GA_TOURNAMENT = 5
GA_SEED       = 42


def run_ga():
    """
    Run the GA to optimise MF boundary parameters.
    Returns the best chromosome and full history.
    """
    random.seed(GA_SEED)
    np.random.seed(GA_SEED)

    # ── Initialise population ─────────────────────────────────
    # Smart init: seed first individual near known good region
    # (low percentile for income/emp, low ratio is good)
    def smart_individual():
        # Spread genes so MFs cover the universe meaningfully
        a = random.uniform(0.10, 0.35)
        b = random.uniform(0.35, 0.65)
        c = random.uniform(0.65, 0.90)
        return [a, b, c,
                random.uniform(0.10, 0.35),
                random.uniform(0.35, 0.65),
                random.uniform(0.65, 0.90),
                random.uniform(0.10, 0.35),
                random.uniform(0.35, 0.65),
                random.uniform(0.65, 0.90)]

    population = [smart_individual() for _ in range(GA_POP)]

    # ── Evaluate initial population ───────────────────────────
    fit_vals    = [fitness(ind) for ind in population]
    best_idx    = int(np.argmin(fit_vals))
    best_ind    = population[best_idx][:]
    best_fit    = fit_vals[best_idx]
    init_fit    = best_fit
    prev_avg    = float(np.mean(fit_vals))

    fit_history = [best_fit]
    avg_history = [prev_avg]

    print(f"\n  Gen   0 | Best={best_fit:.4f} | Avg={prev_avg:.4f}")

    for gen in range(1, GA_GENS + 1):

        # ── Selection + Crossover (Arslan & Kaya §2.1) ────────
        new_pop = [best_ind[:]]   # elitism: keep best

        while len(new_pop) < GA_POP:
            # Tournament selection
            p1 = min(random.sample(list(range(GA_POP)),
                                   min(GA_TOURNAMENT, GA_POP)),
                     key=lambda i: fit_vals[i])
            p2 = min(random.sample(list(range(GA_POP)),
                                   min(GA_TOURNAMENT, GA_POP)),
                     key=lambda i: fit_vals[i])

            parent1 = population[p1]
            parent2 = population[p2]

            # Two-point crossover (paper §2.1: "two point crossover")
            if random.random() < GA_CR:
                pt1 = random.randint(1, N_GENES - 2)
                pt2 = random.randint(pt1 + 1, N_GENES - 1)
                child1 = (parent1[:pt1] +
                          parent2[pt1:pt2] +
                          parent1[pt2:])
                child2 = (parent2[:pt1] +
                          parent1[pt1:pt2] +
                          parent2[pt2:])
            else:
                child1, child2 = parent1[:], parent2[:]

            new_pop.extend([child1, child2])

        population = new_pop[:GA_POP]

        # ── Evaluate new generation ───────────────────────────
        fit_vals  = [fitness(ind) for ind in population]
        new_avg   = float(np.mean(fit_vals))
        gen_best_i = int(np.argmin(fit_vals))
        gen_best_f = fit_vals[gen_best_i]

        # ── Adaptive mutation (Arslan & Kaya §2.1) ────────────
        # Paper: "mutation probability depends on avg fitness of
        #         new generation < avg fitness of old generation"
        # → mutate more aggressively when improving, less when stagnant
        if new_avg < prev_avg:
            mr = GA_MR_BASE * 0.5    # improving: gentle mutation
        else:
            mr = GA_MR_BASE * 2.0    # stagnant: stronger mutation

        for i in range(1, GA_POP):   # skip elite at index 0
            mutated = False
            ind = population[i][:]
            for j in range(N_GENES):
                if random.random() < mr:
                    ind[j] = float(np.clip(
                        ind[j] + np.random.uniform(-0.12, 0.12),
                        0.0, 1.0))
                    mutated = True
            if mutated:
                population[i] = ind

        # Re-evaluate after mutation
        fit_vals  = [fitness(ind) for ind in population]
        new_avg   = float(np.mean(fit_vals))
        gen_best_i = int(np.argmin(fit_vals))
        gen_best_f = fit_vals[gen_best_i]

        if gen_best_f < best_fit:
            best_fit = gen_best_f
            best_ind = population[gen_best_i][:]

        fit_history.append(best_fit)
        avg_history.append(new_avg)
        prev_avg = new_avg

        if gen % 10 == 0 or gen == 1:
            print(f"  Gen {gen:3d} | Best={best_fit:.4f} | "
                  f"Avg={new_avg:.4f} | MR={mr:.4f}")

    improv = (init_fit - best_fit) / init_fit * 100 if init_fit > 0 else 0
    print(f"\n  GA Complete — Improvement: {improv:.1f}%")
    print(f"  Initial fitness: {init_fit:.4f}")
    print(f"  Final fitness  : {best_fit:.4f}")

    return {
        'best_chromosome': best_ind,
        'best_fitness'   : best_fit,
        'init_fitness'   : init_fit,
        'improvement_pct': improv,
        'fit_history'    : fit_history,
        'avg_history'    : avg_history,
        'generations'    : GA_GENS,
        'pop_size'       : GA_POP,
        'crossover_rate' : GA_CR,
        'mutation_base'  : GA_MR_BASE,
    }


# ================================================================
# RUN GA  (cached — the boot GA is deterministic, so we precompute
# it once and ship boot_ga.json. This removes the dominant cold-start
# cost: on Cloud Run every cold start otherwise re-ran the full
# 80-generation GA (~80s) only to have the result immediately
# overwritten by the stored active model. The cache is keyed on the
# dataset + GA hyperparameters and is recomputed automatically if any
# of those change.)
# ================================================================
import json as _json
import os as _os

_GA_CACHE_FILE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "boot_ga.json")


def _ga_signature():
    return {
        "rows": int(len(REF_STATUS)),
        "pop": int(GA_POP), "gens": int(GA_GENS), "seed": int(GA_SEED),
        "cr": float(GA_CR), "tournament": int(GA_TOURNAMENT), "ref_seed": 0,
        "holdout_excluded": True,
    }


def _load_or_run_ga():
    sig = _ga_signature()
    try:
        if _os.path.exists(_GA_CACHE_FILE):
            with open(_GA_CACHE_FILE) as fh:
                cached = _json.load(fh)
            if cached.get("signature") == sig and cached.get("result", {}).get("best_chromosome"):
                print("[engine] Loaded cached boot GA — skipped "
                      f"{GA_GENS}-gen run (deterministic).", flush=True)
                return cached["result"]
    except Exception as exc:
        print(f"[engine] boot GA cache unreadable ({exc}); recomputing.", flush=True)

    print("\n" + "─" * 60)
    print("  Running GA MF Optimisation (no valid cache)...")
    print(f"  Population={GA_POP}  Generations={GA_GENS}  Seed={GA_SEED}")
    print("─" * 60)
    result = run_ga()
    try:
        with open(_GA_CACHE_FILE, "w") as fh:
            _json.dump({"signature": sig, "result": result}, fh)
        print(f"[engine] Saved boot GA cache → {_os.path.basename(_GA_CACHE_FILE)} "
              "(future starts will be instant).", flush=True)
    except Exception as exc:
        print(f"[engine] could not write boot GA cache: {exc}", flush=True)
    return result


ga_result = _load_or_run_ga()

# ================================================================
# DECODE BEST CHROMOSOME → MF PARAMETERS
# ================================================================

best_params = decode_chromosome(ga_result['best_chromosome'])
best_mfs    = build_mfs_from_params(best_params)

inc_a, inc_b, inc_c = best_params['inc']
rat_a, rat_b, rat_c = best_params['rat']
emp_a, emp_b, emp_c = best_params['emp']

print("\n  Optimised MF Breakpoints:")
print(f"    Income   [a={inc_a:,.0f}  b={inc_b:,.0f}  c={inc_c:,.0f}] RM")
print(f"    Ratio    [a={rat_a:.4f}  b={rat_b:.4f}  c={rat_c:.4f}]")
print(f"    Emp Exp  [a={emp_a:.2f}  b={emp_b:.2f}  c={emp_c:.2f}] yrs")

# ================================================================
# BUILD SKFUZZY CONTROL SYSTEM
# ================================================================

inc_u = np.linspace(INC_MIN, INC_MAX, 300)
rat_u = np.linspace(RAT_MIN, RAT_MAX, 200)
emp_u = np.linspace(EMP_MIN, EMP_MAX, 100)

income_fuzzy  = ctrl.Antecedent(inc_u, 'income')
ratio_fuzzy   = ctrl.Antecedent(rat_u, 'ratio')
emp_fuzzy     = ctrl.Antecedent(emp_u, 'emp_exp')
credit_fuzzy  = ctrl.Antecedent(np.arange(300, 851, 1), 'credit')
default_fuzzy = ctrl.Antecedent(np.arange(0, 2, 1),     'default')
risk          = ctrl.Consequent(np.arange(0, 101, 1),    'risk')

# ── GA-optimised MFs ──────────────────────────────────────────
income_fuzzy['low']    = fuzz.trapmf(inc_u, [INC_MIN, INC_MIN, inc_a, inc_b])
income_fuzzy['medium'] = fuzz.trimf( inc_u, [inc_a, inc_b, inc_c])
income_fuzzy['high']   = fuzz.trapmf(inc_u, [inc_b, inc_c, INC_MAX, INC_MAX])

ratio_fuzzy['low']     = fuzz.trapmf(rat_u, [RAT_MIN, RAT_MIN, rat_a, rat_b])
ratio_fuzzy['medium']  = fuzz.trimf( rat_u, [rat_a, rat_b, rat_c])
ratio_fuzzy['high']    = fuzz.trapmf(rat_u, [rat_b, rat_c, RAT_MAX, RAT_MAX])

emp_fuzzy['junior']      = fuzz.trapmf(emp_u, [EMP_MIN, EMP_MIN, emp_a, emp_b])
emp_fuzzy['mid']         = fuzz.trimf( emp_u, [emp_a, emp_b, emp_c])
emp_fuzzy['experienced'] = fuzz.trapmf(emp_u, [emp_b, emp_c, EMP_MAX, EMP_MAX])

# ── CTOS Credit Score (fixed) ─────────────────────────────────
credit_fuzzy['very_poor'] = fuzz.trapmf(credit_fuzzy.universe, [300, 300, 420, 449])
credit_fuzzy['poor']      = fuzz.trapmf(credit_fuzzy.universe, [420, 449, 520, 549])
credit_fuzzy['fair']      = fuzz.trapmf(credit_fuzzy.universe, [520, 549, 620, 649])
credit_fuzzy['good']      = fuzz.trapmf(credit_fuzzy.universe, [620, 649, 720, 749])
credit_fuzzy['excellent'] = fuzz.trapmf(credit_fuzzy.universe, [720, 750, 850, 850])

# ── Binary Default ────────────────────────────────────────────
default_fuzzy['no']  = fuzz.trimf(default_fuzzy.universe, [0, 0, 0])
default_fuzzy['yes'] = fuzz.trimf(default_fuzzy.universe, [1, 1, 1])

# ── Output ────────────────────────────────────────────────────
risk['low']    = fuzz.trimf(risk.universe, [0,   0,  40])
risk['medium'] = fuzz.trimf(risk.universe, [30, 50,  70])
risk['high']   = fuzz.trimf(risk.universe, [60, 100, 100])

# ================================================================
# FUZZY RULES (30 rules)
# ================================================================

r1  = ctrl.Rule(credit_fuzzy['excellent'] & default_fuzzy['no'],                    risk['low'])
r2  = ctrl.Rule(credit_fuzzy['excellent'] & ratio_fuzzy['low'],                     risk['low'])
r3  = ctrl.Rule(credit_fuzzy['good'] & income_fuzzy['high'] & ratio_fuzzy['low'],   risk['low'])
r4  = ctrl.Rule(credit_fuzzy['good'] & default_fuzzy['no'] & ratio_fuzzy['low'],    risk['low'])
r5  = ctrl.Rule(income_fuzzy['high'] & ratio_fuzzy['low'] & default_fuzzy['no'],    risk['low'])
r6  = ctrl.Rule(credit_fuzzy['good'] & emp_fuzzy['experienced'] & default_fuzzy['no'], risk['low'])
r7  = ctrl.Rule(income_fuzzy['high'] & credit_fuzzy['good'],                        risk['low'])
r8  = ctrl.Rule(emp_fuzzy['experienced'] & income_fuzzy['high'] & ratio_fuzzy['low'], risk['low'])
r9  = ctrl.Rule(credit_fuzzy['excellent'] & income_fuzzy['medium'],                 risk['low'])
r10 = ctrl.Rule(credit_fuzzy['good'] & ratio_fuzzy['low'] & emp_fuzzy['experienced'], risk['low'])

r11 = ctrl.Rule(credit_fuzzy['fair'] & income_fuzzy['medium'] & default_fuzzy['no'], risk['medium'])
r12 = ctrl.Rule(credit_fuzzy['fair'] & ratio_fuzzy['medium'],                       risk['medium'])
r13 = ctrl.Rule(credit_fuzzy['good'] & ratio_fuzzy['medium'] & income_fuzzy['medium'], risk['medium'])
r14 = ctrl.Rule(income_fuzzy['medium'] & ratio_fuzzy['medium'] & default_fuzzy['no'], risk['medium'])
r15 = ctrl.Rule(credit_fuzzy['fair'] & emp_fuzzy['mid'] & default_fuzzy['no'],      risk['medium'])
r16 = ctrl.Rule(income_fuzzy['low'] & credit_fuzzy['fair'] & ratio_fuzzy['low'],    risk['medium'])
r17 = ctrl.Rule(emp_fuzzy['junior'] & income_fuzzy['medium'] & credit_fuzzy['fair'], risk['medium'])
r18 = ctrl.Rule(credit_fuzzy['good'] & ratio_fuzzy['medium'] & default_fuzzy['no'], risk['medium'])
r19 = ctrl.Rule(emp_fuzzy['mid'] & credit_fuzzy['good'] & ratio_fuzzy['medium'],    risk['medium'])

r20 = ctrl.Rule(credit_fuzzy['very_poor'],                                           risk['high'])
r21 = ctrl.Rule(credit_fuzzy['poor'],                                                risk['high'])
r22 = ctrl.Rule(default_fuzzy['yes'],                                                risk['high'])
r23 = ctrl.Rule(credit_fuzzy['poor'] & default_fuzzy['yes'],                        risk['high'])
r24 = ctrl.Rule(credit_fuzzy['very_poor'] & default_fuzzy['yes'],                   risk['high'])
r25 = ctrl.Rule(income_fuzzy['low'] & ratio_fuzzy['high'],                          risk['high'])
r26 = ctrl.Rule(ratio_fuzzy['high'] & credit_fuzzy['fair'],                         risk['high'])
r27 = ctrl.Rule(income_fuzzy['low'] & default_fuzzy['yes'],                         risk['high'])
r28 = ctrl.Rule(emp_fuzzy['junior'] & income_fuzzy['low'] & ratio_fuzzy['high'],    risk['high'])
r29 = ctrl.Rule(credit_fuzzy['poor'] & ratio_fuzzy['high'],                         risk['high'])
r30 = ctrl.Rule(income_fuzzy['low'] & credit_fuzzy['poor'] & emp_fuzzy['junior'],   risk['high'])

risk_ctrl = ctrl.ControlSystem([
    r1,  r2,  r3,  r4,  r5,  r6,  r7,  r8,  r9,  r10,
    r11, r12, r13, r14, r15, r16, r17, r18, r19,
    r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)


# ================================================================
# RUNTIME MODEL ACTIVATION
# ================================================================
def _normalise_model_params(params):
    """Validate and normalise promoted MF params before applying them."""
    if not isinstance(params, dict):
        raise ValueError("Model params must be a dictionary.")
    clean = {}
    for key in ("inc", "rat", "emp"):
        vals = params.get(key)
        if not isinstance(vals, (list, tuple)) or len(vals) != 3:
            raise ValueError(f"Model params must contain {key} as a 3-value list.")
        clean[key] = [float(v) for v in vals]
    return clean


def _build_control_system_from_params(params):
    """Rebuild the 30-rule Mamdani control system using supplied MF breakpoints.

    This keeps the original engine design and rule base unchanged. Only the
    GA-tuned membership-function breakpoints are replaced.
    """
    p = _normalise_model_params(params)

    inc_u_new = np.linspace(INC_MIN, INC_MAX, 300)
    rat_u_new = np.linspace(RAT_MIN, RAT_MAX, 200)
    emp_u_new = np.linspace(EMP_MIN, EMP_MAX, 100)

    income_new = ctrl.Antecedent(inc_u_new, 'income')
    ratio_new = ctrl.Antecedent(rat_u_new, 'ratio')
    emp_new = ctrl.Antecedent(emp_u_new, 'emp_exp')
    credit_new = ctrl.Antecedent(np.arange(300, 851, 1), 'credit')
    default_new = ctrl.Antecedent(np.arange(0, 2, 1), 'default')
    risk_new = ctrl.Consequent(np.arange(0, 101, 1), 'risk')

    ia, ib, ic = p['inc']
    ra, rb, rc = p['rat']
    ea, eb, ec = p['emp']

    income_new['low'] = fuzz.trapmf(inc_u_new, [INC_MIN, INC_MIN, ia, ib])
    income_new['medium'] = fuzz.trimf(inc_u_new, [ia, ib, ic])
    income_new['high'] = fuzz.trapmf(inc_u_new, [ib, ic, INC_MAX, INC_MAX])

    ratio_new['low'] = fuzz.trapmf(rat_u_new, [RAT_MIN, RAT_MIN, ra, rb])
    ratio_new['medium'] = fuzz.trimf(rat_u_new, [ra, rb, rc])
    ratio_new['high'] = fuzz.trapmf(rat_u_new, [rb, rc, RAT_MAX, RAT_MAX])

    emp_new['junior'] = fuzz.trapmf(emp_u_new, [EMP_MIN, EMP_MIN, ea, eb])
    emp_new['mid'] = fuzz.trimf(emp_u_new, [ea, eb, ec])
    emp_new['experienced'] = fuzz.trapmf(emp_u_new, [eb, ec, EMP_MAX, EMP_MAX])

    credit_new['very_poor'] = fuzz.trapmf(credit_new.universe, [300, 300, 420, 449])
    credit_new['poor'] = fuzz.trapmf(credit_new.universe, [420, 449, 520, 549])
    credit_new['fair'] = fuzz.trapmf(credit_new.universe, [520, 549, 620, 649])
    credit_new['good'] = fuzz.trapmf(credit_new.universe, [620, 649, 720, 749])
    credit_new['excellent'] = fuzz.trapmf(credit_new.universe, [720, 750, 850, 850])

    default_new['no'] = fuzz.trimf(default_new.universe, [0, 0, 0])
    default_new['yes'] = fuzz.trimf(default_new.universe, [1, 1, 1])

    risk_new['low'] = fuzz.trimf(risk_new.universe, [0, 0, 40])
    risk_new['medium'] = fuzz.trimf(risk_new.universe, [30, 50, 70])
    risk_new['high'] = fuzz.trimf(risk_new.universe, [60, 100, 100])

    rules_new = [
        ctrl.Rule(credit_new['excellent'] & default_new['no'], risk_new['low']),
        ctrl.Rule(credit_new['excellent'] & ratio_new['low'], risk_new['low']),
        ctrl.Rule(credit_new['good'] & income_new['high'] & ratio_new['low'], risk_new['low']),
        ctrl.Rule(credit_new['good'] & default_new['no'] & ratio_new['low'], risk_new['low']),
        ctrl.Rule(income_new['high'] & ratio_new['low'] & default_new['no'], risk_new['low']),
        ctrl.Rule(credit_new['good'] & emp_new['experienced'] & default_new['no'], risk_new['low']),
        ctrl.Rule(income_new['high'] & credit_new['good'], risk_new['low']),
        ctrl.Rule(emp_new['experienced'] & income_new['high'] & ratio_new['low'], risk_new['low']),
        ctrl.Rule(credit_new['excellent'] & income_new['medium'], risk_new['low']),
        ctrl.Rule(credit_new['good'] & ratio_new['low'] & emp_new['experienced'], risk_new['low']),

        ctrl.Rule(credit_new['fair'] & income_new['medium'] & default_new['no'], risk_new['medium']),
        ctrl.Rule(credit_new['fair'] & ratio_new['medium'], risk_new['medium']),
        ctrl.Rule(credit_new['good'] & ratio_new['medium'] & income_new['medium'], risk_new['medium']),
        ctrl.Rule(income_new['medium'] & ratio_new['medium'] & default_new['no'], risk_new['medium']),
        ctrl.Rule(credit_new['fair'] & emp_new['mid'] & default_new['no'], risk_new['medium']),
        ctrl.Rule(income_new['low'] & credit_new['fair'] & ratio_new['low'], risk_new['medium']),
        ctrl.Rule(emp_new['junior'] & income_new['medium'] & credit_new['fair'], risk_new['medium']),
        ctrl.Rule(credit_new['good'] & ratio_new['medium'] & default_new['no'], risk_new['medium']),
        ctrl.Rule(emp_new['mid'] & credit_new['good'] & ratio_new['medium'], risk_new['medium']),

        ctrl.Rule(credit_new['very_poor'], risk_new['high']),
        ctrl.Rule(credit_new['poor'], risk_new['high']),
        ctrl.Rule(default_new['yes'], risk_new['high']),
        ctrl.Rule(credit_new['poor'] & default_new['yes'], risk_new['high']),
        ctrl.Rule(credit_new['very_poor'] & default_new['yes'], risk_new['high']),
        ctrl.Rule(income_new['low'] & ratio_new['high'], risk_new['high']),
        ctrl.Rule(ratio_new['high'] & credit_new['fair'], risk_new['high']),
        ctrl.Rule(income_new['low'] & default_new['yes'], risk_new['high']),
        ctrl.Rule(emp_new['junior'] & income_new['low'] & ratio_new['high'], risk_new['high']),
        ctrl.Rule(credit_new['poor'] & ratio_new['high'], risk_new['high']),
        ctrl.Rule(income_new['low'] & credit_new['poor'] & emp_new['junior'], risk_new['high']),
    ]

    control = ctrl.ControlSystem(rules_new)
    return {
        'params': p,
        'inc_u': inc_u_new,
        'rat_u': rat_u_new,
        'emp_u': emp_u_new,
        'income_fuzzy': income_new,
        'ratio_fuzzy': ratio_new,
        'emp_fuzzy': emp_new,
        'credit_fuzzy': credit_new,
        'default_fuzzy': default_new,
        'risk': risk_new,
        'rules': rules_new,
        'risk_ctrl': control,
        'risk_sim': ctrl.ControlSystemSimulation(control),
    }


def apply_model_params(params, source='active'):
    """Apply promoted MF params to the live fuzzy engine.

    Used by app.py during boot and after rollback/promote so the served
    decision path uses the promoted artifact without editing the rule base.
    """
    global best_params, best_mfs
    global inc_a, inc_b, inc_c, rat_a, rat_b, rat_c, emp_a, emp_b, emp_c
    global inc_u, rat_u, emp_u
    global income_fuzzy, ratio_fuzzy, emp_fuzzy, credit_fuzzy, default_fuzzy, risk
    global risk_ctrl, risk_sim

    built = _build_control_system_from_params(params)
    clean = built['params']

    best_params = clean
    best_mfs = build_mfs_from_params(clean)

    inc_a, inc_b, inc_c = clean['inc']
    rat_a, rat_b, rat_c = clean['rat']
    emp_a, emp_b, emp_c = clean['emp']

    inc_u = built['inc_u']
    rat_u = built['rat_u']
    emp_u = built['emp_u']
    income_fuzzy = built['income_fuzzy']
    ratio_fuzzy = built['ratio_fuzzy']
    emp_fuzzy = built['emp_fuzzy']
    credit_fuzzy = built['credit_fuzzy']
    default_fuzzy = built['default_fuzzy']
    risk = built['risk']
    risk_ctrl = built['risk_ctrl']
    risk_sim = built['risk_sim']

    # Keep the old rule variable names available for compatibility/debugging.
    for i, rule in enumerate(built['rules'], start=1):
        globals()[f'r{i}'] = rule

    print(f"[model] Applied MF params from {source}: inc={clean['inc']} rat={clean['rat']} emp={clean['emp']}", flush=True)
    return clean
