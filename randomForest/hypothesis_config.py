"""
hypotheses_config.py

Configuration for Random Forest hyperparameter hypotheses testing.
Each hypothesis isolates one variable to test specific research questions.
"""

# Each entry is a named hypothesis + the params that test it.
# Variants within a hypothesis isolate one variable at a time.
HYPOTHESES = [
    {
        "hypothesis": "baseline",
        "rationale": "Reproduce the default config as the reference point. "
                     "All other trials are compared against this.",
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 4,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
    # --- Hypothesis 1: deeper trees memorise soil texture patches ---
    {
        "hypothesis": "depth_control_mild",
        "rationale": "Uncapped trees can memorise individual soil texture patches. "
                     "Capping at depth=20 forces more generalised splits.",
        "params": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 4,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
    {
        "hypothesis": "depth_control_aggressive",
        "rationale": "More aggressive cap (depth=12). Tests whether shallower "
                     "trees lose too much recall on thin wheat stems.",
        "params": {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 4,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
    # --- Hypothesis 2: leaf size controls boundary smoothness ---
    {
        "hypothesis": "smooth_boundary_small_leaf",
        "rationale": "Larger min_samples_leaf prevents splits on tiny pixel clusters, "
                     "producing smoother wheat/soil boundaries and less salt-pepper noise.",
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 16,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
    {
        "hypothesis": "smooth_boundary_large_leaf",
        "rationale": "Even larger leaf size (32). Tests the trade-off between "
                     "smoothness and losing fine wheat detail.",
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 32,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
    # --- Hypothesis 3: feature subset affects vegetation index usage ---
    {
        "hypothesis": "more_features_per_split",
        "rationale": "sqrt(13)~3 features per split may consistently skip ExG/NDI. "
                     "Raising to 6 gives vegetation indices a fairer chance each split.",
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 4,
                   "max_features": 6, "class_weight": "balanced"},
    },
    {
        "hypothesis": "all_features_per_split",
        "rationale": "Using all 13 features per split turns RF into a deterministic "
                     "boosted tree. Tests whether feature diversity actually helps.",
        "params": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 4,
                   "max_features": None, "class_weight": "balanced"},
    },
    # --- Hypothesis 4: more trees reduce variance on hard images ---
    {
        "hypothesis": "more_trees",
        "rationale": "Hard images (dense occlusion, low contrast) have high prediction "
                     "variance. 200 trees should reduce this at the cost of training time.",
        "params": {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 4,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
    # --- Hypothesis 5: combine best depth + leaf insights ---
    {
        "hypothesis": "combined_depth_and_leaf",
        "rationale": "If depth=20 and leaf=16 individually reduce noise, combining "
                     "them may give cleaner predictions without sacrificing recall.",
        "params": {"n_estimators": 100, "max_depth": 20, "min_samples_leaf": 16,
                   "max_features": "sqrt", "class_weight": "balanced"},
    },
]


# def get_hypothesis_names():
#     """Return list of all hypothesis names"""
#     return [h["hypothesis"] for h in HYPOTHESES]

# def get_hypothesis_params(hypothesis_name):
#     """Get parameters for a specific hypothesis by name"""
#     for h in HYPOTHESES:
#         if h["hypothesis"] == hypothesis_name:
#             return h["params"]
#     raise ValueError(f"Hypothesis '{hypothesis_name}' not found")

# def get_hypothesis_rationale(hypothesis_name):
#     """Get rationale for a specific hypothesis by name"""
#     for h in HYPOTHESES:
#         if h["hypothesis"] == hypothesis_name:
#             return h["rationale"]
#     raise ValueError(f"Hypothesis '{hypothesis_name}' not found")