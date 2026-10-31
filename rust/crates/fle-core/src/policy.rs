//! The FLE engine's policy variables, taken from FLE Studio's launch.py (REFERENCE_ENV minus the per-machine ones).
//! The user never types these: `plan` sets them for the FLE engine.
pub const FLE_POLICY_ENV: &[(&str, &str)] = &[
    ("FLE_MOE_CACHE_BUDGET_FIT", "1"),
    ("FLE_MOE_PROFILE_SAVE_SEC", "600"),
    ("FLE_MOE_ADMIT_FREQ", "2000"),
    ("FLE_MOE_EVICT_OLDEST", "32"),
    ("FLE_MOE_CACHE_BIAS", "0.001"),
    ("FLE_MOE_OBS_MAXTOK", "8"),
    ("FLE_MC_OVERLAP", "2"),
    ("FLE_HC_FUSED", "1"),
    ("FLE_CKPT_TAIL", "512"),
    ("FLE_MTP_VOCAB_N", "32768"),
    ("FLE_MTP_PENALTIES", "1"),
    ("FLE_PREFILL_LEAN", "1"),
    ("FLE_SLOT_EARLY_SAVE", "1"),
    ("FLE_CACHE_GUARD_SLOT", "0"),
    ("FLE_UBATCH_LEND", "1"),
    ("FLE_UBATCH_PREFETCH", "1"),
    ("FLE_UBATCH_PREFETCH_SLOTS", "3"),
];
