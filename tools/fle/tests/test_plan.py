from __future__ import annotations

import dataclasses

import pytest
from fle import config, gguf, plan
from fle.util import FleError, GiB, MiB
from helpers import FLE_ENGINE, UPSTREAM_ENGINE, settings


def build(model, machine, profile="auto", **kw):
    extra = kw.pop("settings", {})
    return plan.build(settings(profile=profile, **extra), model, engine_path=kw.pop("engine_path", FLE_ENGINE),
                      model_path=str(model.first), **kw)


def flag(args, name):
    return args[args.index(name) + 1]


@pytest.mark.parametrize("profile", config.PROFILES)
def test_every_profile_builds_on_a_fle_engine(profile, moe_model, machine):
    p = build(moe_model, machine, profile)
    assert p.profile == profile
    assert p.binary == FLE_ENGINE
    assert flag(p.args, "-m") == str(moe_model.first)
    assert flag(p.args, "-c") == str(p.context)
    assert p.context == plan._context(profile, moe_model, 0) == min(plan.PROFILE_CONTEXT[profile], 4096)


def test_off_profile_is_a_plain_baseline(moe_model, machine):
    p = build(moe_model, machine, "off")
    assert p.cache_mib == 0
    assert p.fit_target_mib == 0
    assert not any(k.startswith("FLE_") for k in p.env)
    for forbidden in ("-fit", "--fit-target", "-ctk", "--moe-expert-cache-inserts"):
        assert forbidden not in p.args
    assert p.spec == "none"


def test_off_profile_rejects_ngram(moe_model, machine):
    with pytest.raises(FleError, match="reproducibility baseline"):
        build(moe_model, machine, "off", spec="ngram-mod")


def test_auto_profile_sets_cache_and_env(moe_model, machine):
    p = build(moe_model, machine, "auto")
    assert p.cache_mib > 0
    assert p.env["FLE_MOE_CACHE_BUDGET_MIB"] == str(p.cache_mib)
    assert p.env["FLE_MOE_CACHE_BIAS"] == "0.001"
    assert "FLE_MOE_ADMIT_FREQ" not in p.env
    assert flag(p.args, "-ctk") == "q8_0"
    assert flag(p.args, "--fit-target") == str(plan.DEFAULT_FIT_TARGET_MIB)
    assert "-cmoe" in p.args and "-fit" in p.args


def test_cache_budget_arithmetic(moe_model, machine):
    machine.set_gpu_free(30 * GiB)
    p = build(moe_model, machine, "auto")
    est = p.estimate
    expected = (30 * GiB - moe_model.gpu_bytes - est["kv_cache"] - plan.ENGINE_OVERHEAD_MIB * MiB
                - plan.DEFAULT_FIT_TARGET_MIB * MiB) // MiB
    assert p.cache_mib == expected
    assert est["base_mib"] == expected


def test_balanced_leaves_part_of_the_cache_free_and_max_takes_all(moe_model, machine):
    auto = build(moe_model, machine, "auto")
    balanced = build(moe_model, machine, "balanced")
    mx = build(moe_model, machine, "max")
    assert balanced.fit_target_mib == 1536
    assert mx.fit_target_mib == 512
    assert balanced.cache_mib < auto.cache_mib < mx.cache_mib
    assert balanced.env["FLE_MOE_ADMIT_FREQ"] == "2000"
    assert mx.env["FLE_MOE_ADMIT_FREQ"] == "2000"
    assert mx.context == min(plan.PROFILE_CONTEXT["max"], 4096)


def test_explicit_fit_target_beats_the_profile_default(moe_model, machine):
    p = build(moe_model, machine, "balanced", settings={"fit_target_mib": 2048})
    assert p.fit_target_mib == 2048


def test_nvme_profile_small_cache_mmap_no_warmup(moe_model, machine):
    p = build(moe_model, machine, "nvme")
    assert p.cache_mib <= 2048
    assert "--no-warmup" in p.args
    assert "--load-mode" not in p.args  # nvme never pins
    assert p.fit_target_mib == 1024
    assert any("memory-mapped" in n for n in p.notes)


def test_debug_profile_adds_diagnostics(moe_model, machine):
    p = build(moe_model, machine, "debug")
    assert flag(p.args, "--log-verbosity") == "4"
    assert p.env["FLE_TIME_PROFILE"] == "20000"
    assert p.env["FLE_NAN_GUARD"] == "1"


def test_explicit_cache_above_what_fits_warns(moe_model, machine):
    p = build(moe_model, machine, "auto", settings={"gpu_cache_mib": 10_000_000})
    assert p.cache_mib == 10_000_000
    assert any("larger than" in w for w in p.warnings)


def test_no_gpu_memory_left_warns(moe_model, machine):
    machine.set_gpu_free(1 * GiB)
    p = build(moe_model, machine, "auto")
    assert p.cache_mib == 0
    assert any("No GPU memory is left" in w for w in p.warnings)


def test_dense_model_without_gpu_memory_does_not_talk_about_experts(make_model, machine):
    """A dense model has no experts: no 'experts will run on the CPU' warning, only the note that the cache stays empty."""
    machine.set_gpu_free(1 * GiB)
    p = build(make_model(experts=0, layers=2, sidecars=False), machine, "auto")
    assert not any("expert" in w.lower() for w in p.warnings)
    assert any("Dense model" in n for n in p.notes)


def test_pinned_load_setting(moe_model, machine):
    on = build(moe_model, machine, "auto", settings={"pinned_load": "on"})
    off = build(moe_model, machine, "auto", settings={"pinned_load": "off"})
    assert flag(on.args, "--load-mode") == "none"
    assert "--load-mode" not in off.args


def test_fle_profile_needs_an_fle_engine(moe_model, machine):
    with pytest.raises(FleError, match="not an FLE engine"):
        build(moe_model, machine, "auto", engine_path=UPSTREAM_ENGINE)


def test_upstream_engine_with_off_profile_uses_cpu_moe(moe_model, machine):
    p = build(moe_model, machine, "off", engine_path=UPSTREAM_ENGINE)
    assert "--cpu-moe" in p.args
    assert "-cmoe" not in p.args


def test_fle_engine_without_gpu_fails(moe_model, machine):
    machine.gpus = []
    with pytest.raises(FleError, match="needs an NVIDIA GPU"):
        build(moe_model, machine, "auto")


def test_unknown_profile_and_missing_engine(moe_model, machine):
    with pytest.raises(FleError, match="Unknown profile"):
        build(moe_model, machine, "turbo")
    machine.default_engine = ""
    with pytest.raises(FleError, match="No llama-server"):
        plan.build(settings(), moe_model, model_path=str(moe_model.first))
    with pytest.raises(FleError, match="does not start"):
        plan.build(settings(), moe_model, engine_path="/fake/nothing", model_path=str(moe_model.first))


def test_gpu_selection(moe_model, machine):
    with pytest.raises(FleError, match="No GPU with index 5"):
        build(moe_model, machine, "auto", gpu_ids="5")
    p = build(moe_model, machine, "auto", gpu_ids="0")
    assert p.cache_mib > 0


def test_context_is_capped_by_the_model(make_model, machine):
    model = make_model(experts=8, context=2048)
    p = build(model, machine, "auto", settings={"context": 100_000})
    assert p.context == 2048
    assert any("capped" in n for n in p.notes)


def test_warnings_for_model_bigger_than_memory(moe_model, machine):
    big = dataclasses.replace(moe_model, size=500 * GiB, expert_bytes=400 * GiB)
    p = build(big, machine, "auto")
    assert any("larger than RAM plus VRAM" in w for w in p.warnings)
    assert any("Experts need" in w for w in p.warnings)


def test_incomplete_model_warns(make_model, machine):
    model = make_model(experts=8, truncate_data_by=64)
    p = build(model, machine, "auto")
    assert any("missing or incomplete" in w for w in p.warnings)


def test_busy_port_warns(moe_model, machine):
    machine.port_is_free = False
    p = build(moe_model, machine, "auto", settings={"port": 9999})
    assert any("Port 9999" in w for w in p.warnings)


def test_threads(moe_model, machine):
    machine.cpu = {"name": "x", "physical": 8, "logical": 16}
    assert build(moe_model, machine).threads == 8  # SMT: one per core
    machine.cpu = {"name": "x", "physical": 8, "logical": 8}
    assert build(moe_model, machine).threads == 7  # no SMT: leave one core
    assert build(moe_model, machine, settings={"threads": 3}).threads == 3


def test_dense_model_notes(make_model, machine):
    model = make_model(experts=0)
    p = build(model, machine, "max")
    assert any("Dense model" in n for n in p.notes)
    assert any("Max does not add an expert cache" in n for n in p.notes)


def test_extra_args_env_override_and_settings_env(moe_model, machine):
    p = build(moe_model, machine, "auto", settings={"env": {"FLE_X": "1", "FLE_MC_OVERLAP": "9"}},
              env_override={"FLE_X": "2"}, extra_args=["--foo", "bar"])
    assert p.env["FLE_X"] == "2"
    assert p.env["FLE_MC_OVERLAP"] == "9"
    assert p.args[-2 - 2: -2] == ["--foo", "bar"]
    assert p.args[-2:] == ["--alias", p.alias]


def test_plan_serialises(moe_model, machine):
    d = build(moe_model, machine, "auto").as_dict()
    assert d["profile"] == "auto" and d["model"]["arch"] == "testarch"
    assert "llama-server" in d["command"]


# ---- speculation -----------------------------------------------------------------------------------------


def test_auto_uses_mtp_only_with_both_sidecars(make_model, machine):
    with_side = make_model(experts=8, sidecars=True)
    p = build(with_side, machine, "auto")
    assert p.spec == "mtp"
    assert "draft-mtp" in p.args and "-md" in p.args
    assert p.env["FLE_MTP_VOCAB"].endswith("fle-mtp-vocab-32k.txt")
    without = make_model(experts=8)
    q = build(without, machine, "auto")
    assert q.spec == "none"
    assert "--spec-type" not in q.args
    assert any("MTP is off" in n for n in q.notes)


def test_mtp_is_not_used_in_nvme_or_off(make_model, machine):
    model = make_model(experts=8, sidecars=True)
    for profile in ("off", "nvme"):
        p = build(model, machine, profile)
        assert p.spec == "none", profile
    p = build(model, machine, "nvme")
    assert any("MTP stays off in the NVMe profile" in n for n in p.notes)


def test_half_a_sidecar_pair_warns_and_stays_off(make_model, machine):
    model = make_model(experts=8)
    p = build(model, machine, "auto", mtp_head="/tmp/some-head.gguf")
    assert p.spec == "none"
    assert any("needs both a head file and a vocabulary" in w for w in p.warnings)


def test_explicit_head_and_vocab_paths(make_model, machine):
    model = make_model(experts=8)
    p = build(model, machine, "auto", mtp_head="/h/head.gguf", mtp_vocab="/h/vocab.txt")
    assert p.spec == "mtp"
    assert flag(p.args, "-md") == "/h/head.gguf"


def test_ngram_modes_on_engine_with_and_without_spec_type(make_model, machine):
    model = make_model(experts=8)
    p = build(model, machine, "auto", spec="ngram-mod")
    assert p.spec == "ngram-mod"
    assert flag(p.args, "--spec-type") == "ngram-mod"
    assert "-md" not in p.args
    machine.engines[FLE_ENGINE]["has_spec_mtp"] = False
    with pytest.raises(FleError, match="--spec-type"):
        build(model, machine, "auto", spec="ngram-simple")


def test_spec_none_beats_sidecars(make_model, machine):
    p = build(make_model(experts=8, sidecars=True), machine, "auto", spec="none")
    assert p.spec == "none"


def test_spec_mtp_in_nvme_warns(make_model, machine):
    p = build(make_model(experts=8, sidecars=True), machine, "nvme", spec="mtp")
    assert p.spec == "none"
    assert any("not used in the nvme profile" in w for w in p.warnings)


def test_spec_from_settings_and_cli_argument(make_model, machine):
    model = make_model(experts=8)
    assert build(model, machine, "auto", settings={"spec": "ngram-mod"}).spec == "ngram-mod"
    # the explicit argument beats the setting
    assert build(model, machine, "auto", settings={"spec": "ngram-mod"}, spec="none").spec == "none"


def test_mtp_false_in_settings_turns_auto_off(make_model, machine):
    p = build(make_model(experts=8, sidecars=True), machine, "auto", settings={"mtp": False})
    assert p.spec == "none"


# ---- helpers ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "args,expected",
    [
        ([], 0),
        (["--parallel", "4"], 4),
        (["--parallel=3"], 3),
        (["-np", "2"], 0),  # only the long form is understood
        (["--parallel"], 0),
        (["--parallel", "many"], 0),
        (["--foo", "--parallel", "2", "--bar"], 2),
    ],
)
def test_extra_parallel(args, expected):
    assert plan._extra_parallel(args) == expected


def _model_named(name, first="/m/some-file-IQ3_M.gguf"):
    return type("M", (), {"name": name, "first": first})()


@pytest.mark.parametrize(
    "name,path,alias",
    [
        ("Qwen3.8 Flash", "/m/x.gguf", "Qwen3.8 Flash"),
        ("IQ3_M", "/m/Real-Name-IQ3_M.gguf", "Real-Name-IQ3_M"),
        ("Q4_K_M", "/m/Real-Name.gguf", "Real-Name"),
        ("BF16", "/m/a.gguf", "a"),
        ("mxfp4", "/m/b.gguf", "b"),
        ("", "/m/c.gguf", "c"),
        ("  ", "", "some-file-IQ3_M"),
    ],
)
def test_stable_alias(name, path, alias):
    assert plan._stable_alias(_model_named(name), path) == alias


def test_alias_ends_up_in_the_command(make_model, machine):
    model = make_model(experts=8, name="Q8_0", filename="Real-Name-Q8_0.gguf")
    p = build(model, machine, "auto")
    assert p.alias == "Real-Name-Q8_0"
    assert p.args[-2:] == ["--alias", "Real-Name-Q8_0"]


# ---- slots and the KV note -------------------------------------------------------------------------------


def test_single_slot_default(moe_model, machine):
    p = build(moe_model, machine, "auto")
    assert (p.slots, p.kv_unified, p.context_per_slot) == (1, False, p.context)
    assert not any("slot" in w for w in p.warnings)


def test_parallel_without_unified_kv_divides_the_context_and_warns(make_model, machine):
    model = make_model(experts=8, context=100_000)
    p = build(model, machine, "auto", extra_args=["--parallel", "4"], settings={"context": 40_000})
    assert p.slots == 4
    assert not p.kv_unified
    assert p.context_per_slot == 10_000
    assert any("--kv-unified" in w for w in p.warnings)


def test_parallel_with_unified_kv_keeps_the_total(make_model, machine):
    model = make_model(experts=8, context=100_000)
    p = build(model, machine, "auto", extra_args=["--parallel=2", "--kv-unified"], settings={"context": 40_000})
    assert p.slots == 2 and p.kv_unified
    assert p.context_per_slot == 40_000
    assert any("share one unified KV cache" in n for n in p.notes)
    assert not any("--kv-unified" in w for w in p.warnings)
