//! Short facts about a plan, for the chip row above the notes in Engine > Advanced. Port of `fle_studio/planview.py`.
//!
//! Pure: it reads the public form of a plan, the model record and two settings, and returns `[{"label", "value", "tone"}]`
//! with tone `ok` (as wanted), `warn` (something is not as asked, or a fallback) or `info` (a plain fact).
use crate::launch::commas;
use serde_json::{json, Value};

/// The value of the first of these flags, in `--flag value` or `--flag=value` form.
fn value(args: &[String], names: &[&str]) -> Option<String> {
    for (i, arg) in args.iter().enumerate() {
        for name in names {
            if arg == name && i + 1 < args.len() {
                return Some(args[i + 1].clone());
            }
            if let Some(v) = arg.strip_prefix(&format!("{name}=")) {
                return Some(v.to_string());
            }
        }
    }
    None
}

/// Python's `int(text)` for the digits of an argument; `None` when it is not a whole number.
fn int(text: Option<String>) -> Option<i64> {
    text.and_then(|t| t.trim().parse::<i64>().ok())
}

/// What the command really does: `mtp`, `ngram-mod`, `ngram-simple` or another --spec-type, else `none`.
pub fn effective_speculation(plan: &Value) -> String {
    let args = args_of(plan);
    let kind = value(&args, &["--spec-type"])
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    if kind.is_empty() || kind == "none" {
        "none".into()
    } else if kind == "draft-mtp" {
        "mtp".into()
    } else {
        kind
    }
}

fn args_of(plan: &Value) -> Vec<String> {
    plan["args"]
        .as_array()
        .map(|a| a.iter().map(crate::settings::py_str).collect())
        .unwrap_or_default()
}

/// The chips for one plan. `spec_requested` and `mtp_on` are the saved Speculation and MTP settings.
pub fn summary(
    plan: &Value,
    model: Option<&Value>,
    spec_requested: &str,
    mtp_on: bool,
) -> Vec<Value> {
    let args = args_of(plan);
    let mut chips: Vec<Value> = vec![];
    let mut add = |label: &str, value: &str, tone: &str| {
        chips.push(json!({"label": label, "value": value, "tone": tone}))
    };
    let fallback = plan.get("fallback").is_some_and(crate::settings::py_truthy);
    let engine = plan["engine"].as_str().unwrap_or("");
    if engine == "fle" {
        add("Engine", "FLE", "info");
    } else if engine == "upstream" {
        add(
            "Engine",
            "Upstream llama.cpp",
            if fallback { "warn" } else { "info" },
        );
    } else {
        // a build of your own: what it is shows in the command (the FLE engine's flags and settings)
        let fle = plan["env"]
            .as_object()
            .is_some_and(|e| e.keys().any(|k| k.starts_with("FLE_")))
            || args.iter().any(|a| a.starts_with("--moe-expert-cache"));
        add(
            "Engine",
            if fle {
                "FLE (own build)"
            } else {
                "llama.cpp (own build)"
            },
            "info",
        );
    }
    let spec = effective_speculation(plan);
    let asked = if spec_requested.is_empty() {
        "auto".to_string()
    } else {
        spec_requested.to_lowercase()
    };
    let own_auto = plan
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("auto")
        == "auto";
    if spec != "none" {
        add("Speculation", &spec, "ok");
    } else if asked != "auto" && asked != "none" && own_auto {
        add("Speculation", "none (not as asked)", "warn");
    } else {
        add("Speculation", "none", "info");
    }
    let mtp = model
        .map(|m| m["mtp"].clone())
        .filter(|m| m.is_object())
        .unwrap_or(json!({}));
    if mtp["supported"].as_bool() == Some(true) {
        let have = mtp["head"].is_string() && mtp["vocab"].is_string();
        if have {
            add("MTP files", "present", "ok");
        } else if mtp_on && (asked == "auto" || asked == "mtp") && own_auto {
            add("MTP files", "missing", "warn");
        } else {
            add("MTP files", "missing (not needed)", "info");
        }
    }
    if let Some(context) = int(value(&args, &["-c", "--ctx-size"])).filter(|c| *c > 0) {
        let parallel = int(value(&args, &["--parallel", "-np"]))
            .unwrap_or(1)
            .max(1);
        let unified = args.iter().any(|a| a == "--kv-unified" || a == "-kvu");
        let (ctx, per_slot) = crate::launch::effective_context(context, parallel, unified);
        add("Context", &format!("{} tokens", commas(ctx)), "info");
        if parallel > 1 {
            let share = if unified {
                format!("up to {} (shared)", commas(per_slot))
            } else {
                commas(per_slot)
            };
            add("Per slot", &share, "info");
        } else {
            add("Per slot", &commas(per_slot), "info");
        }
        add("Slots", &parallel.to_string(), "info");
    }
    add(
        "Fallback",
        if fallback { "yes" } else { "no" },
        if fallback { "warn" } else { "ok" },
    );
    chips
}
