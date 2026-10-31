//! What a catalog download would fetch (addresses, destinations, sizes, checksums) against the Python Studio, and the fit estimate.
#![cfg(unix)]
mod harness;
use fle_api::catalog::{by_id, catalog, fit, models};
use fle_api::server::Studio;
use harness::studio_src;
use serde_json::{json, Value};
use std::path::Path;

const DRIVER: &str = r#"
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from fle_studio import catalog, hf
sc = json.load(open(sys.argv[2]))
out = {"files": [], "fit": []}
root = Path("/m")
for ident in sc["ids"]:
    for with_mtp in (True, False):
        model = catalog.BY_ID.get(ident)
        if not model:
            out["files"].append(None); continue
        refs = list(model.files)
        if with_mtp and model.mtp:
            refs += [catalog.MTP_HEAD, catalog.MTP_VOCAB]
        out["files"].append({"name": model.name, "files": [{"url": hf.resolve_url(r.repo, r.path), "dest": str(root / r.repo / r.path), "size": r.size, "sha256": r.sha256} for r in refs]})
for s in sc["systems"]:
    for size in sc["sizes"]:
        out["fit"].append(catalog.fit(size, s))
out["catalog"] = catalog.catalog(sc["systems"][0])
print(json.dumps(out))
"#;

#[test]
fn catalog_downloads_and_fit_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found: skipped");
        return;
    };
    let g = 1u64 << 30;
    let systems = vec![
        json!({"memory": {"total": 128 * g}, "gpus": [{"memory_total": 32 * g}]}),
        json!({"memory": {"total": 16 * g}, "gpus": []}),
        json!({"memory": {"total": 64 * g}, "gpus": [{"memory_total": 8 * g}, {"memory_total": 24 * g}]}),
        json!({"memory": {"total": 16 * g}, "gpus": [{"unified": true, "memory_total": 16 * g, "metal_limit": 11 * g}]}),
        json!({"memory": {"total": 192 * g}, "gpus": [{"unified": true, "memory_total": 192 * g, "metal_limit": 144 * g}]}),
        json!({"memory": {"total": 0}, "gpus": []}),
    ];
    let sizes: Vec<u64> = vec![0, g, 5 * g + 12345, 20 * g, 60 * g, 100 * g, 250 * g];
    let ids: Vec<String> = models()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .chain(["nope".to_string()])
        .collect();
    let d = harness::tmp("catfiles");
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    std::fs::write(
        d.join("in.json"),
        json!({"ids": ids, "systems": systems, "sizes": sizes}).to_string(),
    )
    .unwrap();
    let out = std::process::Command::new(
        std::env::var("FLE_STUDIO_PYTHON").unwrap_or_else(|_| "python3".into()),
    )
    .arg(d.join("driver.py"))
    .arg(&src)
    .arg(d.join("in.json"))
    .output()
    .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let py: Value = serde_json::from_slice(&out.stdout).unwrap();
    let mut n = 0;
    for ident in &ids {
        for with_mtp in [true, false] {
            let want = &py["files"][n];
            match Studio::catalog_files(ident, with_mtp, Path::new("/m")) {
                Ok((name, files)) => assert_eq!(
                    json!({"name": name, "files": files}),
                    *want,
                    "{ident} mtp={with_mtp}"
                ),
                Err(e) => assert!(
                    want.is_null() && e.0 == 404 && e.1 == "Unknown catalog model",
                    "{ident}"
                ),
            }
            n += 1;
        }
    }
    let mut k = 0;
    for s in &systems {
        for size in &sizes {
            assert_eq!(fit(*size as i64, s), py["fit"][k], "fit {size} on {s}");
            k += 1;
        }
    }
    let ours: Value = catalog(&systems[0])
        .into_iter()
        .map(Value::Object)
        .collect();
    assert!(ours.to_string().len() > 100);
    assert_eq!(ours, py["catalog"]);
    assert!(by_id("fle-iq3_m").is_some());
    eprintln!("compared with the Python Studio: {n} download plans, {k} fit verdicts and the whole catalog identical");
}
