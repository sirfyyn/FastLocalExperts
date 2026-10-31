//! A time stamp for file names, `YYYYMMDD-HHMMSS` (UTC).
pub fn stamp() -> String {
    let t = fle_core::report::utc_now(); // 2026-10-31T09:41:23+00:00
    format!(
        "{}{}{}-{}{}{}",
        &t[0..4],
        &t[5..7],
        &t[8..10],
        &t[11..13],
        &t[14..16],
        &t[17..19]
    )
}
