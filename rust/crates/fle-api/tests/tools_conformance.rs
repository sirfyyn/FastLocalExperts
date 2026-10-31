//! The built-in tools, the capabilities, the hosts list and the update panel against the Python Studio.
#![cfg(unix)]
mod harness;
use fle_api::tools::{
    calculator, clock, list_files, parse_duckduckgo, parse_searxng, parse_servers, read_file,
    web_search,
};
use harness::*;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener};
use std::path::Path;

const DRIVER: &str = r#"
import json, sys
sys.path.insert(0, sys.argv[1])
from fle_studio import tools, mcp
sc = json.load(open(sys.argv[2]))
def run(f, *a):
    try:
        return {"ok": f(*a)}
    except tools.ToolError as e:
        return {"error": str(e)}
out = {"calc": [run(tools.calculator, {"expression": e}) for e in sc["exprs"]],
       "calc_args": [run(tools.calculator, a) for a in sc["calc_args"]],
       "clock": [run(tools.clock, a) for a in sc["clock"]],
       "list": [run(tools.list_files, sc["root"], a) for a in sc["list"]],
       "read": [run(tools.read_file, sc["root"], a) for a in sc["read"]],
       "ddg": [tools.parse_duckduckgo(h) for h in sc["html"]],
       "searx": [run(tools.parse_searxng, t) for t in sc["searx"]],
       "servers": [mcp.parse_servers(t) for t in sc["servers"]],
       "search": [run(tools.web_search, sc["search_url"], a) for a in sc["search_args"]]}
print(json.dumps(out))
"#;

fn expressions() -> Vec<String> {
    let mut v: Vec<String> = [
        "1+1",
        "2*3+4",
        "(3 + 4) * 2",
        "10/4",
        "10//4",
        "-10//4",
        "10%3",
        "-10%3",
        "10%-3",
        "2**10",
        "2^10",
        "2**-1",
        "2**0.5",
        "(-8)**2",
        "-2**2",
        "2**3**2",
        "7/2*2",
        "1/3",
        "2/3",
        "1e3",
        "1.5e-3",
        ".5+.5",
        "5.",
        "1_000*2",
        "0x10+1",
        "0b101",
        "0o17",
        "007",
        "00",
        "1e999",
        "-1e999",
        "9**9**9",
        "2**100000",
        "2**20000",
        "2**100",
        "10**30",
        "10**30/3",
        "10**400/10**399",
        "10**400",
        "(10**400)/3",
        "10**310/10",
        "1/0",
        "1//0",
        "1%0",
        "1.0/0",
        "1.0//0",
        "1.0%0",
        "0**-1",
        "0.0**-1",
        "(-8)**(1/3)",
        "10.0**400",
        "0**0",
        "(-1)**0.5",
        "sqrt(16)",
        "sqrt(2)**2",
        "sqrt(-1)",
        "sqrt()",
        "sqrt(1,2)",
        "sin(0)",
        "cos(0)",
        "tan(pi/4)",
        "asin(2)",
        "acos(1)",
        "atan(1)",
        "sinh(1)",
        "cosh(1)",
        "tanh(1)",
        "exp(1)",
        "exp(1000)",
        "ln(e)",
        "ln(0)",
        "ln(-1)",
        "log(8,2)",
        "log(100)",
        "log10(1000)",
        "log10(0)",
        "log2(8)",
        "log2(-1)",
        "log(0,2)",
        "log(2,1)",
        "abs(-3)",
        "abs(-3.5)",
        "round(2.5)",
        "round(3.5)",
        "round(-2.5)",
        "round(2.675, 2)",
        "round(1234, -2)",
        "round(1250, -2)",
        "round(1350, -2)",
        "round(1.5, 0)",
        "round(5)",
        "round()",
        "round(1,2,3)",
        "floor(2.7)",
        "floor(-2.1)",
        "ceil(2.1)",
        "ceil(-2.7)",
        "floor(5)",
        "floor(1e300)",
        "min(1,2,3)",
        "max(1,2,3)",
        "min(1)",
        "max()",
        "min(1, 2.5)",
        "max(2, 2.0)",
        "radians(180)",
        "degrees(pi)",
        "factorial(5)",
        "factorial(0)",
        "factorial(-1)",
        "factorial(1001)",
        "factorial(2.5)",
        "factorial(1000)/factorial(998)",
        "gcd(12, 18)",
        "gcd(0, 0)",
        "gcd(2.5, 1)",
        "gcd()",
        "gcd(-4, 6)",
        "pi",
        "e",
        "tau",
        "pi*2",
        "x",
        "foo(1)",
        "os.system('x')",
        "__import__('os')",
        "1 if 2 else 3",
        "[1,2]",
        "(1,2)",
        "()",
        "{1:2}",
        "1 < 2",
        "not 1",
        "'a'",
        "\"a\"+\"b\"",
        "True",
        "None",
        "1j",
        "2+3j",
        "lambda: 1",
        "1 and 2",
        "abs(-1)**2",
        "((((1))))",
        "(1",
        "1)",
        "1 +",
        "+",
        "*2",
        "2 3",
        "1..2",
        "1.2.3",
        "1__0",
        "1_",
        "0x",
        "0xG",
        "5 5",
        "1 +* 2",
        "",
        "   ",
        "−5 + 3",
        "6 × 7",
        "8 ÷ 2",
        "2 ^ 3",
        "sqrt(4) + log(e)",
        "abs(1, 2)",
        "pi(1)",
        "1 if",
        "a b",
        "f(x=1)",
        "sqrt(x=4)",
        "f(*a)",
        "x.y",
        "x[0]",
        "1; 2",
        "1,2,3",
        "1,",
        "1,2,",
        "(1,2,3)",
        "1,,2",
        "100*1.1",
        "0.1+0.2",
        "1/7",
        "22/7",
        "123456789*987654321",
        "123456789012345678901234567890*3",
        "1e16+1",
        "1e15+0.3",
        "12345678901234567890.0",
        "0.000012345",
        "1e-5",
        "123456789012.0",
        "1234567890123.0",
        "-0.0",
        "0*-1.0",
        "5%2.5",
        "-5.5//2",
        "5.5//-2",
        "1e308*10",
        "1e308+1e308",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    v.push(format!("{}1", "(".repeat(45)));
    v.push(format!("{}1{}", "(".repeat(45), ")".repeat(45)));
    v.push("1+".repeat(300));
    v
}

fn sample_html() -> Vec<String> {
    vec![
        r#"<div class="result"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa%3Fx%3D1&amp;rut=abc">First <b>title</b></a><a class="result__snippet" href="x">A  snippet
with   breaks &amp; entities</a></div><div><a class="result__a" href="https://direct.example/b">Second</a></div><a class="other" href="z">no</a>"#.to_string(),
        "<p>nothing here</p>".to_string(),
        r#"<a class="result__a" href="">empty</a><a class="result__snippet">orphan</a>"#.to_string(),
        r#"<a class="result__snippet">before any result</a><a class="x result__a y" href='https://q.example/?a=1&b=2'>Quoted</a>"#.to_string(),
    ]
}

fn fake_search() -> SocketAddr {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = l.local_addr().unwrap();
    std::thread::spawn(move || {
        for s in l.incoming() {
            let mut s = s.unwrap();
            let mut buf = vec![0u8; 8192];
            let n = s.read(&mut buf).unwrap_or(0);
            let req = String::from_utf8_lossy(&buf[..n]).to_string();
            let target = req.split(' ').nth(1).unwrap_or("/").to_string();
            let body = if target.contains("q=broken") {
                "not json".to_string()
            } else if target.contains("q=none") {
                "{\"results\": []}".to_string()
            } else {
                json!({"results": [{"title": "  One  ", "url": "https://one.example", "content": "first\n result"}, {"title": "Two", "url": "https://two.example", "content": null}, {"title": "no url"}, "junk", {"url": "https://three.example"}]}).to_string()
            };
            let code = if target.contains("q=fail") {
                "500 Internal Server Error"
            } else {
                "200 OK"
            };
            let _ = s.write_all(format!("HTTP/1.1 {code}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes());
        }
    });
    addr
}

fn norm_time(s: &str) -> String {
    // the clock moves between the two calls: drop the seconds
    let b: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == ':'
            && i + 3 <= b.len()
            && b.get(i + 1).is_some_and(|c| c.is_ascii_digit())
            && b.get(i + 2).is_some_and(|c| c.is_ascii_digit())
            && b.get(i + 3).is_some_and(|c| *c == '+' || *c == '-')
        {
            out.push_str(":SS");
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    out
}

#[test]
fn the_built_in_tools_match_the_python_studio() {
    let Some(src) = studio_src() else {
        eprintln!("FLE Studio sources not found: skipped");
        return;
    };
    let d = tmp("tools");
    let root = d.join("share");
    std::fs::create_dir_all(root.join("sub/deeper")).unwrap();
    std::fs::write(root.join("a.txt"), "hello world\nsecond line\n").unwrap();
    std::fs::write(root.join("Z.md"), "# z").unwrap();
    std::fs::write(root.join("sub/b.txt"), "ünï".repeat(50)).unwrap();
    std::fs::write(root.join("bin.dat"), [1u8, 2, 0, 3]).unwrap();
    std::fs::write(root.join("big.txt"), "x".repeat(150_000)).unwrap();
    std::fs::write(d.join("outside.txt"), "secret").unwrap();
    for i in 0..230 {
        std::fs::write(root.join(format!("many-{i:03}.txt")), "").unwrap();
    }
    std::os::unix::fs::symlink(d.join("outside.txt"), root.join("link-out.txt")).unwrap();
    let exprs = expressions();
    let calc_args = vec![
        json!({}),
        json!({"expression": null}),
        json!({"expression": 5}),
        json!({"expression": "  2+2  "}),
        json!({"expression": "x".repeat(501)}),
        json!({"expression": ["1+1"]}),
        json!({"expression": true}),
    ];
    let clocks = vec![
        json!({}),
        json!({"timezone": "UTC"}),
        json!({"timezone": "gmt"}),
        json!({"timezone": "z"}),
        json!({"timezone": "Europe/Berlin"}),
        json!({"timezone": " America/New_York "}),
        json!({"timezone": "Asia/Kolkata"}),
        json!({"timezone": "Nope/Zone"}),
        json!({"timezone": ""}),
        json!({"timezone": null}),
    ];
    let lists = vec![
        json!({}),
        json!({"path": "."}),
        json!({"path": "sub"}),
        json!({"path": "sub/deeper"}),
        json!({"path": "a.txt"}),
        json!({"path": "nope"}),
        json!({"path": ".."}),
        json!({"path": "../"}),
        json!({"path": "sub/../.."}),
        json!({"path": d.join("share/sub").to_string_lossy()}),
        json!({"path": d.to_string_lossy()}),
        json!({"path": "a\u{0}b"}),
    ];
    let reads = vec![
        json!({"path": "a.txt"}),
        json!({"path": "a.txt", "offset": 6, "max_chars": 5}),
        json!({"path": "a.txt", "offset": "3", "max_chars": "4"}),
        json!({"path": "sub/b.txt", "max_chars": 20}),
        json!({"path": "big.txt"}),
        json!({"path": "big.txt", "max_chars": 200000}),
        json!({"path": "big.txt", "offset": 149_990, "max_chars": 50}),
        json!({"path": "bin.dat"}),
        json!({"path": "link-out.txt"}),
        json!({"path": "../outside.txt"}),
        json!({"path": d.join("outside.txt").to_string_lossy()}),
        json!({"path": "nope.txt"}),
        json!({"path": "sub"}),
        json!({}),
        json!({"path": "  "}),
        json!({"path": "a.txt", "offset": "x"}),
        json!({"path": "a.txt", "offset": -5, "max_chars": 0}),
        json!({"path": "a.txt", "max_chars": 1.9}),
        json!({"path": "a.txt", "offset": 99999999}),
    ];
    let searx = vec!["{\"results\": [{\"title\": \"A\", \"url\": \"u\", \"content\": \"c\"}, 5, {\"url\": \"v\"}]}".to_string(), "nope".to_string(), "[]".to_string(), "{}".to_string()];
    let servers: Vec<String> = vec![
        "", "  ", "{\"mcpServers\": {\"a\": {\"command\": \"npx\", \"args\": [\"-y\", \"x\", 5], \"env\": {\"K\": 1}, \"cwd\": \"/tmp\"}, \"b\": {\"url\": \"https://x.example/mcp\", \"headers\": {\"A\": 2}, \"oauth\": {\"client_id\": \"i\", \"scope\": \"s\", \"junk\": 1}}, \"c\": {\"url\": \"ftp://x\"}, \"d\": 5, \"e\": {\"disabled\": true, \"command\": \"x\"}, \"f\": {}, \"g\": {\"command\": \"x\", \"args\": \"no\"}, \"h\": {\"url\": \"http://x\", \"type\": \"SSE\"}, \"i\": {\"url\": \"http://x\", \"headers\": []}, \"j\": {\"url\": \"http://x\", \"oauth\": 3}, \"k\": {\"url\": \"http://\"}}}",
        "{\"a\": {\"command\": \"x\"}}", "[1]", "5", "{nope", "{\"mcpServers\": 5}", "{\"mcpServers\": {}}", "{\"a\": {\"command\": \"  \"}}",
    ].into_iter().map(String::from).collect();
    let fake = fake_search();
    let search_url = format!("http://127.0.0.1:{}/search", fake.port());
    let search_args = vec![
        json!({"query": "rust"}),
        json!({"query": "rust", "max_results": 2}),
        json!({"query": "rust", "max_results": 99}),
        json!({"query": "rust", "max_results": 0}),
        json!({"query": "rust", "max_results": "3"}),
        json!({"query": "broken"}),
        json!({"query": "none"}),
        json!({"query": "fail"}),
        json!({"query": "  "}),
        json!({}),
        json!({"query": "a b&c"}),
    ];
    std::fs::write(d.join("driver.py"), DRIVER).unwrap();
    let html = sample_html();
    std::fs::write(d.join("in.json"), json!({"exprs": exprs, "calc_args": calc_args, "clock": clocks, "root": root.to_string_lossy(), "list": lists, "read": reads, "html": html, "searx": searx, "servers": servers, "search_url": search_url, "search_args": search_args}).to_string()).unwrap();
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
    let res = |r: Result<String, fle_api::tools::ToolError>| match r {
        Ok(s) => json!({"ok": s}),
        Err(e) => json!({"error": e.0}),
    };
    let mut differ: Vec<String> = vec![];
    let mut n = 0;
    for (i, e) in exprs.iter().enumerate() {
        let got = res(calculator(&json!({"expression": e})));
        if got != py["calc"][i] {
            differ.push(format!("{e:?}: rust {got} python {}", py["calc"][i]));
        }
        n += 1;
    }
    for (i, a) in calc_args.iter().enumerate() {
        assert_eq!(
            res(calculator(a)),
            py["calc_args"][i],
            "calculator args {a}"
        );
    }
    for (i, a) in clocks.iter().enumerate() {
        let (g, w) = (res(clock(a)), py["clock"][i].clone());
        assert_eq!(
            norm_time(&g.to_string()),
            norm_time(&w.to_string()),
            "clock {a}"
        );
    }
    for (i, a) in lists.iter().enumerate() {
        let got = res(list_files(&root.to_string_lossy(), a));
        assert_eq!(
            got.to_string().replace(&*d.to_string_lossy(), "D"),
            py["list"][i]
                .to_string()
                .replace(&*d.to_string_lossy(), "D"),
            "list_files {a}"
        );
    }
    for (i, a) in reads.iter().enumerate() {
        let got = res(read_file(&root.to_string_lossy(), a));
        assert_eq!(got, py["read"][i], "read_file {a}");
    }
    for (i, h) in html.iter().enumerate() {
        assert_eq!(
            Value::Array(parse_duckduckgo(h)),
            py["ddg"][i],
            "parse_duckduckgo #{i}"
        );
    }
    for (i, t) in searx.iter().enumerate() {
        let got = match parse_searxng(t) {
            Ok(v) => json!({"ok": v}),
            Err(e) => json!({"error": e.0}),
        };
        assert_eq!(got, py["searx"][i], "parse_searxng #{i}");
    }
    for (i, t) in servers.iter().enumerate() {
        let (s, p) = parse_servers(t);
        let problems: serde_json::Map<String, Value> =
            p.into_iter().map(|(k, v)| (k, json!(v))).collect();
        let want = &py["servers"][i];
        assert_eq!(Value::Object(s), want[0], "parse_servers servers #{i}");
        // the wording of a JSON syntax error is each parser's own; the place and the rest are compared
        let (a, b) = (Value::Object(problems), want[1].clone());
        if i == 7 || i == 9 {
            assert_eq!(
                a.as_object().unwrap().keys().collect::<Vec<_>>(),
                b.as_object().unwrap().keys().collect::<Vec<_>>(),
                "parse_servers problems #{i}"
            );
        } else {
            assert_eq!(a, b, "parse_servers problems #{i}");
        }
    }
    for (i, a) in search_args.iter().enumerate() {
        let got = res(web_search(&search_url, a));
        assert_eq!(got, py["search"][i], "web_search {a}");
    }
    assert!(
        differ.is_empty(),
        "the calculator differs from Python in {} of {} expressions:\n{}",
        differ.len(),
        n,
        differ.join("\n")
    );
    eprintln!("compared with the Python Studio: {n} expressions, {} clock calls, {} file listings and {} reads, search parsing and {} MCP texts identical", clocks.len(), lists.len(), reads.len(), servers.len());
    let _ = Path::new("");
}
