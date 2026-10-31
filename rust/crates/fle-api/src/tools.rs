//! Tools the model may call in a chat: a few built-in ones (the MCP servers come with the MCP stage). Port of the built-in
//! part of `fle_studio/tools.py`.
//!
//! Each tool has a name, a description and a JSON schema for its arguments, which go to llama-server in the OpenAI-style
//! `tools` field of the chat request. The page runs the loop (model → tool → model); this module runs a single call.
//! Built-in tools are careful by default: the calculator and the clock work on nothing but their arguments; the file tools
//! read inside one folder the user picked and write nothing; web search is off until it is turned on, and asks before each
//! search.
use crate::settings::{py_str, py_truthy};
use num_bigint::{BigInt, Sign};
use num_integer::Integer;
use num_traits::{One, Signed, ToPrimitive, Zero};
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};

pub const MAX_FILE_CHARS: usize = 100_000;
pub const MAX_LIST_ENTRIES: usize = 200;
const MAX_EXPRESSION: usize = 500;

/// A tool call failed in a way the model should read ("file not found"); not a bug of FLE Studio.
#[derive(Debug)]
pub struct ToolError(pub String);
type R<T> = Result<T, ToolError>;
fn terr<T>(m: impl Into<String>) -> R<T> {
    Err(ToolError(m.into()))
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: Value,
    pub group: &'static str,
    pub confirm: bool,
}

impl Tool {
    pub fn openai(&self) -> Value {
        json!({"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}})
    }
    pub fn public(&self) -> Value {
        json!({"name": self.name, "description": self.description, "group": self.group, "source": "builtin", "server": "", "confirm": self.confirm, "ui": "", "schema": self.openai()})
    }
}

// ---------------------------------------------------------------- calculator

#[derive(Clone, Debug)]
enum Num {
    Int(BigInt),
    Float(f64),
}

#[derive(Debug)]
enum Fail {
    Tool(String),
    /// a power that has a complex result (a negative base to a fractional exponent): `(re, im)`
    Complex(f64, f64),
}

#[derive(Debug, Clone)]
enum Node {
    Num(Num),
    Name(String),
    Neg(Box<Node>),
    Pos(Box<Node>),
    Bin(char, Box<Node>, Box<Node>), // + - * / F(//) % P(**)
    Call(String, Vec<Node>),
    /// valid Python, but not something the calculator allows
    Other,
}

struct Parser {
    t: Vec<Tok>,
    i: usize,
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(String),
    Float(f64),
    Name(String),
    Op(String),
    Str,
    Imag,
}

fn tokenize(s: &str) -> Result<Vec<Tok>, ()> {
    let c: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut out = vec![];
    while i < c.len() {
        let ch = c[i];
        if ch.is_whitespace() {
            i += 1;
        } else if ch.is_ascii_digit()
            || (ch == '.' && c.get(i + 1).is_some_and(|d| d.is_ascii_digit()))
        {
            let start = i;
            if ch == '0'
                && c.get(i + 1)
                    .is_some_and(|x| matches!(x, 'x' | 'X' | 'o' | 'O' | 'b' | 'B'))
            {
                let radix = match c[i + 1].to_ascii_lowercase() {
                    'x' => 16,
                    'o' => 8,
                    _ => 2,
                };
                i += 2;
                let ds = i;
                while i < c.len() && (c[i].is_digit(radix) || c[i] == '_') {
                    i += 1;
                }
                let digits: String = c[ds..i].iter().filter(|x| **x != '_').collect();
                if digits.is_empty()
                    || c[ds..i].first() == Some(&'_')
                    || c[ds..i].last() == Some(&'_')
                {
                    return Err(());
                }
                if i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                    return Err(());
                }
                let v = BigInt::parse_bytes(digits.as_bytes(), radix).ok_or(())?;
                out.push(Tok::Int(v.to_string()));
                continue;
            }
            let mut is_float = false;
            let digits = |i: &mut usize| -> Result<(), ()> {
                let from = *i;
                while *i < c.len() && (c[*i].is_ascii_digit() || c[*i] == '_') {
                    *i += 1;
                }
                let seg = &c[from..*i];
                if seg.first() == Some(&'_')
                    || seg.last() == Some(&'_')
                    || seg.windows(2).any(|w| w[0] == '_' && w[1] == '_')
                {
                    return Err(());
                }
                Ok(())
            };
            digits(&mut i)?;
            if i < c.len() && c[i] == '.' {
                is_float = true;
                i += 1;
                if i < c.len() && c[i] == '_' {
                    return Err(());
                }
                digits(&mut i)?;
            }
            if i < c.len() && (c[i] == 'e' || c[i] == 'E') {
                let mut j = i + 1;
                if j < c.len() && (c[j] == '+' || c[j] == '-') {
                    j += 1;
                }
                if j < c.len() && c[j].is_ascii_digit() {
                    is_float = true;
                    i = j;
                    digits(&mut i)?;
                }
            }
            let text: String = c[start..i].iter().filter(|x| **x != '_').collect();
            if i < c.len() && (c[i] == 'j' || c[i] == 'J') {
                i += 1;
                out.push(Tok::Imag);
                continue;
            }
            if i < c.len() && (c[i].is_alphabetic() || c[i] == '_') {
                return Err(());
            }
            if is_float {
                out.push(Tok::Float(text.parse::<f64>().map_err(|_| ())?));
            } else {
                if text.len() > 1 && text.starts_with('0') && text.chars().any(|x| x != '0') {
                    return Err(()); // leading zeros in decimal integer literals are not permitted
                }
                out.push(Tok::Int(text));
            }
        } else if ch.is_alphabetic() || ch == '_' {
            let start = i;
            while i < c.len() && (c[i].is_alphanumeric() || c[i] == '_') {
                i += 1;
            }
            out.push(Tok::Name(c[start..i].iter().collect()));
        } else if ch == '"' || ch == '\'' {
            let q = ch;
            i += 1;
            while i < c.len() && c[i] != q {
                if c[i] == '\\' {
                    i += 1;
                }
                i += 1;
            }
            if i >= c.len() {
                return Err(());
            }
            i += 1;
            out.push(Tok::Str);
        } else {
            let three: String = c[i..(i + 3).min(c.len())].iter().collect();
            let two: String = c[i..(i + 2).min(c.len())].iter().collect();
            let op = if ["**=", "//=", ">>=", "<<="].contains(&three.as_str()) {
                three
            } else if ["**", "//", "<<", ">>", "<=", ">=", "==", "!=", "->", ":="]
                .contains(&two.as_str())
            {
                two
            } else {
                ch.to_string()
            };
            if !"+-*/%()[]{},.:<>=!&|^~@;".contains(&op[..1]) {
                return Err(());
            }
            i += op.chars().count();
            out.push(Tok::Op(op));
        }
    }
    Ok(out)
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.t.get(self.i)
    }
    fn op(&self, o: &str) -> bool {
        matches!(self.peek(), Some(Tok::Op(x)) if x == o)
    }
    fn name(&self, n: &str) -> bool {
        matches!(self.peek(), Some(Tok::Name(x)) if x == n)
    }
    fn eat_op(&mut self, o: &str) -> bool {
        if self.op(o) {
            self.i += 1;
            true
        } else {
            false
        }
    }
    // test := or_test ['if' or_test 'else' test] | lambda
    fn test(&mut self) -> Result<Node, ()> {
        if self.name("lambda") {
            return Err(()).or_else(|_: ()| self.lambda());
        }
        let a = self.or_test()?;
        if self.name("if") {
            self.i += 1;
            self.or_test()?;
            if !self.name("else") {
                return Err(());
            }
            self.i += 1;
            self.test()?;
            return Ok(Node::Other);
        }
        Ok(a)
    }
    fn lambda(&mut self) -> Result<Node, ()> {
        self.i += 1;
        while !self.op(":") {
            if self.peek().is_none() {
                return Err(());
            }
            self.i += 1;
        }
        self.i += 1;
        self.test()?;
        Ok(Node::Other)
    }
    fn or_test(&mut self) -> Result<Node, ()> {
        let mut a = self.and_test()?;
        while self.name("or") {
            self.i += 1;
            self.and_test()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn and_test(&mut self) -> Result<Node, ()> {
        let mut a = self.not_test()?;
        while self.name("and") {
            self.i += 1;
            self.not_test()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn not_test(&mut self) -> Result<Node, ()> {
        if self.name("not") {
            self.i += 1;
            self.not_test()?;
            return Ok(Node::Other);
        }
        self.comparison()
    }
    fn comparison(&mut self) -> Result<Node, ()> {
        let mut a = self.bit_or()?;
        loop {
            let cmp = matches!(self.peek(), Some(Tok::Op(o)) if ["<", ">", "<=", ">=", "==", "!="].contains(&o.as_str()))
                || self.name("in")
                || self.name("is")
                || (self.name("not")
                    && matches!(self.t.get(self.i + 1), Some(Tok::Name(n)) if n == "in"));
            if !cmp {
                break;
            }
            if self.name("not") {
                self.i += 1;
            }
            self.i += 1;
            if self.name("not") {
                self.i += 1;
            }
            self.bit_or()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn bit_or(&mut self) -> Result<Node, ()> {
        let mut a = self.bit_xor()?;
        while self.eat_op("|") {
            self.bit_xor()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn bit_xor(&mut self) -> Result<Node, ()> {
        let mut a = self.bit_and()?;
        while self.eat_op("^") {
            self.bit_and()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn bit_and(&mut self) -> Result<Node, ()> {
        let mut a = self.shift()?;
        while self.eat_op("&") {
            self.shift()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn shift(&mut self) -> Result<Node, ()> {
        let mut a = self.arith()?;
        while self.op("<<") || self.op(">>") {
            self.i += 1;
            self.arith()?;
            a = Node::Other;
        }
        Ok(a)
    }
    fn arith(&mut self) -> Result<Node, ()> {
        let mut a = self.term()?;
        loop {
            let o = if self.op("+") {
                '+'
            } else if self.op("-") {
                '-'
            } else {
                break;
            };
            self.i += 1;
            let b = self.term()?;
            a = Node::Bin(o, Box::new(a), Box::new(b));
        }
        Ok(a)
    }
    fn term(&mut self) -> Result<Node, ()> {
        let mut a = self.factor()?;
        loop {
            let o = if self.op("*") {
                '*'
            } else if self.op("/") {
                '/'
            } else if self.op("//") {
                'F'
            } else if self.op("%") {
                '%'
            } else if self.op("@") {
                '@'
            } else {
                break;
            };
            self.i += 1;
            let b = self.factor()?;
            a = if o == '@' {
                Node::Other
            } else {
                Node::Bin(o, Box::new(a), Box::new(b))
            };
        }
        Ok(a)
    }
    fn factor(&mut self) -> Result<Node, ()> {
        if self.eat_op("-") {
            return Ok(Node::Neg(Box::new(self.factor()?)));
        }
        if self.eat_op("+") {
            return Ok(Node::Pos(Box::new(self.factor()?)));
        }
        if self.eat_op("~") {
            self.factor()?;
            return Ok(Node::Other);
        }
        self.power()
    }
    fn power(&mut self) -> Result<Node, ()> {
        let a = self.atom_expr()?;
        if self.eat_op("**") {
            let b = self.factor()?;
            return Ok(Node::Bin('P', Box::new(a), Box::new(b)));
        }
        Ok(a)
    }
    fn atom_expr(&mut self) -> Result<Node, ()> {
        let mut a = self.atom()?;
        loop {
            if self.eat_op("(") {
                let mut args = vec![];
                let mut other = false;
                while !self.op(")") {
                    // `**x` and `NAME=value` are keywords (the calculator refuses a call with them); `*x` is an argument it cannot evaluate
                    if self.eat_op("**") {
                        other = true;
                        args.push(self.test()?);
                    } else if self.eat_op("*") {
                        args.push(Node::Other);
                        self.test()?;
                    } else {
                        if matches!(self.peek(), Some(Tok::Name(_)))
                            && matches!(self.t.get(self.i + 1), Some(Tok::Op(o)) if o == "=")
                        {
                            self.i += 2;
                            other = true;
                        }
                        args.push(self.test()?);
                    }
                    if !self.eat_op(",") {
                        break;
                    }
                }
                if !self.eat_op(")") {
                    return Err(());
                }
                a = match (&a, other) {
                    (Node::Name(n), false) => Node::Call(n.clone(), args),
                    _ => Node::Other,
                };
            } else if self.eat_op("[") {
                self.test()?;
                if !self.eat_op("]") {
                    return Err(());
                }
                a = Node::Other;
            } else if self.eat_op(".") {
                if !matches!(self.peek(), Some(Tok::Name(_))) {
                    return Err(());
                }
                self.i += 1;
                a = Node::Other;
            } else {
                break;
            }
        }
        Ok(a)
    }
    fn atom(&mut self) -> Result<Node, ()> {
        let t = self.peek().cloned().ok_or(())?;
        self.i += 1;
        match t {
            Tok::Int(s) => Ok(Node::Num(Num::Int(s.parse::<BigInt>().map_err(|_| ())?))),
            Tok::Float(f) => Ok(Node::Num(Num::Float(f))),
            Tok::Imag | Tok::Str => Ok(Node::Other),
            Tok::Name(n) => {
                if ["True", "False", "None"].contains(&n.as_str()) {
                    return Ok(Node::Other);
                }
                if [
                    "if", "else", "and", "or", "not", "in", "is", "lambda", "for", "while", "def",
                    "class", "return", "import", "from", "as", "with", "try", "except", "finally",
                    "raise", "pass", "break", "continue", "del", "global", "nonlocal", "assert",
                    "yield", "await", "async", "elif",
                ]
                .contains(&n.as_str())
                {
                    return Err(());
                }
                Ok(Node::Name(n))
            }
            Tok::Op(o) if o == "(" => {
                if self.eat_op(")") {
                    return Ok(Node::Other); // ()
                }
                let inner = self.test()?;
                let mut tuple = false;
                while self.eat_op(",") {
                    tuple = true;
                    if self.op(")") {
                        break;
                    }
                    self.test()?;
                }
                if !self.eat_op(")") {
                    return Err(());
                }
                Ok(if tuple { Node::Other } else { inner })
            }
            Tok::Op(o) if o == "[" || o == "{" => {
                let close = if o == "[" { "]" } else { "}" };
                let mut depth = 1;
                while depth > 0 {
                    match self.peek() {
                        None => return Err(()),
                        Some(Tok::Op(x)) if x == "[" || x == "{" || x == "(" => depth += 1,
                        Some(Tok::Op(x)) if x == "]" || x == "}" || x == ")" => depth -= 1,
                        _ => {}
                    }
                    self.i += 1;
                }
                let _ = close;
                Ok(Node::Other)
            }
            _ => Err(()),
        }
    }
}

fn parse_expression(text: &str) -> Result<Node, ()> {
    let toks = tokenize(text)?;
    if toks.is_empty() {
        return Err(());
    }
    let mut p = Parser { t: toks, i: 0 };
    let n = p.test()?;
    // a bare top-level comma makes a tuple
    if p.eat_op(",") {
        while p.peek().is_some() {
            p.test()?;
            if !p.eat_op(",") {
                break;
            }
        }
        return if p.peek().is_none() {
            Ok(Node::Other)
        } else {
            Err(())
        };
    }
    if p.peek().is_some() {
        return Err(());
    }
    Ok(n)
}

fn to_f64(i: &BigInt) -> Result<f64, String> {
    match i.to_f64() {
        Some(f) if f.is_finite() => Ok(f),
        _ => Err("int too large to convert to float".into()),
    }
}

fn as_f(n: &Num) -> Result<f64, String> {
    match n {
        Num::Int(i) => to_f64(i),
        Num::Float(f) => Ok(*f),
    }
}

/// Correctly rounded `a / b` for big integers (Python's `int.__truediv__`).
fn int_truediv(a: &BigInt, b: &BigInt) -> Result<f64, String> {
    if let (Some(x), Some(y)) = (a.to_i64(), b.to_i64()) {
        if x.unsigned_abs() < (1 << 53) && y.unsigned_abs() < (1 << 53) {
            return Ok(x as f64 / y as f64);
        }
    }
    if a.is_zero() {
        return Ok(0.0);
    }
    let neg = (a.sign() == Sign::Minus) != (b.sign() == Sign::Minus);
    let (a, b) = (a.abs(), b.abs());
    // scale so the integer quotient has at least 55 significant bits, keep a sticky bit for the rest
    let shift = 55i64 - (a.bits() as i64 - b.bits() as i64);
    let (num, den) = if shift >= 0 {
        (&a << shift as usize, b.clone())
    } else {
        (a.clone(), &b << (-shift) as usize)
    };
    let (q, r) = num.div_rem(&den);
    let mut q = q;
    if !r.is_zero() {
        q = (q << 1usize) + BigInt::one();
        let e = -(shift + 1);
        let f = q
            .to_f64()
            .ok_or("integer division result too large for a float")?;
        let v = f * 2f64.powi(e as i32);
        if !v.is_finite() {
            return Err("integer division result too large for a float".into());
        }
        return Ok(if neg { -v } else { v });
    }
    let f = q
        .to_f64()
        .ok_or("integer division result too large for a float")?;
    let v = f * 2f64.powi(-shift as i32);
    if !v.is_finite() {
        return Err("integer division result too large for a float".into());
    }
    Ok(if neg { -v } else { v })
}

fn float_mod(a: f64, b: f64) -> f64 {
    let mut m = a % b;
    if m != 0.0 {
        if (b < 0.0) != (m < 0.0) {
            m += b;
        }
    } else {
        m = 0f64.copysign(b);
    }
    m
}

fn float_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut fl = div.floor();
        if div - fl > 0.5 {
            fl += 1.0;
        }
        fl
    } else {
        0f64.copysign(a / b)
    }
}

fn pow_num(base: &Num, exp: &Num) -> Result<Num, Fail> {
    let tool = |m: &str| Fail::Tool(m.to_string());
    if let (Num::Int(b), Num::Int(e)) = (base, exp) {
        if e.is_positive()
            && b.abs() > BigInt::one()
            && (b.abs().bits() as u128) * e.to_u128().unwrap_or(u128::MAX) > 20_000
        {
            return Err(tool("That power is too large to compute"));
        }
        if !e.is_negative() {
            let ee = e
                .to_u32()
                .ok_or_else(|| tool("Cannot calculate that: exponent too large"))?;
            return Ok(Num::Int(num_traits::pow::pow(b.clone(), ee as usize)));
        }
        if b.is_zero() {
            return Err(tool("Division by zero"));
        }
        let (bf, ef) = (
            to_f64(b).map_err(|m| tool(&format!("Cannot calculate that: {m}")))?,
            e.to_f64().unwrap_or(f64::NEG_INFINITY),
        );
        return Ok(Num::Float(bf.powf(ef)));
    }
    let (b, e) = (
        as_f(base).map_err(|m| tool(&format!("Cannot calculate that: {m}")))?,
        as_f(exp).map_err(|m| tool(&format!("Cannot calculate that: {m}")))?,
    );
    if b == 0.0 && e < 0.0 {
        return Err(tool("Division by zero"));
    }
    if b < 0.0 && e.fract() != 0.0 && e.is_finite() {
        // CPython's complex power for a real exponent: polar form
        let (vabs, at) = (b.abs(), std::f64::consts::PI);
        let len = vabs.powf(e);
        let phase = at * e;
        return Err(Fail::Complex(len * phase.cos(), len * phase.sin()));
    }
    let r = b.powf(e);
    if r.is_infinite() && b.is_finite() && e.is_finite() {
        return Err(tool("Cannot calculate that: (34, 'Result too large')"));
    }
    Ok(Num::Float(r))
}

fn eval(n: &Node, depth: usize) -> Result<Num, Fail> {
    let tool = |m: String| Fail::Tool(m);
    if depth > 40 {
        return Err(tool("The expression is nested too deeply".into()));
    }
    let cannot = |m: String| Fail::Tool(format!("Cannot calculate that: {m}"));
    match n {
        Node::Num(x) => Ok(x.clone()),
        Node::Name(id) => match id.as_str() {
            "pi" => Ok(Num::Float(std::f64::consts::PI)),
            "e" => Ok(Num::Float(std::f64::consts::E)),
            "tau" => Ok(Num::Float(std::f64::consts::TAU)),
            _ => Err(tool(format!("Unknown name '{id}'. Known constants: pi, e, tau"))),
        },
        Node::Neg(x) => Ok(match eval(x, depth + 1)? {
            Num::Int(i) => Num::Int(-i),
            Num::Float(f) => Num::Float(-f),
        }),
        Node::Pos(x) => eval(x, depth + 1),
        Node::Bin(op, l, r) => {
            let (a, b) = (eval(l, depth + 1)?, eval(r, depth + 1)?);
            if *op == 'P' {
                return pow_num(&a, &b);
            }
            if let (Num::Int(x), Num::Int(y)) = (&a, &b) {
                return Ok(match op {
                    '+' => Num::Int(x + y),
                    '-' => Num::Int(x - y),
                    '*' => Num::Int(x * y),
                    '/' => {
                        if y.is_zero() {
                            return Err(tool("Division by zero".into()));
                        }
                        Num::Float(int_truediv(x, y).map_err(cannot)?)
                    }
                    'F' => {
                        if y.is_zero() {
                            return Err(tool("Division by zero".into()));
                        }
                        Num::Int(x.div_floor(y))
                    }
                    _ => {
                        if y.is_zero() {
                            return Err(tool("Division by zero".into()));
                        }
                        Num::Int(x.mod_floor(y))
                    }
                });
            }
            let (x, y) = (as_f(&a).map_err(cannot)?, as_f(&b).map_err(cannot)?);
            Ok(Num::Float(match op {
                '+' => x + y,
                '-' => x - y,
                '*' => x * y,
                '/' => {
                    if y == 0.0 {
                        return Err(tool("Division by zero".into()));
                    }
                    x / y
                }
                'F' => {
                    if y == 0.0 {
                        return Err(tool("Division by zero".into()));
                    }
                    float_floordiv(x, y)
                }
                _ => {
                    if y == 0.0 {
                        return Err(tool("Division by zero".into()));
                    }
                    float_mod(x, y)
                }
            }))
        }
        Node::Call(name, args) => {
            const KNOWN: [&str; 26] = ["abs", "acos", "asin", "atan", "ceil", "cos", "cosh", "degrees", "exp", "factorial", "floor", "gcd", "ln", "log", "log10", "log2", "max", "min", "radians", "round", "sin", "sinh", "sqrt", "tan", "tanh", "tau"];
            let known: Vec<&str> = KNOWN.iter().copied().filter(|k| *k != "tau").collect();
            if !known.contains(&name.as_str()) {
                return Err(tool(format!("Unknown function '{name}'. Known: {}", known.join(", "))));
            }
            let vals: Vec<Num> = args.iter().map(|a| eval(a, depth + 1)).collect::<Result<_, _>>()?;
            if name == "factorial" {
                if let Some(Num::Int(i)) = vals.first() {
                    if *i > BigInt::from(1000) {
                        return Err(tool("factorial is limited to 1000".into()));
                    }
                }
            }
            call_fn(name, &vals).map_err(|m| if m == "\0DIV" { tool("Division by zero".into()) } else { cannot(m) })
        }
        Node::Other => Err(tool("Only numbers, + - * / // % ** , parentheses, constants and common functions are allowed".into())),
    }
}

fn one_arg<'a>(name: &str, v: &'a [Num]) -> Result<&'a Num, String> {
    if v.len() != 1 {
        let shown = if name == "abs" {
            name.to_string()
        } else {
            format!("math.{name}")
        };
        return Err(format!(
            "{shown}() takes exactly one argument ({} given)",
            v.len()
        ));
    }
    Ok(&v[0])
}

fn int_of(n: &Num) -> Result<&BigInt, String> {
    match n {
        Num::Int(i) => Ok(i),
        Num::Float(_) => Err("'float' object cannot be interpreted as an integer".into()),
    }
}

fn float_to_int(f: f64) -> Result<Num, String> {
    if f.is_nan() {
        return Err("cannot convert float NaN to integer".into());
    }
    if f.is_infinite() {
        return Err("cannot convert float infinity to integer".into());
    }
    num_traits::FromPrimitive::from_f64(f)
        .map(Num::Int)
        .ok_or_else(|| "cannot convert float to integer".to_string())
}

fn call_fn(name: &str, v: &[Num]) -> Result<Num, String> {
    let domain = || "math domain error".to_string();
    let unary = |f: fn(f64) -> f64, check: fn(f64) -> bool| -> Result<Num, String> {
        let x = as_f(one_arg(name, v)?)?;
        if !check(x) {
            return Err("math domain error".into());
        }
        let r = f(x);
        if r.is_infinite() && x.is_finite() {
            return Err("math range error".into());
        }
        Ok(Num::Float(r))
    };
    let any = |_: f64| true;
    match name {
        "sqrt" => unary(f64::sqrt, |x| x >= 0.0 || x.is_nan()),
        "sin" | "cos" | "tan" => {
            let x = as_f(one_arg(name, v)?)?;
            if x.is_infinite() {
                return Err(domain());
            }
            Ok(Num::Float(match name {
                "sin" => x.sin(),
                "cos" => x.cos(),
                _ => x.tan(),
            }))
        }
        "asin" => unary(f64::asin, |x| (-1.0..=1.0).contains(&x) || x.is_nan()),
        "acos" => unary(f64::acos, |x| (-1.0..=1.0).contains(&x) || x.is_nan()),
        "atan" => unary(f64::atan, any),
        "sinh" => unary(f64::sinh, any),
        "cosh" => unary(f64::cosh, any),
        "tanh" => unary(f64::tanh, any),
        "exp" => unary(f64::exp, any),
        "log10" => unary(f64::log10, |x| x > 0.0 || x.is_nan()),
        "log2" => unary(f64::log2, |x| x > 0.0 || x.is_nan()),
        "radians" => unary(f64::to_radians, any),
        "degrees" => unary(f64::to_degrees, any),
        "ln" | "log" => {
            if v.is_empty() || v.len() > 2 {
                return Err(format!("{name} expected 1 or 2 arguments, got {}", v.len()));
            }
            let ln = |n: &Num| -> Result<f64, String> {
                match n {
                    Num::Int(i) => {
                        if i.is_positive() {
                            let bits = i.bits();
                            if bits > 1000 {
                                let shifted = i >> (bits - 60) as usize;
                                Ok(to_f64(&shifted)?.ln()
                                    + (bits - 60) as f64 * std::f64::consts::LN_2)
                            } else {
                                Ok(to_f64(i)?.ln())
                            }
                        } else {
                            Err("math domain error".into())
                        }
                    }
                    Num::Float(f) => {
                        if *f > 0.0 {
                            Ok(f.ln())
                        } else if f.is_nan() {
                            Ok(*f)
                        } else {
                            Err("math domain error".into())
                        }
                    }
                }
            };
            let x = ln(&v[0])?;
            if v.len() == 1 {
                return Ok(Num::Float(x));
            }
            let b = ln(&v[1])?;
            if b == 0.0 {
                return Err("\0DIV".into());
            }
            Ok(Num::Float(x / b))
        }
        "abs" => Ok(match one_arg(name, v)? {
            Num::Int(i) => Num::Int(i.abs()),
            Num::Float(f) => Num::Float(f.abs()),
        }),
        "floor" | "ceil" => match one_arg(name, v)? {
            Num::Int(i) => Ok(Num::Int(i.clone())),
            Num::Float(f) => float_to_int(if name == "floor" { f.floor() } else { f.ceil() }),
        },
        "round" => {
            if v.is_empty() || v.len() > 2 {
                return Err(if v.is_empty() {
                    "round() missing required argument 'number' (pos 1)".into()
                } else {
                    format!("round() takes at most 2 arguments ({} given)", v.len())
                });
            }
            match (&v[0], v.get(1)) {
                (Num::Int(i), None) => Ok(Num::Int(i.clone())),
                (Num::Float(f), None) => float_to_int(f.round_ties_even()),
                (n, Some(d)) => {
                    let nd = int_of(d)?
                        .to_i64()
                        .ok_or("Python int too large to convert to C ssize_t")?;
                    match n {
                        Num::Int(i) => {
                            if nd >= 0 {
                                Ok(Num::Int(i.clone()))
                            } else {
                                let p = num_traits::pow::pow(BigInt::from(10), (-nd) as usize);
                                let (q, r) = i.div_rem(&p);
                                let twice: BigInt = (&r * BigInt::from(2)).abs();
                                let mut q2 = q;
                                if twice > p || (twice == p && q2.is_odd()) {
                                    q2 += if i.is_negative() { -1 } else { 1 };
                                }
                                Ok(Num::Int(q2 * p))
                            }
                        }
                        Num::Float(f) => {
                            if nd > 308 || !f.is_finite() {
                                return Ok(Num::Float(*f));
                            }
                            if nd < -308 {
                                return Ok(Num::Float(0.0 * f));
                            }
                            // correctly rounded decimal rounding, like Python's round(x, n)
                            let s = format!("{:.*e}", (nd.max(0) as usize).saturating_add(30), f);
                            let _ = s;
                            let r = if nd >= 0 {
                                format!("{:.*}", nd as usize, f)
                                    .parse::<f64>()
                                    .unwrap_or(*f)
                            } else {
                                let p = 10f64.powi((-nd) as i32);
                                (f / p).round_ties_even() * p
                            };
                            Ok(Num::Float(r))
                        }
                    }
                }
            }
        }
        "min" | "max" => {
            if v.is_empty() {
                return Err(format!("{name} expected at least 1 argument, got 0"));
            }
            if v.len() == 1 {
                return Err(match &v[0] {
                    Num::Int(_) => "'int' object is not iterable".into(),
                    Num::Float(_) => "'float' object is not iterable".into(),
                });
            }
            let mut best = v[0].clone();
            for x in &v[1..] {
                let less = match (x, &best) {
                    (Num::Int(a), Num::Int(b)) => a < b,
                    _ => as_f(x)? < as_f(&best)?,
                };
                let greater = match (x, &best) {
                    (Num::Int(a), Num::Int(b)) => a > b,
                    _ => as_f(x)? > as_f(&best)?,
                };
                if (name == "min" && less) || (name == "max" && greater) {
                    best = x.clone();
                }
            }
            Ok(best)
        }
        "factorial" => {
            let i = int_of(one_arg(name, v)?)
                .map_err(|_| "'float' object cannot be interpreted as an integer".to_string())?;
            if i.is_negative() {
                return Err("factorial() not defined for negative values".into());
            }
            let n = i
                .to_u32()
                .ok_or("factorial() argument should not exceed 9223372036854775807")?;
            let mut r = BigInt::one();
            for k in 2..=n {
                r *= k;
            }
            Ok(Num::Int(r))
        }
        "gcd" => {
            let mut g = BigInt::zero();
            for x in v {
                g = g.gcd(int_of(x)?);
            }
            Ok(Num::Int(g))
        }
        _ => Err(format!("unknown function {name}")),
    }
}

/// Python's `format(x, ".12g")`
fn fmt_12g(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".into()
        } else {
            "0".into()
        };
    }
    let exp = x.abs().log10().floor() as i32;
    // the exponent after rounding to 12 significant digits decides the style
    let sci = format!("{:.11e}", x);
    let (mant, e) = sci.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    let _ = exp;
    if !(-4..12).contains(&e) {
        let m = if mant.contains('.') {
            mant.trim_end_matches('0').trim_end_matches('.')
        } else {
            mant
        };
        return format!("{m}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let decimals = (11 - e).max(0) as usize;
    let s = format!("{x:.decimals$}");
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        s
    }
}

pub fn calculator(args: &Value) -> R<String> {
    let text = args
        .get("expression")
        .filter(|v| py_truthy(v))
        .map(py_str)
        .unwrap_or_default();
    let text = text.trim().to_string();
    if text.is_empty() {
        return terr("Give the \"expression\" to calculate, for example \"(3 + 4) * 2\".");
    }
    if text.chars().count() > MAX_EXPRESSION {
        return terr("The expression is too long");
    }
    let text: String = text
        .chars()
        .map(|c| match c {
            '×' => "*".to_string(),
            '÷' => "/".to_string(),
            '−' => "-".to_string(),
            c => c.to_string(),
        })
        .collect::<String>()
        .replace('^', "**");
    let tree = match parse_expression(&text) {
        Ok(t) => t,
        Err(()) => return terr(format!("Cannot read the expression '{text}'")),
    };
    match eval(&tree, 0) {
        Ok(Num::Int(i)) => Ok(i.to_string()),
        Ok(Num::Float(f)) => {
            if f.is_nan() || f.is_infinite() {
                terr("The result is not a finite number")
            } else {
                Ok(fmt_12g(f))
            }
        }
        Err(Fail::Tool(m)) => terr(m),
        // a complex number as the result of the whole expression is shown like Python shows it; inside a longer
        // expression it is not supported here
        Err(Fail::Complex(re, im)) if matches!(tree, Node::Bin('P', _, _)) => {
            Ok(complex_repr(re, im))
        }
        Err(Fail::Complex(..)) => terr("Cannot calculate that: complex results are not supported"),
    }
}

/// Python's `repr(float)`: the shortest digits, scientific below 1e-4 and from 1e16 on.
fn py_float_repr(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    let sci = format!("{x:e}"); // shortest round-trip digits: 6.123233995736766e-17
    let (mant, e) = sci.split_once('e').unwrap();
    let e: i32 = e.parse().unwrap();
    if !(-4..16).contains(&e) {
        return format!("{mant}e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    }
    let s = format!("{x}");
    if s.contains('.') {
        s
    } else {
        format!("{s}.0")
    }
}

/// Python's `str(complex)`: `(1+2j)`, or `1j` when the real part is a plain zero.
fn complex_repr(re: f64, im: f64) -> String {
    let part = |x: f64| {
        let r = py_float_repr(x);
        r.strip_suffix(".0").map(String::from).unwrap_or(r)
    };
    if re == 0.0 && !re.is_sign_negative() {
        return format!("{}j", part(im));
    }
    let sign = if im.is_sign_negative() && !im.is_nan() {
        "-"
    } else {
        "+"
    };
    format!("({}{sign}{}j)", part(re), part(im.abs()))
}

// ---------------------------------------------------------------- clock

pub fn clock(args: &Value) -> R<String> {
    use chrono::{Datelike, Local, Offset, Utc};
    let name = args
        .get("timezone")
        .filter(|v| py_truthy(v))
        .map(py_str)
        .unwrap_or_default();
    let name = name.trim().to_string();
    let (iso, weekday, tzname, where_) = if name.is_empty() {
        let now = Local::now();
        let abbrev = iana_time_zone::get_timezone()
            .ok()
            .and_then(|z| z.parse::<chrono_tz::Tz>().ok())
            .map(|tz| now.with_timezone(&tz).format("%Z").to_string())
            .unwrap_or_else(|| now.format("%Z").to_string());
        (
            now.format("%Y-%m-%dT%H:%M:%S%:z").to_string(),
            now.format("%A").to_string(),
            abbrev,
            "this computer's local time".to_string(),
        )
    } else if ["UTC", "Z", "GMT"].contains(&name.to_uppercase().as_str()) {
        let now = Utc::now();
        (
            now.format("%Y-%m-%dT%H:%M:%S+00:00").to_string(),
            now.format("%A").to_string(),
            "UTC".to_string(),
            name.clone(),
        )
    } else {
        match name.parse::<chrono_tz::Tz>() {
            Ok(tz) => {
                let now = Utc::now().with_timezone(&tz);
                let _ = (now.year(), now.offset().fix());
                (now.format("%Y-%m-%dT%H:%M:%S%:z").to_string(), now.format("%A").to_string(), now.format("%Z").to_string(), name.clone())
            }
            Err(_) => return terr(format!("Unknown time zone '{name}'. Use a name like Europe/Berlin or UTC; leave it out for this computer's time.")),
        }
    };
    Ok(format!("{iso} ({weekday}, {tzname}, {where_})"))
}

// ---------------------------------------------------------------- files

fn inside(root: &Path, rel: &str) -> R<PathBuf> {
    if rel.contains('\0') || (cfg!(windows) && rel.contains(':') && !Path::new(rel).is_absolute()) {
        return terr("That path is not allowed");
    }
    let cand = Path::new(rel);
    let full = crate::library::py_resolve(&if cand.is_absolute() {
        cand.to_path_buf()
    } else {
        root.join(cand)
    });
    if full != root && !full.ancestors().skip(1).any(|a| a == root) {
        return terr("That path is outside the folder the user shared with the model");
    }
    Ok(full)
}

fn files_root(folder: &str) -> R<PathBuf> {
    if folder.trim().is_empty() {
        return terr("No folder is shared with the model");
    }
    let root = match folder.strip_prefix("~/") {
        Some(r) => dirs::home_dir().unwrap_or_default().join(r),
        None => PathBuf::from(folder),
    };
    if !root.is_dir() {
        return terr("The shared folder does not exist");
    }
    Ok(crate::library::py_resolve(&root))
}

pub fn list_files(folder: &str, args: &Value) -> R<String> {
    let root = files_root(folder)?;
    let rel = args
        .get("path")
        .filter(|v| py_truthy(v))
        .map(py_str)
        .unwrap_or_else(|| ".".into());
    let target = inside(&root, &rel)?;
    if !target.is_dir() {
        return terr("That is not a folder");
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&target)
        .map(|r| r.flatten().map(|e| e.path()).collect())
        .unwrap_or_default();
    entries.sort_by_key(|p| {
        (
            !p.is_dir(),
            p.file_name()
                .map(|n| n.to_string_lossy().to_lowercase())
                .unwrap_or_default(),
        )
    });
    let rows: Vec<String> = entries
        .iter()
        .take(MAX_LIST_ENTRIES)
        .map(|p| {
            let name = p.file_name().unwrap().to_string_lossy().to_string();
            if p.is_dir() {
                format!("{name}/")
            } else {
                match std::fs::metadata(p) {
                    Ok(m) => format!("{name}  ({} bytes)", m.len()),
                    Err(_) => name,
                }
            }
        })
        .collect();
    let more = if entries.len() > MAX_LIST_ENTRIES {
        format!("\n… and {} more", entries.len() - MAX_LIST_ENTRIES)
    } else {
        String::new()
    };
    let shown = if target != root {
        target
            .strip_prefix(&root)
            .unwrap()
            .to_string_lossy()
            .replace('\\', "/")
    } else {
        ".".into()
    };
    Ok(format!(
        "{shown}:\n{}{more}",
        if rows.is_empty() {
            "(empty)".to_string()
        } else {
            rows.join("\n")
        }
    ))
}

pub fn read_file(folder: &str, args: &Value) -> R<String> {
    let root = files_root(folder)?;
    let path = args
        .get("path")
        .filter(|v| py_truthy(v))
        .map(py_str)
        .unwrap_or_default();
    let path = path.trim();
    if path.is_empty() {
        return terr("Give the \"path\" of the file, relative to the shared folder.");
    }
    let target = inside(&root, path)?;
    if !target.is_file() {
        return terr("There is no such file in the shared folder");
    }
    let int_arg = |k: &str, default: i64| -> Result<i64, ()> {
        match args.get(k).filter(|v| py_truthy(v)) {
            None => Ok(default),
            Some(Value::Number(n)) => Ok(n
                .as_i64()
                .or_else(|| n.as_f64().map(|f| f as i64))
                .unwrap_or(0)),
            Some(Value::String(s)) => s.trim().parse::<f64>().map(|f| f as i64).map_err(|_| ()),
            Some(Value::Bool(b)) => Ok(i64::from(*b)),
            Some(_) => Err(()),
        }
    };
    let (Ok(offset), Ok(limit)) = (int_arg("offset", 0), int_arg("max_chars", 20_000)) else {
        return terr("\"offset\" and \"max_chars\" must be whole numbers");
    };
    let offset = offset.clamp(0, 10_000_000) as usize;
    let limit = limit.clamp(1, MAX_FILE_CHARS as i64) as usize;
    let want = offset * 4 + limit * 4 + 8;
    let mut raw = vec![];
    match std::fs::File::open(&target).and_then(|f| {
        std::io::Read::read_to_end(&mut std::io::Read::take(f, want as u64), &mut raw)
    }) {
        Ok(_) => {}
        Err(e) => {
            return terr(format!(
                "Cannot read the file: {}",
                e.to_string().split(" (os error").next().unwrap_or("")
            ))
        }
    }
    if raw.iter().take(8000).any(|b| *b == 0) {
        return terr("That is a binary file");
    }
    let text = String::from_utf8_lossy(&raw).to_string();
    let size = std::fs::metadata(&target).map(|m| m.len()).unwrap_or(0) as usize;
    let chars: Vec<char> = text.chars().collect();
    let chunk: String = chars.iter().skip(offset).take(limit).collect();
    let more = chars.len() > offset + limit || size > raw.len();
    let note = if more {
        format!(
            "\n[more follows: ask again with offset={}]",
            offset + chunk.chars().count()
        )
    } else {
        String::new()
    };
    Ok(chunk + &note)
}

// ---------------------------------------------------------------- web search

fn squash(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn html_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&amp;", "&")
}

/// Results of DuckDuckGo's plain HTML page: links with class result__a, snippets with class result__snippet.
pub fn parse_duckduckgo(html: &str) -> Vec<Value> {
    let mut results: Vec<(String, String, String)> = vec![];
    let mut mode = "";
    let b = html;
    let mut i = 0;
    while i < b.len() {
        let rest = &b[i..];
        if rest.starts_with('<') {
            let end = rest.find('>').map(|e| e + 1).unwrap_or(rest.len());
            let tag = &rest[..end];
            let low = tag.to_lowercase();
            if low.starts_with("<a ") || low.starts_with("<a\n") {
                let attr = |name: &str| -> String {
                    let key = format!("{name}=");
                    low.find(&key)
                        .and_then(|p| {
                            let after = &tag[p + key.len()..];
                            let q = after.chars().next()?;
                            if q == '"' || q == '\'' {
                                after[1..].find(q).map(|e| html_unescape(&after[1..1 + e]))
                            } else {
                                after
                                    .split(|c: char| c.is_whitespace() || c == '>')
                                    .next()
                                    .map(String::from)
                            }
                        })
                        .unwrap_or_default()
                };
                let classes = attr("class");
                if classes.split_whitespace().any(|c| c == "result__a") {
                    let href = attr("href");
                    let target = href
                        .split_once('?')
                        .and_then(|(_, q)| {
                            q.split('&')
                                .find_map(|kv| kv.strip_prefix("uddg=").map(percent_decode))
                        })
                        .unwrap_or(href);
                    results.push((String::new(), target, String::new()));
                    mode = "title";
                } else if classes.split_whitespace().any(|c| c == "result__snippet")
                    && !results.is_empty()
                {
                    mode = "snippet";
                }
            } else if low.starts_with("</a") {
                mode = "";
            }
            i += end;
        } else {
            let end = rest.find('<').unwrap_or(rest.len());
            if !mode.is_empty() {
                if let Some(last) = results.last_mut() {
                    let text = html_unescape(&rest[..end]);
                    if mode == "title" {
                        last.0 += &text
                    } else {
                        last.2 += &text
                    }
                }
            }
            i += end;
        }
    }
    results
        .into_iter()
        .filter(|r| !r.1.is_empty())
        .map(|(t, u, s)| json!({"title": squash(&t), "url": squash(&u), "snippet": squash(&s)}))
        .collect()
}

fn percent_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = vec![];
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(if b[i] == b'+' { b' ' } else { b[i] });
        i += 1;
    }
    String::from_utf8_lossy(&out).to_string()
}

pub fn parse_searxng(text: &str) -> R<Vec<Value>> {
    let doc: Value = match serde_json::from_str(text) {
        Ok(d) => d,
        Err(_) => {
            return terr(
                "The search server did not answer with JSON (is format=json allowed there?)",
            )
        }
    };
    let mut out = vec![];
    if let Some(items) = doc.get("results").and_then(Value::as_array) {
        for item in items {
            if item.is_object() && item.get("url").is_some_and(py_truthy) {
                out.push(json!({"title": squash(&item.get("title").filter(|v| py_truthy(v)).map(py_str).unwrap_or_default()), "url": py_str(&item["url"]),
                    "snippet": squash(&item.get("content").filter(|v| py_truthy(v)).map(py_str).unwrap_or_default())}));
            }
        }
    }
    Ok(out)
}

/// The reason phrase Python's `HTTPError.reason` carries.
fn http_reason(c: u16) -> String {
    match c {
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        410 => "Gone",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => return format!("HTTP Error {c}"),
    }
    .to_string()
}

fn quote(text: &str) -> String {
    text.bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"_.-~/".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

pub fn web_search(base_url: &str, args: &Value) -> R<String> {
    let query = args
        .get("query")
        .filter(|v| py_truthy(v))
        .map(py_str)
        .unwrap_or_default();
    let query = query.trim().to_string();
    if query.is_empty() {
        return terr("Give the \"query\" to search for.");
    }
    let count = match args.get("max_results").filter(|v| py_truthy(v)) {
        None => 5,
        Some(Value::Number(n)) => n.as_i64().unwrap_or(5).clamp(1, 10),
        Some(Value::String(s)) => s
            .trim()
            .parse::<f64>()
            .map(|f| (f as i64).clamp(1, 10))
            .unwrap_or(5),
        Some(_) => 5,
    } as usize;
    let custom = base_url.trim();
    let url = if !custom.is_empty() {
        if !(custom.starts_with("http://") || custom.starts_with("https://")) {
            return terr("The search server address must start with http:// or https://");
        }
        format!(
            "{custom}{}q={}&format=json",
            if custom.contains('?') { '&' } else { '?' },
            quote(&query)
        )
    } else {
        format!("https://html.duckduckgo.com/html/?q={}", quote(&query))
    };
    let agent = ureq::AgentBuilder::new()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent(&format!("fle-studio/{}", env!("CARGO_PKG_VERSION")))
        .build();
    let body = match agent
        .get(&url)
        .set("Accept", "text/html,application/json")
        .call()
    {
        Ok(r) => {
            let mut buf = vec![];
            let _ = std::io::Read::read_to_end(
                &mut std::io::Read::take(r.into_reader(), 1_500_000),
                &mut buf,
            );
            String::from_utf8_lossy(&buf).to_string()
        }
        Err(ureq::Error::Status(c, _)) => {
            return terr(format!("The search failed: {}", http_reason(c)))
        }
        Err(e) => return terr(format!("The search failed: {e}")),
    };
    let all = if !custom.is_empty() {
        parse_searxng(&body)?
    } else {
        parse_duckduckgo(&body)
    };
    let results: Vec<&Value> = all.iter().take(count).collect();
    if results.is_empty() {
        return Ok("No results.".into());
    }
    Ok(results
        .iter()
        .enumerate()
        .map(|(i, r)| {
            format!(
                "{}. {}\n   {}\n   {}",
                i + 1,
                r["title"].as_str().unwrap_or(""),
                r["url"].as_str().unwrap_or(""),
                r["snippet"].as_str().unwrap_or("")
            )
            .trim_end()
            .to_string()
        })
        .collect::<Vec<_>>()
        .join("\n\n"))
}

// ---------------------------------------------------------------- the set

fn schema(props: Value, required: &[&str]) -> Value {
    let mut o = Map::new();
    o.insert("type".into(), json!("object"));
    o.insert("properties".into(), props);
    if !required.is_empty() {
        o.insert("required".into(), json!(required));
    }
    Value::Object(o)
}

/// What is on right now, read from the settings every time (`ToolBox.tools`; MCP tools come with the MCP stage).
pub fn tools(s: &Value) -> Vec<Tool> {
    let on = |k: &str| s.get(k).is_some_and(py_truthy);
    let mut out = vec![];
    if !on("enabled") {
        return out;
    }
    if on("calculator") {
        out.push(Tool { name: "calculator".into(), description: "Calculate an arithmetic expression exactly. Use it for any maths instead of working it out yourself.".into(),
            parameters: schema(json!({"expression": {"type": "string", "description": "For example (12.5 + 7) * 3 / 4 or sqrt(2)**2"}}), &["expression"]), group: "calculator", confirm: false });
    }
    if on("datetime") {
        out.push(Tool { name: "get_current_time".into(), description: "Get the current date and time.".into(),
            parameters: schema(json!({"timezone": {"type": "string", "description": "Optional IANA time zone such as Europe/Berlin or UTC. Omit it for this computer's local time."}}), &[]), group: "clock", confirm: false });
    }
    if on("files") && files_dir_ok(s) {
        out.push(Tool { name: "list_files".into(), description: "List the files and folders in a folder that the user shared with you.".into(),
            parameters: schema(json!({"path": {"type": "string", "description": "Folder relative to the shared folder; omit it for the shared folder itself."}}), &[]), group: "files", confirm: false });
        out.push(Tool { name: "read_file".into(), description: "Read a text file from the folder that the user shared with you.".into(),
            parameters: schema(json!({"path": {"type": "string", "description": "File path relative to the shared folder."}, "offset": {"type": "integer", "description": "First character to return (default 0)."},
                "max_chars": {"type": "integer", "description": "How many characters to return (default 20000)."}}), &["path"]), group: "files", confirm: false });
    }
    if on("web_search") {
        out.push(Tool { name: "web_search".into(), description: "Search the web and return titles, links and short snippets.".into(),
            parameters: schema(json!({"query": {"type": "string"}, "max_results": {"type": "integer", "description": "1 to 10, default 5"}}), &["query"]), group: "web_search", confirm: on("web_search_confirm") });
    }
    out
}

pub fn files_dir_ok(s: &Value) -> bool {
    let f = s
        .get("files_dir")
        .filter(|v| py_truthy(v))
        .map(py_str)
        .unwrap_or_default();
    let f = f.trim();
    if f.is_empty() {
        return false;
    }
    match f.strip_prefix("~/") {
        Some(r) => dirs::home_dir().unwrap_or_default().join(r).is_dir(),
        None => Path::new(f).is_dir(),
    }
}

/// Run one built-in tool.
pub fn run(tool: &Tool, s: &Value, args: &Value) -> R<String> {
    match tool.name.as_str() {
        "calculator" => calculator(args),
        "get_current_time" => clock(args),
        "list_files" => list_files(&py_str(&s["files_dir"]), args),
        "read_file" => read_file(&py_str(&s["files_dir"]), args),
        "web_search" => web_search(
            &s.get("web_search_url")
                .filter(|v| py_truthy(v))
                .map(py_str)
                .unwrap_or_default(),
            args,
        ),
        _ => terr("unknown tool"),
    }
}

// ---------------------------------------------------------------- MCP servers (the text of the settings only)

/// `(servers, problems)` from the `mcpServers` JSON text. A server with a problem is left out and named in `problems`; an
/// unreadable text gives the problem under the empty name. Port of `mcp.parse_servers`.
pub fn parse_servers(text: &str) -> (Map<String, Value>, Vec<(String, String)>) {
    let text = text.trim();
    if text.is_empty() {
        return (Map::new(), vec![]);
    }
    let doc: Value = match serde_json::from_str(text) {
        Ok(d) => d,
        Err(e) => {
            return (
                Map::new(),
                vec![(
                    String::new(),
                    format!(
                        "The MCP servers text is not valid JSON: {}",
                        json_error_text(text, &e)
                    ),
                )],
            )
        }
    };
    let doc = match &doc {
        Value::Object(o) if o.get("mcpServers").is_some_and(Value::is_object) => {
            o["mcpServers"].clone()
        }
        _ => doc,
    };
    let Value::Object(entries) = doc else {
        return (
            Map::new(),
            vec![(
                String::new(),
                "Expected {\"mcpServers\": {\"name\": {\"command\": …} or {\"url\": …}}}".into(),
            )],
        );
    };
    let (mut servers, mut problems) = (Map::new(), vec![]);
    for (name, entry) in &entries {
        let Value::Object(e) = entry else {
            problems.push((name.clone(), "The entry must be an object".to_string()));
            continue;
        };
        if e.get("disabled") == Some(&json!(true)) {
            continue;
        }
        let kind = e
            .get("type")
            .filter(|v| py_truthy(v))
            .map(py_str)
            .unwrap_or_default()
            .to_lowercase();
        let cmd = e
            .get("command")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|c| !c.is_empty());
        let url = e
            .get("url")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|c| !c.is_empty());
        if let Some(cmd) = cmd {
            let args = e
                .get("args")
                .filter(|v| py_truthy(v))
                .cloned()
                .unwrap_or(json!([]));
            let env = e
                .get("env")
                .filter(|v| py_truthy(v))
                .cloned()
                .unwrap_or(json!({}));
            let (Value::Array(args), Value::Object(env)) = (&args, &env) else {
                problems.push((
                    name.clone(),
                    "\"args\" must be a list and \"env\" an object".into(),
                ));
                continue;
            };
            servers.insert(name.clone(), json!({"transport": "stdio", "command": cmd, "args": args.iter().map(py_str).collect::<Vec<_>>(),
                "env": env.iter().map(|(k, v)| (k.clone(), json!(py_str(v)))).collect::<Map<String, Value>>(),
                "cwd": e.get("cwd").filter(|v| py_truthy(v)).map(py_str).unwrap_or_default()}));
        } else if let Some(url) = url {
            let headers = e
                .get("headers")
                .filter(|v| py_truthy(v))
                .cloned()
                .unwrap_or(json!({}));
            let host_ok = url
                .split_once("://")
                .map(|(sc, rest)| {
                    (sc == "http" || sc == "https")
                        && !rest
                            .split(['/', '?', '#'])
                            .next()
                            .unwrap_or("")
                            .rsplit('@')
                            .next()
                            .unwrap_or("")
                            .split(':')
                            .next()
                            .unwrap_or("")
                            .is_empty()
                })
                .unwrap_or(false);
            if !host_ok {
                problems.push((
                    name.clone(),
                    "The URL must start with http:// or https://".into(),
                ));
            } else if !headers.is_object() {
                problems.push((name.clone(), "\"headers\" must be an object".into()));
            } else if e
                .get("oauth")
                .is_some_and(|o| !o.is_null() && !o.is_object())
            {
                problems.push((
                    name.clone(),
                    "\"oauth\" must be an object like {\"client_id\": \"…\", \"scope\": \"…\"}"
                        .into(),
                ));
            } else {
                let mut spec = json!({"transport": if kind == "sse" { "sse" } else { "http" }, "url": url,
                    "headers": headers.as_object().unwrap().iter().map(|(k, v)| (k.clone(), json!(py_str(v)))).collect::<Map<String, Value>>()});
                if let Some(o) = e
                    .get("oauth")
                    .filter(|v| py_truthy(v))
                    .and_then(Value::as_object)
                {
                    let picked: Map<String, Value> = ["client_id", "client_secret", "scope"]
                        .iter()
                        .filter(|k| o.get(**k).is_some_and(py_truthy))
                        .map(|k| (k.to_string(), json!(py_str(&o[*k]))))
                        .collect();
                    spec["oauth"] = Value::Object(picked);
                }
                servers.insert(name.clone(), spec);
            }
        } else {
            problems.push((
                name.clone(),
                "Needs a \"command\" (a program to start) or a \"url\" (a Streamable HTTP server)"
                    .into(),
            ));
        }
    }
    (servers, problems)
}

/// Python's wording of a JSON error: `Expecting value: line 1 column 1 (char 0)`.
fn json_error_text(text: &str, e: &serde_json::Error) -> String {
    let (line, col) = (e.line(), e.column());
    let pos: usize = text
        .split('\n')
        .take(line.saturating_sub(1))
        .map(|l| l.chars().count() + 1)
        .sum::<usize>()
        + col.saturating_sub(1);
    let msg = match e.classify() {
        serde_json::error::Category::Eof => "Expecting value",
        _ => {
            let s = e.to_string();
            if s.contains("key must be a string") {
                "Expecting property name enclosed in double quotes"
            } else if s.contains("expected `,` or") {
                "Expecting ',' delimiter"
            } else if s.contains("expected `:`") {
                "Expecting ':' delimiter"
            } else if s.contains("trailing characters") {
                "Extra data"
            } else {
                "Expecting value"
            }
        }
    };
    format!("{msg}: line {line} column {col} (char {pos})")
}

/// What the page shows about the tools (`ToolBox.status`). MCP servers are listed but not connected yet.
pub fn status(s: &Value) -> Value {
    let on = |k: &str| s.get(k).is_some_and(py_truthy);
    let (servers, mut problems) = parse_servers(
        &s.get("mcp_json")
            .filter(|v| py_truthy(v))
            .map(py_str)
            .unwrap_or_default(),
    );
    let mcp_on = on("enabled") && on("mcp_enabled");
    let servers = if mcp_on { servers } else { Map::new() };
    if !mcp_on {
        problems.clear(); // the servers are off, so a mistake in their text does not matter yet
    }
    let trusted: Vec<String> = s
        .get("mcp_trusted")
        .and_then(Value::as_array)
        .map(|a| a.iter().map(py_str).collect())
        .unwrap_or_default();
    let mut list: Vec<Value> = servers
        .iter()
        .map(|(name, spec)| {
            let remote = spec["transport"] != "stdio";
            let auth = if remote { json!({"remote": true, "required": false, "client_id_set": spec.get("oauth").and_then(|o| o.get("client_id")).is_some_and(py_truthy)}) } else { json!({"remote": false}) };
            json!({"name": name, "status": "error", "error": "MCP servers are not supported by this server yet.", "tools": 0, "transport": spec["transport"], "trusted": trusted.contains(name),
                "info": "", "auth": auth, "legacy_sse": false})
        })
        .collect();
    for (name, why) in problems {
        list.push(json!({"name": if name.is_empty() { "(settings)".to_string() } else { name }, "status": "error", "error": why, "tools": 0, "transport": "", "trusted": false, "info": "", "auth": {"remote": false}, "legacy_sse": false}));
    }
    let max_rounds = match s.get("max_rounds") {
        Some(v) if py_truthy(v) => v
            .as_i64()
            .or_else(|| v.as_f64().map(|f| f as i64))
            .unwrap_or(8),
        _ => 8,
    };
    json!({"enabled": on("enabled"), "max_rounds": max_rounds, "tools": tools(s).iter().map(Tool::public).collect::<Vec<_>>(), "servers": list, "connecting": false, "files_dir_ok": files_dir_ok(s)})
}
