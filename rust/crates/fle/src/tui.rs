use anyhow::Result;
use crossterm::event::{self, Event as CEvent, KeyCode, KeyEventKind, KeyModifiers};
use fle_core::{catalog, chat, engines, flow, server::Server, system, Event};
use ratatui::{
    layout::{Constraint, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Gauge, List, ListItem, ListState, Paragraph, Wrap},
    DefaultTerminal, Frame,
};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Duration;

enum Msg {
    Ev(Event),
    Srv(Server),
    Err(String),
    Tok(bool, String),
    Done(Result<chat::Stats, String>),
}

#[derive(PartialEq, Clone, Copy)]
enum Screen {
    Machine,
    Models,
    Loading,
    Chat,
}

struct Ready {
    base: String,
    engine: String,
    model: String,
    command: String,
}

struct App {
    home: PathBuf,
    screen: Screen,
    sys: system::SysInfo,
    models: Vec<catalog::Model>,
    list: ListState,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    steps: Vec<String>,
    progress: Option<(String, u64, u64)>,
    logline: String,
    ready: Option<Ready>,
    server: Option<Server>,
    history: Vec<(String, String)>,
    reasoning: String,
    input: String,
    busy: bool,
    stats: chat::Stats,
    err: Option<String>,
    show_cmd: bool,
    quit: bool,
}

pub fn run(home: &Path) -> Result<()> {
    let sys = system::detect(home);
    let (tx, rx) = channel();
    let models = catalog::builtin();
    let mut list = ListState::default();
    list.select(Some(0));
    let mut app = App {
        home: home.to_path_buf(),
        screen: Screen::Machine,
        sys,
        models,
        list,
        tx,
        rx,
        steps: vec![],
        progress: None,
        logline: String::new(),
        ready: None,
        server: None,
        history: vec![],
        reasoning: String::new(),
        input: String::new(),
        busy: false,
        stats: Default::default(),
        err: None,
        show_cmd: false,
        quit: false,
    };
    let mut term = ratatui::init();
    let res = main_loop(&mut term, &mut app);
    ratatui::restore();
    drop(app.server.take()); // stops the engine
    res
}

fn main_loop(term: &mut DefaultTerminal, app: &mut App) -> Result<()> {
    while !app.quit {
        while let Ok(m) = app.rx.try_recv() {
            app.on_msg(m);
        }
        term.draw(|f| app.draw(f))?;
        if event::poll(Duration::from_millis(60))? {
            if let CEvent::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    app.on_key(k.code, k.modifiers);
                }
            }
        }
    }
    Ok(())
}

impl App {
    fn on_msg(&mut self, m: Msg) {
        match m {
            Msg::Ev(Event::Step(s)) => {
                self.steps.push(s);
                self.progress = None;
            }
            Msg::Ev(Event::Progress { label, done, total }) => {
                self.progress = Some((label, done, total))
            }
            Msg::Ev(Event::Log(l)) => self.logline = l,
            Msg::Ev(Event::Ready {
                base,
                engine,
                model,
                command,
            }) => {
                self.ready = Some(Ready {
                    base,
                    engine,
                    model,
                    command,
                });
                self.screen = Screen::Chat;
                self.progress = None;
            }
            Msg::Srv(s) => self.server = Some(s),
            Msg::Err(e) => {
                self.err = Some(e);
                self.screen = Screen::Models;
            }
            Msg::Tok(r, t) => {
                if r {
                    self.reasoning.push_str(&t)
                } else if let Some(last) = self.history.last_mut() {
                    last.1.push_str(&t)
                }
            }
            Msg::Done(r) => {
                self.busy = false;
                match r {
                    Ok(s) => self.stats = s,
                    Err(e) => self.err = Some(e),
                }
            }
        }
    }

    fn on_key(&mut self, code: KeyCode, mods: KeyModifiers) {
        if mods.contains(KeyModifiers::CONTROL) && code == KeyCode::Char('c') {
            self.quit = true;
            return;
        }
        match self.screen {
            Screen::Machine => match code {
                KeyCode::Enter => self.screen = Screen::Models,
                KeyCode::Char('q') | KeyCode::Esc => self.quit = true,
                _ => {}
            },
            Screen::Models => match code {
                KeyCode::Up => {
                    let i = self.list.selected().unwrap_or(0);
                    self.list.select(Some(i.saturating_sub(1)));
                }
                KeyCode::Down => {
                    let i = self.list.selected().unwrap_or(0);
                    self.list.select(Some((i + 1).min(self.models.len() - 1)));
                }
                KeyCode::Enter => self.start_load(),
                KeyCode::Esc | KeyCode::Left => self.screen = Screen::Machine,
                KeyCode::Char('q') => self.quit = true,
                _ => {}
            },
            Screen::Loading => {
                if code == KeyCode::Char('q') {
                    self.quit = true
                }
            }
            Screen::Chat => match code {
                KeyCode::Enter if !self.busy && !self.input.trim().is_empty() => self.send(),
                KeyCode::Char('o') if mods.contains(KeyModifiers::CONTROL) => {
                    self.show_cmd = !self.show_cmd
                }
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Esc => self.input.clear(),
                KeyCode::Char(c) if !mods.contains(KeyModifiers::CONTROL) => self.input.push(c),
                _ => {}
            },
        }
    }

    fn start_load(&mut self) {
        let m = self.models[self.list.selected().unwrap_or(0)].clone();
        if catalog::fit(&m, &self.sys) == catalog::Fit::TooBig {
            self.err = Some(format!(
                "{} does not fit this machine ({} memory).",
                m.title,
                system::gib(self.sys.ram_bytes)
            ));
            return;
        }
        self.err = None;
        self.steps.clear();
        self.progress = None;
        self.logline.clear();
        self.screen = Screen::Loading;
        let (tx, home, id) = (self.tx.clone(), self.home.clone(), m.id.to_string());
        std::thread::spawn(move || {
            let tx2 = tx.clone();
            match flow::ensure_and_start(&home, &id, &mut |e| {
                let _ = tx2.send(Msg::Ev(e));
            }) {
                Ok(srv) => {
                    let _ = tx.send(Msg::Srv(srv));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Err(format!("{e:#}")));
                }
            }
        });
    }

    fn send(&mut self) {
        let Some(r) = &self.ready else { return };
        let text = std::mem::take(&mut self.input);
        self.history.push(("user".into(), text));
        let hist = self.history.clone();
        self.history.push(("assistant".into(), String::new()));
        self.reasoning.clear();
        self.busy = true;
        self.err = None;
        let (tx, base, model) = (self.tx.clone(), r.base.clone(), r.model.clone());
        std::thread::spawn(move || {
            let t2 = tx.clone();
            let res = chat::stream(&base, &model, &hist, &mut |r, t| {
                let _ = t2.send(Msg::Tok(r, t.to_string()));
            });
            let _ = tx.send(Msg::Done(res.map_err(|e| format!("{e:#}"))));
        });
    }

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let [head, body, foot] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(3),
            Constraint::Length(1),
        ])
        .areas(area);
        let title = Line::from(vec![
            Span::styled(
                " FLE ",
                Style::new()
                    .bg(Color::Magenta)
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(match self.screen {
                Screen::Machine => " this machine",
                Screen::Models => " choose a model",
                Screen::Loading => " loading",
                Screen::Chat => " chat",
            }),
        ]);
        f.render_widget(Paragraph::new(title), head);
        let hint = match self.screen {
            Screen::Machine => "Enter: continue   q: quit",
            Screen::Models => "up/down: choose   Enter: load   Esc: back   q: quit",
            Screen::Loading => "q: quit (stops the engine)",
            Screen::Chat => {
                "Enter: send   Ctrl+O: show the start command   Esc: clear   Ctrl+C: quit"
            }
        };
        f.render_widget(
            Paragraph::new(hint).style(Style::new().fg(Color::DarkGray)),
            foot,
        );
        match self.screen {
            Screen::Machine => self.machine(f, body),
            Screen::Models => self.models_view(f, body),
            Screen::Loading => self.loading(f, body),
            Screen::Chat => self.chat_view(f, body),
        }
    }

    fn machine(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let s = &self.sys;
        let gpu = if s.gpus.is_empty() {
            "none".to_string()
        } else {
            s.gpus
                .iter()
                .map(|g| g.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let vs = engines::variants(s);
        let best = vs.first();
        let have = engines::installed(&self.home);
        let mut t = vec![
            Line::from(""),
            kv("Processor", &format!("{} ({} cores)", s.cpu, s.cores)),
            kv("Memory", &system::gib(s.ram_bytes)),
            kv("GPU", &gpu),
            kv("GPU may use", &system::gib(s.gpu_usable_bytes)),
            kv("Disk free", &system::gib(s.disk_free_bytes)),
            Line::from(""),
        ];
        if let Some(b) = best {
            t.push(Line::from(Span::styled(
                format!("  Best engine for this machine: {}", b.label),
                Style::new().add_modifier(Modifier::BOLD).fg(Color::Green),
            )));
            t.push(Line::from(format!("  {}", b.why)));
            t.push(Line::from(if have.is_empty() {
                "  Not installed yet: the first load installs it for you.".to_string()
            } else {
                format!(
                    "  Installed: {}",
                    have.iter()
                        .map(|(n, _)| n.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }));
        }
        f.render_widget(
            Paragraph::new(t).block(Block::bordered().title(" Your machine ")),
            area,
        );
    }

    fn models_view(&mut self, f: &mut Frame, area: ratatui::layout::Rect) {
        let [top, bottom] =
            Layout::vertical([Constraint::Min(5), Constraint::Length(5)]).areas(area);
        let items: Vec<ListItem> = self
            .models
            .iter()
            .map(|m| {
                let fit = catalog::fit(m, &self.sys);
                let col = match fit {
                    catalog::Fit::Fits => Color::Green,
                    catalog::Fit::Tight => Color::Yellow,
                    catalog::Fit::TooBig => Color::Red,
                };
                let got = flow::model_path(&self.home, m).exists();
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!(" {:<8}", fit.label()),
                        Style::new().fg(col).add_modifier(Modifier::BOLD),
                    ),
                    Span::raw(format!(" {:>9}  {}", system::gib(m.size), m.title)),
                    Span::styled(
                        if got { "  [downloaded]" } else { "" },
                        Style::new().fg(Color::Cyan),
                    ),
                ]))
            })
            .collect();
        let list = List::new(items)
            .block(
                Block::bordered()
                    .title(" Models (whether they fit is checked before anything is downloaded) "),
            )
            .highlight_style(Style::new().add_modifier(Modifier::REVERSED))
            .highlight_symbol("> ");
        f.render_stateful_widget(list, top, &mut self.list);
        let sel = &self.models[self.list.selected().unwrap_or(0)];
        let mut lines = vec![Line::from(sel.note.to_string())];
        if let Some(e) = &self.err {
            lines.push(Line::from(Span::styled(
                e.clone(),
                Style::new().fg(Color::Red),
            )));
        }
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::bordered().title(" About ")),
            bottom,
        );
    }

    fn loading(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let [steps, gauge, log] = Layout::vertical([
            Constraint::Min(4),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .areas(area);
        let lines: Vec<Line> = self
            .steps
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let last = i + 1 == self.steps.len();
                Line::from(vec![
                    Span::styled(
                        if last { " ... " } else { " ok  " },
                        Style::new().fg(if last { Color::Yellow } else { Color::Green }),
                    ),
                    Span::raw(s.clone()),
                ])
            })
            .collect();
        f.render_widget(
            Paragraph::new(lines)
                .wrap(Wrap { trim: true })
                .block(Block::bordered().title(" Getting ready ")),
            steps,
        );
        let (label, ratio) = match &self.progress {
            Some((l, d, t)) if *t > 0 => (
                format!("{l}  {} / {}", system::gib(*d), system::gib(*t)),
                *d as f64 / *t as f64,
            ),
            _ => (String::new(), 0.0),
        };
        f.render_widget(
            Gauge::default()
                .block(Block::bordered().title(" Download "))
                .gauge_style(Style::new().fg(Color::Magenta))
                .ratio(ratio.clamp(0.0, 1.0))
                .label(label),
            gauge,
        );
        f.render_widget(
            Paragraph::new(self.logline.clone())
                .style(Style::new().fg(Color::DarkGray))
                .block(Block::bordered().title(" Engine ")),
            log,
        );
    }

    fn chat_view(&self, f: &mut Frame, area: ratatui::layout::Rect) {
        let r = self.ready.as_ref();
        let extra = if self.show_cmd { 6 } else { 0 };
        let [status, convo, input] = Layout::vertical([
            Constraint::Length(3 + extra),
            Constraint::Min(3),
            Constraint::Length(3),
        ])
        .areas(area);
        let tps = |v: Option<f64>, d: usize| v.map(|x| format!("{x:.d$}")).unwrap_or("-".into());
        let mut sl = vec![Line::from(format!(
            " {}  on  {}   |   decode {} tok/s   prompt {} tok/s   {} tokens",
            r.map(|r| r.model.as_str()).unwrap_or(""),
            r.map(|r| r.engine.as_str()).unwrap_or(""),
            tps(self.stats.decode_tps, 1),
            tps(self.stats.prefill_tps, 0),
            self.stats.tokens
        ))];
        if self.show_cmd {
            if let Some(r) = r {
                sl.push(Line::from(Span::styled(
                    format!(" API {}/v1 \n {}", r.base, r.command),
                    Style::new().fg(Color::DarkGray),
                )));
            }
        }
        f.render_widget(
            Paragraph::new(sl)
                .wrap(Wrap { trim: true })
                .block(Block::bordered().title(" Live ")),
            status,
        );
        let mut t = Text::default();
        for (role, text) in &self.history {
            let (name, col) = if role == "user" {
                ("you", Color::Cyan)
            } else {
                ("fle", Color::Green)
            };
            t.lines.push(Line::from(Span::styled(
                name,
                Style::new().fg(col).add_modifier(Modifier::BOLD),
            )));
            if role == "assistant" && !self.reasoning.is_empty() && text.is_empty() {
                t.lines.push(Line::from(Span::styled(
                    format!(
                        "(thinking) {}",
                        self.reasoning
                            .chars()
                            .rev()
                            .take(300)
                            .collect::<String>()
                            .chars()
                            .rev()
                            .collect::<String>()
                    ),
                    Style::new().fg(Color::DarkGray),
                )));
            }
            for l in text.lines() {
                t.lines.push(Line::from(l.to_string()));
            }
            if text.is_empty() && self.busy && role == "assistant" {
                t.lines.push(Line::from("..."));
            }
            t.lines.push(Line::from(""));
        }
        if let Some(e) = &self.err {
            t.lines.push(Line::from(Span::styled(
                e.clone(),
                Style::new().fg(Color::Red),
            )));
        }
        let h = convo.height.saturating_sub(2) as usize;
        let wrapped_est: usize = t
            .lines
            .iter()
            .map(|l| 1 + l.width() / (convo.width.max(10) as usize - 2))
            .sum();
        let scroll = wrapped_est.saturating_sub(h) as u16;
        f.render_widget(
            Paragraph::new(t)
                .wrap(Wrap { trim: false })
                .scroll((scroll, 0))
                .block(Block::bordered().title(" Conversation ")),
            convo,
        );
        f.render_widget(
            Paragraph::new(format!(
                "> {}{}",
                self.input,
                if self.busy { "" } else { "_" }
            ))
            .block(Block::bordered().title(if self.busy {
                " answering ... "
            } else {
                " your message "
            })),
            input,
        );
    }
}

fn kv(k: &str, v: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("  {k:<14}"), Style::new().fg(Color::DarkGray)),
        Span::raw(v.to_string()),
    ])
}
