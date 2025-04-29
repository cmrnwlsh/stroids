#![feature(array_windows)]

use std::{
    collections::{HashMap, HashSet},
    f64::consts::FRAC_PI_2,
    io::{Stdout, stdout},
    iter,
    ops::{AddAssign, ControlFlow, Mul, SubAssign},
    panic::{set_hook, take_hook},
    thread,
    time::{Duration, Instant},
};

use color_eyre::eyre::Result;
use ratatui::{
    Frame, Terminal,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    layout::Rect,
    prelude::CrosstermBackend,
    style::Color,
    text::Line,
    widgets::{
        Block,
        canvas::{self, Canvas, Context},
    },
};

const FRAME_TIME: f64 = 1. / 60.;
const UPDATE_INTERVAL: f64 = 1. / 120.;
const THRUST_POWER: f64 = 0.005;
const TURN_POWER: f64 = 0.015;
const LINEAR_DAMPING: f64 = 0.99;
const ANGULAR_DAMPING: f64 = 0.95;

fn main() -> Result<()> {
    color_eyre::install()?;
    Game::init()?.run()
}

struct Game {
    tui: Tui,
    ents: Ents,
}

impl Game {
    fn init() -> Result<Self> {
        Ok(Self {
            tui: Tui::init()?,
            ents: Ents::init(),
        })
    }

    fn run(&mut self) -> Result<()> {
        let g = self;
        let target_frame_time = Duration::from_secs_f64(FRAME_TIME);
        let interval = Duration::from_secs_f64(UPDATE_INTERVAL);

        let mut last_time = Instant::now();
        let mut accumulator = Duration::default();
        let mut dt;

        Ok(loop {
            let current_time = Instant::now();
            dt = current_time.duration_since(last_time);
            last_time = current_time;

            if g.handle_input()?.is_break() {
                break;
            }

            for Entity {
                xfs: (last, current),
                ..
            } in g.ents.iter_mut()
            {
                *last = current.clone();
            }

            accumulator += dt;
            while accumulator >= interval {
                g.update()?;
                accumulator -= interval;
            }

            g.draw(accumulator.as_secs_f64() / UPDATE_INTERVAL)?;

            let frame_time = current_time.elapsed();
            if frame_time < target_frame_time {
                thread::sleep(target_frame_time - frame_time);
            }
        })
    }

    fn handle_input(&mut self) -> Result<ControlFlow<()>> {
        let input = &mut self.tui.input;
        let player = &mut self.ents.player;

        input.update();

        while event::poll(Duration::from_millis(0))? {
            if let Event::Key(ev) = event::read()? {
                input.process(ev);
                if ev.modifiers.contains(KeyModifiers::CONTROL) && ev.code == KeyCode::Char('c') {
                    return Ok(ControlFlow::Break(()));
                }
            }
        }

        if input.active(Action::InputLeft) {
            player.w = TURN_POWER;
        } else if input.active(Action::InputRight) {
            player.w = -TURN_POWER;
        } else {
            player.w = 0.0;
        }
        if input.active(Action::InputForward) {
            player.v += Vec2::from(player.xfs.1.rot) * THRUST_POWER;
        } else if input.active(Action::InputReverse) {
            player.v -= Vec2::from(player.xfs.1.rot) * THRUST_POWER;
        }
        if input.active(Action::InputFire) {
            todo!()
        }

        Ok(ControlFlow::Continue(()))
    }

    fn update(&mut self) -> Result<()> {
        self.ents.player.v.x *= LINEAR_DAMPING;
        self.ents.player.v.y *= LINEAR_DAMPING;
        self.ents.player.w *= ANGULAR_DAMPING;
        let xf = &mut self.ents.player.xfs.1;
        xf.pos += self.ents.player.v;
        xf.rot += self.ents.player.w;
        Ok(())
    }

    fn draw(&mut self, alpha: f64) -> Result<()> {
        let block = Block::bordered()
            .title("STROIDS")
            .title(Line::from(" - WASD: Movement - Space: Fire - Ctrl+C: Exit - ").right_aligned());
        let painter = |ctx: &mut Context| {
            for shape in self.ents.iter().map(|ent| {
                let (last, current) = &ent.xfs;
                let xf = last.interpolate(current, alpha);
                let (r, s, p) = (xf.rot, xf.scale, xf.pos);
                ent.vertices
                    .iter()
                    .chain(iter::once(&ent.vertices[0]))
                    .map(|v| Vec2 {
                        x: ((v.x * r.cos() - v.y * r.sin()) * s) + p.x,
                        y: ((v.x * r.sin() + v.y * r.cos()) * s) + p.y,
                    })
                    .collect::<Vec<_>>()
            }) {
                for [v1, v2] in shape.array_windows() {
                    ctx.draw(&canvas::Line::new(v1.x, v1.y, v2.x, v2.y, Color::White))
                }
            }
        };
        let render_callback = |frame: &mut Frame| {
            let Rect { width, height, .. } = frame.area();
            let (w, h) = (width as f64 / 2. + 4., height as f64 / 2. + 4.);
            frame.render_widget(
                Canvas::default()
                    .block(block)
                    .x_bounds([-w, w])
                    .y_bounds([-h, h])
                    .paint(painter),
                frame.area(),
            );
        };
        Ok(self.tui.term.draw(render_callback)?).map(|_| ())
    }
}

#[derive(Clone, Copy)]
enum Action {
    InputLeft,
    InputRight,
    InputForward,
    InputReverse,
    InputFire,
}

#[derive(Default)]
struct Entity {
    xfs: (Transform, Transform),
    v: Vec2,
    w: f64,
    vertices: Vec<Vec2>,
}

struct Ents {
    player: Entity,
    asteroids: Vec<Entity>,
    projectiles: Vec<Entity>,
}

impl Ents {
    fn init() -> Self {
        let xf = Transform {
            pos: Vec2::default(),
            rot: FRAC_PI_2,
            scale: 1.,
        };
        let xfs = (xf.clone(), xf.clone());
        Self {
            player: Entity {
                xfs,
                v: Vec2::default(),
                w: 0.,
                vertices: [(-1., 1.), (1., 0.), (-1., -1.)].map(Vec2::from).to_vec(),
            },
            asteroids: vec![],
            projectiles: vec![],
        }
    }

    fn iter(&self) -> impl Iterator<Item = &Entity> {
        let g = self;
        iter::once(&g.player)
            .chain(&g.asteroids)
            .chain(&g.projectiles)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Entity> {
        let g = self;
        iter::once(&mut g.player)
            .chain(&mut g.asteroids)
            .chain(&mut g.projectiles)
    }
}

#[derive(Default, Clone)]
struct Transform {
    pos: Vec2,
    rot: f64,
    scale: f64,
}

impl Transform {
    fn interpolate(&self, other: &Transform, alpha: f64) -> Transform {
        let t = self;
        Transform {
            pos: Vec2 {
                x: t.pos.x + (other.pos.x - t.pos.x) * alpha,
                y: t.pos.y + (other.pos.y - t.pos.y) * alpha,
            },
            rot: t.rot + (other.rot - t.rot) * alpha,
            scale: t.scale + (other.scale - t.scale) * alpha,
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl From<[f64; 2]> for Vec2 {
    fn from(value: [f64; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl From<(f64, f64)> for Vec2 {
    fn from(value: (f64, f64)) -> Self {
        Self {
            x: value.0,
            y: value.1,
        }
    }
}

impl From<f64> for Vec2 {
    fn from(value: f64) -> Self {
        Self {
            x: value.cos(),
            y: value.sin(),
        }
    }
}

impl<T> AddAssign<T> for Vec2
where
    T: Into<Vec2>,
{
    fn add_assign(&mut self, rhs: T) {
        let Vec2 { x: x_rhs, y: y_rhs } = rhs.into();
        *self = Self {
            x: self.x + x_rhs,
            y: self.y + y_rhs,
        }
    }
}

impl<T> SubAssign<T> for Vec2
where
    T: Into<Vec2>,
{
    fn sub_assign(&mut self, rhs: T) {
        let Vec2 { x: x_rhs, y: y_rhs } = rhs.into();
        *self = Self {
            x: self.x - x_rhs,
            y: self.y - y_rhs,
        }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f64) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

struct InputState {
    pressed_keys: HashSet<KeyCode>,
    key_timers: HashMap<KeyCode, Instant>, // Track time for each key
    modifiers: KeyModifiers,
    auto_release_duration: Duration,
    release_enabled: bool,
}

impl InputState {
    fn init() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            key_timers: HashMap::new(),
            modifiers: KeyModifiers::empty(),
            auto_release_duration: Duration::from_millis(500),
            release_enabled: false, // Initially assume terminal doesn't send releases
        }
    }

    fn active(&self, action: Action) -> bool {
        match action {
            Action::InputLeft => {
                self.pressed_keys.contains(&KeyCode::Char('a'))
                    || self.pressed_keys.contains(&KeyCode::Left)
            }
            Action::InputRight => {
                self.pressed_keys.contains(&KeyCode::Char('d'))
                    || self.pressed_keys.contains(&KeyCode::Right)
            }
            Action::InputForward => {
                self.pressed_keys.contains(&KeyCode::Char('w'))
                    || self.pressed_keys.contains(&KeyCode::Up)
            }
            Action::InputReverse => {
                self.pressed_keys.contains(&KeyCode::Char('s'))
                    || self.pressed_keys.contains(&KeyCode::Down)
            }
            Action::InputFire => self.pressed_keys.contains(&KeyCode::Char(' ')),
        }
    }

    fn process(&mut self, key_event: KeyEvent) {
        match key_event.kind {
            KeyEventKind::Press => {
                self.pressed_keys.insert(key_event.code);
                self.key_timers.insert(key_event.code, Instant::now());
                self.modifiers = key_event.modifiers;
            }
            KeyEventKind::Release => {
                self.pressed_keys.remove(&key_event.code);
                self.key_timers.remove(&key_event.code);
                self.release_enabled = true;
            }
            KeyEventKind::Repeat => {
                if let Some(timer) = self.key_timers.get_mut(&key_event.code) {
                    *timer = Instant::now();
                }
            }
        }
    }

    fn update(&mut self) {
        if !self.release_enabled {
            let now = Instant::now();
            let expired_keys: Vec<KeyCode> = self
                .key_timers
                .iter()
                .filter(|&(_, time)| now.duration_since(*time) > self.auto_release_duration)
                .map(|(key, _)| *key)
                .collect();

            for key in expired_keys {
                self.pressed_keys.remove(&key);
                self.key_timers.remove(&key);
            }
        }
    }
}

struct Tui {
    term: Terminal<CrosstermBackend<Stdout>>,
    input: InputState,
}

impl Tui {
    fn init() -> Result<Self> {
        execute!(stdout(), EnterAlternateScreen)?;
        enable_raw_mode()?;
        let hook = take_hook();
        set_hook(Box::new(move |panic_info| {
            let _ = Self::restore();
            hook(panic_info);
        }));
        Ok(Self {
            term: Terminal::new(CrosstermBackend::new(stdout()))?,
            input: InputState::init(),
        })
    }

    fn restore() -> Result<()> {
        execute!(stdout(), LeaveAlternateScreen)?;
        Ok(disable_raw_mode()?)
    }
}

impl Drop for Tui {
    fn drop(&mut self) {
        let _ = Self::restore();
    }
}
