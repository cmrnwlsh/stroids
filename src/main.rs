#![feature(array_windows)]

use std::{
    f64::consts::FRAC_PI_2,
    io::{Stdout, stdout},
    iter,
    ops::{AddAssign, ControlFlow, Mul, SubAssign},
    panic::{set_hook, take_hook},
    sync::mpsc::{Receiver, channel},
    thread,
    time::Duration,
};

use color_eyre::eyre::Result;
use ratatui::{
    Frame, Terminal,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
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

const THRUST_POWER: f64 = 2.;
const TURN_POWER: f64 = 2.;
const FRAME_TIME: f64 = 1. / 60.;
const UPDATE_INTERVAL: f64 = 1. / 120.;

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

        let mut last_time = std::time::Instant::now();
        let mut accumulator = Duration::default();
        let mut dt;

        Ok(loop {
            let current_time = std::time::Instant::now();
            dt = current_time.duration_since(last_time);
            last_time = current_time;

            if g.handle_input().is_break() {
                break;
            }

            for Entity {
                xf, xfs: (last, _), ..
            } in g.ents.iter_mut()
            {
                *last = xf.clone();
            }

            accumulator += dt;
            while accumulator >= interval {
                g.update(accumulator.as_secs_f64() / interval.as_secs_f64())?;
                accumulator -= interval;
            }
            g.draw()?;

            let frame_time = current_time.elapsed();
            if frame_time < target_frame_time {
                thread::sleep(target_frame_time - frame_time);
            }
        })
    }

    fn handle_input(&mut self) -> ControlFlow<()> {
        let player = &mut self.ents.player;
        ControlFlow::Continue(for action in self.tui.input.try_iter() {
            match action {
                Action::InputLeft => player.w -= TURN_POWER,
                Action::InputRight => player.w += TURN_POWER,
                Action::InputForward => player.v += Vec2::from(THRUST_POWER) * player.xf.rot,
                Action::InputReverse => player.v -= Vec2::from(THRUST_POWER) * player.xf.rot,
                Action::InputFire => todo!(),
                Action::Exit => return ControlFlow::Break(()),
            }
        })
    }

    fn update(&mut self, alpha: f64) -> Result<()> {
        Ok(())
    }

    fn draw(&mut self) -> Result<()> {
        let block = Block::bordered()
            .title("STROIDS")
            .title(Line::from(" - WASD: Movement - Space: Fire - Ctrl+C: Exit - ").right_aligned());
        let painter = |ctx: &mut Context| {
            for shape in self.ents.iter().map(|ent| {
                ent.vertices
                    .iter()
                    .chain(iter::once(&ent.vertices[0]))
                    .map(|v| {
                        let (r, s, p) = (&ent.xf.rot, &ent.xf.scale, &ent.xf.pos);
                        Vec2 {
                            x: ((v.x * r.cos() - v.y * r.sin()) * s) + p.x,
                            y: ((v.x * r.sin() + v.y * r.cos()) * s) + p.y,
                        }
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
            let (w, h) = (width as f64 / 2., height as f64 / 2.);
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

enum Action {
    InputLeft,
    InputRight,
    InputForward,
    InputReverse,
    InputFire,
    Exit,
}

#[derive(Clone, Copy, Default)]
enum EntityKind {
    #[default]
    Player,
    Asteroid,
    Projectile,
}

#[derive(Default)]
struct Entity {
    kind: EntityKind,
    xf: Transform,
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
                kind: EntityKind::Player,
                xf,
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

#[derive(Default, Clone)]
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

impl<T> Mul<T> for Vec2
where
    T: Into<Vec2>,
{
    type Output = Vec2;
    fn mul(self, rhs: T) -> Self::Output {
        let Vec2 { x: x_rhs, y: y_rhs } = rhs.into();
        Self {
            x: self.x * x_rhs,
            y: self.y * y_rhs,
        }
    }
}

struct Tui {
    term: Terminal<CrosstermBackend<Stdout>>,
    input: Receiver<Action>,
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
        let (tx, rx) = channel();
        thread::spawn(move || -> Result<()> {
            Ok(loop {
                let ev = event::read()?;
                if let Event::Key(KeyEvent {
                    code: KeyCode::Char('c'),
                    modifiers: KeyModifiers::CONTROL,
                    ..
                }) = ev
                {
                    break tx.send(Action::Exit)?;
                }
                if let Event::Key(KeyEvent {
                    code: KeyCode::Char(c),
                    ..
                }) = ev
                {
                    match c {
                        'w' => tx.send(Action::InputForward)?,
                        'a' => tx.send(Action::InputLeft)?,
                        's' => tx.send(Action::InputReverse)?,
                        'd' => tx.send(Action::InputRight)?,
                        ' ' => tx.send(Action::InputFire)?,
                        _ => (),
                    }
                }
            })
        });
        Ok(Self {
            term: Terminal::new(CrosstermBackend::new(stdout()))?,
            input: rx,
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
