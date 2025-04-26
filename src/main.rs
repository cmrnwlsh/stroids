#![feature(iter_map_windows)]

use std::{
    f64::consts::FRAC_PI_2,
    io::{Stdout, stdout},
    iter,
    ops::ControlFlow,
    panic::{set_hook, take_hook},
    sync::mpsc::{Receiver, channel},
    thread,
    time::Duration,
};

use color_eyre::eyre::Result;
use ratatui::{
    Terminal,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    prelude::CrosstermBackend,
};

const TURN_SPEED: f64 = 2.;

fn main() -> Result<()> {
    color_eyre::install()?;
    Game::init()?.run()
}

struct Game {
    tui: Tui,
    ents: Ents,
    dt: Duration,
}

impl Game {
    fn init() -> Result<Self> {
        Ok(Self {
            tui: Tui::init()?,
            ents: Ents::default(),
            dt: Duration::default(),
        })
    }

    fn run(&mut self) -> Result<()> {
        let g = self;
        let target_frame_time = Duration::from_secs_f64(1.0 / 60.0); // 60 FPS
        let interval = Duration::from_secs_f64(1.0 / 120.0); // Physics at 120Hz

        let mut last_time = std::time::Instant::now();
        let mut accumulator = Duration::default();

        loop {
            // Calculate delta time
            let current_time = std::time::Instant::now();
            g.dt = current_time.duration_since(last_time);
            last_time = current_time;

            // Handle input
            if g.handle_input().is_break() {
                break; // Exit loop if handle_input returns false
            }

            // Store previous state before updating
            for Entity { xf_last, xf, .. } in g.ents.iter_mut() {
                *xf_last = xf.clone();
            }

            // Fixed timestep physics updates
            accumulator += g.dt;
            while accumulator >= interval {
                // Update game state with fixed timestep
                g.update()?;

                accumulator -= interval;
            }

            // Calculate interpolation factor (alpha) between 0.0 and 1.0
            let alpha = accumulator.as_secs_f64() / interval.as_secs_f64();

            // Render with interpolation between previous and current state
            g.draw(alpha)?;

            // Cap frame rate
            let frame_time = current_time.elapsed();
            if frame_time < target_frame_time {
                thread::sleep(target_frame_time - frame_time);
            }
        }

        Ok(())
    }

    fn handle_input(&mut self) -> ControlFlow<()> {
        let g = self;
        ControlFlow::Continue(for action in g.tui.input.try_iter() {
            match action {
                Action::InputLeft => g.ents.player.w -= TURN_SPEED * g.dt.as_secs_f64(),
                Action::InputRight => todo!(),
                Action::InputForward => todo!(),
                Action::InputReverse => todo!(),
                Action::InputFire => todo!(),
                Action::Exit => return ControlFlow::Break(()),
            }
        })
    }

    fn update(&mut self) -> Result<()> {
        Ok(())
    }

    fn draw(&mut self, alpha: f64) -> Result<()> {
        Ok(())
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

enum EntityKind {
    Player,
    Asteroid,
    Projectile,
}

struct Entity {
    kind: EntityKind,
    xf: Transform,
    xf_last: Transform,
    v: Vec2,
    w: f64,
    vertices: Vec<Vec2>,
}

impl Entity {
    fn new(kind: EntityKind) -> Self {
        let (xf, v, w, vertices) = match kind {
            EntityKind::Player => (
                Transform {
                    pos: Vec2::default(),
                    rot: -FRAC_PI_2,
                    scale: 1.,
                },
                Vec2::default(),
                0.,
                [(-1., 1.), (1., 0.), (-1., -1.)].map(Vec2::from).to_vec(),
            ),
            EntityKind::Asteroid => todo!(),
            EntityKind::Projectile => todo!(),
        };
        let xf_last = xf.clone();
        Self {
            kind,
            xf,
            xf_last,
            v,
            w,
            vertices,
        }
    }
}

impl Default for Entity {
    fn default() -> Self {
        Self::new(EntityKind::Player)
    }
}

#[derive(Default)]
struct Ents {
    player: Entity,
    asteroids: Vec<Entity>,
    projectiles: Vec<Entity>,
}

impl Ents {
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
