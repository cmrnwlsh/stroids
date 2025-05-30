#![feature(array_windows, random)]

use std::{
    collections::{HashMap, HashSet},
    convert::identity,
    f64::consts::{FRAC_PI_2, FRAC_PI_4, PI},
    io::{Stdout, stdout},
    iter,
    ops::{Add, AddAssign, ControlFlow, Mul, Sub, SubAssign},
    panic::{set_hook, take_hook},
    random::random,
    thread,
    time::{Duration, Instant},
};

use color_eyre::eyre::{Report, Result, bail};
use ratatui::{
    Terminal,
    crossterm::{
        event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
        execute,
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    layout::{Rect, Size},
    prelude::CrosstermBackend,
    style::Color,
    text::Line,
    widgets::{
        Block, Padding, Paragraph,
        canvas::{self, Canvas, Context},
    },
};
use wyrand::{RandomWyHashState, WyRand};

const FRAME_TIME: f64 = 1. / 60.;
const UPDATE_INTERVAL: f64 = 1. / 120.;
const THRUST_POWER: f64 = 0.005;
const TURN_POWER: f64 = 0.0155;
const LINEAR_DAMPING: f64 = 0.99;
const ANGULAR_DAMPING: f64 = 0.93;
const INPUT_MS: u64 = 500;
const STROID_V: f64 = 0.1;
const STROID_TIME: Duration = Duration::from_secs(1);
const FIRE_DELAY: Duration = Duration::from_millis(500);
const PROJECTILE_V: f64 = 0.25;

fn main() -> Result<()> {
    color_eyre::install()?;
    Game::init()?.run()
}

struct Game {
    tui: Tui,
    ents: Ents,
    last_stroid: Instant,
    last_projectile: Instant,
    score: i32,
    lives: i32,
}

impl Game {
    fn init() -> Result<Self> {
        Ok(Self {
            tui: Tui::init()?,
            ents: Ents::init(),
            last_stroid: Instant::now(),
            last_projectile: Instant::now() - FIRE_DELAY,
            score: 0,
            lives: 3,
        })
    }

    fn restart(&mut self) {
        self.ents = Ents::init();
        self.last_stroid = Instant::now();
        self.last_projectile = Instant::now();
        self.score = 0;
        self.lives = 3;
    }

    fn run(&mut self) -> Result<()> {
        let target_frame_time = Duration::from_secs_f64(FRAME_TIME);
        let interval = Duration::from_secs_f64(UPDATE_INTERVAL);

        let mut last_time = Instant::now();
        let mut accumulator = Duration::default();
        let mut dt;

        Ok(loop {
            let now = Instant::now();
            dt = now - last_time;
            last_time = now;

            if self.handle_input()?.is_break() {
                break;
            }

            if self.lives > 0 {
                if now - self.last_stroid > STROID_TIME {
                    self.ents.random_stroid(self.tui.term.size()?)?;
                    self.last_stroid = now;
                }

                for Entity {
                    xfs: (last, current),
                    ..
                } in self.ents.iter_mut()
                {
                    *last = *current;
                }

                accumulator += dt;
                while accumulator >= interval {
                    self.update()?;
                    accumulator -= interval;
                }
            }

            self.draw(accumulator.as_secs_f64() / UPDATE_INTERVAL)?;

            let frame_time = now.elapsed();
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
        }
        if input.active(Action::InputForward) {
            player.v += Vec2::from(player.xfs.1.rot) * THRUST_POWER;
        } else if input.active(Action::InputReverse) {
            player.v -= Vec2::from(player.xfs.1.rot) * THRUST_POWER * 0.25;
        }
        Ok(ControlFlow::Continue(
            if input.active(Action::InputFire) && Instant::now() - self.last_projectile > FIRE_DELAY
            {
                if self.lives == 0 {
                    self.restart();
                } else {
                    self.ents.spawn_projectile();
                    self.last_projectile = Instant::now()
                }
            },
        ))
    }

    fn check_bounds(&mut self) -> Result<()> {
        let Size { width, height } = self.tui.term.size()?;
        let (w, h) = (width as f64 / 2., height as f64 / 2.);
        let bounds = |pos: &Vec2, offset: f64| {
            [
                pos.x < -w - offset,
                pos.y > h + offset,
                pos.x > w + offset,
                pos.y < -h - offset,
            ]
        };
        for ents in [&mut self.ents.stroids, &mut self.ents.projectiles] {
            ents.retain(|ent| {
                let pos = ent.xfs.1.pos;
                !bounds(&pos, 4.).into_iter().any(identity)
            });
        }
        let player = &mut self.ents.player;
        player.v.x *= LINEAR_DAMPING;
        player.v.y *= LINEAR_DAMPING;
        player.w *= ANGULAR_DAMPING;
        let (last, curr) = &mut player.xfs;
        let pos = &mut curr.pos;
        Ok(
            for (i, wrapped) in bounds(pos, 1.).into_iter().enumerate() {
                if wrapped {
                    match Side::try_from(i)? {
                        Side::Left => pos.x = w,
                        Side::Top => pos.y = -h,
                        Side::Right => pos.x = -w,
                        Side::Bottom => pos.y = h,
                    }
                    last.pos = *pos
                }
            },
        )
    }

    fn check_collisions(&mut self) {
        for pos in self
            .ents
            .stroids
            .iter_mut()
            .map(|stroid| {
                let (st, pt) = (stroid.xfs.1, self.ents.player.xfs.1);
                if st.pos.distance(&pt.pos) < st.scale + pt.scale {
                    stroid.should_remove = true;
                    self.lives -= 1;
                }
                self.ents
                    .projectiles
                    .iter_mut()
                    .filter_map(|projectile| {
                        let pt = projectile.xfs.1;
                        if st.pos.distance(&pt.pos) < st.scale + pt.scale {
                            stroid.should_remove = true;
                            projectile.should_remove = true;
                            self.score += 1;
                            if st.scale > 2. { Some(st.pos) } else { None }
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .flatten()
            .collect::<Vec<Vec2>>()
        {
            self.ents.explode_stroid(&pos);
        }
        for ents in [&mut self.ents.stroids, &mut self.ents.projectiles] {
            ents.retain(|e| !e.should_remove)
        }
    }

    fn update(&mut self) -> Result<()> {
        for ent in self.ents.iter_mut() {
            let xf = &mut ent.xfs.1;
            xf.pos += ent.v;
            xf.rot += ent.w;
        }
        self.check_bounds()?;
        Ok(self.check_collisions())
    }

    fn draw(&mut self, alpha: f64) -> Result<()> {
        let size = self.tui.term.size()?;
        let block = Block::bordered()
            .title("STROIDS")
            .title(
                Line::from(format!(
                    " - Lives: {} - Score: {} - ",
                    (0..self.lives).map(|_| 'Δ').collect::<String>(),
                    self.score,
                ))
                .centered(),
            )
            .title(Line::from(" - WASD: Movement - Space: Fire - Ctrl+C: Exit - ").right_aligned());
        Ok(self.tui.term.draw(|frame| {
            let Rect { width, height, .. } = frame.area();
            let (w, h) = (width as f64 / 2., height as f64 / 2.);
            let painter = |ctx: &mut Context| {
                for shape in self.ents.iter().map(|ent| {
                    let (last, current) = &ent.xfs;
                    let xf = last.interpolate(current, alpha);
                    let (r, s, p) = (xf.rot, xf.scale, xf.pos);
                    ent.vertices
                        .iter()
                        .chain(iter::once(&ent.vertices[0]))
                        .map(|v| Vec2 {
                            x: (((v.x * r.cos() - v.y * r.sin()) * s) + p.x).clamp(-w, w),
                            y: (((v.x * r.sin() + v.y * r.cos()) * s) + p.y).clamp(-h, h),
                        })
                        .collect::<Vec<_>>()
                }) {
                    for [v1, v2] in shape.array_windows() {
                        ctx.draw(&canvas::Line::new(v1.x, v1.y, v2.x, v2.y, Color::White))
                    }
                }
            };
            if self.lives > 0 {
                frame.render_widget(
                    Canvas::default()
                        .block(block)
                        .x_bounds([-w, w])
                        .y_bounds([-h, h])
                        .paint(painter),
                    frame.area(),
                )
            } else {
                frame.render_widget(
                    Paragraph::new(format!(
                        "GAME OVER\n\nFinal Score: {}\n\nPress Space To Restart",
                        self.score,
                    ))
                    .centered()
                    .block(block.padding(Padding {
                        top: size.height / 4,
                        ..Default::default()
                    })),
                    frame.area(),
                )
            };
        })?)
        .map(|_| ())
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
    should_remove: bool,
}

enum Side {
    Left,
    Top,
    Right,
    Bottom,
}

impl TryFrom<usize> for Side {
    type Error = Report;
    fn try_from(value: usize) -> Result<Self> {
        Ok(match value {
            0 => Self::Left,
            1 => Self::Top,
            2 => Self::Right,
            3 => Self::Bottom,
            _ => bail!("invalid side"),
        })
    }
}

struct Ents {
    player: Entity,
    stroids: Vec<Entity>,
    projectiles: Vec<Entity>,
    rng: WyRand,
}

impl Ents {
    fn init() -> Self {
        let xf = Transform {
            pos: Vec2::default(),
            rot: FRAC_PI_2,
            scale: 1.,
        };
        let xfs = (xf, xf);
        Self {
            player: Entity {
                xfs,
                vertices: [(-1., 1.), (1., 0.), (-1., -1.)].map(Vec2::from).into(),
                ..Default::default()
            },
            stroids: vec![],
            projectiles: vec![],
            rng: WyRand::new(random()),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &Entity> {
        iter::once(&self.player)
            .chain(&self.stroids)
            .chain(&self.projectiles)
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = &mut Entity> {
        iter::once(&mut self.player)
            .chain(&mut self.stroids)
            .chain(&mut self.projectiles)
    }

    fn rand_normalized(&mut self) -> f64 {
        self.rng.rand() as f64 / u64::MAX as f64
    }

    fn rand_range(&mut self, min: f64, max: f64) -> f64 {
        min + self.rand_normalized() * (max - min)
    }

    fn random_stroid(&mut self, bounds: Size) -> Result<()> {
        let (width, height) = (bounds.width as f64, bounds.height as f64);
        let side = Side::try_from((self.rng.rand() % 4) as usize)?;
        let r = self.rand_normalized();
        let pos = Vec2::from(match side {
            Side::Left => (-width / 2. - 3., r * height - height / 2.),
            Side::Top => (r * width - width / 2., height / 2. + 3.),
            Side::Right => (width / 2. + 3., r * height - height / 2.),
            Side::Bottom => (r * width - width / 2., -height / 2. - 3.),
        });
        let rot = match side {
            Side::Left => self.rand_range(-FRAC_PI_4, FRAC_PI_4),
            Side::Top => self.rand_range(5. * FRAC_PI_4, 7. * FRAC_PI_4),
            Side::Right => self.rand_range(3. * FRAC_PI_4, 5. * FRAC_PI_4),
            Side::Bottom => self.rand_range(FRAC_PI_4, 3. * FRAC_PI_4),
        };
        Ok(self.spawn_stroid(Transform {
            pos,
            rot,
            scale: 4.,
        }))
    }

    fn explode_stroid(&mut self, pos: &Vec2) {
        for i in 0..3 {
            let rot = i as f64 * 2. * PI / 3. + self.rand_range(0., PI);
            let xf = Transform {
                pos: *pos,
                rot,
                scale: 2.,
            };
            self.spawn_stroid(xf);
        }
    }

    fn spawn_stroid(&mut self, xf: Transform) {
        let stroid = Entity {
            xfs: (xf, xf),
            v: Vec2::from(xf.rot).normalized() * self.rand_range(STROID_V * 0.7, STROID_V * 1.3),
            w: self.rand_range(-0.01, 0.01),
            vertices: (0..12)
                .map(|i| {
                    let p = Vec2::from(2. * PI * i as f64 / 12.);
                    [p.x, p.y].map(|c| c + self.rand_range(-0.25, 0.25)).into()
                })
                .collect(),
            should_remove: false,
        };
        self.stroids.push(stroid);
    }

    fn spawn_projectile(&mut self) {
        let mut xf = self.player.xfs.1;
        xf.pos += Vec2::from(xf.rot) * 2.;
        self.projectiles.push(Entity {
            xfs: (xf, xf),
            v: Vec2::from(self.player.xfs.1.rot) * PROJECTILE_V,
            w: 0.,
            vertices: [(-1., 0.), (1., 0.)].map(Vec2::from).into(),
            should_remove: false,
        })
    }
}

#[derive(Default, Clone, Copy)]
struct Transform {
    pos: Vec2,
    rot: f64,
    scale: f64,
}

impl Transform {
    fn interpolate(&self, other: &Transform, alpha: f64) -> Transform {
        Transform {
            pos: self.pos + (other.pos - self.pos) * alpha,
            rot: self.rot + (other.rot - self.rot) * alpha,
            scale: self.scale + (other.scale - self.scale) * alpha,
        }
    }
}

#[derive(Default, Clone, Copy)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl Vec2 {
    fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    fn normalized(&mut self) -> Vec2 {
        let len = self.length();
        if len > 0.0 {
            Vec2 {
                x: self.x / len,
                y: self.y / len,
            }
        } else {
            Vec2 {
                x: self.x,
                y: self.y,
            }
        }
    }

    fn distance(&self, other: &Vec2) -> f64 {
        ((other.x - self.x).powi(2) + (other.y - self.y).powi(2)).sqrt()
    }
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

impl Add for Vec2 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Vec2 {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Vec2 {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Vec2 {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f64> for Vec2 {
    type Output = Self;
    fn mul(self, rhs: f64) -> Self {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

struct InputState {
    pressed: HashSet<KeyCode, RandomWyHashState>,
    timers: HashMap<KeyCode, Instant, RandomWyHashState>,
    modifiers: KeyModifiers,
    release: Duration,
}

impl InputState {
    fn init() -> Self {
        Self {
            pressed: HashSet::with_capacity_and_hasher(5, RandomWyHashState::new()),
            timers: HashMap::with_capacity_and_hasher(5, RandomWyHashState::new()),
            modifiers: KeyModifiers::empty(),
            release: Duration::from_millis(INPUT_MS),
        }
    }

    fn active(&self, action: Action) -> bool {
        let pressed = |key| self.pressed.contains(key);
        match action {
            Action::InputLeft => pressed(&KeyCode::Char('a')),
            Action::InputRight => pressed(&KeyCode::Char('d')),
            Action::InputForward => pressed(&KeyCode::Char('w')),
            Action::InputReverse => pressed(&KeyCode::Char('s')),
            Action::InputFire => pressed(&KeyCode::Char(' ')),
        }
    }

    fn process(&mut self, ev: KeyEvent) {
        match ev.kind {
            KeyEventKind::Press => {
                self.pressed.insert(ev.code);
                self.timers.insert(ev.code, Instant::now());
                self.modifiers = ev.modifiers;
            }
            KeyEventKind::Release => {
                self.pressed.remove(&ev.code);
                self.timers.remove(&ev.code);
            }
            KeyEventKind::Repeat => {
                if let Some(timer) = self.timers.get_mut(&ev.code) {
                    *timer = Instant::now();
                }
            }
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let expired_keys = self
            .timers
            .iter()
            .filter(|&(_, time)| now - *time > self.release)
            .map(|(key, _)| *key)
            .collect::<Vec<KeyCode>>();

        for key in expired_keys {
            self.pressed.remove(&key);
            self.timers.remove(&key);
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
