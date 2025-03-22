#![feature(array_windows)]

mod log;

use bevy::{
    app::ScheduleRunnerPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    state::app::StatesPlugin,
    time::common_conditions::on_timer,
};
use bevy_rand::{global::GlobalEntropy, plugin::EntropyPlugin, prelude::WyRand};
use bevy_ratatui::{
    RatatuiPlugins,
    event::KeyEvent,
    terminal::{self, RatatuiContext},
};
use log::{LogPlugin, LogStore};
use paste::paste;
use rand_core::RngCore;
use ratatui::{
    layout::Size,
    style::Color,
    text::Line,
    widgets::{
        Block, Paragraph, Wrap,
        canvas::{self, Canvas, Context},
    },
};
use std::{iter, ops::Neg, time::Duration};

const MAX_VELOCITY: f32 = 20.0;
const MAX_ANGULAR_VELOCITY: f32 = 4.0;
const LINEAR_DAMPING: f32 = 0.6;
const ANGULAR_DAMPING: f32 = 5.;
const THRUST_POWER: f32 = 0.4;
const ROTATION_POWER: f32 = 0.15;

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                1. / 60.,
            ))),
            EntropyPlugin::<WyRand>::default(),
            StatesPlugin,
            DiagnosticsPlugin,
            FrameTimeDiagnosticsPlugin,
            RatatuiPlugins {
                enable_mouse_capture: false,
                enable_kitty_protocol: false,
                enable_input_forwarding: true,
            },
            LogPlugin,
        ))
        .init_state::<ViewState>()
        .init_state::<PauseState>()
        .insert_resource(Time::from_hz(60.))
        .init_resource::<LogScroll>()
        .add_systems(Startup, (init.after(terminal::setup), print_random_values))
        .add_systems(
            Update,
            (
                listen_exit,
                listen_log,
                (listen_scroll, draw_logs).run_if(in_state(ViewState::Log)),
                (
                    spawn_asteroid.run_if(on_timer(Duration::from_secs(2))),
                    listen_movement,
                    (interpolate, draw_game).chain(),
                )
                    .run_if(
                        |view: Res<State<ViewState>>, pause: Res<State<PauseState>>| {
                            matches!(
                                (view.get(), pause.get()),
                                (ViewState::Game, PauseState::Resume)
                            )
                        },
                    ),
            ),
        )
        .add_systems(
            FixedUpdate,
            (apply_velocity, dampen_player, wrap_player)
                .chain()
                .run_if(in_state(ViewState::Game)),
        )
        .run();
}

fn print_random_values(mut rng: GlobalEntropy<WyRand>) {
    info!(
        "Random values: {:#?}",
        (0..10)
            .map(|_| (rng.next_u64() % 20) as i32 - 10)
            .collect::<Vec<_>>()
    );
}

fn title_block(diag: Res<DiagnosticsStore>) -> Block {
    let fps = diag
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed())
        .unwrap_or(0.);
    Block::bordered()
        .title(format!(" -STROIDS- FPS: {:.0} ", fps))
        .title(Line::from(" Movement: WASD • Shoot: Space • Log: ~/` • Pause: P ").right_aligned())
}

fn init(mut commands: Commands, term: Res<RatatuiContext>) {
    info!("hello world");
    let size = term.size().unwrap();
    let (row, col) = ((size.width / 2).into(), (size.height / 2).into());
    commands.spawn(PlayerBundle::new(row, col));
}

fn draw_logs(
    mut term: ResMut<RatatuiContext>,
    diag: Res<DiagnosticsStore>,
    logs: Res<LogStore>,
    scroll: Res<LogScroll>,
) {
    term.draw(|frame| {
        frame.render_widget(
            Paragraph::new(
                logs.iter()
                    .map(|log| log.message().into())
                    .collect::<Vec<String>>()
                    .join("\n"),
            )
            .scroll((**scroll, 0))
            .block(title_block(diag))
            .wrap(Wrap { trim: false }),
            frame.area(),
        )
    })
    .unwrap();
}

fn draw_game(
    mut term: ResMut<RatatuiContext>,
    diag: Res<DiagnosticsStore>,
    shapes: Query<(&Xf, &Vertices)>,
) {
    let Size { width, height } = term.size().unwrap();
    let painter = |ctx: &mut Context| {
        for (
            Isometry2d {
                translation: Vec2 { x: px, y: py },
                rotation: r,
            },
            vs,
        ) in shapes.into_iter().map(|(xf, vs)| (xf.into(), vs))
        {
            for pair in vs
                .array_windows::<2>()
                .chain(iter::once(&[*vs.last().unwrap(), vs[0]]))
            {
                let [(x1, y1), (x2, y2)] = pair.map(|Vec2 { x, y }| {
                    (
                        (px + (x * r.cos - y * r.sin)) as f64,
                        (py + (x * r.sin + y * r.cos)) as f64,
                    )
                });
                let [x1, x2] = [x1, x2].map(|p| p.clamp(0., width as f64 - 2.));
                let [y1, y2] = [y1, y2].map(|p| p.clamp(0., height as f64 - 2.));
                ctx.draw(&canvas::Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    color: Color::White,
                });
            }
        }
    };
    term.draw(|frame| {
        let block = title_block(diag);
        let area = block.inner(frame.area());
        let (w, h) = (area.width.into(), area.height.into());

        frame.render_widget(
            Canvas::default()
                .block(block)
                .x_bounds([0., w])
                .y_bounds([0., h])
                .paint(painter),
            frame.area(),
        )
    })
    .unwrap();
}

fn wrap_player(term: Res<RatatuiContext>, mut query: Query<(&mut Xf, &mut XfState), With<Player>>) {
    let (mut xf, mut xfs) = query.single_mut();
    let Size { width, height } = term.size().unwrap();
    let (width, height) = (width as f32, height as f32);

    if xf.translation.x < 0. || xf.translation.x > width {
        let new_x = if xf.translation.x < 0. { width } else { 0. };
        xf.translation.x = new_x;
        xfs.current.translation.x = new_x;
        xfs.last.translation.x = new_x;
    }

    if xf.translation.y < 0. || xf.translation.y > height {
        let new_y = if xf.translation.y < 0. { height } else { 0. };
        xf.translation.y = new_y;
        xfs.current.translation.y = new_y;
        xfs.last.translation.y = new_y;
    }
}

fn interpolate(time: Res<Time<Fixed>>, mut query: Query<(&mut Xf, &XfState)>) {
    for (mut xf, XfState { current, last }) in &mut query {
        let a = time.overstep_fraction();
        xf.translation = last.translation.lerp(current.translation, a);
        xf.rotation = last.rotation.slerp(current.rotation, a);
    }
}

fn apply_velocity(time: Res<Time>, mut query: Query<(&AngularVelocity, &Velocity, &mut XfState)>) {
    for (w, v, xf_state) in &mut query {
        let XfState { current, last } = xf_state.into_inner();
        *last = *current;

        current.translation += **v * time.delta_secs();
        current.rotation = (current.rotation.as_radians() + **w * time.delta_secs()).into();
    }
}

fn dampen_player(
    time: Res<Time>,
    mut query: Query<(&mut AngularVelocity, &mut Velocity), With<Player>>,
) {
    let (mut w, mut v) = query.single_mut();
    **w *= 1.0 - (ANGULAR_DAMPING * time.delta_secs()).min(1.0);
    **v *= 1.0 - (LINEAR_DAMPING * time.delta_secs()).min(1.0);

    let speed = Vec2::new(v.x, v.y).length();
    if speed > MAX_VELOCITY {
        **v *= MAX_VELOCITY / speed;
    }

    **w = w.clamp(-MAX_ANGULAR_VELOCITY, MAX_ANGULAR_VELOCITY);
}

fn spawn_asteroid(term: Res<RatatuiContext>, rng: GlobalEntropy<WyRand>, mut commands: Commands) {
    commands.spawn(AsteroidBundle::from_random(term.size().unwrap(), rng));
}

fn cleanup(term: Res<RatatuiContext>, query: Query<(Entity, &Xf), With<Asteroid>>) {}

fn listen_movement(
    input: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut AngularVelocity, &mut Velocity, &Xf), With<Player>>,
) {
    let (mut w, mut v, xf) = query.single_mut();
    let r = xf.rotation * Rot2::FRAC_PI_2;

    for k in input.get_pressed() {
        match k {
            KeyCode::KeyW | KeyCode::KeyS => {
                let dir = if *k == KeyCode::KeyW {
                    THRUST_POWER
                } else {
                    -THRUST_POWER
                };
                v.x += r.cos * dir;
                v.y += r.sin * dir;
            }
            KeyCode::KeyA => **w += ROTATION_POWER,
            KeyCode::KeyD => **w -= ROTATION_POWER,
            _ => {}
        }
    }
}

fn listen_log(
    mut events: EventReader<KeyEvent>,
    state: Res<State<ViewState>>,
    mut next_state: ResMut<NextState<ViewState>>,
) {
    use ratatui::crossterm::event::KeyCode;
    for ev in events.read() {
        if let KeyCode::Char('`') | KeyCode::Char('~') = ev.code {
            next_state.set(state.get().next())
        }
    }
}

fn listen_scroll(input: Res<ButtonInput<KeyCode>>, mut scroll: ResMut<LogScroll>) {
    for ev in input.get_pressed() {
        let s = &mut **scroll;
        *s = match ev {
            KeyCode::ArrowUp => s.saturating_sub(1),
            KeyCode::ArrowDown => s.saturating_add(1),
            KeyCode::PageUp => s.saturating_sub(10),
            KeyCode::PageDown => s.saturating_add(10),
            _ => *s,
        }
    }
}

fn listen_exit(mut events: EventReader<KeyEvent>, mut exit: EventWriter<AppExit>) {
    use ratatui::crossterm::event::*;
    for ev in events.read() {
        if let KeyEvent {
            code: KeyCode::Char('c'),
            modifiers: KeyModifiers::CONTROL,
            ..
        } = **ev
        {
            exit.send_default();
        }
    }
}

#[derive(Resource, Deref, DerefMut, Default)]
struct LogScroll(u16);

#[derive(States, Default, Clone, PartialEq, Eq, Hash, Debug)]
enum PauseState {
    #[default]
    Resume,
    Pause,
}

#[derive(States, Default, Clone, PartialEq, Eq, Hash, Debug)]
enum ViewState {
    #[default]
    Game,
    Log,
}

impl ViewState {
    fn next(&self) -> Self {
        match self {
            Self::Log => Self::Game,
            _ => Self::Log,
        }
    }
}

macro_rules! bundle_ent {
    ($($marker:ident),*) => {
        $(paste! {
            #[derive(Bundle)]
            struct [<$marker Bundle>] {
                marker: $marker,
                velocity: Velocity,
                isometry: Xf,
                angular_velocity: AngularVelocity,
                vertices: Vertices,
                xf_state: XfState,
            }
        })*
    };
}

#[derive(Component, Default)]
struct Player;

#[derive(Component, Default)]
struct Asteroid;

#[derive(Component, Default)]
struct Projectile;

bundle_ent![Player, Asteroid, Projectile];

impl PlayerBundle {
    fn new(x: f32, y: f32) -> Self {
        let isometry = Isometry2d::from_xy(x, y);
        Self {
            marker: Player,
            velocity: Velocity(Vec2::from((0., 0.))),
            angular_velocity: AngularVelocity(0.),
            vertices: [(-1., -1.), (0., 1.5), (1., -1.)].into(),
            isometry: isometry.into(),
            xf_state: XfState {
                current: isometry,
                last: isometry,
            },
        }
    }
}

impl AsteroidBundle {
    fn from_random(bounds: Size, mut rng: GlobalEntropy<WyRand>) -> Self {
        enum Side {
            Left,
            Right,
            Top,
            Bottom,
        }
        let side = match rng.next_u64() % 4 {
            0 => Side::Left,
            1 => Side::Right,
            2 => Side::Top,
            3 => Side::Bottom,
            _ => unreachable!(),
        };

        let rx = (rng.next_u64() % bounds.width as u64) as f32;
        let ry = (rng.next_u64() % bounds.height as u64) as f32;
        let (px, py) = match side {
            Side::Left => (0., ry),
            Side::Right => (bounds.width as f32, ry),
            Side::Top => (rx, bounds.height as f32),
            Side::Bottom => (rx, 0.),
        };

        let z10 = (rng.next_u64() % 10) as f32;
        let n10p10 = rng.next_u64() as f32 % 20. - 10.;
        let (vx, vy) = match side {
            Side::Left => (z10, n10p10),
            Side::Right => (z10.neg(), n10p10),
            Side::Top => (n10p10, z10.neg()),
            Side::Bottom => (n10p10, z10),
        };
        let isometry = Isometry2d {
            rotation: Rot2::degrees(rng.next_u64() as f32 % 360.),
            translation: Vec2 { x: px, y: py },
        };

        Self {
            marker: Asteroid,
            velocity: Vec2 { x: vx, y: vy }.into(),
            isometry: isometry.into(),
            angular_velocity: (rng.next_u64() as f32 % 5.).into(),
            vertices: [
                (4., 0.),
                (3.46, 2.),
                (2., 3.46),
                (0., 4.),
                (-2., 3.46),
                (-3.46, 2.),
                (-4., 0.),
                (-3.46, -2.),
                (-2., -3.46),
                (0., -4.),
                (2., -3.46),
                (3.46, -2.),
            ]
            .map(|(x, y)| {
                (
                    (x + rng.next_u64() as f32 % 3.) - 1.,
                    (y + rng.next_u64() as f32 % 3.) - 1.,
                )
            })
            .into(),
            xf_state: XfState {
                current: isometry,
                last: isometry,
            },
        }
    }
}

#[derive(Component, Deref, DerefMut, Default)]
struct Velocity(Vec2);

impl From<Vec2> for Velocity {
    fn from(value: Vec2) -> Self {
        Self(value)
    }
}

#[derive(Component, Deref, DerefMut, Default)]
struct AngularVelocity(f32);

impl From<f32> for AngularVelocity {
    fn from(value: f32) -> Self {
        Self(value)
    }
}

#[derive(Component, Default, Debug, Deref, DerefMut)]
struct Vertices(Vec<Vec2>);

impl<T> From<T> for Vertices
where
    Vec<(f32, f32)>: From<T>,
{
    fn from(value: T) -> Self {
        Self(Vec::from(value).into_iter().map(Vec2::from).collect())
    }
}

#[derive(Component, Default, Debug)]
struct XfState {
    current: Isometry2d,
    last: Isometry2d,
}

#[derive(Component, Default, Debug, Deref, DerefMut)]
struct Xf(Isometry2d);

impl From<Isometry2d> for Xf {
    fn from(value: Isometry2d) -> Self {
        Self(value)
    }
}

impl From<&Xf> for Isometry2d {
    fn from(value: &Xf) -> Self {
        **value
    }
}
