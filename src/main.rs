#![feature(array_windows)]

mod tui;

use std::{iter, time::Duration};

use bevy::{
    app::ScheduleRunnerPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    state::app::StatesPlugin,
};
use ratatui::{
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    style::Color,
    text::Line,
    widgets::{
        Block, Paragraph, Wrap,
        canvas::{self, Canvas},
    },
};
use tui::{Input, LogStore, Terminal, TuiPlugin};

fn main() {
    App::new()
        .add_plugins((
            MinimalPlugins.set(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
                1. / 60.,
            ))),
            StatesPlugin,
            DiagnosticsPlugin,
            FrameTimeDiagnosticsPlugin,
            TuiPlugin,
        ))
        .init_state::<ViewState>()
        .init_state::<PauseState>()
        .insert_resource(Time::from_hz(60.))
        .init_resource::<LogScroll>()
        .add_systems(Startup, init)
        .add_systems(
            Update,
            (
                listen_exit,
                listen_log,
                (listen_scroll, draw_logs).run_if(in_state(ViewState::Log)),
                (listen_interact, (interpolate, draw_game).chain())
                    .run_if(in_state(ViewState::Game)),
            ),
        )
        .add_systems(
            FixedUpdate,
            apply_velocity.run_if(in_state(ViewState::Game)),
        )
        .run();
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

fn init(mut commands: Commands, term: Res<Terminal>) {
    info!("hello world");
    let size = term.size().unwrap();
    let (row, col) = ((size.width / 2).into(), (size.height / 2).into());
    commands.spawn(PlayerBundle::new(row, col));
}

fn draw_logs(
    mut term: ResMut<Terminal>,
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
    mut term: ResMut<Terminal>,
    diag: Res<DiagnosticsStore>,
    shapes: Query<(&Isometry, &Vertices)>,
) {
    term.draw(|frame| {
        let block = title_block(diag);
        let area = block.inner(frame.area());
        let (w, h) = (area.width.into(), area.height.into());
        frame.render_widget(
            Canvas::default()
                .block(block)
                .x_bounds([0., w])
                .y_bounds([0., h])
                .paint(|ctx| {
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
                            let [(x1, y1), (x2, y2)] = pair
                                .map(|Vec2 { x, y }| {
                                    (px + (x * r.cos - y * r.sin), py + (x * r.sin + y * r.cos))
                                })
                                .map(|(x, y)| (x as f64, y as f64));
                            ctx.draw(&canvas::Line {
                                x1,
                                y1,
                                x2,
                                y2,
                                color: Color::White,
                            });
                        }
                    }
                }),
            frame.area(),
        )
    })
    .unwrap();
}

fn interpolate(time: Res<Time<Fixed>>, mut query: Query<(&mut Isometry, &XfState)>) {
    for (mut xf, XfState { current, last }) in &mut query {
        let a = time.overstep_fraction();
        xf.translation = last.translation.lerp(current.translation, a);
    }
}

fn apply_velocity(time: Res<Time>, mut query: Query<(&Velocity, &mut XfState)>) {
    for (velocity, xf_state) in &mut query {
        let XfState { current, last } = xf_state.into_inner();
        last.translation = current.translation;
        current.translation += **velocity * time.delta_secs();
    }
}

fn listen_exit(mut input: EventReader<Input>, mut exit: EventWriter<AppExit>) {
    for ev in input.read() {
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

fn listen_interact(
    mut input: EventReader<Input>,
    mut query: Query<(&mut AngularVelocity, &mut Velocity, &Isometry), With<Player>>,
) {
    let (mut w, mut v, xf) = query.single_mut();
    for ev in input.read() {
        if let KeyCode::Char(c @ 'd' | c @ 'a') = ev.code {
            **w += if c == 'd' { 0.5 } else { -0.5 }
        };
        if let KeyCode::Char(c @ 'w' | c @ 's') = ev.code {
            let dir = if c == 'w' { 1. } else { -1. };
            v.x += dir * xf.rotation.sin;
            v.y += dir * xf.rotation.cos;
        }
    }
}

fn listen_log(
    mut input: EventReader<Input>,
    state: Res<State<ViewState>>,
    mut next_state: ResMut<NextState<ViewState>>,
) {
    for ev in input.read() {
        if let KeyCode::Char('`') | KeyCode::Char('~') = ev.code {
            next_state.set(state.get().next())
        }
    }
}

fn listen_scroll(mut input: EventReader<Input>, mut scroll: ResMut<LogScroll>) {
    for ev in input.read() {
        let s = &mut **scroll;
        *s = match ev.code {
            KeyCode::Up => s.saturating_sub(1),
            KeyCode::Down => s.saturating_add(1),
            KeyCode::PageUp => s.saturating_sub(10),
            KeyCode::PageDown => s.saturating_add(10),
            _ => *s,
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

#[derive(Component, Default)]
struct Player;

#[derive(Bundle)]
struct PlayerBundle {
    marker: Player,
    velocity: Velocity,
    angular_velocity: AngularVelocity,
    isometry: Isometry,
    vertices: Vertices,
    xf_state: XfState,
}

impl PlayerBundle {
    fn new(x: f32, y: f32) -> Self {
        let isometry = Isometry2d::from_xy(x, y);
        Self {
            marker: Player,
            velocity: Velocity(Vec2::from((0., 0.))),
            angular_velocity: AngularVelocity(0.),
            vertices: Vertices([(-1., 0.), (0., 2.), (1., 0.)].map(Vec2::from).to_vec()),
            isometry: isometry.into(),
            xf_state: XfState {
                current: isometry,
                last: isometry,
            },
        }
    }
}

#[derive(Component)]
struct Asteroid;

#[derive(Bundle)]
struct AsteroidBundle {
    marker: Asteroid,
    transform: Transform,
}

#[derive(Component, Deref, DerefMut)]
struct Velocity(Vec2);

#[derive(Component, Deref, DerefMut)]
struct AngularVelocity(f32);

#[derive(Component, Default, Debug, Deref, DerefMut)]
struct Vertices(Vec<Vec2>);

#[derive(Component, Default, Debug)]
struct XfState {
    current: Isometry2d,
    last: Isometry2d,
}

#[derive(Component, Default, Debug, Deref, DerefMut)]
struct Isometry(Isometry2d);

impl From<Isometry2d> for Isometry {
    fn from(value: Isometry2d) -> Self {
        Self(value)
    }
}

impl From<&Isometry> for Isometry2d {
    fn from(value: &Isometry) -> Self {
        **value
    }
}
