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
        .init_resource::<LogScroll>()
        .add_systems(Startup, init)
        .add_systems(
            Update,
            (
                listen_exit,
                listen_log,
                (listen_scroll, draw_logs).run_if(in_state(ViewState::Log)),
                draw_game.run_if(in_state(ViewState::Game)),
            ),
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
    commands.spawn(PlayerBundle::new(Position { x: row, y: col }));
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
    shapes: Query<(&Position, &Rotation)>,
) {
    const PLAYER_VERTS: [(f64, f64); 3] = [(-1., 0.), (0., 2.5), (1., 0.)];
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
                    shapes.iter().for_each(|shape| {
                        let (pos, rot) = shape;
                        PLAYER_VERTS
                            .array_windows::<2>()
                            .chain(iter::once(&[PLAYER_VERTS[2], PLAYER_VERTS[0]]))
                            .for_each(|&[p1, p2]| {
                                let [(x1, y1), (x2, y2)] = [p1, p2];
                                ctx.draw(&canvas::Line {
                                    x1: x1 + pos.x,
                                    y1: y1 + pos.y,
                                    x2: x2 + pos.x,
                                    y2: y2 + pos.y,
                                    color: Color::White,
                                });
                            })
                    });
                }),
            frame.area(),
        )
    })
    .unwrap();
}

fn listen_exit(mut input: EventReader<Input>, mut exit: EventWriter<AppExit>) {
    input.read().for_each(|ev| {
        if let KeyEvent {
            code: KeyCode::Char('c'),
            modifiers: KeyModifiers::CONTROL,
            ..
        } = **ev
        {
            exit.send_default();
        }
    })
}

fn listen_log(
    mut input: EventReader<Input>,
    state: Res<State<ViewState>>,
    mut next_state: ResMut<NextState<ViewState>>,
) {
    input.read().for_each(|ev| {
        if let KeyCode::Char('`') | KeyCode::Char('~') = ev.code {
            next_state.set(state.get().next())
        }
    })
}

fn listen_scroll(mut events: EventReader<Input>, mut scroll: ResMut<LogScroll>) {
    events.read().for_each(|ev| {
        let s = &mut **scroll;
        *s = match ev.code {
            KeyCode::Up => s.saturating_sub(1),
            KeyCode::Down => s.saturating_add(1),
            KeyCode::PageUp => s.saturating_sub(10),
            KeyCode::PageDown => s.saturating_add(10),
            _ => *s,
        }
    })
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
    pos: Position,
    rot: Rotation,
}

impl PlayerBundle {
    fn new(pos: Position) -> Self {
        Self {
            marker: Player,
            rot: Rotation(0.),
            pos,
        }
    }
}

#[derive(Component)]
struct Asteroid;

#[derive(Bundle)]
struct AsteroidBundle {
    marker: Asteroid,
    pos: Position,
    rot: Rotation,
}

#[derive(Component, Deref, DerefMut)]
struct Velocity(f64);

#[derive(Component, Default, Debug)]
struct Position {
    x: f64,
    y: f64,
}

impl From<(f64, f64)> for Position {
    fn from(value: (f64, f64)) -> Self {
        Self {
            x: value.0,
            y: value.0,
        }
    }
}

#[derive(Component, Default, Deref, DerefMut, Debug)]
struct Rotation(f64);
