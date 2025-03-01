mod tui;

use std::time::Duration;

use bevy::{
    app::ScheduleRunnerPlugin,
    diagnostic::{DiagnosticsPlugin, DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    state::app::StatesPlugin,
};
use ratatui::{
    crossterm::event::{KeyCode, KeyEvent, KeyModifiers},
    text::Line,
    widgets::{Block, Paragraph, Wrap},
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
        .add_systems(Startup, || info!("hello world"))
        .add_systems(
            Update,
            (
                listen_exit,
                listen_log,
                render_logs.run_if(in_state(ViewState::Log)),
                render_game.run_if(in_state(ViewState::Game)),
            ),
        )
        .run();
}

pub fn title_block(diag: Res<DiagnosticsStore>) -> Block {
    let fps = diag
        .get(&FrameTimeDiagnosticsPlugin::FPS)
        .and_then(|fps| fps.smoothed())
        .unwrap_or(0.);
    Block::bordered()
        .title(format!(" -STROIDS- FPS: {:.0} ", fps))
        .title(Line::from(" Movement: WASD • Shoot: Space • Log: ~/` • Pause: P ").right_aligned())
}

fn render_logs(
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

fn render_game(mut term: ResMut<Terminal>, diag: Res<DiagnosticsStore>) {
    term.draw(|frame| frame.render_widget(title_block(diag), frame.area()))
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

#[derive(Component)]
struct Player;

#[derive(Bundle)]
struct PlayerBundle {
    marker: Player,
    pos: Position,
    rot: Rotation,
}

#[derive(Component)]
struct Asteroid;

#[derive(Bundle)]
struct AsteroidBundle {
    marker: Asteroid,
    pos: Position,
    rot: Rotation,
}

#[derive(Component)]
struct Vertices(Vec<Vec2>);

#[derive(Component)]
struct Velocity(f32);

#[derive(Component)]
struct Position(Vec2);

#[derive(Component)]
struct Rotation(Rot2);
