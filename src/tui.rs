use better_panic::Settings;
use bevy::prelude::*;
use ratatui::{
    crossterm::{
        ExecutableCommand,
        event::{self, KeyEvent},
        terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
    },
    prelude::CrosstermBackend,
};
use std::{
    io::{Stdout, stdout},
    panic::set_hook,
    time::Duration,
};

pub struct TuiPlugin;
impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(Terminal::init())
            .add_event::<Input>()
            .add_systems(PreUpdate, read_events);
    }
}

#[derive(Event, Deref, DerefMut)]
pub struct Input(KeyEvent);

#[derive(Resource, Deref, DerefMut)]
pub struct Terminal(ratatui::Terminal<CrosstermBackend<Stdout>>);
impl Terminal {
    fn init() -> Self {
        set_hook(Box::new(move |panic_info| {
            Self::restore();
            Settings::auto()
                .most_recent_first(false)
                .lineno_suffix(true)
                .create_panic_handler()(panic_info)
        }));
        (|| -> std::io::Result<Terminal> {
            enable_raw_mode()?;
            stdout().execute(EnterAlternateScreen)?;
            Ok(Self(ratatui::Terminal::new(CrosstermBackend::new(
                stdout(),
            ))?))
        })()
        .unwrap()
    }

    fn restore() {
        let _ = (|| -> std::io::Result<()> {
            disable_raw_mode()?;
            stdout().execute(LeaveAlternateScreen).map(|_| ())
        })();
    }
}

impl Drop for Terminal {
    fn drop(&mut self) {
        Self::restore();
    }
}

fn read_events(mut event: EventWriter<Input>) {
    (|| -> std::io::Result<()> {
        while event::poll(Duration::ZERO)? {
            if let event::Event::Key(key) = event::read()? {
                event.send(Input(key));
            }
        }
        Ok(())
    })()
    .unwrap()
}
