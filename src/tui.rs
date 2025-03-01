use better_panic::Settings;
use bevy::{
    log::{
        tracing_subscriber::{
            layer::Context, layer::SubscriberExt, registry, util::SubscriberInitExt, Layer,
        },
        Level,
    },
    prelude::*,
    utils::tracing::{
        field::{Field, Visit},
        Event, Subscriber,
    },
};
use ratatui::{
    crossterm::{
        event::{self, KeyEvent},
        terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
        ExecutableCommand,
    },
    prelude::CrosstermBackend,
};
use std::{
    io::{stdout, Stdout},
    panic::set_hook,
    sync::mpsc::{channel, Receiver, Sender},
    time::Duration,
};

pub struct TuiPlugin;
impl Plugin for TuiPlugin {
    fn build(&self, app: &mut App) {
        let (sender, receiver) = channel();
        app.insert_resource(LogStore(vec![]))
            .insert_resource(Terminal::init())
            .add_event::<Input>()
            .add_systems(PreUpdate, read_events)
            .insert_non_send_resource(CapturedLogEvents(receiver))
            .add_event::<LogEvent>()
            .add_systems(Update, (transfer_log_events, store_logs));
        registry().with(Some(CaptureLayer(sender))).init();
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

#[derive(Debug, Event, Clone)]
pub enum LogEvent {
    Info(String),
    Debug(String),
    Error(String),
}

impl LogEvent {
    pub fn message(&self) -> &str {
        use LogEvent::*;
        match self {
            Info(s) | Debug(s) | Error(s) => s.as_ref(),
        }
    }
}

#[derive(Deref, DerefMut)]
struct CapturedLogEvents(pub Receiver<LogEvent>);

#[derive(Resource, Deref, DerefMut)]
pub struct LogStore(Vec<LogEvent>);

struct CaptureLayer(pub Sender<LogEvent>);
impl<S: Subscriber> Layer<S> for CaptureLayer {
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut message = None;
        event.record(&mut CaptureLayerVisitor(&mut message));
        if let Some(message) = message {
            let metadata = event.metadata();
            let s = format!(
                "[{}::{}][{}] {}",
                metadata.target(),
                metadata.line().unwrap_or(0),
                metadata.level(),
                message
            );
            let log = match *metadata.level() {
                Level::INFO => Some(LogEvent::Info(s)),
                Level::DEBUG => Some(LogEvent::Debug(s)),
                Level::ERROR => Some(LogEvent::Error(s)),
                _ => None,
            };
            if let Some(log) = log {
                self.0.send(log).unwrap();
            }
        }
    }
}

struct CaptureLayerVisitor<'a>(&'a mut Option<String>);
impl Visit for CaptureLayerVisitor<'_> {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            *self.0 = Some(format!("{value:?}"));
        }
    }
}

fn store_logs(mut events: EventReader<LogEvent>, mut log_store: ResMut<LogStore>) {
    for event in events.read() {
        log_store.push(event.clone());
    }
}

fn transfer_log_events(
    receiver: NonSend<CapturedLogEvents>,
    mut log_events: EventWriter<LogEvent>,
) {
    log_events.send_batch(receiver.try_iter());
}
