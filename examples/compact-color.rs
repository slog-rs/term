#[macro_use]
extern crate slog;
extern crate slog_term;
extern crate slog_async;

use slog::Drain;
use std::sync::Arc;

mod common;

fn main() {
    let decorator = slog_term::TermDecorator::new().build();
    let drain = slog_term::CompactFormat::new(decorator).build().fuse();
    let drain = slog_async::Async::new(drain).build().fuse();

    let log = slog::Logger::root(Arc::new(drain), o!("version" => "0.5"));

    common::simulate_server(log.to_erased());
}
