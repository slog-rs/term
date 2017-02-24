#[macro_use]
extern crate slog;
extern crate slog_term;

use slog::Drain;
use std::sync::Arc;

fn main() {
    let decorator = slog_term::PlainSyncDecorator::new(std::io::stdout());
    let drain = slog_term::Term::new(decorator).build().fuse();

    let root_log = slog::Logger::root(Arc::new(drain), o!("version" => "0.5"));
    let server_log = root_log.new(o!("host" => "localhost", "port" => "8080"));
    let peer1_log =
        server_log.new(o!("peer_addr" => "8.8.8.8", "port" => "18230"));
    let peer2_log =
        server_log.new(o!("peer_addr" => "82.9.9.9", "port" => "42381"));

    info!(server_log, "starting");
    info!(server_log, "listening");
    debug!(peer2_log, "connected");
    debug!(peer2_log, "message received"; "length" => 2);
    debug!(peer1_log, "connected");
    debug!(peer2_log, "response sent"; "length" => 8);
    debug!(peer2_log, "disconnected");
    debug!(peer1_log, "message received"; "length" => 2);
    debug!(peer1_log, "response sent"; "length" => 8);
    debug!(peer1_log, "disconnected");
    info!(server_log, "exit");
}
