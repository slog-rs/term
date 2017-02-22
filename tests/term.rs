#[macro_use]
extern crate slog;
extern crate slog_term;
extern crate slog_async;

use slog::Drain;

use slog_async::Async;

fn main() {
    let decorator = slog_term::PlainSyncDecorator::new(std::io::stdout());
    let _drain = slog_term::Term::new(decorator).build().fuse();

    let decorator = slog_term::PlainDecorator::new(std::io::stdout());
    let _drain = Async::new(slog_term::Term::new(decorator).build().fuse())
        .build()
        .fuse();
}
