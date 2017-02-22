// {{{ Module docs.
//! `slog-rs`'a `Drain` for terminal output
//!
//! This crate implements output formatting targeting logging to
//! terminal/console/shell or similar text-based IO.
//!
//! Using `Decorator` open trait, user can implement outputting
//! using different colors, terminal types and so on.
//!
//! ```
//! #[macro_use]
//! extern crate slog;
//! extern crate slog_term;
//!
//! use slog::*;
//!
//! fn main() {
//!     let root = Logger::root(slog_term::Term::new().build().fuse(), o!());
//! }
//! ```
// }}}

// {{{ Imports & meta
#![warn(missing_docs)]

extern crate slog;
extern crate isatty;
extern crate chrono;
extern crate thread_local;
extern crate term;

use slog::{OwnedKVList, KV, Record};
use slog::Drain;

use std::{io, fmt, sync};
use std::io::Write as IoWrite;
use std::panic::{UnwindSafe, RefUnwindSafe};

use std::result;
// }}}

// {{{ Decorator
/// Output decorator
///
/// Trait implementing strategy of output formating in terms of IO,
/// colors, etc.
pub trait Decorator: Send + Sync + UnwindSafe + RefUnwindSafe {
    /// Get a `RecordDecorator` for a given `record`
    ///
    /// This allows `Decorator` to have on-stack data per processed `Record`s
    fn decorate(&self,
                record: &Record,
                logger_values: &OwnedKVList)
                -> Box<RecordDecorator>;
}

/// Per-record decorator
pub trait RecordDecorator: io::Write {
    /// Format normal text
    fn start_text(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format `Record` message
    fn start_msg(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format timestamp
    fn start_timestamp(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format `Record` level
    fn start_level(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format `Record` message
    fn start_comma(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format key
    fn start_key(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format value
    fn start_value(&mut self) -> io::Result<()> {
        Ok(())
    }

    /// Format value
    fn start_separator(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl RecordDecorator for Box<RecordDecorator> {
    fn start_text(&mut self) -> io::Result<()> {
        (**self).start_text()
    }

    /// Format `Record` message
    fn start_msg(&mut self) -> io::Result<()> {
        (**self).start_msg()
    }

    /// Format timestamp
    fn start_timestamp(&mut self) -> io::Result<()> {
        (**self).start_timestamp()
    }

    /// Format `Record` level
    fn start_level(&mut self) -> io::Result<()> {
        (**self).start_level()
    }

    /// Format `Record` message
    fn start_comma(&mut self) -> io::Result<()> {
        (**self).start_comma()
    }

    /// Format key
    fn start_key(&mut self) -> io::Result<()> {
        (**self).start_key()
    }

    /// Format value
    fn start_value(&mut self) -> io::Result<()> {
        (**self).start_value()
    }

    /// Format value
    fn start_separator(&mut self) -> io::Result<()> {
        (**self).start_separator()
    }
}
// }}}

/// Formatting mode
pub enum FormatMode {
    /// Compact logging format
    Compact,
    /// Full logging format
    Full,
}

// {{{ Term
/// Terminal-output formatting `Drain`
pub struct Term {
    mode: FormatMode,
    decorator: Box<Decorator>,
    fn_timestamp: Box<ThreadSafeTimestampFn<Output = io::Result<()>>>,
}

/// Streamer builder
pub struct TermBuilder {
    mode: FormatMode,
    decorator: Box<Decorator>,
    fn_timestamp: Box<ThreadSafeTimestampFn<Output = io::Result<()>>>,
}

impl TermBuilder {
    /// Output using full mode
    pub fn full(mut self) -> Self {
        self.mode = FormatMode::Full;
        self
    }

    /// Output using compact mode (default)
    pub fn compact(mut self) -> Self {
        self.mode = FormatMode::Compact;
        self
    }

    /// Use the UTC time zone for the timestamp
    pub fn use_utc_timestamp(mut self) -> Self {
        self.fn_timestamp = Box::new(timestamp_utc);
        self
    }

    /// Use the local time zone for the timestamp (default)
    pub fn use_local_timestamp(mut self) -> Self {
        self.fn_timestamp = Box::new(timestamp_local);
        self
    }

    /// Provide a custom function to generate the timestamp
    pub fn use_custom_timestamp<F>(mut self, f: F) -> Self
        where F: ThreadSafeTimestampFn
    {
        self.fn_timestamp = Box::new(f);
        self
    }

    /// Build the streamer
    pub fn build(self) -> Term {
        Term {
            mode: self.mode,
            decorator: self.decorator,
            fn_timestamp: self.fn_timestamp,
        }
    }
}


impl Drain for Term {
    type Ok = ();
    type Err = io::Error;

    fn log(&self,
           record: &Record,
           values: &OwnedKVList)
           -> result::Result<Self::Ok, Self::Err> {
        match self.mode {
            FormatMode::Full => self.format_full(record, values),
            FormatMode::Compact => Ok(()), // TODO: self.format_compact(record, values),
        }
    }
}

impl Term {
    /// New `TermBuilder`
    pub fn new() -> TermBuilder {
        TermBuilder {
            mode: FormatMode::Full,
            fn_timestamp: Box::new(timestamp_local),
            decorator: Box::new(PlainDecorator::new(std::io::stderr())),
        }
    }

    fn format_full(&self,
                   record: &Record,
                   values: &OwnedKVList)
                   -> io::Result<()> {

        let mut decorator = self.decorator.decorate(record, values);


        let comma_needed = try!(self.print_msg_header(&mut *decorator, record));
        let mut serializer = Serializer::new(decorator, comma_needed);

        try!(record.kv().serialize(record, &mut serializer));

        try!(values.serialize(record, &mut serializer));

        let mut decorator = serializer.finish();

        try!(decorator.start_text());
        try!(write!(decorator, "\n"));

        try!(decorator.flush());

        Ok(())
    }

    /*
    fn format_compact(&self,
                      io: &mut io::Write,
                      record: &Record,
                      logger_kvs: &OwnedKVList)
                      -> io::Result<()> {


        let mut iter = logger_kvs.iter_groups();
        let kv = iter.next();
        let indent = try!(self.format_recurse(io, record, kv, &mut iter));

        try!(self.print_indent(io, indent));

        let r_decorator = self.decorator.decorate(record);
        let mut ser = Serializer::new(io, r_decorator);
        let mut comma_needed =
            try!(self.print_msg_header(ser.io, &mut ser.decorator, record));

        for kv in record.kvs() {
            if comma_needed {
                try!(ser.print_comma());
            }
            try!(kv.serialize(record, &mut ser));
            comma_needed |= true;
        }
        try!(write!(&mut ser.io, "\n"));

        Ok(())
    }
    */

    /// Returns `true` if message was not empty
    fn print_msg_header(&self,
                        mut rd: &mut RecordDecorator,
                        record: &Record)
                        -> io::Result<bool> {
        try!(rd.start_timestamp());
        try!((self.fn_timestamp)(&mut rd));

        try!(rd.start_level());
        try!(write!(rd, " {} ", record.level().as_short_str()));

        try!(rd.start_msg());
        let mut count_rd = CountingWriter::new(&mut rd);
        try!(write!(count_rd, "{}", record.msg()));
        Ok(count_rd.count() != 0)
    }

    /*
    fn print_indent(&self,
                    io: &mut io::Write,
                    indent: usize)
                    -> io::Result<()> {
        for _ in 0..indent {
            try!(write!(io, "  "));
        }
        Ok(())
    }
    */

    /*
    // record in the history, and check if should print
    // given set of values
    fn should_print(&self, line: &[u8], indent: usize) -> bool {
        let mut history = self.history.lock().unwrap();
        if history.len() <= indent {
            debug_assert_eq!(history.len(), indent);
            history.push(line.into());
            true
        } else {
            let should = history[indent] != line;
            if should {
                history[indent] = line.into();
                history.truncate(indent + 1);
            }
            should
        }
    }
    */

    /*
    /// Recursively format given `logger_values_ref`
    ///
    /// Returns it's indent level
    fn format_recurse(&self,
                      io : &mut io::Write,
                      record: &slog::Record,
                      kv : Option<&KV>,
                      kv_iter: &mut slog::OwnedKVGroupIterator)
                                    -> io::Result<usize> {
        let (kv, indent) = if let Some(kv) = kv {
            let next_kv = kv_iter.next();
            (kv, try!(self.format_recurse(io, record, next_kv, kv_iter)))
        } else {
            return Ok(0);
        };


        let res : io::Result<()> = TL_BUF.with(|line| {
            let mut line = line.borrow_mut();
            line.clear();
            let r_decorator = self.decorator.decorate(record);
            let mut ser = Serializer::new(&mut *line, r_decorator);

            try!(self.print_indent(&mut ser.io, indent));
            let mut clean = true;
            let mut kv  = kv;
            let mut all_kvs = vec!();
            loop {
                kv = if let Some((v, rest)) = kv.split_first() {
                    all_kvs.push(v);
                    rest
                } else {
                    break;
                }

            }

            for kv in all_kvs.iter().rev() {
                if !clean {
                    try!(ser.print_comma());
                }
                try!(kv.serialize(record, &mut ser));
                clean = false;
            }

            let (mut line, _) = ser.finish();

            if self.should_print(line, indent) {
                try!(write!(line, "\n"));
                try!(io.write_all(line));
            }
            Ok(())
        });
        try!(res);

        Ok(indent + 1)
    }
    */
}
// }}}

// {{{ Serializer
struct Serializer<D: RecordDecorator> {
    comma_needed: bool,
    decorator: D,
}

impl<D: RecordDecorator> Serializer<D> {
    fn new(d: D, comma_needed: bool) -> Self {
        Serializer {
            comma_needed: comma_needed,
            decorator: d,
        }
    }

    fn maybe_print_comma(&mut self) -> io::Result<()> {
        if self.comma_needed {
            try!(self.decorator.start_comma());
            try!(write!(self.decorator, ", "));
            self.comma_needed |= true
        }
        Ok(())
    }

    fn finish(self) -> D {
        self.decorator
    }
}

macro_rules! s(
    ($s:expr, $k:expr, $v:expr) => {

        try!($s.maybe_print_comma());
        try!($s.decorator.start_key());
        try!(write!($s.decorator, "{}", $k));
        try!($s.decorator.start_separator());
        try!(write!($s.decorator, ": "));
        try!($s.decorator.start_value());
        try!(write!($s.decorator, "{}", $v));
    };
);


impl<D: RecordDecorator> slog::ser::Serializer for Serializer<D> {
    fn emit_none(&mut self, key: &str) -> slog::Result {
        s!(self, key, "None");
        Ok(())
    }
    fn emit_unit(&mut self, key: &str) -> slog::Result {
        s!(self, key, "()");
        Ok(())
    }

    fn emit_bool(&mut self, key: &str, val: bool) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }

    fn emit_char(&mut self, key: &str, val: char) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }

    fn emit_usize(&mut self, key: &str, val: usize) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_isize(&mut self, key: &str, val: isize) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }

    fn emit_u8(&mut self, key: &str, val: u8) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_i8(&mut self, key: &str, val: i8) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_u16(&mut self, key: &str, val: u16) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_i16(&mut self, key: &str, val: i16) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_u32(&mut self, key: &str, val: u32) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_i32(&mut self, key: &str, val: i32) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_f32(&mut self, key: &str, val: f32) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_u64(&mut self, key: &str, val: u64) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_i64(&mut self, key: &str, val: i64) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_f64(&mut self, key: &str, val: f64) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_str(&mut self, key: &str, val: &str) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
    fn emit_arguments(&mut self,
                      key: &str,
                      val: &fmt::Arguments)
                      -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
}
// }}}

// {{{ CountingWriter
// Wrapper for `Write` types that counts total bytes written.
struct CountingWriter<'a> {
    wrapped: &'a mut io::Write,
    count: usize,
}

impl<'a> CountingWriter<'a> {
    fn new(wrapped: &'a mut io::Write) -> CountingWriter {
        CountingWriter {
            wrapped: wrapped,
            count: 0,
        }
    }

    fn count(&self) -> usize {
        self.count
    }
}

impl<'a> io::Write for CountingWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.wrapped.write(buf).map(|n| {
            self.count += n;
            n
        })
    }

    fn flush(&mut self) -> io::Result<()> {
        self.wrapped.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.wrapped.write_all(buf).map(|_| {
            self.count += buf.len();
            ()
        })
    }
}
// }}}

// {{{ Timestamp
/// Threadsafe timestamp formatting function type
///
/// To satify `slog-rs` thread and unwind safety requirements, the
/// bounds expressed by this trait need to satisfied for a function
/// to be used in timestamp formatting.
pub trait ThreadSafeTimestampFn
    : Fn(&mut io::Write) -> io::Result<()> + Send + Sync + UnwindSafe + RefUnwindSafe + 'static {
}

impl<F> ThreadSafeTimestampFn for F
    where F: Fn(&mut io::Write) -> io::Result<()> + Send + Sync,
          F: UnwindSafe + RefUnwindSafe + 'static,
          F: ?Sized
{
}

const TIMESTAMP_FORMAT: &'static str = "%b %d %H:%M:%S%.3f";

/// Default local timezone timestamp function
///
/// The exact format used, is still subject to change.
pub fn timestamp_local(io: &mut io::Write) -> io::Result<()> {
    write!(io, "{}", chrono::Local::now().format(TIMESTAMP_FORMAT))
}

/// Default UTC timestamp function
///
/// The exact format used, is still subject to change.
pub fn timestamp_utc(io: &mut io::Write) -> io::Result<()> {
    write!(io, "{}", chrono::UTC::now().format(TIMESTAMP_FORMAT))
}
// }}}

// {{{ Plain

/// Plain (no-op) `Decorator` implementation
pub struct PlainDecorator<W>(sync::Arc<sync::Mutex<W>>) where W: io::Write;

impl<W> PlainDecorator<W>
    where W: io::Write
{
    /// Create `PlainDecorator` instance
    pub fn new(io: W) -> Self {
        PlainDecorator(sync::Arc::new(sync::Mutex::new(io)))
    }
}

impl<W> Decorator for PlainDecorator<W>
    where W: io::Write + Send + 'static
{
    fn decorate(&self,
                _record: &Record,
                _logger_values: &OwnedKVList)
                -> Box<RecordDecorator> {
        Box::new(PlainRecordDecorator {
            io: self.0.clone(),
            buf: vec![],
        })
    }
}

struct PlainRecordDecorator<W>
    where W: io::Write
{
    io: sync::Arc<sync::Mutex<W>>,
    buf: Vec<u8>,
}

impl<W> io::Write for PlainRecordDecorator<W>
    where W: io::Write
{
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.buf.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        if self.buf.is_empty() {
            return Ok(());
        }

        let mut io = try!(self.io
            .lock()
            .map_err(|_| {
                io::Error::new(io::ErrorKind::Other, "mutex locking error")
            }));

        try!(io.write_all(&self.buf));
        io.flush()
    }
}

impl<W> Drop for PlainRecordDecorator<W>
    where W: io::Write
{
    fn drop(&mut self) {
        let _ = self.flush();
    }
}

impl<W> RecordDecorator for PlainRecordDecorator<W> where W: io::Write {}


// }}}
// vim: foldmethod=marker foldmarker={{{,}}}
