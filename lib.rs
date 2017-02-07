//! Unix terminal formatter and drain for slog-rs
//!
//! ```
//! #[macro_use]
//! extern crate slog;
//! extern crate slog_term;
//!
//! use slog::*;
//!
//! fn main() {
//!     let root = Logger::root(slog_term::streamer().build().fuse(), o!("build-id" => "8dfljdf"));
//! }
//! ```
#![warn(missing_docs)]

extern crate slog;
extern crate slog_stream;
extern crate isatty;
extern crate chrono;
extern crate thread_local;
extern crate term;

use std::{io, fmt, sync, cell};
use std::io::Write;

use isatty::{stderr_isatty, stdout_isatty};

use slog::Record;
use slog::{Level, OwnedKVList, KV};
use slog_stream::Format as StreamFormat;
use slog_stream::{Decorator, RecordDecorator, stream, async_stream};

thread_local! {
    static TL_BUF: cell::RefCell<Vec<u8>> = cell::RefCell::new(Vec::with_capacity(128));
}

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

impl<'a> Write for CountingWriter<'a> {
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

type WriterFn = Fn(&mut io::Write) -> io::Result<()>;

// Wrapper for `Write` types that executes a closure before writing anything,
// but only if the write isn't empty. A `finish` call executes a closure after
// writing, but again, only if something has been written.
struct SurroundingWriter<'a> {
    wrapped: &'a mut io::Write,
    before: Option<&'a WriterFn>,
    after: Option<&'a WriterFn>,
}

impl<'a> SurroundingWriter<'a> {
    fn new(wrapped: &'a mut io::Write,
           before: &'a WriterFn,
           after: &'a WriterFn)
           -> SurroundingWriter<'a> {
        SurroundingWriter {
            wrapped: wrapped,
            before: Some(before),
            after: Some(after),
        }
    }

    fn do_before(&mut self, buf: &[u8]) -> io::Result<()> {
        if buf.len() > 0 {
            if let Some(before) = self.before.take() {
                try!(before(self.wrapped));
            }
        }
        Ok(())
    }

    fn finish(&mut self) -> io::Result<()> {
        if let Some(after) = self.after.take() {
            if self.before.is_none() {
                try!(after(self.wrapped));
            }
        }
        Ok(())
    }
}

impl<'a> Drop for SurroundingWriter<'a> {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

impl<'a> Write for SurroundingWriter<'a> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        try!(self.do_before(buf));
        self.wrapped.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.wrapped.flush()
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        try!(self.do_before(buf));
        self.wrapped.write_all(buf)
    }
}


/// Timestamp function type
pub type TimestampFn = Fn(&mut io::Write) -> io::Result<()> + Send + Sync;

/// Formatting mode
pub enum FormatMode {
    /// Compact logging format
    Compact,
    /// Full logging format
    Full,
}

/// Full formatting with optional color support
pub struct Format<D: Decorator> {
    mode: FormatMode,
    decorator: D,
    history: sync::Mutex<Vec<Vec<u8>>>,
    fn_timestamp: Box<TimestampFn>,
}

impl<D: Decorator> Format<D> {
    /// New Format format that prints using color
    pub fn new(mode: FormatMode, d: D, fn_timestamp: Box<TimestampFn>) -> Self {
        Format {
            decorator: d,
            mode: mode,
            history: sync::Mutex::new(vec![]),
            fn_timestamp: fn_timestamp,
        }
    }

    // Returns `true` if message was not empty
    fn print_msg_header(&self,
                        io: &mut io::Write,
                        rd: &mut D::RecordDecorator,
                        record: &Record)
                        -> io::Result<bool> {
        try!(rd.fmt_timestamp(io, &*self.fn_timestamp));
        try!(rd.fmt_level(io,
                          &|io: &mut io::Write| write!(io, " {} ", record.level().as_short_str())));

        let mut writer = CountingWriter::new(io);
        try!(rd.fmt_msg(&mut writer, |io| write!(io, "{}", record.msg())));
        Ok(writer.count() > 0)
    }

    fn format_full(&self,
                   io: &mut io::Write,
                   record: &Record,
                   logger_values: &OwnedKVList)
                   -> io::Result<()> {

        let mut r_decorator = self.decorator.decorate(record);


        let mut comma_needed = try!(self.print_msg_header(io, &mut r_decorator, record));
        let mut serializer = Serializer::new(io, r_decorator);

        for kv in record.kvs().iter().rev() {
            if comma_needed {
                try!(serializer.print_comma());
            }
            try!(kv.serialize(record, &mut serializer));
            comma_needed |= true;
        }

        for kv in logger_values.iter_single() {
            if comma_needed {
                try!(serializer.print_comma());
            }
            try!(kv.serialize(record, &mut serializer));
            comma_needed |= true;
        }

        let (mut io, _decorator_r) = serializer.finish();

        try!(write!(io, "\n"));

        Ok(())
    }


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
        let mut comma_needed = try!(self.print_msg_header(ser.io, &mut ser.decorator, record));

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

    fn print_indent(&self, io: &mut io::Write, indent: usize) -> io::Result<()> {
        for _ in 0..indent {
            try!(write!(io, "  "));
        }
        Ok(())
    }

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
}

fn severity_to_color(lvl: Level) -> u8 {
    match lvl {
        Level::Critical => 5,
        Level::Error => 1,
        Level::Warning => 3,
        Level::Info => 2,
        Level::Debug => 6,
        Level::Trace => 4,
    }
}

/// Any type of a terminal supported by `term` crate
pub enum AnyTerminal {
    /// Stdout terminal
    Stdout(Box<term::StdoutTerminal>),
    /// Stderr terminal
    Stderr(Box<term::StderrTerminal>),
}

/// Record decorator (color) for terminal output
pub struct ColorDecorator{
    term: Option<sync::Arc<sync::Mutex<AnyTerminal>>>,
}

impl ColorDecorator {
    /// New decorator that does color records
    pub fn new_colored(t : AnyTerminal) -> Self {
        ColorDecorator { term : Some(sync::Arc::new(sync::Mutex::new(t)))}
    }
    /// New decorator that does not color records
    pub fn new_plain() -> Self {
        ColorDecorator { term: None }
    }
}

/// Particular record decorator (color) for terminal output
pub struct ColorRecordDecorator {
    level_color: Option<u8>,
    key_bold: bool,
    term: Option<sync::Arc<sync::Mutex<AnyTerminal>>>,
}


impl Decorator for ColorDecorator {
    type RecordDecorator = ColorRecordDecorator;

    fn decorate(&self, record: &Record) -> ColorRecordDecorator {
        if let Some(ref term) = self.term {
            ColorRecordDecorator {
                term: Some(term.clone()),
                level_color: Some(severity_to_color(record.level())),
                key_bold: true,
            }
        } else {
            ColorRecordDecorator {
                term: None,
                level_color: None,
                key_bold: false,
            }
        }
    }
}


impl RecordDecorator for ColorRecordDecorator {
    fn fmt_level<F>(&mut self,
                 io: &mut io::Write,
                 f: F)
                 -> io::Result<()>
        where F : FnOnce(&mut io::Write) -> io::Result<()> {
        if let Some(level_color) = self.level_color {
            try!(write!(io, "\x1b[3{}m", level_color));
            try!(f(io));
            try!(write!(io, "\x1b[39m"));
        } else {
            try!(f(io));
        }
        Ok(())
    }


    fn fmt_msg<F>(&mut self,
               io: &mut io::Write,
               f: F)
               -> io::Result<()>
        where F : FnOnce(&mut io::Write) -> io::Result<()> {
        if self.key_bold {
            let before = |io: &mut io::Write| write!(io, "\x1b[1m");
            let after = |io: &mut io::Write| write!(io, "\x1b[0m");
            let mut wrapper = SurroundingWriter::new(io, &before, &after);
            try!(f(&mut wrapper));
            try!(wrapper.finish());
        } else {
            try!(f(io));
        }
        Ok(())
    }

    fn fmt_key<F>(&mut self,
               io: &mut io::Write,
               f: F)
               -> io::Result<()>
        where F : FnOnce(&mut io::Write) -> io::Result<()> {
        self.fmt_msg(io, f)
    }
}

struct Serializer<W, D: RecordDecorator> {
    io: W,
    decorator: D,
}

impl<W: io::Write, D: RecordDecorator> Serializer<W, D> {
    fn new(io: W, d: D) -> Self {
        Serializer {
            io: io,
            decorator: d,
        }
    }

    fn print_comma(&mut self) -> io::Result<()> {
        try!(self.decorator.fmt_separator(&mut self.io, &|io: &mut io::Write| write!(io, ", ")));
        Ok(())
    }

    fn finish(self) -> (W, D) {
        (self.io, self.decorator)
    }
}

macro_rules! s(
    ($s:expr, $k:expr, $v:expr) => {
        try!($s.decorator.fmt_key(&mut $s.io, &|io : &mut io::Write| write!(io, "{}", $k)));
        try!($s.decorator.fmt_separator(&mut $s.io, &|io : &mut io::Write| write!(io, ": ")));
        try!($s.decorator.fmt_value(&mut $s.io, &|io : &mut io::Write| write!(io, "{}", $v)));
    };
);


impl<W: io::Write, D: RecordDecorator> slog::ser::Serializer for Serializer<W, D> {
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
    fn emit_arguments(&mut self, key: &str, val: &fmt::Arguments) -> slog::Result {
        s!(self, key, val);
        Ok(())
    }
}

impl<D: Decorator + Send + Sync> StreamFormat for Format<D> {
    fn format(&self,
              io: &mut io::Write,
              record: &Record,
              logger_values: &OwnedKVList)
              -> io::Result<()> {
        match self.mode {
            FormatMode::Compact => self.format_compact(io, record, logger_values),
            FormatMode::Full => self.format_full(io, record, logger_values),
        }
    }
}

const TIMESTAMP_FORMAT: &'static str = "%b %d %H:%M:%S%.3f";

/// Default local timestamp function used by `Format`
///
/// The exact format used, is still subject to change.
pub fn timestamp_local(io: &mut io::Write) -> io::Result<()> {
    write!(io, "{}", chrono::Local::now().format(TIMESTAMP_FORMAT))
}

/// Default UTC timestamp function used by `Format`
///
/// The exact format used, is still subject to change.
pub fn timestamp_utc(io: &mut io::Write) -> io::Result<()> {
    write!(io, "{}", chrono::UTC::now().format(TIMESTAMP_FORMAT))
}

/// Streamer builder
pub struct StreamerBuilder {
    color: Option<bool>, // None = auto
    stdout: bool,
    async: bool,
    mode: FormatMode,
    fn_timestamp: Box<TimestampFn>,
}

impl StreamerBuilder {
    /// New `StreamerBuilder`
    pub fn new() -> Self {
        StreamerBuilder {
            color: None,
            stdout: true,
            async: false,
            mode: FormatMode::Full,
            fn_timestamp: Box::new(timestamp_local),
        }
    }

    /// Force colored output
    pub fn color(mut self) -> Self {
        self.color = Some(true);
        self
    }

    /// Force plain output
    pub fn plain(mut self) -> Self {
        self.color = None;
        self
    }

    /// Auto detect color (default)
    pub fn auto_color(mut self) -> Self {
        self.color = None;
        self
    }

    /// Output to stderr
    pub fn stderr(mut self) -> Self {
        self.stdout = false;
        self
    }

    /// Output to stdout (default)
    pub fn stdout(mut self) -> Self {
        self.stdout = true;
        self
    }

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

    /// Use asynchronous streamer
    pub fn async(mut self) -> Self {
        self.async = true;
        self
    }

    /// Use synchronous streamer (default)
    pub fn sync(mut self) -> Self {
        self.async = false;
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
        where F: Fn(&mut io::Write) -> io::Result<()> + 'static + Send + Sync
    {
        self.fn_timestamp = Box::new(f);
        self
    }

    /// Build the streamer
    pub fn build(self) -> Box<slog::Drain<Error = io::Error> + Send + Sync> {
        let color = self.color.unwrap_or(if self.stdout {
            stdout_isatty()
        } else {
            stderr_isatty()
        });

        let term : Option<AnyTerminal> = if let Some(use_color) = self.color {
            if use_color {
                if self.stdout {
                    term::stdout().map(AnyTerminal::Stdout)
                } else {
                    term::stderr().map(AnyTerminal::Stderr)
                }
            } else {
                None
            }
        } else {
            if self.stdout {
                term::stdout().map(AnyTerminal::Stdout)
            } else {
                term::stderr().map(AnyTerminal::Stderr)
            }
        };

        let format = Format::new(self.mode,
                                 ColorDecorator { term : term.map(sync::Mutex::new).map(sync::Arc::new) },
                                 self.fn_timestamp);

        let io = if self.stdout {
            Box::new(io::stdout()) as Box<io::Write + Send>
        } else {
            Box::new(io::stderr()) as Box<io::Write + Send>
        };

        if self.async {
            Box::new(async_stream(io, format))
        } else {
            Box::new(stream(io, format))
        }
    }
}

impl Default for StreamerBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Build `slog_stream::Streamer`/`slog_stream::AsyncStreamer` that
/// will output logging records to stderr/stderr.
pub fn streamer() -> StreamerBuilder {
    StreamerBuilder::new()
}
