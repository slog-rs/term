//! Timestamp logic for slog-term

/// A threadsafe timestamp formatter
///
/// To satify `slog-rs` thread and unwind safety requirements, the
/// bounds expressed by this trait need to satisfied for a function
/// to be used in timestamp formatting.
pub trait TimestampWriter: Sealed + Send + Sync + UnwindSafe + RefUnwindSafe + 'static {
    fn write_timestamp(&self, writer: &mut dyn io::Write) -> io::Result<()>;
}


impl<F> ThreadSafeTimestampFn for F
where
    F: Fn(&mut dyn io::Write) -> io::Result<()> + Send + Sync,
    F: UnwindSafe + RefUnwindSafe + 'static,
    F: ?Sized,
{}

mod sealed {
    pub trait Sealed {}
}