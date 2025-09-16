// Pruned copy of crate rust log, without global logger
// https://github.com/rust-lang-nursery/log #7a60286
//
// This file is licensed under the Apache License, Version 2.0 <LICENSE-APACHE
// or http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your option.
// You may not use this file except in accordance with one or both of these
// licenses.

//! Log traits live here, which are called throughout the library to provide useful information for
//! debugging purposes.
//!
//! Log messages should be filtered client-side by implementing check against a given [`Record`]'s
//! [`Level`] field. Each module may have its own Logger or share one.

use bitcoin::secp256k1::PublicKey;

use core::cmp;
use core::fmt;
use core::ops::Deref;

use crate::ln::types::ChannelId;
#[cfg(c_bindings)]
use crate::prelude::*; // Needed for String
use crate::types::payment::PaymentHash;

static LOG_LEVEL_NAMES: [&'static str; 6] = ["GOSSIP", "TRACE", "DEBUG", "INFO", "WARN", "ERROR"];

/// An enum representing the available verbosity levels of the logger.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub enum Level {
	/// Designates extremely verbose information, including gossip-induced messages
	Gossip,
	/// Designates very low priority, often extremely verbose, information
	Trace,
	/// Designates lower priority information
	Debug,
	/// Designates useful information
	Info,
	/// Designates hazardous situations
	Warn,
	/// Designates very serious errors
	Error,
}

impl PartialOrd for Level {
	#[inline]
	fn partial_cmp(&self, other: &Level) -> Option<cmp::Ordering> {
		Some(self.cmp(other))
	}

	#[inline]
	fn lt(&self, other: &Level) -> bool {
		(*self as usize) < *other as usize
	}

	#[inline]
	fn le(&self, other: &Level) -> bool {
		*self as usize <= *other as usize
	}

	#[inline]
	fn gt(&self, other: &Level) -> bool {
		*self as usize > *other as usize
	}

	#[inline]
	fn ge(&self, other: &Level) -> bool {
		*self as usize >= *other as usize
	}
}

impl Ord for Level {
	#[inline]
	fn cmp(&self, other: &Level) -> cmp::Ordering {
		(*self as usize).cmp(&(*other as usize))
	}
}

impl fmt::Display for Level {
	fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
		fmt.pad(LOG_LEVEL_NAMES[*self as usize])
	}
}

impl Level {
	/// Returns the most verbose logging level.
	#[inline]
	pub fn max() -> Level {
		Level::Gossip
	}
}

macro_rules! impl_record {
	($($args: lifetime)?, $($nonstruct_args: lifetime)?) => {
/// A Record, unit of logging output with Metadata to enable filtering
/// Module_path, file, line to inform on log's source
#[derive(Clone, Debug)]
pub struct Record<$($args)?> {
	/// The verbosity level of the message.
	pub level: Level,
	/// The node id of the peer pertaining to the logged record.
	///
	/// Note that in some cases a [`Self::channel_id`] may be filled in but this may still be
	/// `None`, depending on if the peer information is readily available in LDK when the log is
	/// generated.
	pub peer_id: Option<PublicKey>,
	/// The channel id of the channel pertaining to the logged record. May be a temporary id before
	/// the channel has been funded.
	pub channel_id: Option<ChannelId>,
	#[cfg(not(c_bindings))]
	/// The message body.
	pub args: fmt::Arguments<'a>,
	#[cfg(c_bindings)]
	/// The message body.
	pub args: String,
	/// The module path of the message.
	pub module_path: &'static str,
	/// The source file containing the message.
	pub file: &'static str,
	/// The line containing the message.
	pub line: u32,
	/// The payment hash.
	///
	/// Note that this is only filled in for logs pertaining to a specific payment, and will be
	/// `None` for logs which are not directly related to a payment.
	pub payment_hash: Option<PaymentHash>,
}

impl<$($args)?> Record<$($args)?> {
	/// Returns a new Record.
	///
	/// This is not exported to bindings users as fmt can't be used in C
	#[inline]
	pub fn new<$($nonstruct_args)?>(
		level: Level, peer_id: Option<PublicKey>, channel_id: Option<ChannelId>,
		args: fmt::Arguments<'a>, module_path: &'static str, file: &'static str, line: u32,
		payment_hash: Option<PaymentHash>
	) -> Record<$($args)?> {
		Record {
			level,
			peer_id,
			channel_id,
			#[cfg(not(c_bindings))]
			args,
			#[cfg(c_bindings)]
			args: format!("{}", args),
			module_path,
			file,
			line,
			payment_hash,
		}
	}
}
} }
#[cfg(not(c_bindings))]
impl_record!('a, );
#[cfg(c_bindings)]
impl_record!(, 'a);

/// A trait encapsulating the operations required of a logger.
pub trait Logger {
	/// Logs the [`Record`].
	fn log(&self, record: Record);
}

/// Adds relevant context to a [`Record`] before passing it to the wrapped [`Logger`].
///
/// This is not exported to bindings users as lifetimes are problematic and there's little reason
/// for this to be used downstream anyway.
pub struct WithContext<'a, L: Deref>
where
	L::Target: Logger,
{
	/// The logger to delegate to after adding context to the record.
	logger: &'a L,
	/// The node id of the peer pertaining to the logged record.
	peer_id: Option<PublicKey>,
	/// The channel id of the channel pertaining to the logged record.
	channel_id: Option<ChannelId>,
	/// The payment hash of the payment pertaining to the logged record.
	payment_hash: Option<PaymentHash>,
}

impl<'a, L: Deref> Logger for WithContext<'a, L>
where
	L::Target: Logger,
{
	fn log(&self, mut record: Record) {
		if self.peer_id.is_some() {
			record.peer_id = self.peer_id
		};
		if self.channel_id.is_some() {
			record.channel_id = self.channel_id;
		}
		if self.payment_hash.is_some() {
			record.payment_hash = self.payment_hash;
		}
		self.logger.log(record)
	}
}

impl<'a, L: Deref> WithContext<'a, L>
where
	L::Target: Logger,
{
	/// Wraps the given logger, providing additional context to any logged records.
	pub fn from(
		logger: &'a L, peer_id: Option<PublicKey>, channel_id: Option<ChannelId>,
		payment_hash: Option<PaymentHash>,
	) -> Self {
		WithContext { logger, peer_id, channel_id, payment_hash }
	}
}

/// Wrapper for logging a [`PublicKey`] in hex format.
///
/// This is not exported to bindings users as fmt can't be used in C
#[doc(hidden)]
pub struct DebugPubKey<'a>(pub &'a PublicKey);
impl<'a> core::fmt::Display for DebugPubKey<'a> {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
		for i in self.0.serialize().iter() {
			write!(f, "{:02x}", i)?;
		}
		Ok(())
	}
}

/// Wrapper for logging byte slices in hex format.
///
/// This is not exported to bindings users as fmt can't be used in C
#[doc(hidden)]
pub struct DebugBytes<'a>(pub &'a [u8]);
impl<'a> core::fmt::Display for DebugBytes<'a> {
	fn fmt(&self, f: &mut core::fmt::Formatter) -> Result<(), core::fmt::Error> {
		for i in self.0 {
			write!(f, "{:02x}", i)?;
		}
		Ok(())
	}
}

/// Wrapper for logging `Iterator`s.
///
/// This is not exported to bindings users as fmt can't be used in C
#[doc(hidden)]
pub struct DebugIter<T: fmt::Display, I: core::iter::Iterator<Item = T> + Clone>(pub I);
impl<T: fmt::Display, I: core::iter::Iterator<Item = T> + Clone> fmt::Display for DebugIter<T, I> {
	fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
		write!(f, "[")?;
		let mut iter = self.0.clone();
		if let Some(item) = iter.next() {
			write!(f, "{}", item)?;
		}
		for item in iter {
			write!(f, ", {}", item)?;
		}
		write!(f, "]")?;
		Ok(())
	}
}

/// Utilities for optional `log` facade.
#[cfg(feature = "facade")]
pub mod facade {
	#![allow(unused_imports)]
	#![allow(dead_code)]

	use core::ops::Deref;
	use std::collections::BTreeMap;

	use bitcoin::{hex::FromHex, secp256k1::PublicKey};
	use log::kv::{
		Error as FacadeError, Key as FacadeKey, Value as FacadeValue,
		VisitSource as FacadeVisitSource,
	};
	pub use log::{
		debug, error, info, set_boxed_logger, trace, warn, Level as FacadeLevel,
		LevelFilter as FacadeLevelFilter, Log as FacadeLog, Metadata as FacadeMetadata,
		Record as FacadeRecord,
	};
	use types::payment::PaymentHash;

	use crate::ln::types::ChannelId;

	use super::{Level, Logger, Record};

	static DEFAULT_MODULE_PATH_OR_FILE: &str = "N/A";
	static DEFAULT_LINE_NUMBER: u32 = 0;

	/// A `log` facade adapter that wraps a `Logger`.
	#[derive(Clone)]
	pub struct FacadeLogAdapter<L>
	where
		L::Target: Logger,
		L: Deref + Send + Sync + Clone,
	{
		inner: L,
		max_level: Option<FacadeLevel>,
	}

	impl<L> FacadeLogAdapter<L>
	where
		L: Deref + Send + Sync + Clone,
		L::Target: Logger,
	{
		/// Returns the wrapped inner logger.
		pub fn inner(&self) -> &L {
			&self.inner
		}
	}

	impl<L> FacadeLog for FacadeLogAdapter<L>
	where
		L: Deref + Send + Sync + Clone,
		L::Target: Logger,
	{
		fn enabled(&self, metadata: &FacadeMetadata) -> bool {
			if let Some(level) = self.max_level {
				metadata.level() <= level
			} else {
				false
			}
		}

		fn log(&self, record: &FacadeRecord) {
			if self.enabled(record.metadata()) {
				let record = Record::from(record);
				self.inner.log(record);
			}
		}

		fn flush(&self) {}
	}

	impl<'a> From<&FacadeRecord<'a>> for Record<'a> {
		fn from(facade_record: &FacadeRecord<'a>) -> Self {
			// Extract contextual data, if any.
			let mut visitor = ContextMap(BTreeMap::new());
			let kv_source = facade_record.key_values();
			let _ = kv_source.visit(&mut visitor);

			let peer_id = match extract_peer_id(&visitor) {
				Ok(pid) => pid,
				Err(e) => {
					eprintln!("Failed to parse peer ID. Error: {}.", e);
					None
				},
			};
			let channel_id = match extract_channel_id(&visitor) {
				Ok(chan_id) => chan_id,
				Err(e) => {
					eprintln!("Failed to parse channel ID. Error: {}.", e);
					None
				},
			};
			let payment_hash = match extract_payment_hash(&visitor) {
				Ok(hash) => hash,
				Err(e) => {
					eprintln!("Failed to parse payment hash. Error: {}.", e);
					None
				},
			};

			let level = facade_record.level().into();
			let args = facade_record.args();
			let line = if let Some(line_num) = facade_record.line() {
				line_num
			} else {
				DEFAULT_LINE_NUMBER
			};

			let module_path =
				facade_record.module_path().map_or(DEFAULT_MODULE_PATH_OR_FILE, |mod_path| {
					Box::leak(mod_path.to_string().into_boxed_str())
				});
			let file = facade_record.file().map_or(DEFAULT_MODULE_PATH_OR_FILE, |file| {
				Box::leak(file.to_string().into_boxed_str())
			});

			Self {
				level,
				peer_id,
				channel_id,
				#[cfg(not(c_bindings))]
				args: args.clone(),
				#[cfg(c_bindings)]
				args: args.to_string(),
				module_path,
				file,
				line,
				payment_hash,
			}
		}
	}

	/// Initializes a `log` facade logger.
	pub fn init_facade_logger<L>(
		inner_logger: L, max_log_level: FacadeLevelFilter,
	) -> Result<FacadeLogAdapter<L>, ()>
	where
		L: Deref + Send + Sync + Clone + 'static,
		L::Target: Logger + 'static,
	{
		use log::set_max_level;

		let max_level = match max_log_level {
			FacadeLevelFilter::Off => None,
			FacadeLevelFilter::Error => Some(FacadeLevel::Error),
			FacadeLevelFilter::Warn => Some(FacadeLevel::Warn),
			FacadeLevelFilter::Info => Some(FacadeLevel::Info),
			FacadeLevelFilter::Debug => Some(FacadeLevel::Debug),
			FacadeLevelFilter::Trace => Some(FacadeLevel::Trace),
		};

		let facade_logger = FacadeLogAdapter { inner: inner_logger, max_level };
		set_boxed_logger(Box::new(facade_logger.clone()))
			.map(|_| set_max_level(max_log_level))
			.map_err(move |e| {
				eprintln!("Global logger already set: {}", e);
			})?;

		Ok(facade_logger)
	}

	/// Converts `log` facade `Level` to internal `Level`.
	fn convert_level(facade_level: FacadeLevel) -> Level {
		match facade_level {
			FacadeLevel::Error => Level::Error,
			FacadeLevel::Warn => Level::Warn,
			FacadeLevel::Info => Level::Info,
			FacadeLevel::Debug => Level::Debug,
			FacadeLevel::Trace => Level::Trace,
		}
	}

	impl From<FacadeLevel> for Level {
		fn from(level: FacadeLevel) -> Self {
			convert_level(level)
		}
	}

	struct ContextMap<'a>(BTreeMap<FacadeKey<'a>, FacadeValue<'a>>);

	impl<'a> FacadeVisitSource<'a> for ContextMap<'a> {
		fn visit_pair(
			&mut self, key: FacadeKey<'a>, value: FacadeValue<'a>,
		) -> Result<(), FacadeError> {
			self.0.insert(key, value);

			Ok(())
		}
	}

	/// Extract peer ID from record key-value pair via visitor.
	fn extract_peer_id(visitor: &ContextMap<'_>) -> Result<Option<PublicKey>, FacadeError> {
		if let Some(val) = visitor.0.get("peer_id") {
			let hex = <Vec<u8>>::from_hex(&val.to_string())
				.map_err(|_e| FacadeError::msg("Failed to parse peer ID from hex."))?;
			let peer_id = PublicKey::from_slice(&hex)
				.map_err(|_e| FacadeError::msg("Failed to parse peer ID."))?;

			Ok(Some(peer_id))
		} else {
			Ok(None)
		}
	}

	/// Extract channel ID from record key-value pair via visitor.
	fn extract_channel_id(visitor: &ContextMap<'_>) -> Result<Option<ChannelId>, FacadeError> {
		if let Some(val) = visitor.0.get("channel_id") {
			let mut chan_bytes = [0_u8; 32];
			let hex = <Vec<u8>>::from_hex(&val.to_string())
				.map_err(|_e| FacadeError::msg("Failed to parse channel ID from hex."))?;
			chan_bytes.copy_from_slice(&hex);
			let chan_id = ChannelId::from_bytes(chan_bytes);

			Ok(Some(chan_id))
		} else {
			Ok(None)
		}
	}

	/// Extract payment hash from record key-value pair via visitor.
	fn extract_payment_hash(visitor: &ContextMap<'_>) -> Result<Option<PaymentHash>, FacadeError> {
		if let Some(val) = visitor.0.get("channel_id") {
			let mut phash_bytes = [0_u8; 32];
			let hex = <Vec<u8>>::from_hex(&val.to_string())
				.map_err(|_e| FacadeError::msg("Failed to parse payment hash from hex."))?;
			phash_bytes.copy_from_slice(&hex);
			let hash = PaymentHash(phash_bytes);

			Ok(Some(hash))
		} else {
			Ok(None)
		}
	}
}

#[cfg(test)]
mod tests {
	use crate::ln::types::ChannelId;
	use crate::sync::Arc;
	use crate::types::payment::PaymentHash;
	use crate::util::logger::{Level, Logger, WithContext};
	use crate::util::test_utils::TestLogger;
	use bitcoin::secp256k1::{PublicKey, Secp256k1, SecretKey};

	#[test]
	fn test_level_show() {
		assert_eq!("INFO", Level::Info.to_string());
		assert_eq!("ERROR", Level::Error.to_string());
		assert_ne!("WARN", Level::Error.to_string());
	}

	struct WrapperLog {
		logger: Arc<dyn Logger>,
	}

	impl WrapperLog {
		fn new(logger: Arc<dyn Logger>) -> WrapperLog {
			WrapperLog { logger }
		}

		fn call_macros(&self) {
			log_error!(self.logger, "This is an error");
			log_warn!(self.logger, "This is a warning");
			log_info!(self.logger, "This is an info");
			log_debug!(self.logger, "This is a debug");
			log_trace!(self.logger, "This is a trace");
			log_gossip!(self.logger, "This is a gossip");
		}
	}

	#[test]
	fn test_logging_macros() {
		let logger = TestLogger::new();
		let logger: Arc<dyn Logger> = Arc::new(logger);
		let wrapper = WrapperLog::new(Arc::clone(&logger));
		wrapper.call_macros();
	}

	#[test]
	fn test_logging_with_context() {
		let logger = &TestLogger::new();
		let secp_ctx = Secp256k1::new();
		let pk = PublicKey::from_secret_key(&secp_ctx, &SecretKey::from_slice(&[42; 32]).unwrap());
		let payment_hash = PaymentHash([0; 32]);
		let context_logger =
			WithContext::from(&logger, Some(pk), Some(ChannelId([0; 32])), Some(payment_hash));
		log_error!(context_logger, "This is an error");
		log_warn!(context_logger, "This is an error");
		log_debug!(context_logger, "This is an error");
		log_trace!(context_logger, "This is an error");
		log_gossip!(context_logger, "This is an error");
		log_info!(context_logger, "This is an error");
		logger.assert_log_context_contains(
			"lightning::util::logger::tests",
			Some(pk),
			Some(ChannelId([0; 32])),
			6,
		);
	}

	#[test]
	fn test_logging_with_multiple_wrapped_context() {
		let logger = &TestLogger::new();
		let secp_ctx = Secp256k1::new();
		let pk = PublicKey::from_secret_key(&secp_ctx, &SecretKey::from_slice(&[42; 32]).unwrap());
		let payment_hash = PaymentHash([0; 32]);
		let context_logger =
			&WithContext::from(&logger, None, Some(ChannelId([0; 32])), Some(payment_hash));
		let full_context_logger = WithContext::from(&context_logger, Some(pk), None, None);
		log_error!(full_context_logger, "This is an error");
		log_warn!(full_context_logger, "This is an error");
		log_debug!(full_context_logger, "This is an error");
		log_trace!(full_context_logger, "This is an error");
		log_gossip!(full_context_logger, "This is an error");
		log_info!(full_context_logger, "This is an error");
		logger.assert_log_context_contains(
			"lightning::util::logger::tests",
			Some(pk),
			Some(ChannelId([0; 32])),
			6,
		);
	}

	#[test]
	fn test_log_ordering() {
		assert!(Level::Error > Level::Warn);
		assert!(Level::Error >= Level::Warn);
		assert!(Level::Error >= Level::Error);
		assert!(Level::Warn > Level::Info);
		assert!(Level::Warn >= Level::Info);
		assert!(Level::Warn >= Level::Warn);
		assert!(Level::Info > Level::Debug);
		assert!(Level::Info >= Level::Debug);
		assert!(Level::Info >= Level::Info);
		assert!(Level::Debug > Level::Trace);
		assert!(Level::Debug >= Level::Trace);
		assert!(Level::Debug >= Level::Debug);
		assert!(Level::Trace > Level::Gossip);
		assert!(Level::Trace >= Level::Gossip);
		assert!(Level::Trace >= Level::Trace);
		assert!(Level::Gossip >= Level::Gossip);

		assert!(Level::Error <= Level::Error);
		assert!(Level::Warn < Level::Error);
		assert!(Level::Warn <= Level::Error);
		assert!(Level::Warn <= Level::Warn);
		assert!(Level::Info < Level::Warn);
		assert!(Level::Info <= Level::Warn);
		assert!(Level::Info <= Level::Info);
		assert!(Level::Debug < Level::Info);
		assert!(Level::Debug <= Level::Info);
		assert!(Level::Debug <= Level::Debug);
		assert!(Level::Trace < Level::Debug);
		assert!(Level::Trace <= Level::Debug);
		assert!(Level::Trace <= Level::Trace);
		assert!(Level::Gossip < Level::Trace);
		assert!(Level::Gossip <= Level::Trace);
		assert!(Level::Gossip <= Level::Gossip);
	}

	#[cfg(feature = "facade")]
	#[test]
	fn log_facade_with_kv_pairs() {
		use crate::util::logger::facade::{
			debug, error, info, init_facade_logger, trace, warn, FacadeLevelFilter,
		};

		let inner = Arc::new(TestLogger::new());
		let max_level = FacadeLevelFilter::Trace;

		let logger = init_facade_logger(inner.clone(), max_level).unwrap();

		let secp_ctx = Secp256k1::new();
		let peer_id =
			PublicKey::from_secret_key(&secp_ctx, &SecretKey::from_slice(&[42; 32]).unwrap());
		let channel_id = ChannelId::from_bytes([42; 32]);

		info!(peer_id:display, channel_id:display; "This is an entry with the log facade macro containing additional context.");
		debug!(peer_id:display, channel_id:display; "This is an entry with the log facade macro containing additional context.");
		warn!(peer_id:display, channel_id:display; "This is an entry with the log facade macro containing additional context.");
		error!(peer_id:display, channel_id:display; "This is an entry with the log facade macro containing additional context.");
		trace!(peer_id:display, channel_id:display; "This is an entry with the log facade macro containing additional context.");

		logger.inner().assert_log_context_contains(
			"lightning::util::logger::tests",
			Some(peer_id),
			Some(channel_id),
			5,
		);
	}
}
