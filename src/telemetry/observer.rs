use crate::observability::traits::{Observer, ObserverEvent, ObserverMetric};
use crate::telemetry::store::{ActionRecord, TelemetrySqliteStore};
use parking_lot::Mutex;
use std::any::Any;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Observer implementation that translates `ObserverEvent`s into telemetry
/// `ActionRecord` submissions for the research database.
pub struct TelemetryObserver {
    store: Arc<TelemetrySqliteStore>,
    session_id: String,
    turn_counter: AtomicU64,
    sequence_counter: AtomicU64,
    previous_action_type: Mutex<Option<String>>,
    turn_action_sequence: Mutex<Vec<String>>,
    is_user_initiated: Mutex<bool>,
}

impl TelemetryObserver {
    pub fn new(store: Arc<TelemetrySqliteStore>, session_id: String) -> Self {
        Self {
            store,
            session_id,
            turn_counter: AtomicU64::new(0),
            sequence_counter: AtomicU64::new(0),
            previous_action_type: Mutex::new(None),
            turn_action_sequence: Mutex::new(Vec::new()),
            is_user_initiated: Mutex::new(false),
        }
    }

    fn now_ts() -> (String, i64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let epoch_ms = i64::try_from(now.as_millis()).unwrap_or(i64::MAX);
        let ts = chrono::Utc::now().to_rfc3339();
        (ts, epoch_ms)
    }

    fn next_sequence(&self) -> i64 {
        self.sequence_counter.fetch_add(1, Ordering::Relaxed) as i64
    }

    fn turn_id(&self) -> String {
        format!(
            "{}-t{}",
            self.session_id,
            self.turn_counter.load(Ordering::Relaxed)
        )
    }

    fn record_action(&self, event_type: &str, record: ActionRecord) {
        let mut prev = self.previous_action_type.lock();
        let mut seq = self.turn_action_sequence.lock();
        seq.push(event_type.to_string());
        *prev = Some(event_type.to_string());
        drop(prev);
        drop(seq);
        self.store.submit_action(record);
    }
}

impl Observer for TelemetryObserver {
    fn record_event(&self, event: &ObserverEvent) {
        match event {
            ObserverEvent::AgentStart { .. } => {
                *self.is_user_initiated.lock() = true;
            }
            ObserverEvent::LlmResponse {
                provider,
                model,
                duration,
                success,
                error_message,
                tokens_in,
                tokens_out,
            } => {
                let (ts, ts_epoch_ms) = Self::now_ts();
                let seq = self.next_sequence();
                let prev = self.previous_action_type.lock().clone();
                let action_seq = serde_json::to_string(&*self.turn_action_sequence.lock()).ok();
                let user_init = *self.is_user_initiated.lock();

                let record = ActionRecord {
                    ts,
                    ts_epoch_ms,
                    session_id: self.session_id.clone(),
                    turn_id: self.turn_id(),
                    sequence_index: seq,
                    event_type: "llm_response".into(),
                    provider: Some(provider.clone()),
                    model: Some(model.clone()),
                    tool_name: None,
                    tool_type_embedding: None,
                    arguments_hash: None,
                    tool_success: Some(*success),
                    duration_ms: Some(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)),
                    tokens_in: tokens_in.map(|t| i64::try_from(t).unwrap_or(i64::MAX)),
                    tokens_out: tokens_out.map(|t| i64::try_from(t).unwrap_or(i64::MAX)),
                    is_user_initiated: user_init,
                    iteration_index: 0,
                    previous_action_type: prev,
                    turn_action_sequence: action_seq,
                    error_message: error_message.clone(),
                };
                self.record_action("llm_response", record);

                // Clear user-initiated flag after first event in a turn
                if user_init {
                    *self.is_user_initiated.lock() = false;
                }
            }
            ObserverEvent::ToolCall {
                tool,
                duration,
                success,
                arguments_hash,
                iteration,
            } => {
                let (ts, ts_epoch_ms) = Self::now_ts();
                let seq = self.next_sequence();
                let prev = self.previous_action_type.lock().clone();
                let action_seq = serde_json::to_string(&*self.turn_action_sequence.lock()).ok();

                let record = ActionRecord {
                    ts,
                    ts_epoch_ms,
                    session_id: self.session_id.clone(),
                    turn_id: self.turn_id(),
                    sequence_index: seq,
                    event_type: "tool_call".into(),
                    provider: None,
                    model: None,
                    tool_name: Some(tool.clone()),
                    tool_type_embedding: None,
                    arguments_hash: arguments_hash.clone(),
                    tool_success: Some(*success),
                    duration_ms: Some(i64::try_from(duration.as_millis()).unwrap_or(i64::MAX)),
                    tokens_in: None,
                    tokens_out: None,
                    is_user_initiated: false,
                    iteration_index: i64::from(iteration.unwrap_or(0)),
                    previous_action_type: prev,
                    turn_action_sequence: action_seq,
                    error_message: None,
                };
                self.record_action("tool_call", record);
            }
            ObserverEvent::TurnComplete => {
                self.turn_counter.fetch_add(1, Ordering::Relaxed);
                self.sequence_counter.store(0, Ordering::Relaxed);
                *self.previous_action_type.lock() = None;
                self.turn_action_sequence.lock().clear();
            }
            // Other events are not recorded in the telemetry store.
            _ => {}
        }
    }

    fn record_metric(&self, _metric: &ObserverMetric) {
        // Telemetry observer does not handle metrics — system samples
        // are collected by the separate collector task.
    }

    fn flush(&self) {
        // The store is backed by a sync_channel writer thread;
        // records are flushed automatically in batches.
    }

    fn name(&self) -> &str {
        "telemetry"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tempfile::TempDir;

    fn make_store(tmp: &TempDir) -> Arc<TelemetrySqliteStore> {
        Arc::new(TelemetrySqliteStore::open(tmp.path(), 64).unwrap())
    }

    #[test]
    fn observer_records_llm_response() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let obs = TelemetryObserver::new(store.clone(), "test-sess".into());

        obs.record_event(&ObserverEvent::LlmResponse {
            provider: "openai".into(),
            model: "gpt-4".into(),
            duration: Duration::from_millis(200),
            success: true,
            error_message: None,
            tokens_in: Some(100),
            tokens_out: Some(50),
        });

        // Give writer thread time to flush.
        std::thread::sleep(Duration::from_millis(300));
        drop(obs);
        drop(store);

        let conn = rusqlite::Connection::open(tmp.path().join("research.db")).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM action_events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);

        let event_type: String = conn
            .query_row(
                "SELECT event_type FROM action_events LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(event_type, "llm_response");
    }

    #[test]
    fn observer_records_tool_call() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let obs = TelemetryObserver::new(store.clone(), "test-sess".into());

        obs.record_event(&ObserverEvent::ToolCall {
            tool: "shell".into(),
            duration: Duration::from_millis(50),
            success: true,
            arguments_hash: Some("abc123".into()),
            iteration: Some(0),
        });

        std::thread::sleep(Duration::from_millis(300));
        drop(obs);
        drop(store);

        let conn = rusqlite::Connection::open(tmp.path().join("research.db")).unwrap();
        let tool_name: String = conn
            .query_row(
                "SELECT tool_name FROM action_events LIMIT 1",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tool_name, "shell");
    }

    #[test]
    fn turn_complete_resets_counters() {
        let tmp = TempDir::new().unwrap();
        let store = make_store(&tmp);
        let obs = TelemetryObserver::new(store, "test-sess".into());

        obs.record_event(&ObserverEvent::LlmResponse {
            provider: "test".into(),
            model: "test".into(),
            duration: Duration::from_millis(1),
            success: true,
            error_message: None,
            tokens_in: None,
            tokens_out: None,
        });

        assert_eq!(obs.sequence_counter.load(Ordering::Relaxed), 1);
        assert_eq!(obs.turn_counter.load(Ordering::Relaxed), 0);

        obs.record_event(&ObserverEvent::TurnComplete);

        assert_eq!(obs.sequence_counter.load(Ordering::Relaxed), 0);
        assert_eq!(obs.turn_counter.load(Ordering::Relaxed), 1);
        assert!(obs.turn_action_sequence.lock().is_empty());
    }
}
