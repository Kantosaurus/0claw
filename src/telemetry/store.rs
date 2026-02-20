use crate::telemetry::schema;
use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::thread;
use std::time::Duration;

/// A single action event record ready for insertion.
#[derive(Debug, Clone)]
pub struct ActionRecord {
    pub ts: String,
    pub ts_epoch_ms: i64,
    pub session_id: String,
    pub turn_id: String,
    pub sequence_index: i64,
    pub event_type: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub tool_name: Option<String>,
    pub tool_type_embedding: Option<Vec<u8>>,
    pub arguments_hash: Option<String>,
    pub tool_success: Option<bool>,
    pub duration_ms: Option<i64>,
    pub tokens_in: Option<i64>,
    pub tokens_out: Option<i64>,
    pub is_user_initiated: bool,
    pub iteration_index: i64,
    pub previous_action_type: Option<String>,
    pub turn_action_sequence: Option<String>,
    pub error_message: Option<String>,
}

/// A single system metrics sample ready for insertion.
#[derive(Debug, Clone)]
pub struct SystemSample {
    pub ts: String,
    pub ts_epoch_ms: i64,
    pub cpu_usage_pct: f64,
    pub memory_used_bytes: i64,
    pub memory_total_bytes: i64,
    pub process_count: i64,
    pub process_spawn_rate: i64,
    pub file_read_bytes: i64,
    pub file_write_bytes: i64,
    pub net_connections: i64,
    pub dest_ip_entropy: f64,
    pub syscall_freq_json: Option<String>,
}

/// Operations the writer thread can perform.
pub enum WriteOp {
    ActionEvent(Box<ActionRecord>),
    SystemSample(SystemSample),
    Shutdown,
}

/// Persistent telemetry store backed by a dedicated SQLite writer thread.
pub struct TelemetrySqliteStore {
    sender: Option<SyncSender<WriteOp>>,
    join_handle: Option<thread::JoinHandle<()>>,
    db_path: PathBuf,
}

impl TelemetrySqliteStore {
    /// Open (or create) the telemetry database at `db_dir/research.db`.
    pub fn open(db_dir: &Path, buffer_capacity: usize) -> Result<Self> {
        std::fs::create_dir_all(db_dir)
            .with_context(|| format!("creating telemetry dir: {}", db_dir.display()))?;

        let db_path = db_dir.join("research.db");
        let conn = Connection::open(&db_path)
            .with_context(|| format!("opening telemetry db: {}", db_path.display()))?;

        conn.execute_batch(schema::PRAGMAS)
            .context("telemetry PRAGMA setup")?;
        conn.execute_batch(schema::ACTION_EVENTS_DDL)
            .context("action_events DDL")?;
        conn.execute_batch(schema::SYSTEM_SAMPLES_DDL)
            .context("system_samples DDL")?;
        conn.execute_batch(schema::TOOL_EMBEDDINGS_CACHE_DDL)
            .context("tool_embeddings_cache DDL")?;

        let (tx, rx) = mpsc::sync_channel::<WriteOp>(buffer_capacity);

        let handle = thread::Builder::new()
            .name("telemetry-writer".into())
            .spawn(move || writer_loop(conn, rx))
            .context("spawning telemetry writer thread")?;

        Ok(Self {
            sender: Some(tx),
            join_handle: Some(handle),
            db_path: db_path.clone(),
        })
    }

    /// Non-blocking submit of an action event. Drops with a warning if the
    /// channel is full.
    pub fn submit_action(&self, record: ActionRecord) {
        if let Some(ref sender) = self.sender {
            if let Err(TrySendError::Full(_)) = sender.try_send(WriteOp::ActionEvent(Box::new(record))) {
                tracing::warn!("telemetry action channel full — dropping record");
            }
        }
    }

    /// Non-blocking submit of a system sample.
    pub fn submit_system_sample(&self, sample: SystemSample) {
        if let Some(ref sender) = self.sender {
            if let Err(TrySendError::Full(_)) = sender.try_send(WriteOp::SystemSample(sample)) {
                tracing::warn!("telemetry system channel full — dropping sample");
            }
        }
    }

    /// Path to the underlying database file.
    pub fn db_path(&self) -> &Path {
        &self.db_path
    }

    /// Graceful shutdown: signal the writer thread and wait for it to finish.
    pub fn shutdown(&mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.try_send(WriteOp::Shutdown);
            // Drop the sender so the writer thread sees a disconnect even if
            // the Shutdown message could not be delivered (channel full).
            drop(sender);
        }
        if let Some(handle) = self.join_handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for TelemetrySqliteStore {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Writer thread main loop: batches writes in transactions.
fn writer_loop(conn: Connection, rx: mpsc::Receiver<WriteOp>) {
    let mut batch: Vec<WriteOp> = Vec::with_capacity(10);

    loop {
        // Block on the first message.
        match rx.recv() {
            Ok(WriteOp::Shutdown) | Err(_) => break,
            Ok(op) => batch.push(op),
        }

        // Drain up to 9 more without blocking (batch of 10 max).
        while batch.len() < 10 {
            match rx.try_recv() {
                Ok(WriteOp::Shutdown) | Err(mpsc::TryRecvError::Disconnected) => {
                    flush_batch(&conn, &batch);
                    return;
                }
                Ok(op) => batch.push(op),
                Err(mpsc::TryRecvError::Empty) => break,
            }
        }

        // If we only got one message, wait briefly for more (up to 1s).
        if batch.len() == 1 {
            let deadline = Duration::from_secs(1);
            match rx.recv_timeout(deadline) {
                Ok(WriteOp::Shutdown) => {
                    flush_batch(&conn, &batch);
                    return;
                }
                Ok(op) => batch.push(op),
                Err(_) => {} // timeout or disconnect — flush what we have
            }
        }

        flush_batch(&conn, &batch);
        batch.clear();
    }
}

fn flush_batch(conn: &Connection, batch: &[WriteOp]) {
    if batch.is_empty() {
        return;
    }
    if let Err(e) = conn.execute_batch("BEGIN") {
        tracing::error!("telemetry BEGIN failed: {e}");
        return;
    }
    for op in batch {
        let result = match op {
            WriteOp::ActionEvent(rec) => insert_action(conn, rec.as_ref()),
            WriteOp::SystemSample(sample) => insert_system_sample(conn, sample),
            WriteOp::Shutdown => Ok(()),
        };
        if let Err(e) = result {
            tracing::error!("telemetry insert failed: {e}");
        }
    }
    if let Err(e) = conn.execute_batch("COMMIT") {
        tracing::error!("telemetry COMMIT failed: {e}");
    }
}

fn insert_action(conn: &Connection, r: &ActionRecord) -> Result<()> {
    conn.execute(
        "INSERT INTO action_events (
            ts, ts_epoch_ms, session_id, turn_id, sequence_index, event_type,
            provider, model, tool_name, tool_type_embedding, arguments_hash,
            tool_success, duration_ms, tokens_in, tokens_out,
            is_user_initiated, iteration_index, previous_action_type,
            turn_action_sequence, error_message
        ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20)",
        rusqlite::params![
            r.ts,
            r.ts_epoch_ms,
            r.session_id,
            r.turn_id,
            r.sequence_index,
            r.event_type,
            r.provider,
            r.model,
            r.tool_name,
            r.tool_type_embedding,
            r.arguments_hash,
            r.tool_success.map(|b| if b { 1 } else { 0 }),
            r.duration_ms,
            r.tokens_in,
            r.tokens_out,
            i32::from(r.is_user_initiated),
            r.iteration_index,
            r.previous_action_type,
            r.turn_action_sequence,
            r.error_message,
        ],
    )?;
    Ok(())
}

fn insert_system_sample(conn: &Connection, s: &SystemSample) -> Result<()> {
    conn.execute(
        "INSERT INTO system_samples (
            ts, ts_epoch_ms, cpu_usage_pct, memory_used_bytes, memory_total_bytes,
            process_count, process_spawn_rate, file_read_bytes, file_write_bytes,
            net_connections, dest_ip_entropy, syscall_freq_json
        ) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12)",
        rusqlite::params![
            s.ts,
            s.ts_epoch_ms,
            s.cpu_usage_pct,
            s.memory_used_bytes,
            s.memory_total_bytes,
            s.process_count,
            s.process_spawn_rate,
            s.file_read_bytes,
            s.file_write_bytes,
            s.net_connections,
            s.dest_ip_entropy,
            s.syscall_freq_json,
        ],
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_action_record() -> ActionRecord {
        ActionRecord {
            ts: "2026-01-01T00:00:00Z".into(),
            ts_epoch_ms: 1_767_225_600_000,
            session_id: "sess-1".into(),
            turn_id: "turn-1".into(),
            sequence_index: 0,
            event_type: "llm_response".into(),
            provider: Some("openai".into()),
            model: Some("gpt-4".into()),
            tool_name: None,
            tool_type_embedding: None,
            arguments_hash: None,
            tool_success: None,
            duration_ms: Some(150),
            tokens_in: Some(100),
            tokens_out: Some(50),
            is_user_initiated: true,
            iteration_index: 0,
            previous_action_type: None,
            turn_action_sequence: Some(r#"["llm_response"]"#.into()),
            error_message: None,
        }
    }

    #[test]
    fn store_open_and_insert_action() {
        let tmp = TempDir::new().unwrap();
        let store = TelemetrySqliteStore::open(tmp.path(), 10).unwrap();
        store.submit_action(make_action_record());
        // Give the writer thread time to process.
        std::thread::sleep(Duration::from_millis(200));
        drop(store);

        // Verify the record was inserted.
        let conn = Connection::open(tmp.path().join("research.db")).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM action_events", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn store_open_and_insert_system_sample() {
        let tmp = TempDir::new().unwrap();
        let store = TelemetrySqliteStore::open(tmp.path(), 10).unwrap();
        store.submit_system_sample(SystemSample {
            ts: "2026-01-01T00:00:01Z".into(),
            ts_epoch_ms: 1_767_225_601_000,
            cpu_usage_pct: 23.5,
            memory_used_bytes: 1_000_000,
            memory_total_bytes: 8_000_000,
            process_count: 120,
            process_spawn_rate: 2,
            file_read_bytes: 4096,
            file_write_bytes: 2048,
            net_connections: 15,
            dest_ip_entropy: 2.3,
            syscall_freq_json: None,
        });
        std::thread::sleep(Duration::from_millis(200));
        drop(store);

        let conn = Connection::open(tmp.path().join("research.db")).unwrap();
        let count: i64 = conn
            .query_row("SELECT COUNT(*) FROM system_samples", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn store_backpressure_does_not_panic() {
        let tmp = TempDir::new().unwrap();
        let store = TelemetrySqliteStore::open(tmp.path(), 2).unwrap();
        // Submit more than the capacity — should not panic, just drop.
        for _ in 0..20 {
            store.submit_action(make_action_record());
        }
        drop(store);
    }
}
