use anyhow::{Context, Result};
use rusqlite::Connection;
use std::path::Path;

/// A read-only view of the telemetry database for export/download.
///
/// Opens a separate read-only SQLite connection so that concurrent reads
/// don't interfere with the writer thread (WAL mode allows this).
pub struct TelemetryReader {
    conn: Connection,
}

/// Action event record for serialization in the download endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ActionEventRow {
    pub ts: String,
    pub ts_epoch_ms: i64,
    pub session_id: String,
    pub turn_id: String,
    pub sequence_index: i64,
    pub event_type: String,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub tool_name: Option<String>,
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

/// System sample record for serialization in the download endpoint.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SystemSampleRow {
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

impl TelemetryReader {
    /// Open a read-only connection to the telemetry database.
    pub fn open(db_path: &Path) -> Result<Self> {
        let conn = Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .with_context(|| format!("opening telemetry db read-only: {}", db_path.display()))?;
        Ok(Self { conn })
    }

    /// Export action events, optionally filtered by timestamp.
    pub fn export_action_events(
        &self,
        since_epoch_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<ActionEventRow>> {
        let since = since_epoch_ms.unwrap_or(0);
        let mut stmt = self.conn.prepare(
            "SELECT ts, ts_epoch_ms, session_id, turn_id, sequence_index, event_type,
                    provider, model, tool_name, arguments_hash, tool_success,
                    duration_ms, tokens_in, tokens_out, is_user_initiated,
                    iteration_index, previous_action_type, turn_action_sequence,
                    error_message
             FROM action_events
             WHERE ts_epoch_ms >= ?1
             ORDER BY ts_epoch_ms ASC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![since, limit as i64], |row| {
            Ok(ActionEventRow {
                ts: row.get(0)?,
                ts_epoch_ms: row.get(1)?,
                session_id: row.get(2)?,
                turn_id: row.get(3)?,
                sequence_index: row.get(4)?,
                event_type: row.get(5)?,
                provider: row.get(6)?,
                model: row.get(7)?,
                tool_name: row.get(8)?,
                arguments_hash: row.get(9)?,
                tool_success: row.get::<_, Option<i32>>(10)?.map(|v| v != 0),
                duration_ms: row.get(11)?,
                tokens_in: row.get(12)?,
                tokens_out: row.get(13)?,
                is_user_initiated: row.get::<_, i32>(14)? != 0,
                iteration_index: row.get(15)?,
                previous_action_type: row.get(16)?,
                turn_action_sequence: row.get(17)?,
                error_message: row.get(18)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Export system samples, optionally filtered by timestamp.
    pub fn export_system_samples(
        &self,
        since_epoch_ms: Option<i64>,
        limit: usize,
    ) -> Result<Vec<SystemSampleRow>> {
        let since = since_epoch_ms.unwrap_or(0);
        let mut stmt = self.conn.prepare(
            "SELECT ts, ts_epoch_ms, cpu_usage_pct, memory_used_bytes, memory_total_bytes,
                    process_count, process_spawn_rate, file_read_bytes, file_write_bytes,
                    net_connections, dest_ip_entropy, syscall_freq_json
             FROM system_samples
             WHERE ts_epoch_ms >= ?1
             ORDER BY ts_epoch_ms ASC
             LIMIT ?2",
        )?;

        let rows = stmt.query_map(rusqlite::params![since, limit as i64], |row| {
            Ok(SystemSampleRow {
                ts: row.get(0)?,
                ts_epoch_ms: row.get(1)?,
                cpu_usage_pct: row.get(2)?,
                memory_used_bytes: row.get(3)?,
                memory_total_bytes: row.get(4)?,
                process_count: row.get(5)?,
                process_spawn_rate: row.get(6)?,
                file_read_bytes: row.get(7)?,
                file_write_bytes: row.get(8)?,
                net_connections: row.get(9)?,
                dest_ip_entropy: row.get(10)?,
                syscall_freq_json: row.get(11)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::store::{ActionRecord, TelemetrySqliteStore};
    use tempfile::TempDir;

    #[test]
    fn reader_exports_action_events() {
        let tmp = TempDir::new().unwrap();
        let store = TelemetrySqliteStore::open(tmp.path(), 64).unwrap();
        store.submit_action(ActionRecord {
            ts: "2026-01-01T00:00:00Z".into(),
            ts_epoch_ms: 1_000,
            session_id: "s1".into(),
            turn_id: "t1".into(),
            sequence_index: 0,
            event_type: "llm_response".into(),
            provider: Some("openai".into()),
            model: Some("gpt-4".into()),
            tool_name: None,
            tool_type_embedding: None,
            arguments_hash: None,
            tool_success: None,
            duration_ms: Some(100),
            tokens_in: Some(50),
            tokens_out: Some(25),
            is_user_initiated: true,
            iteration_index: 0,
            previous_action_type: None,
            turn_action_sequence: None,
            error_message: None,
        });
        // Let writer flush
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(store);

        let reader = TelemetryReader::open(&tmp.path().join("research.db")).unwrap();
        let events = reader.export_action_events(None, 100).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, "llm_response");
        assert_eq!(events[0].tokens_in, Some(50));
    }

    #[test]
    fn reader_filters_by_since() {
        let tmp = TempDir::new().unwrap();
        let store = TelemetrySqliteStore::open(tmp.path(), 64).unwrap();
        for i in 0..3 {
            store.submit_action(ActionRecord {
                ts: format!("2026-01-01T00:00:0{i}Z"),
                ts_epoch_ms: (i + 1) * 1000,
                session_id: "s1".into(),
                turn_id: "t1".into(),
                sequence_index: i,
                event_type: "tool_call".into(),
                provider: None,
                model: None,
                tool_name: Some("shell".into()),
                tool_type_embedding: None,
                arguments_hash: None,
                tool_success: Some(true),
                duration_ms: Some(10),
                tokens_in: None,
                tokens_out: None,
                is_user_initiated: false,
                iteration_index: 0,
                previous_action_type: None,
                turn_action_sequence: None,
                error_message: None,
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(300));
        drop(store);

        let reader = TelemetryReader::open(&tmp.path().join("research.db")).unwrap();
        let events = reader.export_action_events(Some(2000), 100).unwrap();
        assert_eq!(events.len(), 2); // ts_epoch_ms 2000 and 3000
    }
}
