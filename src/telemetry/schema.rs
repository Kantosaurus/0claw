// DDL constants for the research telemetry database.

pub const ACTION_EVENTS_DDL: &str = "\
CREATE TABLE IF NOT EXISTS action_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    ts_epoch_ms         INTEGER NOT NULL,
    session_id          TEXT    NOT NULL,
    turn_id             TEXT    NOT NULL,
    sequence_index      INTEGER NOT NULL,
    event_type          TEXT    NOT NULL,
    provider            TEXT,
    model               TEXT,
    tool_name           TEXT,
    tool_type_embedding BLOB,
    arguments_hash      TEXT,
    tool_success        INTEGER,
    duration_ms         INTEGER,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    is_user_initiated   INTEGER NOT NULL,
    iteration_index     INTEGER NOT NULL,
    previous_action_type TEXT,
    turn_action_sequence TEXT,
    error_message       TEXT
);
CREATE INDEX IF NOT EXISTS idx_ae_session ON action_events(session_id);
CREATE INDEX IF NOT EXISTS idx_ae_turn    ON action_events(turn_id);
CREATE INDEX IF NOT EXISTS idx_ae_epoch   ON action_events(ts_epoch_ms);
";

pub const SYSTEM_SAMPLES_DDL: &str = "\
CREATE TABLE IF NOT EXISTS system_samples (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,
    ts_epoch_ms         INTEGER NOT NULL,
    cpu_usage_pct       REAL    NOT NULL,
    memory_used_bytes   INTEGER NOT NULL,
    memory_total_bytes  INTEGER NOT NULL,
    process_count       INTEGER NOT NULL,
    process_spawn_rate  INTEGER NOT NULL,
    file_read_bytes     INTEGER NOT NULL,
    file_write_bytes    INTEGER NOT NULL,
    net_connections     INTEGER NOT NULL,
    dest_ip_entropy     REAL    NOT NULL,
    syscall_freq_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_ss_epoch ON system_samples(ts_epoch_ms);
";

pub const TOOL_EMBEDDINGS_CACHE_DDL: &str = "\
CREATE TABLE IF NOT EXISTS tool_embeddings_cache (
    tool_name   TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    dimensions  INTEGER NOT NULL,
    computed_at TEXT NOT NULL
);
";

pub const PRAGMAS: &str = "\
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA mmap_size    = 4194304;
PRAGMA cache_size   = -1000;
PRAGMA temp_store   = MEMORY;
";

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    #[test]
    fn ddl_executes_on_in_memory_db() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(PRAGMAS).unwrap();
        conn.execute_batch(ACTION_EVENTS_DDL).unwrap();
        conn.execute_batch(SYSTEM_SAMPLES_DDL).unwrap();
        conn.execute_batch(TOOL_EMBEDDINGS_CACHE_DDL).unwrap();
    }

    #[test]
    fn ddl_is_idempotent() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(PRAGMAS).unwrap();
        // Execute twice to verify IF NOT EXISTS
        conn.execute_batch(ACTION_EVENTS_DDL).unwrap();
        conn.execute_batch(ACTION_EVENTS_DDL).unwrap();
        conn.execute_batch(SYSTEM_SAMPLES_DDL).unwrap();
        conn.execute_batch(SYSTEM_SAMPLES_DDL).unwrap();
    }
}
