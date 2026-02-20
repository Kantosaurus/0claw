use crate::config::TelemetryConfig;
use crate::telemetry::store::{SystemSample, TelemetrySqliteStore};
use std::sync::Arc;

/// Run the system metrics collector as a background tokio task.
///
/// Samples CPU, memory, process count, file I/O, and network connection
/// metrics at the configured interval and submits them to the telemetry store.
pub async fn run_system_collector(store: Arc<TelemetrySqliteStore>, config: TelemetryConfig) {
    use sysinfo::System;

    let interval = std::time::Duration::from_secs(config.system_interval_secs.max(1));
    let mut sys = System::new();

    // Initial refresh to get a baseline for CPU (first reading is always 0).
    sys.refresh_all();
    let mut prev_process_count: i64 = sys.processes().len() as i64;

    #[cfg(target_os = "linux")]
    let mut prev_io = read_proc_self_io();

    loop {
        tokio::time::sleep(interval).await;

        sys.refresh_all();

        let cpu_usage_pct = f64::from(sys.global_cpu_usage());
        let memory_used_bytes = sys.used_memory() as i64;
        let memory_total_bytes = sys.total_memory() as i64;
        let process_count = sys.processes().len() as i64;
        let process_spawn_rate = (process_count - prev_process_count).max(0);
        prev_process_count = process_count;

        // File I/O from /proc/self/io (Linux only)
        #[cfg(target_os = "linux")]
        let (file_read_bytes, file_write_bytes) = {
            let current_io = read_proc_self_io();
            let read_delta = (current_io.0 - prev_io.0).max(0);
            let write_delta = (current_io.1 - prev_io.1).max(0);
            prev_io = current_io;
            (read_delta, write_delta)
        };
        #[cfg(not(target_os = "linux"))]
        let (file_read_bytes, file_write_bytes) = (0i64, 0i64);

        // Network connections + dest IP entropy (Linux only)
        #[cfg(target_os = "linux")]
        let (net_connections, dest_ip_entropy) = read_net_connections();
        #[cfg(not(target_os = "linux"))]
        let (net_connections, dest_ip_entropy) = (0i64, 0.0f64);

        // eBPF syscall frequency (when available)
        let syscall_freq_json = super::ebpf::try_read_syscall_freq();

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let ts_epoch_ms = i64::try_from(now.as_millis()).unwrap_or(i64::MAX);
        let ts = chrono::Utc::now().to_rfc3339();

        store.submit_system_sample(SystemSample {
            ts,
            ts_epoch_ms,
            cpu_usage_pct,
            memory_used_bytes,
            memory_total_bytes,
            process_count,
            process_spawn_rate,
            file_read_bytes,
            file_write_bytes,
            net_connections,
            dest_ip_entropy,
            syscall_freq_json,
        });
    }
}

/// Read /proc/self/io and return (read_bytes, write_bytes).
#[cfg(target_os = "linux")]
fn read_proc_self_io() -> (i64, i64) {
    let content = match std::fs::read_to_string("/proc/self/io") {
        Ok(c) => c,
        Err(_) => return (0, 0),
    };
    let mut read_bytes: i64 = 0;
    let mut write_bytes: i64 = 0;
    for line in content.lines() {
        if let Some(val) = line.strip_prefix("read_bytes: ") {
            read_bytes = val.trim().parse().unwrap_or(0);
        } else if let Some(val) = line.strip_prefix("write_bytes: ") {
            write_bytes = val.trim().parse().unwrap_or(0);
        }
    }
    (read_bytes, write_bytes)
}

/// Read /proc/net/tcp + /proc/net/tcp6 to count connections and compute
/// Shannon entropy of destination IP addresses.
#[cfg(target_os = "linux")]
fn read_net_connections() -> (i64, f64) {
    let mut dest_ips: Vec<String> = Vec::new();

    for path in &["/proc/net/tcp", "/proc/net/tcp6"] {
        if let Ok(content) = std::fs::read_to_string(path) {
            for line in content.lines().skip(1) {
                // Fields: sl local_address rem_address st ...
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 3 {
                    // rem_address is like "0100007F:1F90" (hex IP:port)
                    if let Some(ip_hex) = parts[2].split(':').next() {
                        dest_ips.push(ip_hex.to_string());
                    }
                }
            }
        }
    }

    let net_connections = dest_ips.len() as i64;
    let entropy = shannon_entropy(&dest_ips);
    (net_connections, entropy)
}

/// Compute Shannon entropy of a set of string values.
#[cfg(target_os = "linux")]
fn shannon_entropy(values: &[String]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for v in values {
        *counts.entry(v.as_str()).or_insert(0) += 1;
    }
    let total = values.len() as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}
