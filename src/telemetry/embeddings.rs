use sha2::Digest;

/// Compute a deterministic 256-bit embedding for a tool name using SHA-256.
///
/// Returns the raw hash bytes (32 bytes = 32 dimensions) and the dimension count.
/// This is a hash-based embedding — no external API call required.
pub fn compute_tool_embedding(tool_name: &str) -> (Vec<u8>, usize) {
    let hash = sha2::Sha256::digest(tool_name.as_bytes());
    (hash.to_vec(), 32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_is_deterministic() {
        let (a, dim_a) = compute_tool_embedding("shell");
        let (b, dim_b) = compute_tool_embedding("shell");
        assert_eq!(a, b);
        assert_eq!(dim_a, dim_b);
        assert_eq!(dim_a, 32);
    }

    #[test]
    fn different_tools_produce_different_embeddings() {
        let (a, _) = compute_tool_embedding("shell");
        let (b, _) = compute_tool_embedding("file_read");
        assert_ne!(a, b);
    }

    #[test]
    fn embedding_has_correct_length() {
        let (bytes, dim) = compute_tool_embedding("memory_store");
        assert_eq!(bytes.len(), 32);
        assert_eq!(dim, 32);
    }
}
