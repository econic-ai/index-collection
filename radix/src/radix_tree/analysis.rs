use super::{
    BUCKET_SLOTS, CHUNKS_PER_GROUP, CHUNK_WIDTH, GROUP_SLOTS, GROUPS_PER_BUCKET, RADIX_FP_EMPTY,
    RadixTree,
};
use crate::mix64;

#[derive(Debug)]
pub struct OccupancyStats {
    /// Histogram: how many groups have exactly N slots occupied (index 0..=64).
    pub group_occupancy: [usize; GROUP_SLOTS + 1],
    /// Histogram: how many chunks have exactly N slots occupied (index 0..=16).
    pub chunk_occupancy: [usize; CHUNK_WIDTH + 1],
    pub full_groups: usize,
    pub full_chunks: usize,
    pub full_buckets: usize,
    pub total_groups: usize,
    pub total_chunks: usize,
    pub total_buckets: usize,
}

impl Default for OccupancyStats {
    fn default() -> Self {
        Self {
            group_occupancy: [0; GROUP_SLOTS + 1],
            chunk_occupancy: [0; CHUNK_WIDTH + 1],
            full_groups: 0,
            full_chunks: 0,
            full_buckets: 0,
            total_groups: 0,
            total_chunks: 0,
            total_buckets: 0,
        }
    }
}

#[derive(Debug, Default)]
pub struct ProbeTrace {
    pub total_lookups: usize,
    pub expect_hit: bool,
    /// Resolved by scalar check at preferred_offset (match or empty).
    pub scalar_hit: usize,
    pub scalar_empty: usize,
    /// Resolved in the first chunk (preferred chunk) — scalar + NEON combined.
    pub resolved_chunk_0_scalar: usize,
    pub resolved_chunk_0_neon: usize,
    /// Resolved in chunks 2-4 of the same group.
    pub resolved_chunk_1_plus: usize,
    /// Required advancing to a different bucket.
    pub bucket_overflows: usize,
    /// Total bucket probes across all lookups.
    pub total_bucket_probes: usize,
    /// Lookups that exhausted all buckets without resolving.
    pub unresolved: usize,
}

impl RadixTree {
    /// Static occupancy analysis: scan the table and report fill distributions.
    pub fn occupancy_stats(&self) -> OccupancyStats {
        let mut stats = OccupancyStats::default();
        let bucket_count = self.layout.bucket_count;

        for b in 0..bucket_count {
            let mut bucket_occ = 0usize;
            for g in 0..GROUPS_PER_BUCKET {
                let group = &self.fingerprints[b].groups[g].0;
                let mut group_occ = 0usize;

                for c in 0..CHUNKS_PER_GROUP {
                    let chunk_start = c * CHUNK_WIDTH;
                    let chunk_occ = group[chunk_start..chunk_start + CHUNK_WIDTH]
                        .iter()
                        .filter(|&&fp| fp != RADIX_FP_EMPTY)
                        .count();
                    stats.chunk_occupancy[chunk_occ] += 1;
                    if chunk_occ == CHUNK_WIDTH {
                        stats.full_chunks += 1;
                    }
                    group_occ += chunk_occ;
                }

                stats.group_occupancy[group_occ] += 1;
                if group_occ == GROUP_SLOTS {
                    stats.full_groups += 1;
                }
                bucket_occ += group_occ;
            }
            if bucket_occ == BUCKET_SLOTS {
                stats.full_buckets += 1;
            }
        }

        stats.total_groups = bucket_count * GROUPS_PER_BUCKET;
        stats.total_chunks = stats.total_groups * CHUNKS_PER_GROUP;
        stats.total_buckets = bucket_count;
        stats
    }

    /// Dynamic probe trace: run contains on a set of keys and record where
    /// each lookup resolved. Does NOT use the optimised hot path — this is
    /// a separate instrumented walk for analysis only.
    pub fn probe_trace(&self, keys: &[u64], expect_hit: bool) -> ProbeTrace {
        let mut trace = ProbeTrace::default();

        for &id in keys {
            let h = mix64(id);
            let mut bucket = (h >> self.bucket_shift) as usize & self.bucket_mask;
            let group_slot = ((h >> self.group_slot_shift) & 0xFF) as u8;
            let group = (group_slot >> 6) as usize;
            let preferred_slot = (group_slot & 0x3F) as usize;
            let raw_fp = ((h >> self.fp_shift) & 0xFF) as u8;
            let fp = raw_fp | ((raw_fp == 0) as u8);

            let start_chunk = preferred_slot >> 4;
            let preferred_offset = preferred_slot & 0xF;

            let mut resolved = false;
            let mut buckets_probed = 0usize;

            'outer: for _ in 0..self.layout.bucket_count {
                buckets_probed += 1;
                let grp = &self.fingerprints[bucket].groups[group].0;
                let hash_base = bucket * BUCKET_SLOTS + group * GROUP_SLOTS;

                for ci in 0..CHUNKS_PER_GROUP {
                    let c = (start_chunk + ci) & (CHUNKS_PER_GROUP - 1);
                    let chunk_offset = c * CHUNK_WIDTH;
                    let pref_addr = chunk_offset + preferred_offset;

                    // Check preferred offset first
                    if grp[pref_addr] == fp {
                        if self.hashes[hash_base + pref_addr] == id {
                            trace.scalar_hit += 1;
                            if ci == 0 {
                                trace.resolved_chunk_0_scalar += 1;
                            }
                            resolved = true;
                            break 'outer;
                        }
                    } else if grp[pref_addr] == 0 {
                        trace.scalar_empty += 1;
                        if ci == 0 {
                            trace.resolved_chunk_0_scalar += 1;
                        }
                        resolved = true;
                        break 'outer;
                    }

                    // Scan rest of chunk
                    let mut found_in_chunk = false;
                    let mut has_empty = false;
                    for pos in 0..CHUNK_WIDTH {
                        if pos == preferred_offset {
                            continue;
                        }
                        let slot = chunk_offset + pos;
                        if grp[slot] == fp && self.hashes[hash_base + slot] == id {
                            match ci {
                                0 => trace.resolved_chunk_0_neon += 1,
                                _ => trace.resolved_chunk_1_plus += 1,
                            }
                            resolved = true;
                            found_in_chunk = true;
                            break;
                        }
                        if grp[slot] == 0 {
                            has_empty = true;
                        }
                    }
                    if found_in_chunk {
                        break 'outer;
                    }
                    if has_empty {
                        match ci {
                            0 => trace.resolved_chunk_0_neon += 1,
                            _ => trace.resolved_chunk_1_plus += 1,
                        }
                        resolved = true;
                        break 'outer;
                    }
                }

                bucket = (bucket + 1) & self.bucket_mask;
            }

            if !resolved {
                trace.unresolved += 1;
            }
            if buckets_probed > 1 {
                trace.bucket_overflows += 1;
            }
            trace.total_bucket_probes += buckets_probed;
            trace.total_lookups += 1;
        }

        trace.expect_hit = expect_hit;
        trace
    }
}

#[cfg(test)]
mod tests {
    use super::super::{RadixTree, RadixTreeConfig};
    use crate::IndexTable;

    #[test]
    fn occupancy_analysis() {
        let capacity_bits: u8 = 20;
        let load_factors = [0.01, 0.25, 0.50, 0.75];
        let probe_sample = 10_000usize;

        let mut report = String::new();
        report.push_str("# Radix Tree Occupancy Analysis\n\n");
        report.push_str(&format!(
            "capacity_bits={}, total_slots={}\n\n",
            capacity_bits,
            1usize << capacity_bits
        ));

        for &lf in &load_factors {
            let mut t = RadixTree::new(RadixTreeConfig { capacity_bits }).expect("build");
            let n = ((t.capacity() as f64) * lf) as usize;

            let hit_keys: Vec<u64> = (0..n)
                .map(|i| (i as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
                .collect();
            for &k in &hit_keys {
                t.insert(k);
            }

            let miss_keys: Vec<u64> = (0..probe_sample)
                .map(|i| (i as u64).wrapping_mul(0xdead_beef_cafe_babe).wrapping_add(0xffff_ffff))
                .collect();

            // Static occupancy
            let occ = t.occupancy_stats();

            report.push_str(&format!("## Load Factor {:.0}% ({} / {})\n\n", lf * 100.0, n, t.capacity()));

            report.push_str("### Chunk Occupancy (16 slots each)\n\n");
            report.push_str("| Slots filled | Count | % of chunks |\n");
            report.push_str("|-------------|-------|-------------|\n");
            for i in 0..=16 {
                if occ.chunk_occupancy[i] > 0 {
                    report.push_str(&format!(
                        "| {} | {} | {:.2}% |\n",
                        i,
                        occ.chunk_occupancy[i],
                        occ.chunk_occupancy[i] as f64 / occ.total_chunks as f64 * 100.0
                    ));
                }
            }
            report.push_str(&format!(
                "\nFull chunks (16/16): {} / {} ({:.2}%)\n\n",
                occ.full_chunks,
                occ.total_chunks,
                occ.full_chunks as f64 / occ.total_chunks as f64 * 100.0
            ));

            report.push_str("### Group Occupancy (64 slots each)\n\n");
            report.push_str("| Range | Count | % of groups |\n");
            report.push_str("|-------|-------|-------------|\n");
            let ranges = [(0, 0, "0 (empty)"), (1, 16, "1-16"), (17, 32, "17-32"),
                          (33, 48, "33-48"), (49, 63, "49-63"), (64, 64, "64 (full)")];
            for &(lo, hi, label) in &ranges {
                let count: usize = (lo..=hi).map(|i| occ.group_occupancy[i]).sum();
                if count > 0 {
                    report.push_str(&format!(
                        "| {} | {} | {:.2}% |\n",
                        label,
                        count,
                        count as f64 / occ.total_groups as f64 * 100.0
                    ));
                }
            }
            report.push_str(&format!(
                "\nFull groups (64/64): {} / {} ({:.2}%)\n",
                occ.full_groups, occ.total_groups,
                occ.full_groups as f64 / occ.total_groups as f64 * 100.0
            ));
            report.push_str(&format!(
                "Full buckets (256/256): {} / {} ({:.2}%)\n\n",
                occ.full_buckets, occ.total_buckets,
                occ.full_buckets as f64 / occ.total_buckets as f64 * 100.0
            ));

            // Dynamic probe trace — hits
            let sample_hit_keys: Vec<u64> = hit_keys.iter().take(probe_sample).copied().collect();
            let hit_trace = t.probe_trace(&sample_hit_keys, true);

            // Dynamic probe trace — misses
            let miss_trace = t.probe_trace(&miss_keys, false);

            for trace in [&hit_trace, &miss_trace] {
                let kind = if trace.expect_hit { "Hits" } else { "Misses" };
                let total = trace.total_lookups as f64;
                report.push_str(&format!("### Probe Trace — {} ({} lookups)\n\n", kind, trace.total_lookups));
                report.push_str("| Resolution level | Count | % |\n");
                report.push_str("|-----------------|-------|----|\n");
                report.push_str(&format!(
                    "| Scalar preferred (match) | {} | {:.1}% |\n",
                    trace.scalar_hit, trace.scalar_hit as f64 / total * 100.0
                ));
                report.push_str(&format!(
                    "| Scalar preferred (empty) | {} | {:.1}% |\n",
                    trace.scalar_empty, trace.scalar_empty as f64 / total * 100.0
                ));
                report.push_str(&format!(
                    "| Chunk 0 — NEON remainder | {} | {:.1}% |\n",
                    trace.resolved_chunk_0_neon, trace.resolved_chunk_0_neon as f64 / total * 100.0
                ));
                report.push_str(&format!(
                    "| Chunks 1-3 (same group) | {} | {:.1}% |\n",
                    trace.resolved_chunk_1_plus, trace.resolved_chunk_1_plus as f64 / total * 100.0
                ));
                report.push_str(&format!(
                    "| Bucket overflows | {} | {:.1}% |\n",
                    trace.bucket_overflows, trace.bucket_overflows as f64 / total * 100.0
                ));
                report.push_str(&format!(
                    "| Avg buckets probed | {:.3} |\n\n",
                    trace.total_bucket_probes as f64 / total
                ));
            }
        }

        let out_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("analysis/occupancy.md");
        std::fs::write(&out_path, &report).expect("write occupancy report");
        eprintln!("Wrote occupancy report to {}", out_path.display());
    }
}
