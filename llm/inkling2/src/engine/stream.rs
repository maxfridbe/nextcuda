//! One card for a model that does not fit it (`NS_INK2_PLACE=one`): the routed experts of some layers outside the loop
//! stay in pinned host memory in the silo's packed layout, and the experts a pass chooses are copied into a pool of
//! slots on the card - an LRU cache a matrix shape (gate | up, down; NVFP4, Q8_0), sized to the memory the plan leaves.
//! A pass's ids are read back at each such layer application (the stream is synced to its router then), the missing
//! experts copied over PCIe on the copy lane, the stream ordered after them, and the products run on the pool with the
//! ids remapped to slots - the same kernels, the same arithmetic (a slot holds an expert's bytes as they are).

use std::collections::HashMap;

use super::*;

/// An expert stack kept in host memory: silo.h's packed layout of E experts (every expert's quants, then from
/// `scales_at` every expert's scales)
pub struct Streamed {
    pub host: HostBuf,
    pub qbytes: usize,
    pub sbytes: usize,
    pub scales_at: usize,
}

/// The slots of one matrix shape on the card
pub struct Pool {
    pub ty: GType,
    pub rows: usize,
    pub cols: usize,
    pub slots: usize,
    buf: DevBuf,
    scale: DevBuf,
    scale_h: Vec<f32>,
    qbytes: usize,
    sbytes: usize,
    scales_at: usize,
    /// (stack, expert) in a slot; when it was last asked for
    owner: Vec<Option<(usize, usize)>>,
    used: Vec<u64>,
    map: HashMap<(usize, usize), usize>,
    clock: u64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_in: u64,
}

/// An expert's bytes in silo.h's packed layout: (its quants, its scales)
pub fn expert_bytes(ty: GType, rows: usize, cols: usize) -> (usize, usize) {
    match ty {
        GType::NVFP4 => (rows * cols / 2, rows * cols / 16),
        _ => (rows * cols, rows * (cols / 32) * 2),
    }
}

impl Pool {
    pub fn new(gpu: &Arc<Gpu>, s: &ffi::Silo, ty: GType, rows: usize, cols: usize, slots: usize) -> Result<Pool> {
        let (qbytes, sbytes) = expert_bytes(ty, rows, cols);
        let code = ty.code() as c_int;
        // SAFETY: arithmetic only.
        let (total, scales_at) = unsafe { ((s.packed_bytes)(code, (slots * rows) as i64, cols as i64), (s.scales_at)(code, (slots * rows) as i64, cols as i64)) };
        Ok(Pool { ty, rows, cols, slots, buf: DevBuf::new(gpu, total)?, scale: DevBuf::from_f32(gpu, &vec![1.0; slots])?, scale_h: vec![1.0; slots], qbytes, sbytes,
                  scales_at, owner: vec![None; slots], used: vec![0; slots], map: HashMap::new(), clock: 0, hits: 0, misses: 0, bytes_in: 0 })
    }

    /// The bytes of `slots` slots of a shape
    pub fn bytes(ty: GType, rows: usize, cols: usize, slots: usize) -> usize {
        let (q, s) = expert_bytes(ty, rows, cols);
        slots * (q + s) + 512 + slots * 4
    }

    pub fn buf(&self) -> &DevBuf {
        &self.buf
    }
    pub fn scale(&self) -> &DevBuf {
        &self.scale
    }

    /// Experts `want` of stack `stack` (its host copy `st`, its scales `scales`) in slots: each one's slot, the missing
    /// ones copied in on `gpu`'s copy lane with the stream ordered after them. The stream must be past every kernel
    /// that read the slots this may take (the caller read the pass's ids back).
    pub fn ensure(&mut self, gpu: &Arc<Gpu>, stack: usize, st: &Streamed, scales: &[f32], want: &[usize]) -> Result<Vec<usize>> {
        self.clock += 1;
        let now = self.clock;
        let mut distinct: Vec<usize> = want.to_vec();
        distinct.sort_unstable();
        distinct.dedup();
        if distinct.len() > self.slots {
            return Err(e(format!("inkling2: {} experts asked of a pool of {} slots", distinct.len(), self.slots)));
        }
        // the ones there first (kept from eviction this round)
        let mut missing = Vec::new();
        for &x in &distinct {
            match self.map.get(&(stack, x)) {
                Some(&sl) => {
                    self.used[sl] = now;
                    self.hits += 1;
                }
                None => missing.push(x),
            }
        }
        let mut last: Option<i64> = None;
        let mut scales_changed = false;
        if !missing.is_empty() {
            let after: Vec<i64> = gpu.ticket()?.into_iter().collect();
            for &x in &missing {
                // a free slot, else the least recently asked for (none of this round's)
                let sl = match self.owner.iter().position(|o| o.is_none()) {
                    Some(i) => i,
                    None => (0..self.slots).filter(|i| self.used[*i] < now).min_by_key(|i| self.used[*i]).ok_or_else(|| e("inkling2: no slot to evict"))?,
                };
                if let Some(old) = self.owner[sl].take() {
                    self.map.remove(&old);
                }
                // (the copy lane runs in order: the last copy's ticket covers them all)
                self.buf.copy_from_host_async(sl * self.qbytes, &st.host, x * st.qbytes, self.qbytes, &after)?;
                last = Some(self.buf.copy_from_host_async(self.scales_at + sl * self.sbytes, &st.host, st.scales_at + x * st.sbytes, self.sbytes, &after)?);
                self.owner[sl] = Some((stack, x));
                self.map.insert((stack, x), sl);
                self.used[sl] = now;
                if self.scale_h[sl] != scales[x] {
                    self.scale_h[sl] = scales[x];
                    scales_changed = true;
                }
                self.misses += 1;
                self.bytes_in += (self.qbytes + self.sbytes) as u64;
            }
        }
        if scales_changed {
            // (the stream is synced up to this layer's router: no kernel in flight reads these)
            self.scale.write(0, &self.scale_h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>())?;
        }
        if let Some(t) = last {
            gpu.await_ticket(t)?;
        }
        Ok(want.iter().map(|x| self.map[&(stack, *x)]).collect())
    }
}

/// A stack's raw GGUF blocks repacked on the host into silo.h's layout (`pack_k` in silo/mv.cu: Q8_0 quants in the
/// 0 8 1 9 ... order within each 16, then the half scales; NVFP4 codes as they are, then the UE4M3 bytes) - no copy of
/// the stack on the card while the rest of the model fills it
fn pack_host(ty: GType, raw: &[u8], rows: usize, n: usize, scales_at: usize, dst: &mut [u8]) {
    let lines = n / 32;
    let qpos = |v: usize| (v & 16) + if (v & 15) < 8 { 2 * (v & 15) } else { 2 * ((v & 15) - 8) + 1 };
    for r in 0..rows {
        for l in 0..lines {
            if ty == GType::Q8_0 {
                let b = &raw[(r * lines + l) * 34..(r * lines + l + 1) * 34];
                for v in 0..32 {
                    dst[r * n + l * 32 + qpos(v)] = b[2 + v];
                }
                let at = scales_at + (r * lines + l) * 2;
                dst[at] = b[0];
                dst[at + 1] = b[1];
            } else {
                let blk = r * (n / 64) + l / 2;
                let b = &raw[blk * 36..(blk + 1) * 36];
                let h = l & 1;
                dst[r * (n / 2) + l * 16..r * (n / 2) + l * 16 + 16].copy_from_slice(&b[4 + h * 16..4 + h * 16 + 16]);
                let at = scales_at + r * (n / 16) + l * 2;
                dst[at] = b[h * 2];
                dst[at + 1] = b[h * 2 + 1];
            }
        }
    }
}

/// A stack's experts read and packed into pinned host memory (never on the card whole)
pub fn load_streamed(f: &Gguf, gpu: &Arc<Gpu>, s: &ffi::Silo, name: &str, experts: usize, loaded: &std::cell::Cell<u64>) -> Result<(Streamed, GType, usize, usize)> {
    let t = f.tensor(name).ok_or_else(|| e(format!("no {name}")))?;
    if t.shape.len() != 3 || t.shape[0] as usize != experts {
        return Err(e(format!("{name}: {:?}, not [{experts} experts, rows, cols]", t.shape)));
    }
    let (rows, cols) = (t.shape[1] as usize, t.shape[2] as usize);
    let code = t.ty.code() as c_int;
    // SAFETY: arithmetic only.
    if unsafe { (s.supported)(code, cols as i64) } == 0 || !matches!(t.ty, GType::Q8_0 | GType::NVFP4) {
        return Err(e(format!("{name}: {} experts the silo does not take (streamed experts are repacked)", t.ty.name())));
    }
    let raw = f.read(t).map_err(|x| e(x.0))?;
    loaded.set(loaded.get() + raw.len() as u64);
    let all = experts * rows;
    // SAFETY: arithmetic only.
    let (bytes, scales_at) = unsafe { ((s.packed_bytes)(code, all as i64, cols as i64), (s.scales_at)(code, all as i64, cols as i64)) };
    let mut host = HostBuf::new(gpu, bytes)?;
    pack_host(t.ty, &raw, all, cols, scales_at, host.as_mut_slice());
    let (qbytes, sbytes) = expert_bytes(t.ty, rows, cols);
    Ok((Streamed { host, qbytes, sbytes, scales_at }, t.ty, rows, cols))
}
