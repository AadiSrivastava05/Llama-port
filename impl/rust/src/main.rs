// microgpt inference - Rust port.
//
// Implements the contract in BENCHMARK.md: same weights, same f64 operation order,
// same xorshift64* RNG, same JSON report as impl/python/infer.py and impl/cpp/infer.cpp.
//
// Build (the harness uses exactly this - plain --release, because Rust never
// reassociates floating point or auto-fuses into FMA):
//   cargo build --release --manifest-path impl/rust/Cargo.toml
//
// Two traps this port has to avoid, both documented in BENCHMARK.md section 3:
//   - accumulate with a plain running total, never a compensated sum
//   - never f64::mul_add in the kernels; it fuses and changes the rounding

use std::fmt::Write as _;
use std::time::Instant;

// ------------------------------------------------------------------ sha256

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());

    let mut w = [0u32; 64];
    for chunk in msg.chunks(64) {
        for i in 0..16 {
            w[i] = u32::from_be_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d) = (h[0], h[1], h[2], h[3]);
        let (mut e, mut f, mut g, mut hh) = (h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = String::with_capacity(64);
    for v in h {
        let _ = write!(out, "{v:08x}");
    }
    out
}

// --------------------------------------------------------------------- rng

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Rng {
            state: if seed != 0 { seed } else { 0x9E37_79B9_7F4A_7C15 },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

// ----------------------------------------------------------------- weights

struct Mat {
    rows: usize,
    cols: usize,
    d: Vec<f64>,
}

impl Mat {
    fn row(&self, r: usize) -> &[f64] {
        &self.d[r * self.cols..(r + 1) * self.cols]
    }
}

struct Layer {
    wq: Mat,
    wk: Mat,
    wv: Mat,
    wo: Mat,
    fc1: Mat,
    fc2: Mat,
}

struct Model {
    n_layer: usize,
    n_embd: usize,
    block_size: usize,
    n_head: usize,
    vocab_size: usize,
    head_dim: usize,
    bos: usize,
    uchars: String,
    char_to_id: [i32; 256],
    wte: Mat,
    wpe: Mat,
    lm_head: Mat,
    layers: Vec<Layer>,
    num_weights: usize,
    weights_sum: f64,
    weights_abs_sum: f64,
    sha256: String,
}

fn rd_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

impl Model {
    fn load(path: &str) -> Result<Model, String> {
        let blob = std::fs::read(path).map_err(|e| format!("cannot open weights {path}: {e}"))?;
        if blob.len() < 32 || &blob[0..4] != b"MGPT" {
            return Err(format!("bad magic, not a microgpt-bin file: {path}"));
        }
        let sha256 = sha256_hex(&blob);

        let version = rd_u32(&blob, 4);
        if version != 1 {
            return Err(format!("unsupported format version {version}"));
        }
        let n_layer = rd_u32(&blob, 8) as usize;
        let n_embd = rd_u32(&blob, 12) as usize;
        let block_size = rd_u32(&blob, 16) as usize;
        let n_head = rd_u32(&blob, 20) as usize;
        let vocab_size = rd_u32(&blob, 24) as usize;
        let n_uchars = rd_u32(&blob, 28) as usize;

        let mut off = 32usize;
        let uchars = String::from_utf8(blob[off..off + n_uchars].to_vec())
            .map_err(|e| format!("vocab is not valid ASCII: {e}"))?;
        off += n_uchars;
        off += (8 - (off % 8)) % 8; // align the f64 payload

        let mut char_to_id = [-1i32; 256];
        for (i, c) in uchars.bytes().enumerate() {
            char_to_id[c as usize] = i as i32;
        }

        let mut cursor = off;
        let mut num_weights = 0usize;
        let mut weights_sum = 0.0f64;
        let mut weights_abs_sum = 0.0f64;

        // Plain running totals, matching the exporter and the other ports. A
        // compensated sum here would disagree with weights/config.json.
        let mut take = |rows: usize, cols: usize| -> Result<Mat, String> {
            let n = rows * cols;
            let mut d = Vec::with_capacity(n);
            for _ in 0..n {
                if cursor + 8 > blob.len() {
                    return Err("weight file truncated".to_string());
                }
                let w = f64::from_le_bytes(blob[cursor..cursor + 8].try_into().unwrap());
                cursor += 8;
                d.push(w);
                weights_sum += w;
                weights_abs_sum += w.abs();
                num_weights += 1;
            }
            Ok(Mat { rows, cols, d })
        };

        let wte = take(vocab_size, n_embd)?;
        let wpe = take(block_size, n_embd)?;
        let lm_head = take(vocab_size, n_embd)?;
        let mut layers = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            layers.push(Layer {
                wq: take(n_embd, n_embd)?,
                wk: take(n_embd, n_embd)?,
                wv: take(n_embd, n_embd)?,
                wo: take(n_embd, n_embd)?,
                fc1: take(4 * n_embd, n_embd)?,
                fc2: take(n_embd, 4 * n_embd)?,
            });
        }
        if cursor != blob.len() {
            return Err("weight file has trailing bytes the model does not want".to_string());
        }

        Ok(Model {
            n_layer,
            n_embd,
            block_size,
            n_head,
            vocab_size,
            head_dim: n_embd / n_head,
            bos: n_uchars,
            uchars,
            char_to_id,
            wte,
            wpe,
            lm_head,
            layers,
            num_weights,
            weights_sum,
            weights_abs_sum,
            sha256,
        })
    }

    fn tokenize(&self, doc: &str) -> Result<Vec<usize>, String> {
        let mut t = Vec::with_capacity(doc.len() + 2);
        t.push(self.bos);
        for c in doc.bytes() {
            let id = self.char_to_id[c as usize];
            if id < 0 {
                return Err(format!("character not in vocab: {}", c as char));
            }
            t.push(id as usize);
        }
        t.push(self.bos);
        Ok(t)
    }
}

// ----------------------------------------------------------------- kernels
// Operation order here is load-bearing: see BENCHMARK.md section 3.

fn linear(x: &[f64], w: &Mat, out: &mut [f64]) {
    for o in 0..w.rows {
        let wo = w.row(o);
        let mut acc = 0.0f64;
        for (wi, xi) in wo.iter().zip(x.iter()) {
            acc += wi * xi;
        }
        out[o] = acc;
    }
}

fn rmsnorm_scale(x: &[f64]) -> f64 {
    let mut ms = 0.0f64;
    for &xi in x.iter() {
        ms += xi * xi;
    }
    ms /= x.len() as f64;
    (ms + 1e-5).powf(-0.5) // pow, not 1/sqrt
}

fn rmsnorm_inplace(x: &mut [f64]) {
    let scale = rmsnorm_scale(x);
    for xi in x.iter_mut() {
        *xi *= scale;
    }
}

fn rmsnorm_to(x: &[f64], out: &mut [f64]) {
    let scale = rmsnorm_scale(x);
    for (o, xi) in out.iter_mut().zip(x.iter()) {
        *o = xi * scale;
    }
}

fn softmax(z: &mut [f64]) {
    let mut m = z[0];
    for &v in z.iter().skip(1) {
        if v > m {
            m = v;
        }
    }
    let mut total = 0.0f64;
    for v in z.iter_mut() {
        *v = (*v - m).exp();
        total += *v;
    }
    for v in z.iter_mut() {
        *v /= total;
    }
}

fn sample_from(probs: &[f64], rng: &mut Rng, cum: &mut [f64]) -> usize {
    let mut total = 0.0f64;
    for (i, &p) in probs.iter().enumerate() {
        total += p;
        cum[i] = total;
    }
    let u = rng.next_f64() * total;
    for (i, &c) in cum[..probs.len()].iter().enumerate() {
        if u < c {
            return i;
        }
    }
    probs.len() - 1
}

// ------------------------------------------------------------------- state

struct Cache {
    keys: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    len: usize,
}

impl Cache {
    fn new(m: &Model) -> Cache {
        Cache {
            keys: (0..m.n_layer).map(|_| vec![0.0; m.block_size * m.n_embd]).collect(),
            values: (0..m.n_layer).map(|_| vec![0.0; m.block_size * m.n_embd]).collect(),
            len: 0,
        }
    }
    fn clear(&mut self) {
        self.len = 0;
    }
}

struct Scratch {
    x: Vec<f64>,
    xn: Vec<f64>,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    attn: Vec<f64>,
    hidden: Vec<f64>,
    logits: Vec<f64>,
    attn_logits: Vec<f64>,
    probs: Vec<f64>,
    cum: Vec<f64>,
}

impl Scratch {
    fn new(m: &Model) -> Scratch {
        Scratch {
            x: vec![0.0; m.n_embd],
            xn: vec![0.0; m.n_embd],
            q: vec![0.0; m.n_embd],
            k: vec![0.0; m.n_embd],
            v: vec![0.0; m.n_embd],
            attn: vec![0.0; m.n_embd],
            hidden: vec![0.0; 4 * m.n_embd],
            logits: vec![0.0; m.vocab_size],
            attn_logits: vec![0.0; m.block_size],
            probs: vec![0.0; m.vocab_size],
            cum: vec![0.0; m.vocab_size],
        }
    }
}

// One decode step. Appends this position's k/v to the cache, leaves logits in s.logits.
fn forward(m: &Model, token_id: usize, pos_id: usize, c: &mut Cache, s: &mut Scratch) {
    let e = m.n_embd;
    let hd = m.head_dim;

    let tok = m.wte.row(token_id);
    let pos = m.wpe.row(pos_id);
    for i in 0..e {
        s.x[i] = tok[i] + pos[i];
    }
    rmsnorm_inplace(&mut s.x);

    let t_idx = c.len;
    let t_len = t_idx + 1;
    for li in 0..m.n_layer {
        let layer = &m.layers[li];
        rmsnorm_to(&s.x, &mut s.xn);
        linear(&s.xn, &layer.wq, &mut s.q);
        linear(&s.xn, &layer.wk, &mut s.k);
        linear(&s.xn, &layer.wv, &mut s.v);

        c.keys[li][t_idx * e..(t_idx + 1) * e].copy_from_slice(&s.k);
        c.values[li][t_idx * e..(t_idx + 1) * e].copy_from_slice(&s.v);

        let scale = (hd as f64).powf(0.5);
        for h in 0..m.n_head {
            let hs = h * hd;
            for t in 0..t_len {
                let kt = &c.keys[li][t * e + hs..t * e + hs + hd];
                let mut acc = 0.0f64;
                for (a, b) in s.q[hs..hs + hd].iter().zip(kt.iter()) {
                    acc += a * b;
                }
                s.attn_logits[t] = acc / scale;
            }
            softmax(&mut s.attn_logits[..t_len]);
            for j in 0..hd {
                let mut acc = 0.0f64;
                for t in 0..t_len {
                    acc += s.attn_logits[t] * c.values[li][t * e + hs + j];
                }
                s.attn[hs + j] = acc;
            }
        }
        linear(&s.attn, &layer.wo, &mut s.xn);
        for i in 0..e {
            s.x[i] += s.xn[i];
        }

        rmsnorm_to(&s.x, &mut s.xn);
        linear(&s.xn, &layer.fc1, &mut s.hidden);
        for v in s.hidden.iter_mut() {
            // Written as the contract's `x > 0.0 ? x : 0.0`, not `if v <= 0.0 { 0.0 }`:
            // the latter leaves NaN untouched, where Python and C++ map NaN to 0.0.
            *v = if *v > 0.0 { *v } else { 0.0 };
        }
        linear(&s.hidden, &layer.fc2, &mut s.xn);
        for i in 0..e {
            s.x[i] += s.xn[i];
        }
    }
    c.len = t_len;
    linear(&s.x, &m.lm_head, &mut s.logits);
}

// -------------------------------------------------------------- benchmarks

// Generate a single sample. Returns the number of forward passes it took.
fn gen_one(
    m: &Model,
    temperature: f64,
    rng: &mut Rng,
    s: &mut Scratch,
    c: &mut Cache,
    mut out: Option<&mut String>,
) -> u64 {
    c.clear();
    let mut token_id = m.bos;
    let mut n = 0u64;
    for pos in 0..m.block_size {
        forward(m, token_id, pos, c, s);
        n += 1;
        for i in 0..m.vocab_size {
            s.probs[i] = s.logits[i] / temperature; // divide, never a precomputed reciprocal
        }
        softmax(&mut s.probs);
        token_id = sample_from(&s.probs, rng, &mut s.cum);
        if token_id == m.bos {
            break;
        }
        if let Some(o) = out.as_deref_mut() {
            o.push(m.uchars.as_bytes()[token_id] as char);
        }
    }
    n
}

// Teacher-forced scoring of one document. Returns tokens scored, adds into *nll.
fn score_doc(
    m: &Model,
    doc: &str,
    s: &mut Scratch,
    c: &mut Cache,
    nll: Option<&mut f64>,
) -> Result<u64, String> {
    let t = m.tokenize(doc)?;
    let n = std::cmp::min(m.block_size, t.len() - 1);
    c.clear();
    // Accumulate this document locally and add once, matching the other ports.
    // Folding each token into the running total sums in a different order.
    let mut local = 0.0f64;
    for pos in 0..n {
        forward(m, t[pos], pos, c, s);
        s.probs.copy_from_slice(&s.logits);
        softmax(&mut s.probs);
        local -= s.probs[t[pos + 1]].ln();
    }
    if let Some(acc) = nll {
        *acc += local;
    }
    Ok(n as u64)
}

// -------------------------------------------------------------------- json

struct Json {
    s: String,
    first: Vec<bool>,
}

fn fmt_f64(v: f64) -> String {
    if !v.is_finite() {
        return "null".to_string();
    }
    // Rust's Display for f64 is the shortest representation that round-trips,
    // which is what CPython's repr does too.
    let s = format!("{v}");
    if s.contains('.') || s.contains('e') || s.contains("inf") {
        s
    } else {
        format!("{s}.0")
    }
}

fn quote(v: &str) -> String {
    let mut r = String::with_capacity(v.len() + 2);
    r.push('"');
    for c in v.chars() {
        match c {
            '"' => r.push_str("\\\""),
            '\\' => r.push_str("\\\\"),
            '\n' => r.push_str("\\n"),
            '\r' => r.push_str("\\r"),
            '\t' => r.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(r, "\\u{:04x}", c as u32);
            }
            c => r.push(c),
        }
    }
    r.push('"');
    r
}

impl Json {
    fn new() -> Json {
        Json { s: String::new(), first: Vec::new() }
    }
    fn comma(&mut self) {
        if let Some(f) = self.first.last_mut() {
            if !*f {
                self.s.push(',');
            }
            *f = false;
        }
    }
    fn key(&mut self, k: &str) {
        self.comma();
        self.s.push('"');
        self.s.push_str(k);
        self.s.push_str("\":");
    }
    fn obj_start(&mut self) {
        self.comma();
        self.s.push('{');
        self.first.push(true);
    }
    fn obj(&mut self, k: &str) {
        self.key(k);
        self.s.push('{');
        self.first.push(true);
    }
    fn obj_end(&mut self) {
        self.s.push('}');
        self.first.pop();
    }
    fn arr(&mut self, k: &str) {
        self.key(k);
        self.s.push('[');
        self.first.push(true);
    }
    fn arr_end(&mut self) {
        self.s.push(']');
        self.first.pop();
    }
    fn numf(&mut self, k: &str, v: f64) {
        self.key(k);
        self.s.push_str(&fmt_f64(v));
    }
    fn numi(&mut self, k: &str, v: i64) {
        self.key(k);
        let _ = write!(self.s, "{v}");
    }
    fn strv(&mut self, k: &str, v: &str) {
        self.key(k);
        self.s.push_str(&quote(v));
    }
    fn valf(&mut self, v: f64) {
        self.comma();
        self.s.push_str(&fmt_f64(v));
    }
    fn vali(&mut self, v: i64) {
        self.comma();
        let _ = write!(self.s, "{v}");
    }
    fn vals(&mut self, v: &str) {
        self.comma();
        self.s.push_str(&quote(v));
    }
}

// ---------------------------------------------------------------- affinity

// Pin to one logical CPU. On Intel hybrid parts (P-cores + E-cores) the scheduler
// migrates sustained single-threaded work onto an efficiency core, which changes
// throughput 2-3x mid-run. P-cores enumerate first, so CPU 0 is a P-core in practice.
#[cfg(windows)]
fn pin_cpu(n: i32) -> String {
    if n < 0 {
        return "not pinned".to_string();
    }
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn SetProcessAffinityMask(handle: isize, mask: usize) -> i32;
    }
    let ok = unsafe { SetProcessAffinityMask(GetCurrentProcess(), 1usize << n) };
    if ok != 0 {
        format!("cpu{n}")
    } else {
        format!("pin to cpu{n} failed")
    }
}

#[cfg(target_os = "linux")]
fn pin_cpu(n: i32) -> String {
    if n < 0 {
        return "not pinned".to_string();
    }
    extern "C" {
        fn sched_setaffinity(pid: i32, cpusetsize: usize, mask: *const u64) -> i32;
    }
    let mut mask = [0u64; 16];
    let idx = (n as usize) / 64;
    if idx >= mask.len() {
        return format!("pin to cpu{n} failed");
    }
    mask[idx] = 1u64 << ((n as usize) % 64);
    let rc = unsafe { sched_setaffinity(0, std::mem::size_of_val(&mask), mask.as_ptr()) };
    if rc == 0 {
        format!("cpu{n}")
    } else {
        format!("pin to cpu{n} failed")
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
fn pin_cpu(_n: i32) -> String {
    "not pinned".to_string()
}

// -------------------------------------------------------------------- main

fn read_docs(path: &str, max_docs: usize) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot open data {path}: {e}"))?;
    let mut docs = Vec::new();
    for line in text.lines() {
        // Strip both ends, matching Python's str.strip() and the C++ port.
        let t = line.trim_matches(|c| c == '\r' || c == '\n' || c == ' ' || c == '\t');
        if !t.is_empty() {
            docs.push(t.to_string());
        }
        if max_docs > 0 && docs.len() >= max_docs {
            break;
        }
    }
    Ok(docs)
}

fn median(sorted: &[f64]) -> f64 {
    sorted[sorted.len() / 2]
}

fn run() -> Result<String, String> {
    let mut weights = "weights/microgpt.bin".to_string();
    let mut data = "data/val.txt".to_string();
    let mut mode = "all".to_string();
    let mut json_out = String::new();
    let mut samples: usize = 1000;
    let mut seed: i64 = 1234;
    let mut repeats: usize = 5;
    let mut max_docs: usize = 0;
    let mut pin: i32 = 0;
    let mut temperature: f64 = 0.5;
    let mut time_budget: f64 = 2.0;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        let a = argv[i].as_str();
        let next = |i: &mut usize| -> Result<String, String> {
            *i += 1;
            argv.get(*i)
                .cloned()
                .ok_or_else(|| format!("missing value for {a}"))
        };
        match a {
            "--weights" => weights = next(&mut i)?,
            "--data" => data = next(&mut i)?,
            "--mode" => mode = next(&mut i)?,
            "--json" => json_out = next(&mut i)?,
            "--samples" => samples = next(&mut i)?.parse().map_err(|e| format!("--samples: {e}"))?,
            "--seed" => seed = next(&mut i)?.parse().map_err(|e| format!("--seed: {e}"))?,
            "--repeats" => repeats = next(&mut i)?.parse().map_err(|e| format!("--repeats: {e}"))?,
            "--max-docs" => max_docs = next(&mut i)?.parse().map_err(|e| format!("--max-docs: {e}"))?,
            "--pin" => pin = next(&mut i)?.parse().map_err(|e| format!("--pin: {e}"))?,
            "--temperature" => {
                temperature = next(&mut i)?.parse().map_err(|e| format!("--temperature: {e}"))?
            }
            "--time-budget" => {
                time_budget = next(&mut i)?.parse().map_err(|e| format!("--time-budget: {e}"))?
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if !matches!(mode.as_str(), "all" | "gen" | "ppl" | "check") {
        return Err(format!("unknown --mode {mode}"));
    }

    let cpu_pin = pin_cpu(pin);

    let t0 = Instant::now();
    let m = Model::load(&weights)?;
    let load_seconds = t0.elapsed().as_secs_f64();

    let docs = read_docs(&data, max_docs)?;
    let mut s = Scratch::new(&m);
    let mut c = Cache::new(&m);

    let do_check = mode == "all" || mode == "check";
    let do_gen = mode == "all" || mode == "gen";
    let do_ppl = mode == "all" || mode == "ppl";

    let mut j = Json::new();
    j.obj_start();
    j.strv("impl", "rust");
    j.strv("mode", &mode);
    j.strv("runtime", env!("MICROGPT_RUSTC"));
    j.strv("build", "cargo --release (opt-level 3, no fast-math equivalent exists)");
    j.strv("weights_sha256", &m.sha256);
    j.strv("weights_path", &weights);
    j.strv("data_path", &data);
    j.obj("config");
    j.numi("n_layer", m.n_layer as i64);
    j.numi("n_embd", m.n_embd as i64);
    j.numi("block_size", m.block_size as i64);
    j.numi("n_head", m.n_head as i64);
    j.numi("head_dim", m.head_dim as i64);
    j.numi("vocab_size", m.vocab_size as i64);
    j.numi("bos_token_id", m.bos as i64);
    j.strv("uchars", &m.uchars);
    j.numi("num_weights", m.num_weights as i64);
    j.obj_end();
    j.numf("load_seconds", load_seconds);
    j.strv("cpu_pin", &cpu_pin);

    // ------------------------------------------------------------- check
    if do_check {
        let prompt = "emma";
        let mut pt = m.tokenize(prompt)?;
        pt.pop(); // drop the trailing BOS
        c.clear();
        for (pos, &tid) in pt.iter().enumerate() {
            forward(&m, tid, pos, &mut c, &mut s);
        }
        let logits = s.logits.clone();
        let mut probs = logits.clone();
        softmax(&mut probs);
        let mut argmax = 0usize;
        for i in 1..probs.len() {
            if probs[i] > probs[argmax] {
                argmax = i;
            }
        }

        let take3 = std::cmp::min(3, docs.len());
        let mut nll3 = 0.0f64;
        let mut n3 = 0u64;
        for d in docs.iter().take(take3) {
            n3 += score_doc(&m, d, &mut s, &mut c, Some(&mut nll3))?;
        }

        let mut r = Rng::new(seed as u64);
        let draws: Vec<String> = (0..4).map(|_| format!("0x{:016x}", r.next_u64())).collect();

        j.obj("check");
        j.strv("prompt", prompt);
        j.arr("logits_after_prompt");
        for &v in &logits {
            j.valf(v);
        }
        j.arr_end();
        j.arr("probs_after_prompt");
        for &v in &probs {
            j.valf(v);
        }
        j.arr_end();
        j.numi("argmax_token", argmax as i64);
        j.strv(
            "argmax_char",
            &if argmax == m.bos {
                "<BOS>".to_string()
            } else {
                (m.uchars.as_bytes()[argmax] as char).to_string()
            },
        );
        j.arr("first3_docs");
        for d in docs.iter().take(take3) {
            j.vals(d);
        }
        j.arr_end();
        j.numf("first3_nll", nll3);
        j.numi("first3_tokens", n3 as i64);
        j.numi("rng_seed", seed);
        j.arr("rng_first4_u64");
        for d in &draws {
            j.vals(d);
        }
        j.arr_end();
        j.numf("weights_sum", m.weights_sum);
        j.numf("weights_abs_sum", m.weights_abs_sum);
        j.obj_end();
    }

    // --------------------------------------------------------------- gen
    if do_gen {
        // 1. Deterministic pass: fixed sample count, produces the cross-language
        //    hash. Doubles as warmup so the timed passes below start hot.
        let mut texts: Vec<String> = Vec::with_capacity(samples);
        let mut tokens = 0u64;
        let mut chars = 0u64;
        {
            let mut rng = Rng::new(seed as u64);
            for _ in 0..samples {
                let mut buf = String::new();
                tokens += gen_one(&m, temperature, &mut rng, &mut s, &mut c, Some(&mut buf));
                chars += buf.len() as u64;
                texts.push(buf);
            }
        }
        let body = texts.join("\n");

        // 2. Timed passes under a shared wall-clock budget, so every implementation
        //    is measured in the same CPU clock state.
        let mut seconds: Vec<f64> = Vec::with_capacity(repeats);
        let mut tok_counts: Vec<u64> = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let mut rng = Rng::new(seed as u64);
            let mut n_tok = 0u64;
            let start = Instant::now();
            let mut elapsed;
            loop {
                n_tok += gen_one(&m, temperature, &mut rng, &mut s, &mut c, None);
                elapsed = start.elapsed().as_secs_f64();
                if elapsed >= time_budget {
                    break;
                }
            }
            tok_counts.push(n_tok);
            seconds.push(elapsed);
        }

        let rates: Vec<f64> = tok_counts
            .iter()
            .zip(seconds.iter())
            .map(|(&t, &sec)| t as f64 / sec)
            .collect();
        let mut ordered = rates.clone();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());

        j.obj("gen");
        j.numi("samples", samples as i64);
        j.numf("temperature", temperature);
        j.numi("seed", seed);
        j.numi("repeats", repeats as i64);
        j.numf("time_budget", time_budget);
        j.numi("tokens", tokens as i64);
        j.numi("chars", chars as i64);
        j.strv("output_fnv1a64", &format!("0x{:016x}", fnv1a64(&body)));
        j.arr("first_samples");
        for t in texts.iter().take(20) {
            j.vals(t);
        }
        j.arr_end();
        j.arr("rates_per_repeat");
        for &r in &rates {
            j.valf(r);
        }
        j.arr_end();
        j.arr("tokens_per_repeat");
        for &t in &tok_counts {
            j.vali(t as i64);
        }
        j.arr_end();
        j.arr("seconds_per_repeat");
        for &sec in &seconds {
            j.valf(sec);
        }
        j.arr_end();
        j.numf("tokens_per_sec_best", *ordered.last().unwrap());
        j.numf("tokens_per_sec_median", median(&ordered));
        j.obj_end();
    }

    // --------------------------------------------------------------- ppl
    if do_ppl {
        // 1. Quality pass over the exact evaluation set - this is what perplexity means.
        let mut nll_total = 0.0f64;
        let mut n_tok = 0u64;
        let full_start = Instant::now();
        for d in docs.iter() {
            n_tok += score_doc(&m, d, &mut s, &mut c, Some(&mut nll_total))?;
        }
        let full_seconds = full_start.elapsed().as_secs_f64();

        // 2. Timed passes under a shared wall-clock budget, cycling the corpus.
        let mut seconds: Vec<f64> = Vec::with_capacity(repeats);
        let mut tok_counts: Vec<u64> = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let mut n = 0u64;
            let mut idx = 0usize;
            let start = Instant::now();
            let mut elapsed;
            loop {
                n += score_doc(&m, &docs[idx % docs.len()], &mut s, &mut c, None)?;
                idx += 1;
                elapsed = start.elapsed().as_secs_f64();
                if elapsed >= time_budget {
                    break;
                }
            }
            tok_counts.push(n);
            seconds.push(elapsed);
        }

        let rates: Vec<f64> = tok_counts
            .iter()
            .zip(seconds.iter())
            .map(|(&t, &sec)| t as f64 / sec)
            .collect();
        let mut ordered = rates.clone();
        ordered.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let nll_per_token = nll_total / n_tok as f64;

        j.obj("ppl");
        j.numi("docs", docs.len() as i64);
        j.numi("tokens", n_tok as i64);
        j.numi("repeats", repeats as i64);
        j.numf("time_budget", time_budget);
        j.numf("nll_total", nll_total);
        j.numf("nll_per_token", nll_per_token);
        j.numf("perplexity", nll_per_token.exp());
        j.numf("bits_per_token", nll_per_token / std::f64::consts::LN_2);
        j.numf("full_pass_seconds", full_seconds);
        j.arr("rates_per_repeat");
        for &r in &rates {
            j.valf(r);
        }
        j.arr_end();
        j.arr("tokens_per_repeat");
        for &t in &tok_counts {
            j.vali(t as i64);
        }
        j.arr_end();
        j.arr("seconds_per_repeat");
        for &sec in &seconds {
            j.valf(sec);
        }
        j.arr_end();
        j.numf("tokens_per_sec_best", *ordered.last().unwrap());
        j.numf("tokens_per_sec_median", median(&ordered));
        j.obj_end();
    }

    j.obj_end();

    if !json_out.is_empty() {
        std::fs::write(&json_out, format!("{}\n", j.s))
            .map_err(|e| format!("cannot write json {json_out}: {e}"))?;
    }
    Ok(j.s)
}

fn main() {
    match run() {
        Ok(text) => println!("{text}"),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
