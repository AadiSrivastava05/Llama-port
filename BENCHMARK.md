# microgpt port benchmark — contract

Three implementations of the same inference algorithm — Python, C++, Rust — measured on
**token generation speed** and **held-out perplexity**, loading **identical weights**.

Any port that follows this document can be dropped into `bench/run.py` and compared.

## Current results

Core Ultra 7 155H (mobile, hybrid), pinned to CPU 0, best of 7 time-budgeted 1s repeats, one process per measurement:

| impl | runtime | gen tok/s | speedup | perplexity | gen hash |
|---|---|---:|---:|---:|---|
| python | CPython 3.14.3 | ~4,590 | 1.0x | 10.789614873329409 | `0x6fa78e…dbcb` |
| cpp | g++ 15.2.0 `-O3` | ~584,000 | **~127x** | 10.789614873329409 | `0x6fa78e…dbcb` |
| rust | rustc 1.98.0 `--release` | ~567,000 | **~124x** | 10.789614873329409 | `0x6fa78e…dbcb` |

C++ and Rust are within ~3% of each other - effectively tied, which is the expected result
for the same scalar algorithm compiled by GCC and LLVM. Do not read the gap as meaningful.

Two corrections worth recording, since both inflated an earlier version of this table:

- An earlier revision claimed C++ was ~142x. That was inflated because the Python reference
  used CPython's compensated `sum()`, doing arithmetic the compiled ports never did. See the
  `sum()` warning in section 3.
- An earlier revision had C++ at `-O2` while Rust ran `opt-level 3`, which made Rust look
  ~20% faster than C++. That was the flag, not the language. Both now build at O3.

Measure implementations in **short, isolated sessions**. Running all three through one
`bench/run.py` invocation with high `--repeats` takes long enough that the machine loses
clock across the session, and the numbers spread by 1.6x or more. The per-implementation
figures above come from separate ~2s runs.

Identical weights, identical generated text, and **bit-identical perplexity** - not merely
close: the same f64 down to the last digit, plus identical `weights_sum`.

Note that within each implementation gen and ppl throughput agree to well under 2%. That is
the sanity check that the timing is sound: both run the same forward pass, so they must. An
earlier version of this harness reported gen and ppl differing by 1.5x, which turned out to
be a measurement artifact, not a real difference - see section 5.

Agreement is also verified across model shapes, not just this checkpoint
(`python tools/test_shapes.py`): 1/2/3-layer, 8-embd to 32-embd, 1 to 8 heads, all matching
exactly. The shipped checkpoint has `n_layer=1`, so without that sweep the transformer's
per-layer indexing would never actually be exercised.

Perplexity in context, all fit on the same 1,000 documents microgpt trained on
(`python tools/baselines.py`):

| model | nll/token | perplexity |
|---|---:|---:|
| uniform over 27 tokens | 3.2958 | 27.00 |
| unigram (char frequency) | 2.8332 | 17.00 |
| bigram (count table) | 2.5120 | 12.33 |
| **microgpt (1,000 steps)** | **2.3786** | **10.79** |

---

## 1. There are no pretrained weights to download

`microgpt.py` is a from-scratch teaching file. Its "model" is 4,192 numbers produced by a
random init plus 1,000 Adam steps, and its architecture is bespoke (1 layer, `n_embd=16`,
`block_size=16`, 4 heads, vocab 27, rmsnorm instead of layernorm, no biases, ReLU instead of
GeLU). No checkpoint for it exists publicly, and no unrelated checkpoint is shape-compatible.

This costs nothing, because training is a **one-time ~80 second** job:

```bash
python tools/train_export.py --steps 1000
```

`tools/train_export.py` is a faithful copy of microgpt.py's math, RNG call order and
optimizer, so it produces exactly the weights microgpt.py would hold when its own training
loop ends. It then writes `weights/microgpt.bin`. **After that nothing ever trains again** —
all three implementations are inference-only and just read that file.

Current checkpoint: 1,000 steps, train loss 3.37 → 2.32, held-out perplexity 10.79
(uniform baseline is 27.0).

---

## 2. Weight file format — `microgpt-bin-v1`

Little-endian throughout. `weights/config.json` mirrors all of this in readable form,
including a sha256 and checksums for load verification.

| Offset | Type | Field |
|---|---|---|
| 0 | `char[4]` | magic `"MGPT"` |
| 4 | `u32` | format version (`1`) |
| 8 | `u32` | `n_layer` |
| 12 | `u32` | `n_embd` |
| 16 | `u32` | `block_size` |
| 20 | `u32` | `n_head` |
| 24 | `u32` | `vocab_size` |
| 28 | `u32` | `n_uchars` |
| 32 | `u8[n_uchars]` | vocabulary, ASCII, ascending; token id = index |
| … | `u8[]` | zero padding to the next multiple of 8 (`payload_offset`) |
| `payload_offset` | `f64[]` | all tensors, in the order below, row-major |

`bos_token_id == n_uchars` (there is no character for it), and `vocab_size == n_uchars + 1`.
`head_dim == n_embd / n_head`.

Tensor order — read them back-to-back in exactly this sequence, with `E = n_embd`,
`V = vocab_size`, `B = block_size`:

1. `wte` `[V][E]`
2. `wpe` `[B][E]`
3. `lm_head` `[V][E]`
4. then, for each layer `i` in `0..n_layer`:
   `attn_wq [E][E]`, `attn_wk [E][E]`, `attn_wv [E][E]`, `attn_wo [E][E]`,
   `mlp_fc1 [4E][E]`, `mlp_fc2 [E][4E]`

With the defaults that is 4,192 f64 values = 33,536 payload bytes, 33,600 total.

**Load check.** Sum all weights left-to-right in file order and compare against
`weights_sum` / `weights_abs_sum` in `config.json`. A port that gets these wrong has a
layout bug, and everything downstream is meaningless.

---

## 3. Numerics contract

All three implementations compute in **f64** and must apply operations in the same order, so
that perplexity agrees to ~1e-12 rather than merely "close enough". Reproduce these exactly:

```
linear(x, W)[o]  = accumulate acc = 0.0; for i in 0..nin: acc += W[o][i] * x[i]
                   (ascending i, sequential, no reassociation, no FMA contraction)

rmsnorm(x)       = ms = (accumulate x[i]*x[i], ascending) / len(x)
                   scale = pow(ms + 1e-5, -0.5)      // pow, not 1/sqrt
                   out[i] = x[i] * scale

softmax(z)       = m = max(z)
                   e[i] = exp(z[i] - m)
                   total = accumulate e[i], ascending
                   out[i] = e[i] / total

attention        = logits[t] = (accumulate q_h[j]*k_h[t][j], ascending j) / pow(head_dim, 0.5)
                   weights   = softmax(logits)
                   out[j]    = accumulate weights[t]*v_h[t][j], ascending t

relu(x)          = x > 0.0 ? x : 0.0
                   (write it this way round. `if x <= 0 { x = 0 }` looks equivalent but
                    leaves NaN untouched, where this maps NaN to 0.0)
temperature      = z[i] / temperature        // divide; do NOT precompute 1/temperature
```

Forward pass per token, matching `microgpt.py`:

```
x = wte[token] + wpe[pos]
x = rmsnorm(x)
for each layer:
    r = x;  x = rmsnorm(x)
    q,k,v = linear(x, wq), linear(x, wk), linear(x, wv)
    append k,v to this layer's KV cache
    per head: causal attention over the whole cache, concatenated
    x = linear(concat, wo);  x = x + r
    r = x;  x = rmsnorm(x)
    x = linear(x, fc1);  x = relu(x);  x = linear(x, fc2);  x = x + r
logits = linear(x, lm_head)
```

**Beware your language's `sum()`.** "Plain sequential accumulation" above is a real
constraint, not boilerplate. CPython has used Neumaier *compensated* summation for floats
since 3.12, so `sum(w*x for ...)` is measurably more accurate than a running total - on this
checkpoint it changed **141 of 214** dot products and shifted `weights_sum` by 5e-15. That is
not something C++ or Rust reproduce, so the Python reference uses explicit `acc += ...` loops
instead. Rust's `iter().sum::<f64>()` is a plain fold and is fine; `f64::mul_add` is not, as
it fuses. If a port's `weights_sum` does not match `config.json` **exactly**, this is the
first thing to suspect - `bench/run.py` now compares it exactly for that reason.

**Compiler flags: no fast-math, and match the optimization level across languages.**
`-ffast-math` / `-Ofast` reassociate sums and break cross-language agreement. Build C++ with
`-O3 -fno-fast-math -ffp-contract=off`, and Rust with plain `--release` (no `-C
target-feature` fiddling that enables FMA contraction).

`-O3` rather than `-O2` specifically because Rust's release profile is `opt-level = 3`.
Benchmarking `g++ -O2` against Rust `opt-level 3` measures the flag, not the language: it
cost C++ about 20% here (472k vs 587k tok/s) and made Rust look ~20% faster than C++ when
the two are actually neck and neck.

Residual differences of ~1e-15 per logit are tolerated, since libm `exp`/`log`/`pow` may
differ between implementations by an ulp - though in practice CPython and MSYS2 UCRT agree
exactly here, and Python and C++ currently produce bit-identical perplexity. `tools/validate.py` demonstrates the accepted
noise floor by checking the Python reference against microgpt.py's own autograd graph
(measured: 1.3e-13 max relative logit difference, identical NLL to 15 digits).

---

## 4. RNG and sampling

Sampling must be reproducible across languages, so the benchmark defines its own generator
rather than using each language's stdlib. **xorshift64\***, all arithmetic wrapping mod 2^64:

```
init:  state = seed != 0 ? seed : 0x9E3779B97F4A7C15

next_u64():
    x = state
    x ^= x >> 12
    x ^= (x << 25)          // wrapping
    x ^= x >> 27
    state = x
    return x * 0x2545F4914F6CDD1D    // wrapping

next_f64():
    return (next_u64() >> 11) * (1.0 / 9007199254740992.0)   // 2^-53, in [0,1)
```

Rust: use `wrapping_shl`-free `<<` on `u64` in release is fine, but write
`wrapping_mul` for the multiply. C++: `uint64_t` arithmetic already wraps.

Sanity vector — seed `1234`, first four `next_u64()` draws are reported in every
implementation's `check` output and must match.

Sampling from a probability vector:

```
sample(probs, rng):
    total = 0.0; cum[i] = (total += probs[i])      // ascending i
    u = rng.next_f64() * total
    for i in 0..n: if u < cum[i]: return i
    return n - 1
```

---

## 5. Tokenization and the two measurements

**Tokenizer.** A document is `[BOS] + one token per character + [BOS]`, matching how
microgpt.py trains. Character token id is its index in the vocab string.

### Perplexity (`--mode ppl`)

Teacher-forced over `data/val.txt` — 2,000 documents the model never saw, since training
only ever consumed `docs[:1000]`. No temperature, no sampling.

```
for each doc:
    tokens = tokenize(doc)
    n = min(block_size, len(tokens) - 1)
    fresh KV cache
    for pos in 0..n:
        logits = forward(tokens[pos], pos, cache)
        nll   -= log(softmax(logits)[tokens[pos+1]])
        count += 1
perplexity = exp(nll / count)
```

The `min(block_size, …)` cap is microgpt.py's; with a 15-character longest name it never
actually truncates, so every token of every document is scored. Expect
`docs=2000, tokens=14307`.

### Generation speed (`--mode gen`)

```
for each of --samples samples:
    fresh KV cache; token = BOS
    for pos in 0..block_size:
        logits = forward(token, pos, cache)
        tokens_generated += 1
        token = sample(softmax(logits / temperature), rng)
        if token == BOS: break
        emit vocab[token]
```

The RNG is **re-seeded at the start of every repeat**, so all repeats do identical work and
produce identical text. `tokens_generated` counts forward passes — that is the numerator for
tokens/sec.

Defaults: `--samples 1000 --temperature 0.5 --seed 1234 --repeats 5`.

Each implementation runs this in two parts:

1. a **deterministic pass** of exactly `--samples` samples, which produces the hash and the
   sample text, and warms the code paths; then
2. `--repeats` **timed passes**, each generating until `--time-budget` seconds have elapsed.

Perplexity works the same way: one full pass over the evaluation set for the number itself,
then time-budgeted passes cycling the corpus for the throughput measurement.

### Timing methodology, and why it is this convoluted

Wall clock around the loop only; weight loading is reported separately as `load_seconds` and
excluded. Three measures exist because measuring this naively produced numbers that were
wrong by more than 2x on the development machine:

**Equal time, not equal work.** A fixed workload means Python runs for 20 seconds while C++
runs for 40 milliseconds — and a process's throughput on a modern laptop depends on how long
it has been running. Measured here: the identical forward pass ran at 3,449 tok/s in a 0.7s
workload and 1,601 tok/s in a 23s workload. Comparing those directly measures CPU clock
behaviour, not language performance. Every implementation therefore runs for the same
`--time-budget` per repeat, and throughput is reported as a **rate**.

**CPU pinning (`--pin`, default logical CPU 0).** On hybrid CPUs (Intel P-core/E-core, and
ARM big.LITTLE) the scheduler migrates sustained single-threaded work onto an efficiency
core, which is 2-3x slower for scalar FP. P-cores enumerate first, so CPU 0 is a P-core in
practice. Pass `--pin -1` to disable.

**One process per measurement.** `bench/run.py` invokes each implementation separately for
`gen` and for `ppl` rather than once with `--mode all`. Running them back to back in one
process made whichever went second look ~1.5x slower in *both* languages, because by then the
process had been busy for ~10 seconds and had lost clock. Splitting the processes dropped
measured spread from ~1.6x to ~1.03x and brought gen and ppl into agreement.

**Best of N repeats.** Contamination only ever makes a run slower, so the fastest repeat is
the least-contaminated estimate. The harness also reports **spread** — fastest/slowest repeat
*rate* — and warns above 1.25x. Compare rates rather than durations here: with a time budget
every repeat takes the same number of seconds by construction, so it is the token count that
varies.

**Read the ratios with the spread in mind.** With all three measures in place the
development machine (Core Ultra 7 155H, mobile, hybrid) reaches ~1.03x spread and a stable
~142x. The cross-check that the numbers are sound is that the **gen and ppl throughputs agree
within an implementation** — they measure the same forward pass, so they must. Two successive
versions of this harness reported 211x/124x and then 159x/146x for those two; both
disagreements were measurement bugs, and chasing them is what produced the rules above.

---

## 6. CLI contract

Every implementation accepts the same flags, so `bench/run.py` can drive them uniformly:

| Flag | Default | Meaning |
|---|---|---|
| `--weights PATH` | `weights/microgpt.bin` | weight file |
| `--data PATH` | `data/val.txt` | one document per line |
| `--mode all\|gen\|ppl\|check` | `all` | which measurements to run |
| `--samples N` | `1000` | generation samples |
| `--temperature F` | `0.5` | sampling temperature |
| `--seed N` | `1234` | RNG seed |
| `--repeats N` | `5` | timed repeats |
| `--time-budget F` | `2.0` | wall-clock seconds per timed repeat |
| `--pin N` | `0` | pin to this logical CPU (`-1` disables) |
| `--max-docs N` | `0` (all) | truncate the perplexity set |
| `--json PATH` | — | also write the report to this file |

The JSON report goes to stdout. Nothing else may be printed to stdout; diagnostics go to
stderr.

---

## 7. JSON report contract

```json
{
  "impl": "python",
  "mode": "all",
  "runtime": "CPython 3.14.3",
  "build": "interpreted (no numpy, plain f64 lists)",
  "weights_sha256": "9bf6ac86…",
  "config": { "n_layer": 1, "n_embd": 16, "num_weights": 4192, "…": "…" },
  "load_seconds": 0.0012,
  "cpu_pin": "cpu0",

  "check": {
    "logits_after_prompt": [27 f64],
    "first3_nll": 8.34…, "first3_tokens": 17,
    "rng_first4_u64": ["0x…", "0x…", "0x…", "0x…"],
    "weights_sum": 10.6213…, "weights_abs_sum": 516.1316…
  },
  "gen": {
    "samples": 1000, "tokens": 6021, "chars": 5021,
    "output_fnv1a64": "0x6fa78e6dddfddbcb",
    "first_samples": ["mana", "analen", "…"],
    "rates_per_repeat": [3301.2, 2870.4, …],
    "tokens_per_repeat": [3301, 2870, …],
    "seconds_per_repeat": [1.0, 1.0, …],
    "tokens_per_sec_best": 3301.2, "tokens_per_sec_median": 2954.1
  },
  "ppl": {
    "docs": 2000, "tokens": 14307,
    "nll_total": 34029.5…, "nll_per_token": 2.378584…,
    "perplexity": 10.7896…, "bits_per_token": 3.4316…,
    "full_pass_seconds": 4.69,
    "rates_per_repeat": [2120.3, …], "tokens_per_sec_best": 2120.3
  }
}
```

`output_fnv1a64` is FNV-1a 64 over the generated samples joined by `"\n"` (no trailing
newline), ASCII:

```
h = 0xCBF29CE484222325
for each byte b: h ^= b; h = h * 0x100000001B3   // wrapping
```

---

## 8. What counts as a correct port

`bench/run.py` enforces, across every implementation:

`gen.tokens`, `gen.output_fnv1a64` and `ppl.tokens` all come from the deterministic passes,
never from the time-budgeted ones, so they stay comparable across implementations.

`check.first3_nll` must group its summation per document and then add, the same way the
perplexity path does - all three implementations route it through their `score_doc` helper
for exactly this reason. Folding every token into one running total is a different summation
order and shifts the last digits, which reads as a cross-language mismatch when it is only
the fingerprint disagreeing with itself.

- identical `weights_sha256` and `config`
- `weights_sum` / `weights_abs_sum` matching `config.json`
- `check.rng_first4_u64` identical — proves the RNG is right
- `ppl.perplexity` agreeing within **1e-9** relative
- `gen.output_fnv1a64` identical — proves the ports generate the same text

The hash is the strong check: it can only match if the forward pass, the softmax, the
temperature division, the RNG and the sampling scan all agree. If perplexity matches but the
hash does not, the divergence is in sampling, not the model.

Note that the hash covers the whole generated corpus, so it changes with `--samples`,
`--temperature` and `--seed`. It is only meaningful when comparing implementations **within a
single `bench/run.py` invocation**, which is why the harness passes identical flags to every
implementation. Do not compare a hash against one recorded from a different settings run.

---

## 9. Running it

```bash
python tools/train_export.py --steps 1000     # once, ~80s, produces the weights
python tools/validate.py                      # plain-float forward vs microgpt.py autograd
python bench/run.py --build                   # build C++/Rust, run all, print the table
```

Useful variations:

```bash
python bench/run.py --impls python,cpp --repeats 9 --time-budget 1.0
python bench/run.py --check-only              # correctness agreement, no timing
python bench/run.py --pin -1                  # disable CPU pinning
```

Benchmark hygiene: close other heavy processes, keep the machine on AC power, and prefer the
`_best` numbers. If `spread` stays above ~1.25x, the ratios are indicative rather than
precise - see the timing methodology in section 5. This model is tiny (4,192 weights, everything in L1), so the comparison
measures **language and interpreter overhead per scalar op**, not memory bandwidth or SIMD
throughput. That is the honest framing of the result. For a version that stresses SIMD and
cache instead, retrain wider — `--n-embd 64 --n-layer 4` — and rerun; the format, contract
and harness are all shape-agnostic.
