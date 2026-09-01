# Porting microgpt to another language

`BENCHMARK.md` is the **contract** — the file format, the operation order, the RNG, the CLI
and JSON shape. Read sections 2–7 of it first. This file is the other half: **what actually
goes wrong**, in the order it is likely to bite you.

Every trap listed here was found the hard way while writing the C++ and Rust ports. Seven of
them were real defects that shipped and had to be fixed. Two survived a full benchmark run
looking perfectly correct, because the outputs happened to agree anyway.

The headline lesson: **agreeing outputs are not proof of a correct port.** Floating-point
divergence usually hides until it doesn't, and a discrete decision (which token got sampled)
absorbs small errors silently. Verify the arithmetic, not just the answer.

---

## 0. The two-minute checklist

If you read nothing else:

- [ ] **f64 everywhere.** No f32, no mixed precision, no "close enough".
- [ ] **Never use your language's `sum()`.** Write an explicit `acc += ...` loop. See trap 1.
- [ ] **No FMA, no fast-math.** No `mul_add`, no `-ffast-math`, no `-Ofast`.
- [ ] **`pow(x, -0.5)`**, not `1.0/sqrt(x)`.
- [ ] **Divide by temperature**, never multiply by a precomputed `1/temperature`.
- [ ] **`relu(x) = x > 0.0 ? x : 0.0`** — written that way round, so NaN maps to 0.
- [ ] **Accumulate NLL per document, then add** — grouping changes the result.
- [ ] **Wrapping integer arithmetic** in the RNG, and handle seed 0.
- [ ] **Match the optimization level** of the other ports, or you are benchmarking a flag.
- [ ] **Only JSON on stdout.** Diagnostics to stderr.

Then verify against the golden values in section 2 before trusting any timing number.

---

## 1. Port in this order

Each step is independently verifiable, so a bug is localized when it appears. Do not skip
ahead — a weight-loading bug diagnosed at the perplexity stage costs hours.

| # | Step | Verify by |
|---|---|---|
| 1 | Read the header: magic, version, dims, vocab, 8-byte alignment | dims match `weights/config.json` |
| 2 | Read the f64 payload, accumulating `weights_sum` / `weights_abs_sum` | **exact** match with `config.json` |
| 3 | Implement the RNG | first four draws for seed 1234 match section 2 |
| 4 | Implement the kernels and the forward pass | `--mode check`: logits and `first3_nll` match |
| 5 | Implement generation + sampling | `--mode gen`: `output_fnv1a64` and `tokens` match |
| 6 | Implement perplexity | `--mode ppl`: `perplexity` matches to the last digit |
| 7 | Emit the full JSON report, wire into `bench/run.py` | `python bench/run.py --check-only` passes |
| 8 | Run the shape sweep | `python tools/test_shapes.py` passes all 4 shapes |

Step 2 is the highest-value checkpoint in the whole process. It is a single number that
proves your header parsing, alignment, byte order, tensor ordering and f64 decoding are all
correct, before any model math exists. If it does not match **exactly**, stop and fix it.

---

## 2. Golden reference values

For the shipped checkpoint (`weights/microgpt.bin`, sha256 `9bf6ac8636d6b4b4…`, 1000 training
steps). Regenerate with `python tools/train_export.py --steps 1000` if you lose it — it is
deterministic.

**Config**

```
n_layer 1   n_embd 16   block_size 16   n_head 4   head_dim 4
vocab_size 27   bos_token_id 26   num_weights 4192
uchars "abcdefghijklmnopqrstuvwxyz"
file: 33600 bytes total, payload_offset 64
```

**Load checksums** — must match exactly, not approximately:

```
weights_sum      10.621326738609703
weights_abs_sum  516.1316394186422
```

**RNG**, seed 1234, first four `next_u64()`:

```
0xbc5614c61020390e  0x4dda77751ffcd35f  0x692ee90e3ef071fe  0x530a68df87af205b
```

**`--mode check`** (prompt `"emma"`, first three docs of `data/val.txt`):

```
first3_docs        ["denton", "mikai", "ailis"]
first3_nll         40.86737794408452     (19 tokens)
argmax_token       13  ('n')
logits[0..3]       -0.6980699153942418  -0.7545837963500106
                   -1.857827332313785   -1.0179736552648901
```

**`--mode gen --samples 1000 --temperature 0.5 --seed 1234`**

```
tokens             6021
chars              5021
output_fnv1a64     0x6fa78e6dddfddbcb
first samples      mana analen aris sanar anan jaria lielia gahela
```

**`--mode ppl`** over all of `data/val.txt`:

```
docs               2000
tokens             14307
nll_per_token      2.3785840857071237
perplexity         10.789614873329409
bits_per_token     3.4315714647870763
```

The generation hash covers the whole generated corpus, so it **changes** with `--samples`,
`--temperature` or `--seed`. Only compare it at identical settings.

---

## 3. The traps

### Trap 1 — your language's `sum()` is probably not a plain running total

**This is the one that will get you.** It cost this project a wrong headline number for
several hours, and the wrong number *looked* right.

Modern standard libraries silently use better-than-naive summation:

- **CPython ≥ 3.12** uses Neumaier compensated summation for floats. `sum(w*x for ...)` is
  strictly more accurate than a running total — and neither C++ nor Rust reproduces it.
- **NumPy** and **Julia** use pairwise summation for arrays above a size threshold.
- Anything named `fsum`, `Kahan`, `Neumaier`, `accurate_sum` is compensated by construction.

Measured on this checkpoint: switching Python from `sum(genexp)` to an explicit loop changed
**141 of 214** dot products and moved `weights_sum` by 23 ulp (5e-15 relative). It slipped
through a 1e-12 agreement tolerance for the entire first round of benchmarking.

**Litmus test.** In your language:

```
sum([1e100, 1.0, -1e100, 1.0])
```

- `1.0` → naive running total. **This is what the contract requires.**
- `2.0` → compensated (Kahan / Neumaier). Do not use it.

Pairwise summation does *not* reliably show up at four elements. To detect it, sum a
realistic-length array (16, 27 and 64 are the lengths this model actually uses) both with
your builtin and with an explicit loop, and compare bit-for-bit. On CPython 3.14 the two
already differ at n=16.

**Fix.** Write the loop out. In every kernel — `linear`, `rmsnorm`, `softmax`, the two
attention accumulations, and the load-time checksums:

```
acc = 0.0
for i in 0..n:  acc += a[i] * b[i]
```

**Detect.** `weights_sum` will not match `config.json` exactly. `bench/run.py` compares it
exactly for precisely this reason, and `tools/test_harness.py` has a regression case pinning
the 5e-15 drift.

---

### Trap 2 — FMA contraction and fast-math

`a*b + c` fused into a single instruction rounds **once** instead of twice. That is more
accurate and completely breaks cross-language agreement. Reassociation (`-ffast-math`)
rewrites your summation order outright.

**Fix.**

- C/C++: `-fno-fast-math -ffp-contract=off`. Never `-Ofast`.
- Rust: nothing needed — the language never reassociates or auto-fuses. Just never write
  `f64::mul_add` in a kernel.
- JVM / .NET: `Math.fma` / `Math.FusedMultiplyAdd` — avoid.
- C#: avoid `MathF`/vectorized `Vector<T>` dot products unless you control rounding.
- Anything with a `--ffp-model=fast` or `/fp:fast` equivalent: turn it off.

**Detect.** Perplexity diverges around the 12th–14th digit; the generation hash may still
match, which is why you cannot rely on the hash alone.

---

### Trap 3 — NLL summation grouping

Both scoring paths must accumulate **per document into a local, then add that local to the
total.** Folding every token straight into one running total is a different summation order
and shifts the last digits.

```
for each doc:
    local = 0.0
    for each position:  local -= log(p[target])
    total += local            # <- one add per document, not per token
```

This bit the Rust port: its perplexity path was right, but its `check.first3_nll` fingerprint
folded differently and disagreed at 1e-16. It passed the 1e-9 tolerance and would have sat
there indefinitely as a latent inconsistency.

**Fix.** Route both `check` and `ppl` through one shared `score_doc` helper, as all three
reference implementations now do.

---

### Trap 4 — the shape of your `relu` changes NaN behaviour

These look identical and are not:

```
x > 0.0 ? x : 0.0        // NaN -> 0.0     <- required
if x <= 0.0 { x = 0.0 }  // NaN -> NaN     <- wrong
```

`NaN > 0.0` and `NaN <= 0.0` are *both* false, so the first form replaces NaN and the second
preserves it. `-0.0` behaves the same in both (yields `+0.0`).

This bit the Rust port. It is latent — dot products of finite weights cannot produce NaN
here — but it is a real semantic divergence, and it will surface the moment someone feeds in
a checkpoint with a NaN weight.

**Fix.** Write it as the ternary, matching the contract literally.

---

### Trap 5 — temperature must be a division

```
z[i] / temperature          // required
z[i] * (1.0 / temperature)  // wrong
```

For `temperature = 0.5` these coincide exactly, because `1/0.5` is representable. For 0.7,
0.9, or almost anything else, they do not. Precomputing the reciprocal is the obvious
optimization and it silently breaks agreement at non-default temperatures.

---

### Trap 6 — `pow`, not `1/sqrt`

```
scale = pow(ms + 1e-5, -0.5)     // required
scale = 1.0 / sqrt(ms + 1e-5)    // differs by an ulp
```

Same for the attention scale: `pow(head_dim, 0.5)`. For `head_dim = 4` that is exactly 2.0 so
it cannot bite today, but it will the moment `head_dim` is not a perfect square.

---

### Trap 7 — RNG portability

The benchmark defines its own xorshift64\* precisely so all languages agree. Three things go
wrong:

1. **Overflow on the multiply.** Needs explicit wrapping (`wrapping_mul`, or unsigned types
   that wrap by definition). Languages that trap on overflow in debug builds will panic;
   languages without unsigned 64-bit integers (JavaScript pre-BigInt, Lua) need care.
2. **Shift semantics.** `x << 25` on a 64-bit unsigned must discard the high bits. Check what
   your language does for shifts and whether the shift count is masked.
3. **Seed 0** must fall back to `0x9E3779B97F4A7C15`, or the generator is stuck at zero
   forever.

Also: **actually honour the `--seed` flag everywhere.** The Python reference had `Rng(1234)`
hardcoded in its check path, so at any non-default seed the harness reported a false RNG
mismatch against a perfectly correct C++ port.

**Detect.** The four golden draws in section 2.

---

### Trap 8 — the sampling scan

```
total = 0.0; cum[i] = (total += probs[i])     // ascending
u = rng.next_f64() * total
for i in 0..n:  if u < cum[i]: return i       // strict <
return n - 1                                   // fallback, do not omit
```

Three details: the comparison is strict `<`; `u` is scaled by `total` (not assumed to be 1.0);
and the `n-1` fallback matters when floating-point rounding leaves `u` just above the last
cumulative value. A `<=` here, or omitting the fallback, produces rare off-by-one token
selections that only show up as a hash mismatch on long runs.

---

### Trap 9 — how you strip whitespace from the corpus

Python's `str.strip()` removes whitespace from **both** ends. The C++ port originally stripped
only trailing characters, so a document with a leading space tokenized as a vocab miss and
crashed. Latent on `data/val.txt` (LF-only lowercase ASCII), real for any other corpus.

**Fix.** Strip `\r`, `\n`, space and tab from both ends. Note the reference implementations
use exactly that ASCII set, deliberately — a full Unicode-whitespace `trim()` is a superset
and will diverge on exotic input.

---

### Trap 10 — the weight file

- **8-byte alignment.** The f64 payload starts at the next multiple of 8 after the vocab
  bytes, *not* immediately after them. For the default vocab that is offset 64, not 58.
- **Tensor order is fixed:** `wte`, `wpe`, `lm_head`, then per layer `wq, wk, wv, wo, fc1,
  fc2`. Row-major. Get this wrong and `weights_sum` still matches (addition is commutative)
  while everything downstream is garbage — so also check the perplexity, not just the sum.
- **Little-endian, always.** Decode explicitly rather than casting a pointer, unless you
  are certain of the host.
- `mlp_fc1` is `[4E][E]` and `mlp_fc2` is `[E][4E]` — the shapes are not symmetric.

---

### Trap 11 — flags, and benchmarking a flag instead of a language

The first three-way run had C++ at `-O2` while Rust used its release default of
`opt-level = 3`. That made Rust look ~20% faster than C++. It was the flag. At matched O3 the
two are within ~3%.

**Fix.** Match optimization intent across implementations, and state the flags in your JSON
`build` field so the comparison is auditable.

---

### Trap 12 — output plumbing

- **Only the JSON goes to stdout.** The harness parses stdout; a stray log line breaks it.
  Diagnostics, warnings and errors go to stderr.
- **Floats must round-trip.** Emit shortest-round-trip (Rust `{}`, Python `repr`) or 17
  significant digits (C++ `setprecision(17)`). Fewer digits silently truncates the values the
  harness compares.
- **No `NaN`/`Infinity` literals** — they are not valid JSON.
- **Integer widths.** C++ `std::stoi` narrows `--seed` to `int`; a seed above 2^31 would
  wrap. Parse into a 64-bit type.
- **Paths.** Windows `os.path.relpath` throws across drive letters. The Python reference
  crashed on any weights file outside the repo drive until the shape sweep — which writes to
  a temp dir on `C:` — caught it.

---

## 4. Wiring into the harness

Implement the CLI in `BENCHMARK.md` §6 exactly, then register your port in the `IMPLS` dict
in `bench/run.py`:

```python
"mylang": dict(
    label="MyLang",
    run=[p("impl", "mylang", "build", "microgpt_infer" + EXE)],
    build=["mylang-compile", "-O3", ...],
    build_needs=lambda: shutil.which("mylang-compile") is not None,
    build_missing="mylang-compile not on PATH",
    probe=lambda: os.path.exists(p("impl", "mylang", "build", "microgpt_infer" + EXE)),
    missing="not built yet - run with --build",
    src=p("impl", "mylang", "main.ml"),
),
```

Then add it to `IMPLS` in `tools/test_shapes.py` too, so the shape sweep covers it.

Two structural requirements that are easy to miss:

- **`gen` runs two passes.** A deterministic pass of exactly `--samples` samples produces the
  hash and sample text; then `--repeats` time-budgeted passes produce the throughput. Reseed
  the RNG at the start of every pass.
- **`ppl` runs two passes.** One full pass over the evaluation set for the perplexity number,
  then time-budgeted passes cycling the corpus for throughput.

`gen.tokens`, `gen.output_fnv1a64` and `ppl.tokens` must come from the deterministic passes,
never the timed ones, or they will not be comparable.

---

## 5. Making your timing numbers mean something

Correctness first — but once you get to timing, four things caused errors of 1.5x–2.5x here,
which is larger than most language differences you might be trying to measure:

1. **Equal time, not equal work.** The same forward pass measured 3,449 tok/s in a 0.7-second
   workload and 1,601 tok/s in a 23-second one. Give every implementation the same
   `--time-budget` per repeat and report a **rate**.
2. **Pin to one core.** On hybrid CPUs (Intel P/E-cores, ARM big.LITTLE) the scheduler moves
   sustained single-threaded work onto an efficiency core mid-run, worth 2–3x. `--pin 0`.
3. **One process per measurement.** Running `gen` then `ppl` in the same process made
   whichever ran second look ~1.5x slower in every language. Separate processes dropped
   measured spread from ~1.6x to ~1.03x.
4. **Watch the spread.** `bench/run.py` reports fastest/slowest repeat *rate* and warns above
   1.25x. Above that, the ratios are indicative only.

**The cross-check that tells you the timing is sound:** `gen` and `ppl` throughput must agree
within an implementation, because they run the same forward pass. When they disagree, the
measurement is broken — that disagreement is what exposed both timing bugs in this project.

---

## 6. What the existing checks do NOT cover

Do not read a green suite as proof of a perfect port.

- **One platform per language.** g++ 15.2 (MSYS2 UCRT), rustc 1.98, CPython 3.14.3, all on
  x86-64 Windows. A different libm may shift `exp`/`log`/`pow` by an ulp. The 1e-9 tolerance
  absorbs that, but nothing here proves portability across platforms.
- **The `block_size` truncation branch is never exercised.** The longest name is 15
  characters, so `n = min(block_size, len(tokens)-1)` never actually truncates. A bug in that
  branch is invisible in all three implementations.
- **No fuzzing.** Agreement is verified on 4 fixed shapes and one corpus, not over random
  weights, random documents or random settings.
- **NaN / Inf paths are unexercised.** Trap 4 was fixed by inspection, not by a test.
- **Malformed weight files** — nothing checks that implementations reject truncated or
  bad-magic input consistently.
- **Only `n_layer` up to 3 and `n_embd` up to 32** are covered by the sweep.

If you want stronger assurance, the highest-value additions would be a random-weights fuzz
comparison and a corpus containing a document long enough to trigger truncation.

---

## 7. Definition of done

```bash
python bench/run.py --build --check-only   # all implementations agree
python tools/test_shapes.py                # all 4 shapes, including yours
python tools/test_harness.py               # the gate itself still works
python bench/run.py --repeats 7 --time-budget 1.0
```

Your port is done when:

- [ ] `weights_sum` and `weights_abs_sum` match `config.json` **exactly**
- [ ] the four golden RNG draws match
- [ ] `check.first3_nll` matches to ~1e-15
- [ ] `ppl.perplexity` matches the reference to the last digit or two
- [ ] `ppl.tokens` is exactly 14307 and `gen.tokens` exactly 6021
- [ ] `gen.output_fnv1a64` is **identical** — the generated text is bit-for-bit the same
- [ ] all 4 shapes in `tools/test_shapes.py` pass, including the multi-layer ones
- [ ] `gen` and `ppl` throughput agree within ~2% for your implementation
- [ ] your build flags are recorded in the JSON `build` field

And one last thing: when your port agrees on everything, that is evidence, not proof. Both
adversarial code reviews of these implementations found real defects **after** the benchmark
was reporting full agreement. Have someone — or something — read your arithmetic.
