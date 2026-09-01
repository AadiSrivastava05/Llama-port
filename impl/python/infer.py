"""
microgpt inference — pure Python, plain floats, no autograd, no numpy.

This is the reference implementation of the benchmark contract in BENCHMARK.md.
The C++ and Rust ports must reproduce its numbers. Everything is written in the
same operation order as microgpt.py so that the three languages agree bit-for-bit
(or very nearly: libm exp/log/pow may differ by an ulp).

Usage:
  python impl/python/infer.py --mode all
  python impl/python/infer.py --mode gen --samples 2000 --repeats 5
"""

import argparse
import array
import hashlib
import json
import math
import os
import platform
import struct
import sys
import time

IMPL_NAME = "python"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
U64 = 0xFFFFFFFFFFFFFFFF

# Exactly the characters the compiled ports strip from each data line. Bare
# str.strip() also removes \v, \f and Unicode spaces, which would tokenize
# differently from the C++ and Rust ports on a non-ASCII corpus.
STRIP_CHARS = "\r\n \t"


# ---------------------------------------------------------------- rng (spec'd)

class Rng:
    """xorshift64* — identical integer arithmetic is required of every port."""

    __slots__ = ("state",)

    def __init__(self, seed):
        self.state = seed & U64 if (seed & U64) else 0x9E3779B97F4A7C15

    def next_u64(self):
        x = self.state
        x ^= (x >> 12)
        x ^= (x << 25) & U64
        x ^= (x >> 27)
        self.state = x
        return (x * 0x2545F4914F6CDD1D) & U64

    def next_f64(self):
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))


def short_path(path):
    """Repo-relative when possible. On Windows relpath raises across drive letters,
    which happens whenever weights live outside the repo (a temp dir, another disk)."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return os.path.abspath(path)


def pin_cpu(n):
    """Pin this process to one logical CPU.

    On Intel hybrid parts (P-cores + E-cores) the scheduler migrates sustained
    single-threaded work onto an E-core, which changes throughput by 2-3x mid-run
    and makes cross-language comparison meaningless. Logical CPU 0 is a P-core on
    every hybrid layout in practice, since P-cores enumerate first.
    """
    if n < 0:
        return "not pinned"
    try:
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            k.GetCurrentProcess.restype = ctypes.c_void_p
            k.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            k.SetProcessAffinityMask.restype = ctypes.c_int
            if not k.SetProcessAffinityMask(k.GetCurrentProcess(), 1 << n):
                return f"pin to cpu{n} failed"
            return f"cpu{n}"
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {n})
            return f"cpu{n}"
    except Exception as e:  # pinning is best-effort; never fail the benchmark over it
        return f"pin failed: {e}"
    return "not pinned"


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & U64
    return h


# ------------------------------------------------------------------- weights

class Model:
    def __init__(self, path):
        with open(path, "rb") as f:
            blob = f.read()
        self.sha256 = hashlib.sha256(blob).hexdigest()
        if blob[:4] != b"MGPT":
            raise ValueError(f"bad magic {blob[:4]!r}, not a microgpt-bin file")
        (version, self.n_layer, self.n_embd, self.block_size,
         self.n_head, self.vocab_size, n_uchars) = struct.unpack_from("<7I", blob, 4)
        if version != 1:
            raise ValueError(f"unsupported format version {version}")
        off = 4 + 7 * 4
        self.uchars = blob[off:off + n_uchars].decode("ascii")
        off += n_uchars
        off += (-off) % 8                       # align to 8 bytes
        self.head_dim = self.n_embd // self.n_head
        self.bos = n_uchars
        self.char_to_id = {c: i for i, c in enumerate(self.uchars)}

        flat = array.array("d")
        flat.frombytes(blob[off:])
        if sys.byteorder != "little":
            flat.byteswap()

        self.tensors = {}
        cursor = 0

        def take(name, nout, nin):
            nonlocal cursor
            mat = [list(flat[cursor + r * nin: cursor + (r + 1) * nin]) for r in range(nout)]
            cursor += nout * nin
            self.tensors[name] = mat
            return mat

        E, V, B, L = self.n_embd, self.vocab_size, self.block_size, self.n_layer
        self.wte = take("wte", V, E)
        self.wpe = take("wpe", B, E)
        self.lm_head = take("lm_head", V, E)
        self.layers = []
        for i in range(L):
            self.layers.append(dict(
                wq=take(f"layer{i}.attn_wq", E, E),
                wk=take(f"layer{i}.attn_wk", E, E),
                wv=take(f"layer{i}.attn_wv", E, E),
                wo=take(f"layer{i}.attn_wo", E, E),
                fc1=take(f"layer{i}.mlp_fc1", 4 * E, E),
                fc2=take(f"layer{i}.mlp_fc2", E, 4 * E),
            ))
        if cursor != len(flat):
            raise ValueError(f"weight file has {len(flat)} f64 but model wants {cursor}")
        self.num_weights = cursor
        ws = 0.0
        wabs = 0.0
        for w in flat:                      # plain accumulation, matching the exporter
            ws += w
            wabs += abs(w)
        self.weights_sum = ws
        self.weights_abs_sum = wabs

    def config(self):
        return dict(n_layer=self.n_layer, n_embd=self.n_embd, block_size=self.block_size,
                    n_head=self.n_head, head_dim=self.head_dim, vocab_size=self.vocab_size,
                    bos_token_id=self.bos, uchars=self.uchars, num_weights=self.num_weights)

    def tokenize(self, doc):
        """[BOS] + chars + [BOS], exactly as microgpt.py trains."""
        return [self.bos] + [self.char_to_id[c] for c in doc] + [self.bos]

    # ------------------------------------------------------------- forward

    def forward(self, token_id, pos_id, keys, values):
        E, H, HD = self.n_embd, self.n_head, self.head_dim
        tok_emb = self.wte[token_id]
        pos_emb = self.wpe[pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(self.n_layer):
            layer = self.layers[li]
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, layer["wq"])
            k = linear(x, layer["wk"])
            v = linear(x, layer["wv"])
            keys[li].append(k)
            values[li].append(v)
            k_cache = keys[li]
            v_cache = values[li]
            T = len(k_cache)
            scale = HD ** 0.5
            x_attn = []
            for h in range(H):
                hs = h * HD
                he = hs + HD
                q_h = q[hs:he]
                attn_logits = [None] * T
                for t in range(T):
                    kt = k_cache[t]
                    acc = 0.0
                    for j in range(HD):
                        acc += q_h[j] * kt[hs + j]
                    attn_logits[t] = acc / scale
                attn_weights = softmax(attn_logits)
                for j in range(HD):
                    acc = 0.0
                    for t in range(T):
                        acc += attn_weights[t] * v_cache[t][hs + j]
                    x_attn.append(acc)
            x = linear(x_attn, layer["wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, layer["fc1"])
            x = [xi if xi > 0.0 else 0.0 for xi in x]
            x = linear(x, layer["fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        return linear(x, self.lm_head)

    def new_cache(self):
        return [[] for _ in range(self.n_layer)], [[] for _ in range(self.n_layer)]


# NOTE: these deliberately accumulate with an explicit `acc +=` loop rather than the
# idiomatic sum(...). Since 3.12 CPython's sum() applies Neumaier compensated
# summation to floats, which is *more* accurate than a plain running total - and
# therefore not what C++, Rust, or microgpt.py itself compute. (microgpt.py's sum()
# runs over Value objects, so it never hits CPython's float fast path.) Measured on
# the shipped checkpoint, compensation changed 141 of 214 dot products. BENCHMARK.md
# section 3 mandates plain sequential accumulation; this is that.

def linear(x, w):
    out = []
    for wo in w:
        acc = 0.0
        for wi, xi in zip(wo, x):
            acc += wi * xi
        out.append(acc)
    return out


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = 0.0
    for e in exps:
        total += e
    return [e / total for e in exps]


def rmsnorm(x):
    ms = 0.0
    for xi in x:
        ms += xi * xi
    ms /= len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def sample_from(probs, rng):
    """Cumulative-sum sampling; every port must scan in this exact order."""
    total = 0.0
    cum = [0.0] * len(probs)
    for i, p in enumerate(probs):
        total += p
        cum[i] = total
    u = rng.next_f64() * total
    for i in range(len(cum)):
        if u < cum[i]:
            return i
    return len(cum) - 1


# ----------------------------------------------------------------- benchmarks

def rate_stats(tok_counts, seconds):
    """Throughput is measured as a rate, because time-budgeted repeats do differing work."""
    rates = [t / s for t, s in zip(tok_counts, seconds)]
    ordered = sorted(rates)
    return dict(rates_per_repeat=rates,
                tokens_per_repeat=tok_counts,
                seconds_per_repeat=seconds,
                tokens_per_sec_best=max(rates),
                tokens_per_sec_median=ordered[len(ordered) // 2])


def gen_one(model, temperature, rng, collect=None):
    """Generate a single sample. Returns the number of forward passes it took."""
    keys, values = model.new_cache()
    token_id = model.bos
    n = 0
    buf = [] if collect is not None else None
    for pos_id in range(model.block_size):
        logits = model.forward(token_id, pos_id, keys, values)
        n += 1
        probs = softmax([l / temperature for l in logits])
        token_id = sample_from(probs, rng)
        if token_id == model.bos:
            break
        if buf is not None:
            buf.append(model.uchars[token_id])
    if collect is not None:
        collect.append("".join(buf))
    return n


def bench_gen(model, samples, temperature, seed, repeats, budget):
    # 1. Deterministic pass: fixed sample count, produces the cross-language hash.
    #    Doubles as warmup so the timed passes below start hot.
    rng = Rng(seed)
    texts = []
    tokens = 0
    for _ in range(samples):
        tokens += gen_one(model, temperature, rng, texts)
    body = "\n".join(texts)

    # 2. Timed passes: every implementation runs for the same wall-clock budget, so
    #    all of them are measured under the same CPU clock state. Comparing a
    #    20-second Python run against a 40-millisecond C++ run measures thermal
    #    throttling as much as it measures the language.
    tok_counts, seconds = [], []
    for _ in range(repeats):
        rng = Rng(seed)
        n_tok = 0
        t0 = time.perf_counter()
        while True:
            n_tok += gen_one(model, temperature, rng)
            elapsed = time.perf_counter() - t0
            if elapsed >= budget:
                break
        tok_counts.append(n_tok)
        seconds.append(elapsed)

    out = dict(samples=samples, temperature=temperature, seed=seed, repeats=repeats,
               time_budget=budget, tokens=tokens, chars=sum(len(s) for s in texts),
               output_fnv1a64=f"0x{fnv1a64(body.encode('ascii')):016x}",
               first_samples=texts[:20])
    out.update(rate_stats(tok_counts, seconds))
    return out


def score_doc(model, doc):
    """Teacher-forced NLL for one document. Returns (nll, tokens scored)."""
    tokens = model.tokenize(doc)
    n = min(model.block_size, len(tokens) - 1)
    keys, values = model.new_cache()
    nll = 0.0
    for pos_id in range(n):
        logits = model.forward(tokens[pos_id], pos_id, keys, values)
        probs = softmax(logits)
        nll -= math.log(probs[tokens[pos_id + 1]])
    return nll, n


def bench_ppl(model, docs, repeats, budget):
    # 1. Quality pass over the exact evaluation set - this is what perplexity means.
    nll_total, n_tok = 0.0, 0
    t0 = time.perf_counter()
    for doc in docs:
        nll, n = score_doc(model, doc)
        nll_total += nll
        n_tok += n
    full_seconds = time.perf_counter() - t0

    # 2. Timed passes under a shared wall-clock budget, cycling the corpus.
    tok_counts, seconds = [], []
    for _ in range(repeats):
        n = 0
        i = 0
        t0 = time.perf_counter()
        while True:
            _, k = score_doc(model, docs[i % len(docs)])
            n += k
            i += 1
            elapsed = time.perf_counter() - t0
            if elapsed >= budget:
                break
        tok_counts.append(n)
        seconds.append(elapsed)

    nll_per_token = nll_total / n_tok
    out = dict(docs=len(docs), tokens=n_tok, repeats=repeats, time_budget=budget,
               nll_total=nll_total, nll_per_token=nll_per_token,
               perplexity=math.exp(nll_per_token),
               bits_per_token=nll_per_token / math.log(2.0),
               full_pass_seconds=full_seconds)
    out.update(rate_stats(tok_counts, seconds))
    return out


def bench_check(model, docs, seed):
    """Deterministic numeric fingerprints, for cross-language agreement checks."""
    prompt = "emma"
    tokens = model.tokenize(prompt)[:-1]          # BOS e m m a
    keys, values = model.new_cache()
    logits = None
    for pos_id, tid in enumerate(tokens):
        logits = model.forward(tid, pos_id, keys, values)
    probs = softmax(logits)
    # Reuse score_doc so this fingerprint groups its summation exactly the way the
    # perplexity path does - per document, then added. Folding every token into one
    # running total instead sums in a different order and shifts the last digits,
    # which is a spurious cross-language mismatch waiting to happen.
    sub = docs[:3]
    nll = 0.0
    n = 0
    for doc in sub:
        d_nll, d_n = score_doc(model, doc)
        nll += d_nll
        n += d_n
    rng = Rng(seed)
    draws = [rng.next_u64() for _ in range(4)]
    argmax = max(range(len(probs)), key=lambda i: probs[i])
    return dict(prompt=prompt, logits_after_prompt=logits, probs_after_prompt=probs,
                argmax_token=argmax,
                argmax_char="<BOS>" if argmax == model.bos else model.uchars[argmax],
                first3_docs=sub, first3_nll=nll, first3_tokens=n,
                rng_seed=seed, rng_first4_u64=[f"0x{d:016x}" for d in draws],
                weights_sum=model.weights_sum, weights_abs_sum=model.weights_abs_sum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights", "microgpt.bin"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "val.txt"))
    ap.add_argument("--mode", default="all", choices=["all", "gen", "ppl", "check"])
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all docs in --data")
    ap.add_argument("--pin", type=int, default=0,
                    help="pin to this logical CPU (-1 disables); avoids E-core migration")
    ap.add_argument("--time-budget", type=float, default=2.0,
                    help="wall-clock seconds per timed repeat; equalizes CPU clock state "
                         "across implementations")
    ap.add_argument("--json", default="", help="also write the JSON report here")
    args = ap.parse_args()

    cpu_pin = pin_cpu(args.pin)

    t0 = time.perf_counter()
    model = Model(args.weights)
    load_seconds = time.perf_counter() - t0

    docs = [d for d in (l.strip(STRIP_CHARS) for l in open(args.data)) if d]
    if args.max_docs:
        docs = docs[:args.max_docs]

    report = dict(impl=IMPL_NAME, mode=args.mode,
                  runtime=f"CPython {platform.python_version()}",
                  build="interpreted (no numpy, plain f64 lists)",
                  weights_sha256=model.sha256, weights_path=short_path(args.weights),
                  data_path=short_path(args.data),
                  config=model.config(), load_seconds=load_seconds, cpu_pin=cpu_pin)

    if args.mode in ("all", "check"):
        report["check"] = bench_check(model, docs, args.seed)
    if args.mode in ("all", "gen"):
        report["gen"] = bench_gen(model, args.samples, args.temperature, args.seed,
                                  args.repeats, args.time_budget)
    if args.mode in ("all", "ppl"):
        report["ppl"] = bench_ppl(model, docs, args.repeats, args.time_budget)

    text = json.dumps(report, indent=2)
    if args.json:
        with open(args.json, "w") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
