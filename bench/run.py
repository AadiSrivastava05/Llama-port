"""
Benchmark harness: build every available microgpt implementation, run them under
identical settings, verify they agree numerically, and print the comparison.

  python bench/run.py --build          # build C++/Rust, then run everything
  python bench/run.py --check-only     # agreement only, no timing
  python bench/run.py --impls python,cpp --samples 2000 --repeats 5

Contract for adding an implementation: see BENCHMARK.md. Register it in IMPLS below.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "bench", "results")
EXE = ".exe" if os.name == "nt" else ""

# Tolerances for cross-implementation agreement (see BENCHMARK.md section 8)
PPL_REL_TOL = 1e-9
NLL_REL_TOL = 1e-9

# Flag a run whose slowest repeat rate is this much worse than its fastest: either
# something else used the CPU, or the machine dropped clock partway through.
NOISE_THRESHOLD = 1.25


def p(*parts):
    return os.path.join(ROOT, *parts)


IMPLS = {
    "python": dict(
        label="Python",
        run=[sys.executable, p("impl", "python", "infer.py")],
        build=None,
        probe=lambda: os.path.exists(p("impl", "python", "infer.py")),
        missing="impl/python/infer.py not found",
    ),
    "cpp": dict(
        label="C++",
        run=[p("impl", "cpp", "build", "microgpt_infer" + EXE)],
        build=[
            # -O3, not -O2, to match Rust's release profile (opt-level 3). Comparing
            # g++ -O2 against Rust opt-level 3 measured the flag, not the language:
            # it cost C++ ~20% and made Rust look faster than it is.
            "g++", "-O3", "-std=c++20", "-fno-fast-math", "-ffp-contract=off",
            "-o", p("impl", "cpp", "build", "microgpt_infer" + EXE),
            p("impl", "cpp", "infer.cpp"),
        ],
        build_needs=lambda: shutil.which("g++") is not None,
        build_missing="g++ not on PATH",
        probe=lambda: os.path.exists(p("impl", "cpp", "build", "microgpt_infer" + EXE)),
        missing="not built yet - run with --build (needs impl/cpp/infer.cpp)",
        src=p("impl", "cpp", "infer.cpp"),
    ),
    "rust": dict(
        label="Rust",
        run=[p("impl", "rust", "target", "release", "microgpt_infer" + EXE)],
        build=["cargo", "build", "--release", "--manifest-path", p("impl", "rust", "Cargo.toml")],
        build_needs=lambda: shutil.which("cargo") is not None,
        build_missing="cargo not on PATH - install Rust to enable this implementation",
        probe=lambda: os.path.exists(p("impl", "rust", "target", "release", "microgpt_infer" + EXE)),
        missing="not built yet - run with --build (needs impl/rust/src/main.rs)",
        src=p("impl", "rust", "src", "main.rs"),
    ),
}


def build(name, spec, verbose):
    if spec["build"] is None:
        return True, "nothing to build"
    src = spec.get("src")
    if src and not os.path.exists(src):
        return False, f"source missing: {os.path.relpath(src, ROOT)}"
    needs = spec.get("build_needs")
    if needs and not needs():
        return False, spec.get("build_missing", "toolchain missing")
    os.makedirs(os.path.dirname(spec["run"][0]), exist_ok=True)
    t0 = time.perf_counter()
    proc = subprocess.run(spec["build"], cwd=ROOT, capture_output=True, text=True)
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        return False, f"build failed:\n{proc.stdout}\n{proc.stderr}".strip()
    if verbose and (proc.stdout or proc.stderr):
        sys.stderr.write(proc.stdout + proc.stderr)
    return True, f"built in {dt:.1f}s"


def run_one_mode(name, spec, args, mode):
    out_json = os.path.join(RESULTS, f"{name}.{mode}.json")
    cmd = list(spec["run"]) + [
        "--weights", p("weights", "microgpt.bin"),
        "--data", p("data", args.data),
        "--mode", mode,
        "--samples", str(args.samples),
        "--temperature", str(args.temperature),
        "--seed", str(args.seed),
        "--repeats", str(args.repeats),
        "--max-docs", str(args.max_docs),
        "--time-budget", str(args.time_budget),
        "--pin", str(args.pin),
        "--json", out_json,
    ]
    # Give the timed process the best shot at an uncontended core. This machine
    # showed 2-3x swings between repeats without it.
    extra = {}
    if os.name == "nt":
        extra["creationflags"] = subprocess.HIGH_PRIORITY_CLASS
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **extra)
    if proc.returncode != 0:
        return None, f"{mode} run failed (exit {proc.returncode}):\n{proc.stderr.strip()[:2000]}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError as e:
        if os.path.exists(out_json):
            with open(out_json) as f:
                return json.load(f), None
        return None, f"{mode}: invalid JSON on stdout ({e}); stderr:\n{proc.stderr.strip()[:1000]}"


def run_impl(name, spec, args):
    """Run each measurement in its OWN process.

    Running gen and ppl back to back in one process makes whichever goes second look
    slower: by then the process has been busy for ~10s and the CPU has dropped clock
    or been moved to an efficiency core. Measured here, that alone depressed ppl
    throughput by ~1.5x in both languages. A fresh process per measurement removes the
    ordering bias, and gen/ppl then agree with each other as they should.
    """
    modes = ["check"] if args.check_only else ["check", "gen", "ppl"]
    merged = None
    for mode in modes:
        r, err = run_one_mode(name, spec, args, mode)
        if err:
            return None, err
        if merged is None:
            merged = r
            merged["mode"] = "all" if len(modes) > 1 else mode
        elif mode in r:
            merged[mode] = r[mode]
    with open(os.path.join(RESULTS, f"{name}.json"), "w") as f:
        json.dump(merged, f, indent=2)
    return merged, None


def spread(section):
    """fastest/slowest repeat rate - a proxy for how contaminated the timing was.

    Repeats are time-budgeted, so every repeat takes the same number of seconds and
    it is the throughput that varies. Compare rates, not durations.
    """
    if not section:
        return None
    rates = section.get("rates_per_repeat") or []
    if not rates:
        secs = section.get("seconds_per_repeat") or []
        if len(secs) < 2 or min(secs) <= 0:
            return None
        return max(secs) / min(secs)
    if len(rates) < 2 or min(rates) <= 0:
        return None
    return max(rates) / min(rates)


def rel(a, b):
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom else abs(a - b)


def check_agreement(reports, expected_cfg):
    """Returns (list of problem strings, list of note strings)."""
    problems, notes = [], []
    ref_name, ref = reports[0]

    for name, r in reports:
        if r["weights_sha256"] != ref["weights_sha256"]:
            problems.append(f"{name}: weights sha256 differs from {ref_name}")
        for k in ("n_layer", "n_embd", "block_size", "n_head", "vocab_size", "num_weights"):
            if r["config"].get(k) != ref["config"].get(k):
                problems.append(f"{name}: config.{k}={r['config'].get(k)} != {ref['config'].get(k)}")

        c = r.get("check")
        if c:
            if expected_cfg:
                # Exact, not approximate. Summing the same f64 values in the same order
                # is deterministic under IEEE754, so any difference means the port is
                # not doing what the contract says. The usual culprit is a language
                # whose sum() silently compensates - CPython's does since 3.12, which
                # drifted this reference by 5e-15 and hid under a 1e-12 tolerance.
                for k in ("weights_sum", "weights_abs_sum"):
                    if k in c and k in expected_cfg and c[k] != expected_cfg[k]:
                        problems.append(
                            f"{name}: {k}={c[k]!r} != weights/config.json {expected_cfg[k]!r} "
                            f"(rel {rel(c[k], expected_cfg[k]):.1e}) - weight layout bug, or "
                            f"compensated instead of plain sequential summation")
            rc, rr = ref.get("check", {}), c
            if rc.get("rng_first4_u64") and rr.get("rng_first4_u64") != rc.get("rng_first4_u64"):
                problems.append(f"{name}: RNG draws differ from {ref_name} - xorshift64* mismatch")
            if rc.get("first3_nll") is not None and rr.get("first3_nll") is not None:
                d = rel(rr["first3_nll"], rc["first3_nll"])
                if d > NLL_REL_TOL:
                    problems.append(f"{name}: check.first3_nll differs from {ref_name} by {d:.2e}")

        if "ppl" in r and "ppl" in ref:
            d = rel(r["ppl"]["perplexity"], ref["ppl"]["perplexity"])
            if d > PPL_REL_TOL:
                problems.append(f"{name}: perplexity differs from {ref_name} by {d:.2e} rel "
                                f"({r['ppl']['perplexity']!r} vs {ref['ppl']['perplexity']!r})")
            elif name != ref_name and d > 0:
                notes.append(f"{name}: perplexity matches {ref_name} to {d:.1e} rel (libm noise)")
            if r["ppl"]["tokens"] != ref["ppl"]["tokens"]:
                problems.append(f"{name}: scored {r['ppl']['tokens']} tokens, "
                                f"{ref_name} scored {ref['ppl']['tokens']}")

        if "gen" in r and "gen" in ref and name != ref_name:
            if r["gen"]["output_fnv1a64"] != ref["gen"]["output_fnv1a64"]:
                problems.append(f"{name}: generated text hash {r['gen']['output_fnv1a64']} != "
                                f"{ref_name} {ref['gen']['output_fnv1a64']} - sampling diverges")
            if r["gen"]["tokens"] != ref["gen"]["tokens"]:
                problems.append(f"{name}: generated {r['gen']['tokens']} tokens, "
                                f"{ref_name} generated {ref['gen']['tokens']}")
    return problems, notes


def fmt_table(rows, headers, aligns=None):
    aligns = aligns or ["<"] * len(headers)
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) if rows else len(headers[i])
              for i in range(len(headers))]
    line = "  ".join(f"{headers[i]:{aligns[i]}{widths[i]}}" for i in range(len(headers)))
    sep = "  ".join("-" * widths[i] for i in range(len(headers)))
    body = ["  ".join(f"{r[i]:{aligns[i]}{widths[i]}}" for i in range(len(headers))) for r in rows]
    return "\n".join([line, sep] + body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impls", default="", help="comma-separated subset, default all registered")
    ap.add_argument("--build", action="store_true", help="build compiled implementations first")
    ap.add_argument("--check-only", action="store_true", help="agreement checks, skip timing")
    ap.add_argument("--data", default="val.txt", help="file under data/")
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=2.0,
                    help="wall-clock seconds per timed repeat; equalizes CPU clock state")
    ap.add_argument("--pin", type=int, default=0,
                    help="pin each implementation to this logical CPU (-1 disables)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    wpath = p("weights", "microgpt.bin")
    if not os.path.exists(wpath):
        sys.exit("weights/microgpt.bin missing - run:  python tools/train_export.py --steps 1000")
    with open(p("weights", "config.json")) as f:
        expected_cfg = json.load(f)

    names = [n.strip() for n in args.impls.split(",") if n.strip()] or list(IMPLS)
    for n in names:
        if n not in IMPLS:
            sys.exit(f"unknown implementation {n!r}; known: {', '.join(IMPLS)}")

    print("microgpt port benchmark")
    print(f"weights  {os.path.relpath(wpath, ROOT)}  sha256={expected_cfg['sha256'][:16]}...  "
          f"{expected_cfg['num_weights']} f64  ({expected_cfg['steps_done']} training steps)")
    print(f"data     data/{args.data}")
    print(f"settings samples={args.samples} temperature={args.temperature} seed={args.seed} "
          f"repeats={args.repeats}\n")

    if args.build:
        for n in names:
            ok, msg = build(n, IMPLS[n], args.verbose)
            print(f"build {n:8s} {'ok  ' if ok else 'SKIP'} {msg}")
        print()

    reports, skipped = [], []
    for n in names:
        spec = IMPLS[n]
        if not spec["probe"]():
            skipped.append((n, spec["missing"]))
            continue
        r, err = run_impl(n, spec, args)
        if err:
            skipped.append((n, err))
            continue
        reports.append((n, r))
        print(f"ran   {n:8s} ok")

    if skipped:
        print()
        for n, why in skipped:
            print(f"skip  {n:8s} {why}")
    if not reports:
        sys.exit("\nno implementations ran")

    problems, notes = check_agreement(reports, expected_cfg)

    print("\n" + "=" * 100)
    print("CORRECTNESS")
    print("=" * 100)
    rows = []
    for n, r in reports:
        c = r.get("check", {})
        ppl = r.get("ppl", {})
        rows.append([
            n,
            r["weights_sha256"][:12],
            f"{c.get('weights_sum', float('nan')):.9f}",
            (c.get("rng_first4_u64") or ["-"])[0],
            f"{ppl.get('perplexity', float('nan')):.10f}" if ppl else "-",
            r.get("gen", {}).get("output_fnv1a64", "-"),
        ])
    print(fmt_table(rows, ["impl", "weights", "weights_sum", "rng[0]", "perplexity", "gen hash"]))
    for note in notes:
        print(f"\nnote: {note}")
    if problems:
        print("\nFAILED AGREEMENT:")
        for prob in problems:
            print(f"  - {prob}")
    elif len(reports) > 1:
        print("\nall implementations agree within tolerance")

    if not args.check_only:
        base_gen = next((r["gen"]["tokens_per_sec_best"] for _, r in reports if "gen" in r), None)
        base_ppl = next((r["ppl"]["tokens_per_sec_best"] for _, r in reports if "ppl" in r), None)
        print("\n" + "=" * 100)
        print(f"SPEED  (tok/s = forward passes per second, best of {args.repeats} "
              f"time-budgeted {args.time_budget}s repeats; higher is better)")
        print("=" * 100)
        rows, noisy = [], []
        for n, r in reports:
            g, pp = r.get("gen"), r.get("ppl")
            sp = max([s for s in (spread(g), spread(pp)) if s is not None], default=None)
            if sp is not None and sp > NOISE_THRESHOLD:
                noisy.append((n, sp))
            rows.append([
                n,
                r.get("runtime", "?")[:22],
                f"{g['tokens_per_sec_best']:,.0f}" if g else "-",
                f"{g['tokens_per_sec_best']/base_gen:.2f}x" if g and base_gen else "-",
                f"{pp['tokens_per_sec_best']:,.0f}" if pp else "-",
                f"{pp['tokens_per_sec_best']/base_ppl:.2f}x" if pp and base_ppl else "-",
                f"{r.get('load_seconds', 0)*1000:.1f}ms",
                f"{sp:.2f}x" if sp is not None else "-",
                f"{pp['perplexity']:.4f}" if pp else "-",
            ])
        print(fmt_table(
            rows,
            ["impl", "runtime", "gen tok/s", "vs base", "ppl tok/s", "vs base", "load",
             "spread", "ppl"],
            ["<", "<", ">", ">", ">", ">", ">", ">", ">"]))

        if noisy:
            print("\nWARNING: timing spread (slowest/fastest repeat) exceeded "
                  f"{NOISE_THRESHOLD:.2f}x for: "
                  + ", ".join(f"{n} ({s:.2f}x)" for n, s in noisy))
            print("  Either another process competed for CPU, or this machine drops clock")
            print("  / migrates to an efficiency core partway through a run. The 'best'")
            print("  column is the least contaminated estimate; re-run with more --repeats")
            print("  or close other load before quoting a precise ratio.")
            print("  Cross-check: gen and ppl run the same forward pass, so their tok/s")
            print("  should agree within an implementation. If they do not, distrust both.")

        first = reports[0][1].get("gen", {}).get("first_samples", [])
        if first:
            print(f"\nsample output ({reports[0][0]}): {' '.join(first[:14])}")

    summary = dict(
        weights_sha256=expected_cfg["sha256"], settings=vars(args),
        implementations={n: r for n, r in reports},
        skipped={n: why for n, why in skipped},
        agreement_problems=problems, agreement_notes=notes,
    )
    out = os.path.join(RESULTS, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {os.path.relpath(out, ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
