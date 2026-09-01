"""
Verify the ports agree across model SHAPES, not just the default one.

The shipped checkpoint has n_layer=1, so the transformer's layer loop body executes
exactly once and its per-layer indexing is never really exercised. A port that mixed
up a layer's KV cache, or that read the tensor stream in the wrong order for layer 2,
would pass the default benchmark and still be wrong.

This trains throwaway models at several shapes (weight quality is irrelevant - only
agreement matters) and checks every implementation against the Python reference.

  python tools/test_shapes.py
  python tools/test_shapes.py --keep    # leave the temp models on disk
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE = ".exe" if os.name == "nt" else ""

# (n_layer, n_embd, n_head, steps) - steps are tiny on purpose
SHAPES = [
    (1, 16, 4, 5),    # the default shape, as a control
    (2, 16, 4, 5),    # multi-layer: the case the default checkpoint cannot exercise
    (3, 32, 8, 3),    # deeper, wider, more heads
    (2, 8, 1, 5),     # single head, narrow
]

IMPLS = {
    "python": [sys.executable, os.path.join(ROOT, "impl", "python", "infer.py")],
    "cpp": [os.path.join(ROOT, "impl", "cpp", "build", "microgpt_infer" + EXE)],
    "rust": [os.path.join(ROOT, "impl", "rust", "target", "release", "microgpt_infer" + EXE)],
}


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--max-docs", type=int, default=150)
    ap.add_argument("--samples", type=int, default=60)
    args = ap.parse_args()

    available = {n: c for n, c in IMPLS.items()
                 if n == "python" or os.path.exists(c[0])}
    missing = [n for n in IMPLS if n not in available]
    if missing:
        print(f"note: skipping unbuilt implementations: {', '.join(missing)}\n")
    if len(available) < 2:
        print("need at least two implementations to compare; build the ports first")
        return 1

    tmp = tempfile.mkdtemp(prefix="microgpt_shapes_")
    failures = []
    try:
        for n_layer, n_embd, n_head, steps in SHAPES:
            tag = f"L{n_layer}_E{n_embd}_H{n_head}"
            out_dir = os.path.join(tmp, tag)
            os.makedirs(out_dir, exist_ok=True)

            print(f"--- {tag}: training {steps} steps ...", end=" ", flush=True)
            proc = run([sys.executable, os.path.join(ROOT, "tools", "train_export.py"),
                        "--steps", str(steps), "--n-layer", str(n_layer),
                        "--n-embd", str(n_embd), "--n-head", str(n_head),
                        "--val-docs", "300", "--ckpt-every", "0", "--out-dir", out_dir])
            if proc.returncode != 0:
                print("FAILED")
                print(proc.stderr[:1500])
                failures.append(f"{tag}: training failed")
                continue
            weights = os.path.join(out_dir, "weights", "microgpt.bin")
            data = os.path.join(out_dir, "data", "val.txt")
            cfg = json.load(open(os.path.join(out_dir, "weights", "config.json")))
            print(f"{cfg['num_params']} params")

            results = {}
            for name, base in available.items():
                proc = run(base + ["--weights", weights, "--data", data, "--mode", "all",
                                   "--samples", str(args.samples), "--repeats", "1",
                                   "--time-budget", "0.05", "--max-docs", str(args.max_docs)])
                if proc.returncode != 0:
                    failures.append(f"{tag}/{name}: run failed - {proc.stderr.strip()[:300]}")
                    print(f"    {name:8s} RUN FAILED")
                    continue
                results[name] = json.loads(proc.stdout)

            if "python" not in results or len(results) < 2:
                continue
            ref = results["python"]
            for name, r in results.items():
                ppl = r["ppl"]["perplexity"]
                ref_ppl = ref["ppl"]["perplexity"]
                d = abs(ppl - ref_ppl) / max(abs(ppl), abs(ref_ppl), 1e-300)
                same_hash = r["gen"]["output_fnv1a64"] == ref["gen"]["output_fnv1a64"]
                ok = d < 1e-9 and same_hash and r["gen"]["tokens"] == ref["gen"]["tokens"]
                print(f"    {name:8s} ppl={ppl:.10f} ({d:.1e} rel)  hash={r['gen']['output_fnv1a64']}"
                      f"  {'ok' if ok else 'MISMATCH'}")
                if not ok and name != "python":
                    why = []
                    if d >= 1e-9:
                        why.append(f"perplexity differs by {d:.2e}")
                    if not same_hash:
                        why.append("generated text differs")
                    if r["gen"]["tokens"] != ref["gen"]["tokens"]:
                        why.append(f"token count {r['gen']['tokens']} vs {ref['gen']['tokens']}")
                    failures.append(f"{tag}/{name}: " + "; ".join(why))
            print()
    finally:
        if args.keep:
            print(f"temp models kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"{len(failures)} SHAPE FAILURE(S):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"all {len(SHAPES)} shapes agree across {', '.join(available)} "
          f"- multi-layer indexing and tensor ordering verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
