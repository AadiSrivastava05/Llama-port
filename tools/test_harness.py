"""
Test that the benchmark's correctness gate actually catches a broken port.

The gate in bench/run.py is the thing standing between "the C++ port is fast" and
"the C++ port is fast because it is computing the wrong thing". Feed it synthetic
reports with the bugs a real port would plausibly have, and assert it complains.

  python tools/test_harness.py
"""

import copy
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench"))
import run as harness  # noqa: E402

EXPECTED_CFG = dict(weights_sum=10.621326738609703, weights_abs_sum=516.1316394186422)

GOOD = {
    "impl": "python",
    "weights_sha256": "9bf6ac8636d6b4b4a24eededaca631086d617793f287e7e803437a5b2f7009b8",
    "config": dict(n_layer=1, n_embd=16, block_size=16, n_head=4, vocab_size=27, num_weights=4192),
    "check": dict(weights_sum=10.621326738609703, weights_abs_sum=516.1316394186422,
                  first3_nll=8.34567890123, rng_first4_u64=["0xbc5614c61020390e", "0x1", "0x2", "0x3"]),
    "gen": dict(tokens=5731, output_fnv1a64="0x6342ab3fc84fbfac", tokens_per_sec_best=4030.0),
    "ppl": dict(tokens=14307, perplexity=10.789614873298, nll_per_token=2.378584,
                tokens_per_sec_best=3973.0),
}

CASES = []


def case(name, expect_problem, mutate):
    CASES.append((name, expect_problem, mutate))


def mutated(**path_vals):
    def f(r):
        for dotted, val in path_vals.items():
            keys = dotted.split("__")
            d = r
            for k in keys[:-1]:
                d = d[k]
            d[keys[-1]] = val
        return r
    return f


# A correct port: only libm-level divergence.
case("identical", False, lambda r: r)
case("libm noise in perplexity (1e-14 rel)", False,
     mutated(ppl__perplexity=10.789614873298 * (1 + 1e-14)))

# Bugs a real port would actually have.
case("wrong weight layout (weights_sum off)", True, mutated(check__weights_sum=10.4))
# CPython >=3.12 compensates sum() for floats, which drifted this reference by 5e-15 and
# sat under the old 1e-12 tolerance. weights_sum must now match config.json exactly.
case("compensated summation drift (5e-15 in weights_sum)", True,
     mutated(check__weights_sum=10.621326738609756))
case("perplexity diverges (bad forward pass)", True, mutated(ppl__perplexity=10.9))
case("perplexity off in the 8th digit", True, mutated(ppl__perplexity=10.78961497))
case("RNG mismatch (stdlib rng instead of xorshift64*)", True,
     mutated(check__rng_first4_u64=["0xdeadbeef", "0x1", "0x2", "0x3"]))
case("sampling diverges (gen hash differs)", True, mutated(gen__output_fnv1a64="0xffffffffffffffff"))
case("fewer tokens scored (tokenizer/block_size bug)", True, mutated(ppl__tokens=14000))
case("generated a different number of tokens", True, mutated(gen__tokens=5000))
case("different config (loaded a different checkpoint)", True, mutated(config__n_embd=32))
case("different weight file", True, mutated(weights_sha256="0" * 64))
case("check.first3_nll diverges", True, mutated(check__first3_nll=8.35))


def main():
    failures = []
    for name, expect_problem, mutate in CASES:
        ref = copy.deepcopy(GOOD)
        cand = mutate(copy.deepcopy(GOOD))
        cand["impl"] = "candidate"
        problems, notes = harness.check_agreement(
            [("reference", ref), ("candidate", cand)], EXPECTED_CFG)
        caught = bool(problems)
        ok = caught == expect_problem
        status = "ok  " if ok else "FAIL"
        want = "flag" if expect_problem else "pass"
        got = f"{len(problems)} problem(s)" if caught else f"clean ({len(notes)} note(s))"
        print(f"{status} expect {want:4s} | {name:52s} -> {got}")
        if problems and ok and expect_problem:
            print(f"       {problems[0][:110]}")
        if not ok:
            failures.append(name)

    # the gate must also survive a single implementation with nothing to compare against
    problems, _ = harness.check_agreement([("solo", copy.deepcopy(GOOD))], EXPECTED_CFG)
    if problems:
        print(f"FAIL single implementation should not self-report problems: {problems}")
        failures.append("solo")
    else:
        print("ok   expect pass | single implementation, nothing to compare              -> clean")

    print()
    if failures:
        print(f"{len(failures)} gate test(s) FAILED: {', '.join(failures)}")
        return 1
    print(f"all {len(CASES) + 1} gate tests passed - the harness catches broken ports")
    return 0


if __name__ == "__main__":
    sys.exit(main())
