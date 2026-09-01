"""
Sanity-check the model's perplexity against classical baselines on the *same*
evaluation protocol (same splits, same tokenization, same block_size cap).

A transformer that cannot beat a bigram count model trained on the same 1,000
documents is not doing anything interesting, and a benchmark built on it would be
measuring speed on a broken model.

  python tools/baselines.py
"""

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "impl", "python"))
import infer  # noqa: E402


def load(name):
    with open(os.path.join(ROOT, "data", f"{name}.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def score(pairs_prob, docs, model, block_size):
    """Run the benchmark's scoring protocol with an arbitrary P(next | context) fn."""
    nll, n = 0.0, 0
    for doc in docs:
        tokens = model.tokenize(doc)
        for pos in range(min(block_size, len(tokens) - 1)):
            p = pairs_prob(tokens, pos)
            nll -= math.log(p)
            n += 1
    return nll / n, n


def main():
    model = infer.Model(os.path.join(ROOT, "weights", "microgpt.bin"))
    V, B = model.vocab_size, model.block_size
    train, val = load("train"), load("val")

    # counts from exactly the documents the model was trained on
    uni = [0] * V
    bi = [[0] * V for _ in range(V)]
    for doc in train:
        t = model.tokenize(doc)
        for i in range(len(t) - 1):
            uni[t[i + 1]] += 1
            bi[t[i]][t[i + 1]] += 1

    uni_tot = sum(uni)
    uni_p = [(uni[i] + 1) / (uni_tot + V) for i in range(V)]          # add-1 smoothing
    bi_p = []
    for a in range(V):
        row_tot = sum(bi[a])
        bi_p.append([(bi[a][b] + 1) / (row_tot + V) for b in range(V)])

    results = []
    results.append(("uniform", math.log(V), None))

    nll, n = score(lambda t, pos: uni_p[t[pos + 1]], val, model, B)
    results.append(("unigram (char frequency)", nll, n))

    nll, n = score(lambda t, pos: bi_p[t[pos]][t[pos + 1]], val, model, B)
    results.append(("bigram (count table)", nll, n))

    # the model itself
    nll_m, n_m = 0.0, 0
    for doc in val:
        t = model.tokenize(doc)
        keys, values = model.new_cache()
        for pos in range(min(B, len(t) - 1)):
            logits = model.forward(t[pos], pos, keys, values)
            nll_m -= math.log(infer.softmax(logits)[t[pos + 1]])
            n_m += 1
    results.append(("microgpt (1000 steps)", nll_m / n_m, n_m))

    print(f"eval: {len(val)} held-out docs, protocol identical to bench/run.py")
    print(f"count models fit on the same {len(train)} docs microgpt trained on\n")
    print(f"{'model':28s} {'nll/token':>10s} {'perplexity':>11s} {'bits/token':>11s}")
    print("-" * 64)
    for name, nll, n in results:
        print(f"{name:28s} {nll:10.4f} {math.exp(nll):11.4f} {nll/math.log(2):11.4f}")

    bigram_nll = results[2][1]
    model_nll = results[3][1]
    print()
    if model_nll < bigram_nll:
        print(f"microgpt beats the bigram table by {bigram_nll - model_nll:.4f} nats/token "
              f"({math.exp(bigram_nll)/math.exp(model_nll):.2f}x lower perplexity)")
    else:
        print(f"WARNING: microgpt is WORSE than a bigram count table by "
              f"{model_nll - bigram_nll:.4f} nats/token - suspect a bug or undertraining")
    return 0


if __name__ == "__main__":
    sys.exit(main())
