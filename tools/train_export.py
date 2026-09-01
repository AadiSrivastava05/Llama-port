"""
Train microgpt and export the weights so the Python/C++/Rust ports can run
inference-only benchmarks against identical parameters.

The model, initialization, RNG call order and optimizer are a faithful copy of
microgpt.py, so with default flags this produces exactly the weights microgpt.py
would hold at the end of its own training run.

Outputs:
  weights/microgpt.bin   f64 weights + vocab (format documented in BENCHMARK.md)
  weights/config.json    human-readable mirror + integrity checksums
  data/train.txt         the documents training actually consumed
  data/val.txt           held-out documents, used for perplexity
  data/test.txt          second held-out split
"""

import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import threading
import time

MAGIC = b"MGPT"
VERSION = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000, help="training steps (microgpt.py uses 1000)")
    ap.add_argument("--n-layer", type=int, default=1)
    ap.add_argument("--n-embd", type=int, default=16)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--val-docs", type=int, default=2000)
    ap.add_argument("--input", default="input.txt")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--ckpt-every", type=int, default=100, help="0 to disable intermediate checkpoints")
    args = ap.parse_args()

    random.seed(42)  # Let there be order among chaos

    if not os.path.exists(args.input):
        import urllib.request
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(names_url, args.input)
    docs = [line.strip() for line in open(args.input) if line.strip()]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")

    uchars = sorted(set("".join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"vocab size: {vocab_size}")

    class Value:
        __slots__ = ("data", "grad", "_children", "_local_grads")

        def __init__(self, data, children=(), local_grads=()):
            self.data = data
            self.grad = 0
            self._children = children
            self._local_grads = local_grads

        def __add__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data + other.data, (self, other), (1, 1))

        def __mul__(self, other):
            other = other if isinstance(other, Value) else Value(other)
            return Value(self.data * other.data, (self, other), (other.data, self.data))

        def __pow__(self, other): return Value(self.data ** other, (self,), (other * self.data ** (other - 1),))
        def log(self): return Value(math.log(self.data), (self,), (1 / self.data,))
        def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
        def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
        def __neg__(self): return self * -1
        def __radd__(self, other): return self + other
        def __sub__(self, other): return self + (-other)
        def __rsub__(self, other): return other + (-self)
        def __rmul__(self, other): return self * other
        def __truediv__(self, other): return self * other ** -1
        def __rtruediv__(self, other): return other * self ** -1

        def backward(self):
            topo = []
            visited = set()

            def build_topo(v):
                if v not in visited:
                    visited.add(v)
                    for child in v._children:
                        build_topo(child)
                    topo.append(v)

            build_topo(self)
            self.grad = 1
            for v in reversed(topo):
                for child, local_grad in zip(v._children, v._local_grads):
                    child.grad += local_grad * v.grad

    n_layer, n_embd, block_size, n_head = args.n_layer, args.n_embd, args.block_size, args.n_head
    assert n_embd % n_head == 0
    head_dim = n_embd // n_head
    matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
    state_dict = {"wte": matrix(vocab_size, n_embd), "wpe": matrix(block_size, n_embd),
                  "lm_head": matrix(vocab_size, n_embd)}
    for i in range(n_layer):
        state_dict[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        state_dict[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
    params = [p for mat in state_dict.values() for row in mat for p in row]
    print(f"num params: {len(params)}")

    def linear(x, w):
        return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

    def softmax(logits):
        max_val = max(val.data for val in logits)
        exps = [(val - max_val).exp() for val in logits]
        total = sum(exps)
        return [e / total for e in exps]

    def rmsnorm(x):
        ms = sum(xi * xi for xi in x) / len(x)
        scale = (ms + 1e-5) ** -0.5
        return [xi * scale for xi in x]

    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict["wte"][token_id]
        pos_emb = state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)
        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"layer{li}.attn_wq"])
            k = linear(x, state_dict[f"layer{li}.attn_wk"])
            v = linear(x, state_dict[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs + head_dim]
                k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + head_dim] for vi in values[li]]
                attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim ** 0.5
                               for t in range(len(k_h))]
                attn_weights = softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]
        return linear(x, state_dict["lm_head"])

    cfg = dict(n_layer=n_layer, n_embd=n_embd, block_size=block_size, n_head=n_head,
               head_dim=head_dim, vocab_size=vocab_size, bos_token_id=BOS,
               uchars="".join(uchars), num_params=len(params), train_steps=args.steps,
               trainer="tools/train_export.py (pure-python autograd, microgpt.py faithful)")

    # data splits: training only ever sees docs[:steps] via docs[step % len(docs)]
    seen = min(args.steps, len(docs))
    write_split(args.out_dir, "train", docs[:seen])
    write_split(args.out_dir, "val", docs[seen:seen + args.val_docs])
    write_split(args.out_dir, "test", docs[seen + args.val_docs:seen + 2 * args.val_docs])

    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)
    v = [0.0] * len(params)

    num_steps = args.steps
    t_start = time.perf_counter()
    loss_hist = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = (1 / n) * sum(losses)

        loss.backward()

        lr_t = learning_rate * (1 - step / num_steps)
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
            p.grad = 0

        loss_hist.append(loss.data)
        elapsed = time.perf_counter() - t_start
        rate = (step + 1) / elapsed
        eta = (num_steps - step - 1) / rate
        avg50 = sum(loss_hist[-50:]) / len(loss_hist[-50:])
        print(f"step {step+1:5d}/{num_steps:5d} | loss {loss.data:.4f} | avg50 {avg50:.4f} | "
              f"{rate:6.2f} steps/s | elapsed {elapsed/60:6.2f}m | eta {eta/60:6.2f}m", flush=True)

        if args.ckpt_every and (step + 1) % args.ckpt_every == 0 and step + 1 < num_steps:
            export(args.out_dir, state_dict, cfg, uchars, n_layer, step + 1, avg50, quiet=True)

    export(args.out_dir, state_dict, cfg, uchars, n_layer, num_steps,
           sum(loss_hist[-50:]) / len(loss_hist[-50:]))
    print(f"\ntraining wall time: {(time.perf_counter()-t_start)/60:.2f} min")


def write_split(out_dir, name, docs):
    path = os.path.join(out_dir, "data", f"{name}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="\n") as f:
        for d in docs:
            f.write(d + "\n")
    print(f"wrote {path}: {len(docs)} docs")


def tensor_order(n_layer):
    """Canonical serialization order. Every port must read tensors in exactly this order."""
    names = ["wte", "wpe", "lm_head"]
    for i in range(n_layer):
        names += [f"layer{i}.attn_wq", f"layer{i}.attn_wk", f"layer{i}.attn_wv",
                  f"layer{i}.attn_wo", f"layer{i}.mlp_fc1", f"layer{i}.mlp_fc2"]
    return names


def export(out_dir, state_dict, cfg, uchars, n_layer, steps_done, recent_loss, quiet=False):
    wdir = os.path.join(out_dir, "weights")
    os.makedirs(wdir, exist_ok=True)
    bin_path = os.path.join(wdir, "microgpt.bin")
    cfg_path = os.path.join(wdir, "config.json")

    buf = bytearray()
    buf += MAGIC
    buf += struct.pack("<7I", VERSION, cfg["n_layer"], cfg["n_embd"], cfg["block_size"],
                       cfg["n_head"], cfg["vocab_size"], len(uchars))
    buf += "".join(uchars).encode("ascii")
    while len(buf) % 8:              # align the f64 payload to 8 bytes
        buf += b"\x00"
    payload_offset = len(buf)

    shapes, total, wsum, wabs = {}, 0, 0.0, 0.0
    for name in tensor_order(n_layer):
        mat = state_dict[name]
        shapes[name] = [len(mat), len(mat[0])]
        for row in mat:
            for p in row:
                val = p.data if hasattr(p, "data") else float(p)
                buf += struct.pack("<d", val)
                wsum += val
                wabs += abs(val)
                total += 1

    with open(bin_path, "wb") as f:
        f.write(buf)

    meta = dict(cfg)
    meta.update(
        format="microgpt-bin-v1", dtype="float64", byte_order="little",
        payload_offset=payload_offset, tensor_order=tensor_order(n_layer), tensor_shapes=shapes,
        num_weights=total, weights_sum=wsum, weights_abs_sum=wabs,
        steps_done=steps_done, recent_train_loss=recent_loss,
        sha256=hashlib.sha256(buf).hexdigest(), bytes=len(buf),
    )
    with open(cfg_path, "w") as f:
        json.dump(meta, f, indent=2)
    if not quiet:
        print(f"\nwrote {bin_path} ({len(buf)} bytes, {total} f64 weights)")
        print(f"wrote {cfg_path}  sha256={meta['sha256'][:16]}...")
        print(f"checksums: sum={wsum:.12g} abs_sum={wabs:.12g}")


if __name__ == "__main__":
    sys.setrecursionlimit(1_000_000)
    threading.stack_size(64 * 1024 * 1024)   # backward() recurses through the graph
    t = threading.Thread(target=main)
    t.start()
    t.join()
