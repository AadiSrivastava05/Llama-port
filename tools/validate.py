"""
Validate the plain-float reference forward against microgpt.py's Value-based
forward, using the same exported weights.

This is the correctness gate for the whole benchmark: if the plain-float
implementation in impl/python/infer.py matches microgpt.py's autograd graph, then
it is a legitimate reference for the C++ and Rust ports to be checked against.

Agreement is expected to be ~1e-15 relative, not bit-exact, because microgpt.py's
Value class routes division through multiplication by a reciprocal
(`a / b` is `a * b**-1`), which rounds differently than a plain divide.

  python tools/validate.py
"""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "impl", "python"))
import infer  # noqa: E402


class Value:
    """Verbatim from microgpt.py (forward path only; grads unused here)."""
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


def v_linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def v_softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def v_rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def main():
    model = infer.Model(os.path.join(ROOT, "weights", "microgpt.bin"))
    n_layer, n_embd, n_head = model.n_layer, model.n_embd, model.n_head
    head_dim = model.head_dim
    block_size = model.block_size

    # rebuild microgpt.py's state_dict as Values from the exported weights
    sd = {name: [[Value(w) for w in row] for row in mat] for name, mat in model.tensors.items()}

    def v_gpt(token_id, pos_id, keys, values):
        tok_emb = sd["wte"][token_id]
        pos_emb = sd["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = v_rmsnorm(x)
        for li in range(n_layer):
            x_residual = x
            x = v_rmsnorm(x)
            q = v_linear(x, sd[f"layer{li}.attn_wq"])
            k = v_linear(x, sd[f"layer{li}.attn_wk"])
            v = v_linear(x, sd[f"layer{li}.attn_wv"])
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
                attn_weights = v_softmax(attn_logits)
                head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                            for j in range(head_dim)]
                x_attn.extend(head_out)
            x = v_linear(x_attn, sd[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]
            x_residual = x
            x = v_rmsnorm(x)
            x = v_linear(x, sd[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = v_linear(x, sd[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]
        return v_linear(x, sd["lm_head"])

    docs = [l.strip() for l in open(os.path.join(ROOT, "data", "val.txt")) if l.strip()][:50]

    max_abs, max_rel, n_cmp = 0.0, 0.0, 0
    nll_plain, nll_value, n_tok = 0.0, 0.0, 0
    for doc in docs:
        tokens = model.tokenize(doc)
        n = min(block_size, len(tokens) - 1)
        pk, pv = model.new_cache()
        vk, vv = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        for pos_id in range(n):
            tid, target = tokens[pos_id], tokens[pos_id + 1]
            lp = model.forward(tid, pos_id, pk, pv)
            lv = [z.data for z in v_gpt(tid, pos_id, vk, vv)]
            for a, b in zip(lp, lv):
                d = abs(a - b)
                max_abs = max(max_abs, d)
                if abs(b) > 1e-12:
                    max_rel = max(max_rel, d / abs(b))
                n_cmp += 1
            nll_plain -= math.log(infer.softmax(lp)[target])
            nll_value -= math.log(v_softmax([Value(z) for z in lv])[target].data)
            n_tok += 1

    print(f"compared {n_cmp} logits over {n_tok} token positions in {len(docs)} docs")
    print(f"max abs logit diff : {max_abs:.3e}")
    print(f"max rel logit diff : {max_rel:.3e}")
    print(f"nll/token plain-float : {nll_plain / n_tok:.15f}")
    print(f"nll/token Value graph : {nll_value / n_tok:.15f}")
    print(f"nll/token difference  : {abs(nll_plain - nll_value) / n_tok:.3e}")

    ok = max_rel < 1e-9 and abs(nll_plain - nll_value) / n_tok < 1e-12
    print("\nPASS: plain-float forward is a faithful port of microgpt.py" if ok
          else "\nFAIL: implementations disagree beyond floating-point noise")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
