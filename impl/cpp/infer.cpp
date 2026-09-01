// microgpt inference - C++ port.
//
// Implements the contract in BENCHMARK.md: same weights, same f64 operation
// order, same xorshift64* RNG, same JSON report as impl/python/infer.py.
//
// Build (the harness uses exactly these flags - no fast-math, no FP contraction,
// because reassociating the dot products would break cross-language agreement):
//   g++ -O3 -std=c++20 -fno-fast-math -ffp-contract=off -o build/microgpt_infer infer.cpp
// (-O3 to match Rust's release opt-level 3; -O2 costs about 20% here.)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace {

// Pin to one logical CPU. On Intel hybrid parts (P-cores + E-cores) the scheduler
// migrates sustained single-threaded work onto an E-core, which changes throughput
// 2-3x mid-run. P-cores enumerate first, so logical CPU 0 is a P-core in practice.
std::string pin_cpu(int n) {
    if (n < 0) return "not pinned";
#ifdef _WIN32
    if (SetProcessAffinityMask(GetCurrentProcess(), DWORD_PTR(1) << n))
        return "cpu" + std::to_string(n);
    return "pin to cpu" + std::to_string(n) + " failed";
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(n, &set);
    if (sched_setaffinity(0, sizeof(set), &set) == 0) return "cpu" + std::to_string(n);
    return "pin to cpu" + std::to_string(n) + " failed";
#else
    return "not pinned";
#endif
}

// ---------------------------------------------------------------- sha256

class Sha256 {
public:
    Sha256() { reset(); }

    void update(const uint8_t* data, size_t len) {
        for (size_t i = 0; i < len; ++i) {
            buf_[buflen_++] = data[i];
            if (buflen_ == 64) { transform(buf_); bitlen_ += 512; buflen_ = 0; }
        }
    }

    std::string hex() {
        uint8_t h[32];
        finish(h);
        static const char* d = "0123456789abcdef";
        std::string s;
        s.reserve(64);
        for (int i = 0; i < 32; ++i) { s += d[h[i] >> 4]; s += d[h[i] & 15]; }
        return s;
    }

private:
    uint32_t st_[8];
    uint8_t buf_[64];
    uint32_t buflen_ = 0;
    uint64_t bitlen_ = 0;

    void reset() {
        static const uint32_t iv[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                                       0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
        std::memcpy(st_, iv, sizeof(iv));
        buflen_ = 0;
        bitlen_ = 0;
    }

    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void transform(const uint8_t* d) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t m[64];
        for (int i = 0; i < 16; ++i)
            m[i] = (uint32_t(d[i * 4]) << 24) | (uint32_t(d[i * 4 + 1]) << 16) |
                   (uint32_t(d[i * 4 + 2]) << 8) | uint32_t(d[i * 4 + 3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
            uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
            m[i] = m[i - 16] + s0 + m[i - 7] + s1;
        }
        uint32_t a = st_[0], b = st_[1], c = st_[2], dd = st_[3];
        uint32_t e = st_[4], f = st_[5], g = st_[6], h = st_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + k[i] + m[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = dd + t1;
            dd = c; c = b; b = a; a = t1 + t2;
        }
        st_[0] += a; st_[1] += b; st_[2] += c; st_[3] += dd;
        st_[4] += e; st_[5] += f; st_[6] += g; st_[7] += h;
    }

    void finish(uint8_t* out) {
        uint64_t total = bitlen_ + uint64_t(buflen_) * 8;
        uint32_t i = buflen_;
        buf_[i++] = 0x80;
        if (i > 56) {
            while (i < 64) buf_[i++] = 0;
            transform(buf_);
            i = 0;
        }
        while (i < 56) buf_[i++] = 0;
        for (int j = 7; j >= 0; --j) buf_[56 + (7 - j)] = uint8_t(total >> (j * 8));
        transform(buf_);
        for (int j = 0; j < 8; ++j)
            for (int b = 0; b < 4; ++b) out[j * 4 + b] = uint8_t(st_[j] >> (24 - b * 8));
    }
};

// ------------------------------------------------------------------- rng

struct Rng {
    uint64_t state;
    explicit Rng(uint64_t seed) : state(seed ? seed : 0x9E3779B97F4A7C15ULL) {}

    uint64_t next_u64() {
        uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }

    double next_f64() { return double(next_u64() >> 11) * (1.0 / 9007199254740992.0); }
};

uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 0xCBF29CE484222325ULL;
    for (unsigned char c : s) {
        h ^= uint64_t(c);
        h *= 0x100000001B3ULL;
    }
    return h;
}

// --------------------------------------------------------------- weights

uint32_t rd_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

double rd_f64(const uint8_t* p) {
    uint64_t b = 0;
    for (int i = 7; i >= 0; --i) b = (b << 8) | uint64_t(p[i]);
    double d;
    std::memcpy(&d, &b, sizeof(d));
    return d;
}

struct Mat {
    int rows = 0, cols = 0;
    std::vector<double> d;
    const double* row(int r) const { return d.data() + size_t(r) * cols; }
};

struct Layer {
    Mat wq, wk, wv, wo, fc1, fc2;
};

struct Model {
    int n_layer = 0, n_embd = 0, block_size = 0, n_head = 0, vocab_size = 0, head_dim = 0, bos = 0;
    std::string uchars;
    int char_to_id[256];
    Mat wte, wpe, lm_head;
    std::vector<Layer> layers;
    size_t num_weights = 0;
    double weights_sum = 0.0, weights_abs_sum = 0.0;
    std::string sha256;

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open weights: " + path);
        std::vector<uint8_t> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (blob.size() < 32 || std::memcmp(blob.data(), "MGPT", 4) != 0)
            throw std::runtime_error("bad magic, not a microgpt-bin file: " + path);

        Sha256 sh;
        sh.update(blob.data(), blob.size());
        sha256 = sh.hex();

        uint32_t version = rd_u32(&blob[4]);
        if (version != 1) throw std::runtime_error("unsupported format version");
        n_layer    = int(rd_u32(&blob[8]));
        n_embd     = int(rd_u32(&blob[12]));
        block_size = int(rd_u32(&blob[16]));
        n_head     = int(rd_u32(&blob[20]));
        vocab_size = int(rd_u32(&blob[24]));
        int n_uchars = int(rd_u32(&blob[28]));

        size_t off = 32;
        uchars.assign(reinterpret_cast<const char*>(&blob[off]), size_t(n_uchars));
        off += size_t(n_uchars);
        off += (8 - (off % 8)) % 8;

        head_dim = n_embd / n_head;
        bos = n_uchars;
        std::fill(std::begin(char_to_id), std::end(char_to_id), -1);
        for (int i = 0; i < n_uchars; ++i) char_to_id[uint8_t(uchars[i])] = i;

        size_t cursor = off;
        auto take = [&](Mat& m, int rows, int cols) {
            m.rows = rows;
            m.cols = cols;
            m.d.resize(size_t(rows) * cols);
            for (size_t i = 0; i < m.d.size(); ++i) {
                if (cursor + 8 > blob.size()) throw std::runtime_error("weight file truncated");
                double w = rd_f64(&blob[cursor]);
                cursor += 8;
                m.d[i] = w;
                weights_sum += w;
                weights_abs_sum += std::fabs(w);
                ++num_weights;
            }
        };

        take(wte, vocab_size, n_embd);
        take(wpe, block_size, n_embd);
        take(lm_head, vocab_size, n_embd);
        layers.resize(size_t(n_layer));
        for (auto& L : layers) {
            take(L.wq, n_embd, n_embd);
            take(L.wk, n_embd, n_embd);
            take(L.wv, n_embd, n_embd);
            take(L.wo, n_embd, n_embd);
            take(L.fc1, 4 * n_embd, n_embd);
            take(L.fc2, n_embd, 4 * n_embd);
        }
        if (cursor != blob.size())
            throw std::runtime_error("weight file has trailing bytes the model does not want");
    }

    std::vector<int> tokenize(const std::string& doc) const {
        std::vector<int> t;
        t.reserve(doc.size() + 2);
        t.push_back(bos);
        for (char c : doc) {
            int id = char_to_id[uint8_t(c)];
            if (id < 0) throw std::runtime_error(std::string("character not in vocab: ") + c);
            t.push_back(id);
        }
        t.push_back(bos);
        return t;
    }
};

// --------------------------------------------------------------- kernels
// Operation order here is load-bearing: see BENCHMARK.md section 3.

void linear(const double* x, const Mat& w, double* out) {
    for (int o = 0; o < w.rows; ++o) {
        const double* wo = w.row(o);
        double acc = 0.0;
        for (int i = 0; i < w.cols; ++i) acc += wo[i] * x[i];
        out[o] = acc;
    }
}

void rmsnorm(const double* x, int n, double* out) {
    double ms = 0.0;
    for (int i = 0; i < n; ++i) ms += x[i] * x[i];
    ms /= double(n);
    double scale = std::pow(ms + 1e-5, -0.5);
    for (int i = 0; i < n; ++i) out[i] = x[i] * scale;
}

void softmax(double* z, int n) {
    double m = z[0];
    for (int i = 1; i < n; ++i) m = std::max(m, z[i]);
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        z[i] = std::exp(z[i] - m);
        total += z[i];
    }
    for (int i = 0; i < n; ++i) z[i] /= total;
}

int sample_from(const double* probs, int n, Rng& rng, std::vector<double>& cum) {
    double total = 0.0;
    for (int i = 0; i < n; ++i) {
        total += probs[i];
        cum[i] = total;
    }
    double u = rng.next_f64() * total;
    for (int i = 0; i < n; ++i)
        if (u < cum[i]) return i;
    return n - 1;
}

// ----------------------------------------------------------------- state

struct Cache {
    // per layer, block_size x n_embd, row-major; `len` rows are live
    std::vector<std::vector<double>> keys, values;
    int len = 0;

    Cache(int n_layer, int block_size, int n_embd)
        : keys(size_t(n_layer), std::vector<double>(size_t(block_size) * n_embd)),
          values(size_t(n_layer), std::vector<double>(size_t(block_size) * n_embd)) {}

    void clear() { len = 0; }
};

struct Scratch {
    std::vector<double> x, xn, q, k, v, attn, hidden, logits, attn_logits, cum;

    Scratch(const Model& m)
        : x(size_t(m.n_embd)), xn(size_t(m.n_embd)), q(size_t(m.n_embd)), k(size_t(m.n_embd)),
          v(size_t(m.n_embd)), attn(size_t(m.n_embd)), hidden(size_t(4 * m.n_embd)),
          logits(size_t(m.vocab_size)), attn_logits(size_t(m.block_size)),
          cum(size_t(m.vocab_size)) {}
};

// One decode step. Appends this position's k/v to the cache and returns logits in s.logits.
void forward(const Model& m, int token_id, int pos_id, Cache& c, Scratch& s) {
    const int E = m.n_embd, H = m.n_head, HD = m.head_dim;
    const double* tok = m.wte.row(token_id);
    const double* pos = m.wpe.row(pos_id);
    for (int i = 0; i < E; ++i) s.x[i] = tok[i] + pos[i];
    rmsnorm(s.x.data(), E, s.x.data());

    const int t_idx = c.len;
    for (int li = 0; li < m.n_layer; ++li) {
        const Layer& L = m.layers[size_t(li)];
        rmsnorm(s.x.data(), E, s.xn.data());
        linear(s.xn.data(), L.wq, s.q.data());
        linear(s.xn.data(), L.wk, s.k.data());
        linear(s.xn.data(), L.wv, s.v.data());

        double* kcache = c.keys[size_t(li)].data();
        double* vcache = c.values[size_t(li)].data();
        std::memcpy(kcache + size_t(t_idx) * E, s.k.data(), size_t(E) * sizeof(double));
        std::memcpy(vcache + size_t(t_idx) * E, s.v.data(), size_t(E) * sizeof(double));
        const int T = t_idx + 1;

        const double scale = std::pow(double(HD), 0.5);
        for (int h = 0; h < H; ++h) {
            const int hs = h * HD;
            for (int t = 0; t < T; ++t) {
                const double* kt = kcache + size_t(t) * E;
                double acc = 0.0;
                for (int j = 0; j < HD; ++j) acc += s.q[size_t(hs + j)] * kt[hs + j];
                s.attn_logits[size_t(t)] = acc / scale;
            }
            softmax(s.attn_logits.data(), T);
            for (int j = 0; j < HD; ++j) {
                double acc = 0.0;
                for (int t = 0; t < T; ++t)
                    acc += s.attn_logits[size_t(t)] * vcache[size_t(t) * E + hs + j];
                s.attn[size_t(hs + j)] = acc;
            }
        }
        linear(s.attn.data(), L.wo, s.xn.data());
        for (int i = 0; i < E; ++i) s.x[i] += s.xn[i];

        rmsnorm(s.x.data(), E, s.xn.data());
        linear(s.xn.data(), L.fc1, s.hidden.data());
        for (int i = 0; i < 4 * E; ++i) s.hidden[size_t(i)] = s.hidden[size_t(i)] > 0.0 ? s.hidden[size_t(i)] : 0.0;
        linear(s.hidden.data(), L.fc2, s.xn.data());
        for (int i = 0; i < E; ++i) s.x[i] += s.xn[i];
    }
    c.len = t_idx + 1;
    linear(s.x.data(), m.lm_head, s.logits.data());
}

// ------------------------------------------------------------- benchmarks

// Generate a single sample. Returns the number of forward passes it took.
int gen_one(const Model& m, double temperature, Rng& rng, Scratch& s, Cache& c,
            std::vector<double>& tprobs, std::string* out) {
    c.clear();
    int token_id = m.bos;
    int n = 0;
    for (int pos = 0; pos < m.block_size; ++pos) {
        forward(m, token_id, pos, c, s);
        ++n;
        for (int i = 0; i < m.vocab_size; ++i)
            tprobs[size_t(i)] = s.logits[size_t(i)] / temperature;
        softmax(tprobs.data(), m.vocab_size);
        token_id = sample_from(tprobs.data(), m.vocab_size, rng, s.cum);
        if (token_id == m.bos) break;
        if (out) *out += m.uchars[size_t(token_id)];
    }
    return n;
}

// Teacher-forced scoring of one document. Returns tokens scored, accumulates into *nll.
int score_doc(const Model& m, const std::string& doc, Scratch& s, Cache& c,
              std::vector<double>& probs, double* nll) {
    std::vector<int> t = m.tokenize(doc);
    int n = std::min<int>(m.block_size, int(t.size()) - 1);
    c.clear();
    // Accumulate this document locally and add once, matching the Python reference.
    // Folding each token straight into the running total instead would sum in a
    // different order and shift perplexity in the 15th digit.
    double local = 0.0;
    for (int pos = 0; pos < n; ++pos) {
        forward(m, t[size_t(pos)], pos, c, s);
        std::memcpy(probs.data(), s.logits.data(), size_t(m.vocab_size) * sizeof(double));
        softmax(probs.data(), m.vocab_size);
        local -= std::log(probs[size_t(t[size_t(pos) + 1])]);
    }
    if (nll) *nll += local;
    return n;
}

// ------------------------------------------------------------------ json

struct Json {
    std::ostringstream o;
    std::vector<bool> first;

    Json() { o << std::setprecision(17); }

    void comma() {
        if (!first.empty()) {
            if (!first.back()) o << ",";
            first.back() = false;
        }
    }
    void key(const std::string& k) { comma(); o << "\"" << k << "\":"; }
    void obj(const std::string& k) { key(k); o << "{"; first.push_back(true); }
    void obj() { comma(); o << "{"; first.push_back(true); }
    void end_obj() { o << "}"; first.pop_back(); }
    void arr(const std::string& k) { key(k); o << "["; first.push_back(true); }
    void end_arr() { o << "]"; first.pop_back(); }

    void num(const std::string& k, double v) { key(k); o << v; }
    void num(const std::string& k, long long v) { key(k); o << v; }
    void str(const std::string& k, const std::string& v) { key(k); o << quote(v); }
    void val(double v) { comma(); o << v; }
    void val(long long v) { comma(); o << v; }
    void val(const std::string& v) { comma(); o << quote(v); }

    static std::string quote(const std::string& s) {
        std::string r = "\"";
        for (char c : s) {
            switch (c) {
                case '"': r += "\\\""; break;
                case '\\': r += "\\\\"; break;
                case '\n': r += "\\n"; break;
                case '\t': r += "\\t"; break;
                case '\r': r += "\\r"; break;
                default:
                    if (uint8_t(c) < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); r += b; }
                    else r += c;
            }
        }
        return r + "\"";
    }

    std::string str() const { return o.str(); }
};

std::string hex64(uint64_t v) {
    char b[32];
    std::snprintf(b, sizeof(b), "0x%016llx", static_cast<unsigned long long>(v));
    return b;
}

double now() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

std::vector<std::string> read_docs(const std::string& path, int max_docs) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open data: " + path);
    std::vector<std::string> docs;
    std::string line;
    while (std::getline(f, line)) {
        // Strip both ends, matching Python's str.strip() as used by the reference
        // implementation and by tools/train_export.py when it writes these files.
        // Trailing-only stripping would tokenize a leading space as a vocab miss.
        auto ws = [](char c) { return c == '\r' || c == '\n' || c == ' ' || c == '\t'; };
        size_t b = 0, e = line.size();
        while (b < e && ws(line[b])) ++b;
        while (e > b && ws(line[e - 1])) --e;
        line = line.substr(b, e - b);
        if (!line.empty()) docs.push_back(line);
        if (max_docs > 0 && int(docs.size()) >= max_docs) break;
    }
    return docs;
}

}  // namespace

int main(int argc, char** argv) {
    std::string weights = "weights/microgpt.bin", data = "data/val.txt", mode = "all", json_out;
    int samples = 1000, repeats = 5, max_docs = 0, pin = 0;
    long long seed = 1234;   // long long, matching Rust's i64, so large seeds agree
    double temperature = 0.5, time_budget = 2.0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--weights") weights = next();
        else if (a == "--data") data = next();
        else if (a == "--mode") mode = next();
        else if (a == "--samples") samples = std::stoi(next());
        else if (a == "--temperature") temperature = std::stod(next());
        else if (a == "--seed") seed = std::stoll(next());
        else if (a == "--repeats") repeats = std::stoi(next());
        else if (a == "--max-docs") max_docs = std::stoi(next());
        else if (a == "--time-budget") time_budget = std::stod(next());
        else if (a == "--pin") pin = std::stoi(next());
        else if (a == "--json") json_out = next();
        else {
            std::cerr << "unknown argument: " << a << "\n";
            return 2;
        }
    }

    try {
        std::string cpu_pin = pin_cpu(pin);
        double t0 = now();
        Model m;
        m.load(weights);
        double load_seconds = now() - t0;

        std::vector<std::string> docs = read_docs(data, max_docs);
        Scratch s(m);
        Cache cache(m.n_layer, m.block_size, m.n_embd);

        Json j;
        j.obj();
        j.str("impl", "cpp");
        j.str("mode", mode);
        {
            std::ostringstream rt;
#if defined(__clang__)
            rt << "clang++ " << __clang_major__ << "." << __clang_minor__ << "." << __clang_patchlevel__;
#elif defined(__GNUC__)
            rt << "g++ " << __GNUC__ << "." << __GNUC_MINOR__ << "." << __GNUC_PATCHLEVEL__;
#else
            rt << "c++ (unknown compiler)";
#endif
            j.str("runtime", rt.str());
        }
        j.str("build", "-O3 -std=c++20 -fno-fast-math -ffp-contract=off");
        j.str("weights_sha256", m.sha256);
        j.str("weights_path", weights);
        j.str("data_path", data);
        j.obj("config");
        j.num("n_layer", (long long)m.n_layer);
        j.num("n_embd", (long long)m.n_embd);
        j.num("block_size", (long long)m.block_size);
        j.num("n_head", (long long)m.n_head);
        j.num("head_dim", (long long)m.head_dim);
        j.num("vocab_size", (long long)m.vocab_size);
        j.num("bos_token_id", (long long)m.bos);
        j.str("uchars", m.uchars);
        j.num("num_weights", (long long)m.num_weights);
        j.end_obj();
        j.num("load_seconds", load_seconds);
        j.str("cpu_pin", cpu_pin);

        const bool do_check = (mode == "all" || mode == "check");
        const bool do_gen = (mode == "all" || mode == "gen");
        const bool do_ppl = (mode == "all" || mode == "ppl");

        // ------------------------------------------------------------ check
        if (do_check) {
            const std::string prompt = "emma";
            std::vector<int> pt = m.tokenize(prompt);
            pt.pop_back();  // drop trailing BOS
            cache.clear();
            for (size_t i = 0; i < pt.size(); ++i) forward(m, pt[i], int(i), cache, s);
            std::vector<double> logits = s.logits;
            std::vector<double> probs = logits;
            softmax(probs.data(), int(probs.size()));
            int argmax = 0;
            for (int i = 1; i < int(probs.size()); ++i)
                if (probs[size_t(i)] > probs[size_t(argmax)]) argmax = i;

            // Reuse score_doc so this fingerprint groups its summation exactly the way
            // the perplexity path does - per document, then added. Folding every token
            // into one running total instead sums in a different order and shifts the
            // last digits, which is a spurious cross-language mismatch waiting to happen.
            double nll3 = 0.0;
            long long n3 = 0;
            size_t take3 = std::min<size_t>(3, docs.size());
            std::vector<double> check_probs(size_t(m.vocab_size));
            for (size_t di = 0; di < take3; ++di) {
                n3 += score_doc(m, docs[di], s, cache, check_probs, &nll3);
            }

            Rng r{uint64_t(seed)};
            std::vector<std::string> draws;
            for (int i = 0; i < 4; ++i) draws.push_back(hex64(r.next_u64()));

            j.obj("check");
            j.str("prompt", prompt);
            j.arr("logits_after_prompt");
            for (double d : logits) j.val(d);
            j.end_arr();
            j.arr("probs_after_prompt");
            for (double d : probs) j.val(d);
            j.end_arr();
            j.num("argmax_token", (long long)argmax);
            j.str("argmax_char", argmax == m.bos ? "<BOS>" : std::string(1, m.uchars[size_t(argmax)]));
            j.arr("first3_docs");
            for (size_t di = 0; di < take3; ++di) j.val(docs[di]);
            j.end_arr();
            j.num("first3_nll", nll3);
            j.num("first3_tokens", (long long)n3);
            j.num("rng_seed", seed);
            j.arr("rng_first4_u64");
            for (const auto& d : draws) j.val(d);
            j.end_arr();
            j.num("weights_sum", m.weights_sum);
            j.num("weights_abs_sum", m.weights_abs_sum);
            j.end_obj();
        }

        // -------------------------------------------------------------- gen
        if (do_gen) {
            std::vector<double> tprobs(size_t(m.vocab_size));

            // 1. Deterministic pass: fixed sample count, produces the cross-language
            //    hash. Doubles as warmup so the timed passes below start hot.
            std::vector<std::string> texts;
            texts.reserve(size_t(samples));
            long long tokens = 0, chars = 0;
            {
                Rng rng{uint64_t(seed)};
                for (int si = 0; si < samples; ++si) {
                    std::string buf;
                    tokens += gen_one(m, temperature, rng, s, cache, tprobs, &buf);
                    chars += (long long)buf.size();
                    texts.push_back(std::move(buf));
                }
            }
            std::string body;
            for (size_t i = 0; i < texts.size(); ++i) {
                if (i) body += "\n";
                body += texts[i];
            }

            // 2. Timed passes under a shared wall-clock budget, so every
            //    implementation is measured in the same CPU clock state.
            std::vector<double> seconds;
            std::vector<long long> tok_counts;
            for (int rep = 0; rep < repeats; ++rep) {
                Rng rng{uint64_t(seed)};
                long long n_tok = 0;
                double start = now(), elapsed = 0.0;
                do {
                    n_tok += gen_one(m, temperature, rng, s, cache, tprobs, nullptr);
                    elapsed = now() - start;
                } while (elapsed < time_budget);
                tok_counts.push_back(n_tok);
                seconds.push_back(elapsed);
            }

            std::vector<double> rates;
            for (size_t i = 0; i < seconds.size(); ++i)
                rates.push_back(double(tok_counts[i]) / seconds[i]);
            std::vector<double> ordered = rates;
            std::sort(ordered.begin(), ordered.end());

            j.obj("gen");
            j.num("samples", (long long)samples);
            j.num("temperature", temperature);
            j.num("seed", seed);
            j.num("repeats", (long long)repeats);
            j.num("time_budget", time_budget);
            j.num("tokens", tokens);
            j.num("chars", chars);
            j.str("output_fnv1a64", hex64(fnv1a64(body)));
            j.arr("first_samples");
            for (size_t i = 0; i < texts.size() && i < 20; ++i) j.val(texts[i]);
            j.end_arr();
            j.arr("rates_per_repeat");
            for (double d : rates) j.val(d);
            j.end_arr();
            j.arr("tokens_per_repeat");
            for (long long d : tok_counts) j.val(d);
            j.end_arr();
            j.arr("seconds_per_repeat");
            for (double d : seconds) j.val(d);
            j.end_arr();
            j.num("tokens_per_sec_best", ordered.back());
            j.num("tokens_per_sec_median", ordered[ordered.size() / 2]);
            j.end_obj();
        }

        // -------------------------------------------------------------- ppl
        if (do_ppl) {
            std::vector<double> probs(size_t(m.vocab_size));

            // 1. Quality pass over the exact evaluation set - this is what perplexity means.
            double nll_total = 0.0;
            long long n_tok = 0;
            double full_start = now();
            for (const auto& doc : docs) n_tok += score_doc(m, doc, s, cache, probs, &nll_total);
            double full_seconds = now() - full_start;

            // 2. Timed passes under a shared wall-clock budget, cycling the corpus.
            std::vector<double> seconds;
            std::vector<long long> tok_counts;
            for (int rep = 0; rep < repeats; ++rep) {
                long long n = 0;
                size_t i = 0;
                double start = now(), elapsed = 0.0;
                do {
                    n += score_doc(m, docs[i % docs.size()], s, cache, probs, nullptr);
                    ++i;
                    elapsed = now() - start;
                } while (elapsed < time_budget);
                tok_counts.push_back(n);
                seconds.push_back(elapsed);
            }

            std::vector<double> rates;
            for (size_t i = 0; i < seconds.size(); ++i)
                rates.push_back(double(tok_counts[i]) / seconds[i]);
            std::vector<double> ordered = rates;
            std::sort(ordered.begin(), ordered.end());
            double nll_per_token = nll_total / double(n_tok);

            j.obj("ppl");
            j.num("docs", (long long)docs.size());
            j.num("tokens", n_tok);
            j.num("repeats", (long long)repeats);
            j.num("time_budget", time_budget);
            j.num("nll_total", nll_total);
            j.num("nll_per_token", nll_per_token);
            j.num("perplexity", std::exp(nll_per_token));
            j.num("bits_per_token", nll_per_token / std::log(2.0));
            j.num("full_pass_seconds", full_seconds);
            j.arr("rates_per_repeat");
            for (double d : rates) j.val(d);
            j.end_arr();
            j.arr("tokens_per_repeat");
            for (long long d : tok_counts) j.val(d);
            j.end_arr();
            j.arr("seconds_per_repeat");
            for (double d : seconds) j.val(d);
            j.end_arr();
            j.num("tokens_per_sec_best", ordered.back());
            j.num("tokens_per_sec_median", ordered[ordered.size() / 2]);
            j.end_obj();
        }

        j.end_obj();
        std::string text = j.str();
        if (!json_out.empty()) {
            std::ofstream f(json_out);
            if (!f) throw std::runtime_error("cannot write json: " + json_out);
            f << text << "\n";
        }
        std::cout << text << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
