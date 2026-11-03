#include "llama-moe-fused.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <mutex>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FLE_PAUSE() _mm_pause()
#else
#define FLE_PAUSE() ((void) 0)
#endif

// ggml_get_type_traits_cpu lives in the CPU backend; with GGML_BACKEND_DL that is a module, so look it up at run time
typedef const struct ggml_type_traits_cpu * (*fle_traits_fn_t)(enum ggml_type);
static const struct ggml_type_traits_cpu * fle_traits(ggml_type t) {
    static fle_traits_fn_t fn = [] {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CPU");
        return reg ? (fle_traits_fn_t) ggml_backend_reg_get_proc_address(reg, "ggml_get_type_traits_cpu") : nullptr;
    }();
    return fn ? fn(t) : nullptr;
}

namespace {

struct pair_t { int32_t t, slot, e; };

struct fused_ud {
    int   il    = -1;
    int   dummy = 0;
    float limit = 0.0f;
    // shared across the threads of one call (written by thread 0 before the first barrier)
    std::vector<pair_t> pairs;
    std::vector<float>  act;        // [n_pairs][n_ff]
    std::atomic<int>    arrive{0};
    std::atomic<int>    gen{0};
};

static void fused_barrier(fused_ud * u, int nth) {
    const int g = u->gen.load(std::memory_order_acquire);
    if (u->arrive.fetch_add(1, std::memory_order_acq_rel) + 1 == nth) {
        u->arrive.store(0, std::memory_order_relaxed);
        u->gen.fetch_add(1, std::memory_order_acq_rel);
    } else {
        while (u->gen.load(std::memory_order_acquire) == g) { FLE_PAUSE(); }
    }
}

static const char * expert_data(const ggml_tensor * w, int32_t e) {
    void * ud = nullptr;
    const ggml_moe_data_cb_t cb = ggml_get_moe_data_callback(&ud);
    if (cb) {
        const void * p = cb(w, e, ud);
        if (p) { return (const char *) p; }
    }
    return (const char *) w->data + (size_t) e * w->nb[2];
}

static void fused_fn(ggml_tensor * dst, int ith, int nth, void * userdata) {
    fused_ud * u = (fused_ud *) userdata;
    const ggml_tensor * cur   = dst->src[0];
    const ggml_tensor * ids   = dst->src[1];
    const ggml_tensor * gate  = dst->src[2];
    const ggml_tensor * up    = dst->src[3];
    const ggml_tensor * down  = dst->src[4];
    const ggml_tensor * table = dst->src[5];
    const ggml_tensor * pred  = dst->src[6];

    const int64_t n_embd = cur->ne[0];
    const int64_t n_tok  = ids->ne[1];
    const int64_t n_used = ids->ne[0];
    const int64_t n_ff   = gate->ne[1];
    const int32_t * tbl  = (const int32_t *) table->data;

    // phase 0 (thread 0): routing observation (expert cache, expert store, prediction) and the list of CPU experts
    if (ith == 0) {
        void * oud = nullptr;
        ggml_moe_obs_cb_t obs = ggml_get_moe_obs_callback(&oud);
        if (obs) { obs(gate->name, ids, oud); }
        void * pud = nullptr;
        ggml_moe_pred_cb_t pcb = ggml_get_moe_pred_callback(&pud);
        if (pcb && pred) { pcb(gate->name, pred, pud); }
        u->pairs.clear();
        for (int64_t t = 0; t < n_tok; ++t) {
            for (int64_t i = 0; i < n_used; ++i) {
                const int32_t e = *(const int32_t *) ((const char *) ids->data + t*ids->nb[1] + i*ids->nb[0]);
                if (e < 0 || e >= gate->ne[2]) { GGML_ABORT("fused moe: expert id %d outside [0, %lld)", e, (long long) gate->ne[2]); }
                if (tbl[e] != u->dummy) { continue; }   // the GPU cache computes it
                u->pairs.push_back({ (int32_t) t, (int32_t) i, e });
            }
        }
        u->act.resize(u->pairs.size() * (size_t) n_ff);
    }
    // zero this thread's share of the output (the GPU-cached slots stay 0, the CPU ones are written in phase 2)
    {
        const int64_t n_rows = n_used * n_tok;
        for (int64_t r = ith; r < n_rows; r += nth) {
            memset((char *) dst->data + (r % n_used)*dst->nb[1] + (r / n_used)*dst->nb[2], 0, n_embd * sizeof(float));
        }
    }
    fused_barrier(u, nth);

    const int64_t np = (int64_t) u->pairs.size();
    if (np == 0) {
        return;
    }

    // the activation quantized once per token, as mul_mat_id does (from_float of the weights' vec_dot_type)
    const auto * tg = fle_traits(gate->type);
    const auto * tu = fle_traits(up->type);
    const auto * td = fle_traits(down->type);
    GGML_ASSERT(tg && tu && td && tg->vec_dot_type == tu->vec_dot_type);
    const auto * tq  = fle_traits(tg->vec_dot_type);
    const size_t xrs = ggml_row_size(tg->vec_dot_type, n_embd);
    thread_local std::vector<uint8_t> xq;
    xq.resize((size_t) n_tok * xrs);
    for (int64_t t = 0; t < n_tok; ++t) {
        tq->from_float((const float *) ((const char *) cur->data + t*cur->nb[2]), xq.data() + t*xrs, n_embd);
    }

    // phase 1: gate and up rows of every CPU expert, clamped SwiGLU on the spot (ggml_swiglu_clamp's arithmetic)
    {
        const int64_t n_rows = np * n_ff;
        const int64_t r0 = n_rows * ith / nth, r1 = n_rows * (ith + 1) / nth;
        const float lim = u->limit;
        for (int64_t r = r0; r < r1; ++r) {
            const pair_t & p = u->pairs[(size_t) (r / n_ff)];
            const int64_t row = r % n_ff;
            float g = 0.0f, v = 0.0f;
            tg->vec_dot((int) n_embd, &g, 0, expert_data(gate, p.e) + row*gate->nb[1], 0, xq.data() + p.t*xrs, 0, 1);
            tu->vec_dot((int) n_embd, &v, 0, expert_data(up,   p.e) + row*up->nb[1],   0, xq.data() + p.t*xrs, 0, 1);
            const float gc = std::min(g, lim);
            const float uc = std::clamp(v, -lim, lim);
            u->act[(size_t) r] = gc / (1.f + expf(-gc)) * uc;
        }
    }
    fused_barrier(u, nth);

    // phase 2: down rows; every thread quantizes the activations of all CPU experts for itself (small)
    {
        const auto * tqd = fle_traits(td->vec_dot_type);
        const size_t ars = ggml_row_size(td->vec_dot_type, n_ff);
        thread_local std::vector<uint8_t> aq;
        aq.resize((size_t) np * ars);
        for (int64_t p = 0; p < np; ++p) {
            tqd->from_float(u->act.data() + p*n_ff, aq.data() + p*ars, n_ff);
        }
        const int64_t n_rows = np * n_embd;
        const int64_t r0 = n_rows * ith / nth, r1 = n_rows * (ith + 1) / nth;
        for (int64_t r = r0; r < r1; ++r) {
            const pair_t & p = u->pairs[(size_t) (r / n_embd)];
            const int64_t row = r % n_embd;
            float * o = (float *) ((char *) dst->data + p.slot*dst->nb[1] + p.t*dst->nb[2]) + row;
            td->vec_dot((int) n_ff, o, 0, expert_data(down, p.e) + row*down->nb[1], 0, aq.data() + (r / n_embd)*ars, 0, 1);
        }
    }
}

} // namespace

ggml_tensor * llama_moe_fused_cpu(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * ids,
        ggml_tensor * gate_exps, ggml_tensor * up_exps, ggml_tensor * down_exps,
        ggml_tensor * table, int32_t dummy, ggml_tensor * pred, float clamp_limit, int il) {
    static std::mutex mtx;
    static std::vector<fused_ud *> uds;   // one per layer, never freed (graphs are rebuilt per batch shape)
    fused_ud * u = nullptr;
    {
        std::lock_guard<std::mutex> lk(mtx);
        if ((size_t) (il + 1) > uds.size()) { uds.resize((size_t) il + 1, nullptr); }
        if (!uds[(size_t) il]) { uds[(size_t) il] = new fused_ud(); }
        u = uds[(size_t) il];
    }
    u->il = il; u->dummy = dummy; u->limit = clamp_limit;
    ggml_tensor * args[7] = { cur, ids, gate_exps, up_exps, down_exps, table, pred };
    return ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], ids->ne[0], ids->ne[1], 1, args, pred ? 7 : 6, fused_fn, GGML_N_TASKS_MAX, u);
}
