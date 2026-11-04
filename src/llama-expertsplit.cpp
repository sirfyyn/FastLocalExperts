#include "llama-expertsplit.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <atomic>
#include <cinttypes>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct split_state {
    std::mutex mtx;
    std::set<const llama_model *> done;
    std::map<const ggml_tensor *, const llama_expert_split_layer *> by_up_cold;
    std::map<const llama_model *, std::vector<const llama_expert_split_layer *>> by_model;
    std::atomic<bool> any{false};
};

// never destroyed: the tables live as long as the process (graphs of any context may reference them, as the expert store)
split_state & st() {
    static split_state * s = new split_state();
    return *s;
}

} // namespace

void llama_expert_split_init(const llama_model & model) {
    split_state & s = st();
    std::lock_guard<std::mutex> lk(s.mtx);
    if (!s.done.insert(&model).second) {
        return;
    }

    std::vector<llama_expert_split_layer *> ls;
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (!l.ffn_up_exps_hot) {
            continue;
        }
        GGML_ASSERT(l.ffn_gate_exps_hot && l.ffn_down_exps_hot && l.ffn_up_exps_cold && l.ffn_gate_exps_cold && l.ffn_down_exps_cold &&
                    l.ffn_exps_slot && l.ffn_exps_id && "split layer without all its tensors");
        GGML_ASSERT(!l.ffn_up_exps && !l.ffn_gate_exps && !l.ffn_down_exps && !l.ffn_gate_up_exps && "split layer with normal experts");
        auto * x = new llama_expert_split_layer();
        x->il        = (int) il;
        x->n_hot     = (int32_t) l.ffn_up_exps_hot->ne[2] - 1;
        x->n_cold    = (int32_t) l.ffn_up_exps_cold->ne[2] - 1;
        x->up_hot    = l.ffn_up_exps_hot;
        x->gate_hot  = l.ffn_gate_exps_hot;
        x->down_hot  = l.ffn_down_exps_hot;
        x->up_cold   = l.ffn_up_exps_cold;
        x->gate_cold = l.ffn_gate_exps_cold;
        x->down_cold = l.ffn_down_exps_cold;
        if ((int64_t) x->n_hot + x->n_cold != l.ffn_exps_slot->ne[0] || l.ffn_exps_slot->type != GGML_TYPE_I32 ||
                l.ffn_exps_id->type != GGML_TYPE_I32 || l.ffn_exps_id->ne[0] != l.ffn_exps_slot->ne[0]) {
            throw std::runtime_error(format("split layer %zu: %d hot + %d cold rows do not match the slot tables (%" PRId64 " experts)",
                    il, x->n_hot, x->n_cold, l.ffn_exps_slot->ne[0]));
        }
        ls.push_back(x);
    }
    if (ls.empty()) {
        return;
    }

    // the id tables next to the hot half (the router's device: the get_rows run there), the skip tables in host memory
    std::map<ggml_backend_buffer_type_t, std::vector<llama_expert_split_layer *>> groups;
    for (auto * x : ls) {
        ggml_backend_buffer_type_t bt = x->up_hot->buffer ? ggml_backend_buffer_get_type(x->up_hot->buffer) : ggml_backend_cpu_buffer_type();
        groups[bt].push_back(x);
    }
    auto alloc = [&](ggml_backend_buffer_type_t bt, const std::vector<llama_expert_split_layer *> & xs, bool host) {
        ggml_init_params ip = { ggml_tensor_overhead() * (xs.size() * 2 + 4), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            throw std::runtime_error("split experts: ggml_init failed");
        }
        for (auto * x : xs) {
            const int64_t n_expert = (int64_t) x->n_hot + x->n_cold;
            if (host) {
                x->cold_skip = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, x->n_cold + 1);
                ggml_format_name(x->cold_skip, "split_cold_skip.%d", x->il);
            } else {
                x->hot_of  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
                x->cold_of = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, n_expert);
                ggml_format_name(x->hot_of,  "split_hot_of.%d",  x->il);
                ggml_format_name(x->cold_of, "split_cold_of.%d", x->il);
            }
        }
        if (!ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt)) {
            throw std::runtime_error(format("split experts: no memory for the tables on %s", ggml_backend_buft_name(bt)));
        }
    };
    for (auto & g : groups) {
        alloc(g.first, g.second, false);
    }
    alloc(ggml_backend_cpu_buffer_type(), ls, true);

    size_t n_hot_sum = 0, n_cold_sum = 0, n_placeholder = 0;
    for (auto * x : ls) {
        const auto & l = model.layers[(size_t) x->il];
        const int64_t n_expert = (int64_t) x->n_hot + x->n_cold;
        std::vector<int32_t> slot((size_t) n_expert, 0), id((size_t) n_expert, 0);
        std::vector<int32_t> hot_of((size_t) n_expert, x->n_hot), cold_of((size_t) n_expert, x->n_cold);
        // a memory-estimation model has no weights: every id maps to the zero rows (only the graph shape matters there)
        const bool have = l.ffn_exps_slot->buffer && l.ffn_exps_slot->data && l.ffn_exps_id->buffer && l.ffn_exps_id->data;
        if (have) {
            ggml_backend_tensor_get(l.ffn_exps_slot, slot.data(), 0, slot.size() * sizeof(int32_t));
            ggml_backend_tensor_get(l.ffn_exps_id,   id.data(),   0, id.size()   * sizeof(int32_t));
            // the two tables must be inverse permutations, else an expert would be computed twice or never
            for (int64_t e = 0; e < n_expert; ++e) {
                const int32_t sl = slot[(size_t) e];
                if (sl < 0 || sl >= n_expert || id[(size_t) sl] != (int32_t) e) {
                    throw std::runtime_error(format("split layer %d: expert %" PRId64 " has slot %d, the id table disagrees", x->il, e, sl));
                }
                if (sl < x->n_hot) { hot_of[(size_t) e] = sl; } else { cold_of[(size_t) e] = sl - x->n_hot; }
            }
        } else {
            n_placeholder++;
        }
        std::vector<int32_t> skip((size_t) x->n_cold + 1, LLAMA_EXPERT_SPLIT_RUN);
        skip[(size_t) x->n_cold] = 0;   // the zero row: skipped by the CPU, its dst rows zeroed
        ggml_backend_tensor_set(x->hot_of,    hot_of.data(),  0, hot_of.size()  * sizeof(int32_t));
        ggml_backend_tensor_set(x->cold_of,   cold_of.data(), 0, cold_of.size() * sizeof(int32_t));
        ggml_backend_tensor_set(x->cold_skip, skip.data(),    0, skip.size()    * sizeof(int32_t));
        n_hot_sum  += (size_t) x->n_hot;
        n_cold_sum += (size_t) x->n_cold;

        s.by_up_cold[x->up_cold] = x;
        s.by_model[&model].push_back(x);
    }
    s.any.store(true, std::memory_order_release);

    LLAMA_LOG_INFO("%s: FLE split experts: %zu layers, %zu hot experts on %s, %zu cold experts on %s%s\n", __func__, ls.size(),
            n_hot_sum, ls[0]->up_hot->buffer ? ggml_backend_buffer_name(ls[0]->up_hot->buffer) : "?",
            n_cold_sum, ls[0]->up_cold->buffer ? ggml_backend_buffer_name(ls[0]->up_cold->buffer) : "?",
            n_placeholder ? " (no weights loaded: placeholder tables)" : "");
}

bool llama_expert_split_any() {
    return st().any.load(std::memory_order_acquire);
}

const llama_expert_split_layer * llama_expert_split_lookup(const ggml_tensor * up_cold) {
    if (!up_cold || !llama_expert_split_any()) {
        return nullptr;
    }
    split_state & s = st();
    std::lock_guard<std::mutex> lk(s.mtx);
    const auto it = s.by_up_cold.find(up_cold);
    return it == s.by_up_cold.end() ? nullptr : it->second;
}

std::vector<const llama_expert_split_layer *> llama_expert_split_layers(const llama_model & model) {
    if (!llama_expert_split_any()) {
        return {};
    }
    split_state & s = st();
    std::lock_guard<std::mutex> lk(s.mtx);
    const auto it = s.by_model.find(&model);
    return it == s.by_model.end() ? std::vector<const llama_expert_split_layer *>() : it->second;
}
