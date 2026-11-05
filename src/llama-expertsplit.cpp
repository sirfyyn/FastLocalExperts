#include "llama-expertsplit.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// everything the tables of one model own; destroyed with the model (llama_model::fle_expert_split)
struct split_owner {
    std::vector<std::unique_ptr<llama_expert_split_layer>> layers;
    std::vector<ggml_context *>         ctxs;
    std::vector<ggml_backend_buffer_t>  bufs;

    ~split_owner() {
        for (auto * b : bufs) { ggml_backend_buffer_free(b); }
        for (auto * c : ctxs) { ggml_free(c); }
    }
};

std::mutex g_split_mtx;   // two contexts of one model may start at once

// host memory of the CPU backend, including its repack ("extra") buffer types, which do not report is_host
bool on_cpu(ggml_backend_buffer_t b) {
    if (!b || ggml_backend_buffer_is_host(b)) {
        return true;
    }
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(b));
    return !dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

} // namespace

void llama_expert_split_init(const llama_model & model) {
    std::lock_guard<std::mutex> lk(g_split_mtx);
    if (model.fle_expert_split) {
        return;
    }

    auto own = std::make_unique<split_owner>();
    for (size_t il = 0; il < model.layers.size(); ++il) {
        const auto & l = model.layers[il];
        if (!l.ffn_up_exps_hot) {
            continue;
        }
        GGML_ASSERT(l.ffn_gate_exps_hot && l.ffn_down_exps_hot && l.ffn_up_exps_cold && l.ffn_gate_exps_cold && l.ffn_down_exps_cold &&
                    l.ffn_exps_slot && l.ffn_exps_id && "split layer without all its tensors");
        GGML_ASSERT(!l.ffn_up_exps && !l.ffn_gate_exps && !l.ffn_down_exps && !l.ffn_gate_up_exps && "split layer with normal experts");
        auto x = std::make_unique<llama_expert_split_layer>();
        x->il        = (int) il;
        x->n_hot     = (int32_t) l.ffn_up_exps_hot->ne[2] - 1;
        x->n_cold    = (int32_t) l.ffn_up_exps_cold->ne[2] - 1;
        x->up_hot    = l.ffn_up_exps_hot;
        x->gate_hot  = l.ffn_gate_exps_hot;
        x->down_hot  = l.ffn_down_exps_hot;
        x->up_cold   = l.ffn_up_exps_cold;
        x->gate_cold = l.ffn_gate_exps_cold;
        x->down_cold = l.ffn_down_exps_cold;
        // plain host memory only: the fused CPU op and the expert store read the raw rows (not a repack layout, not a device)
        x->cold_host = l.ffn_up_exps_cold->buffer && ggml_backend_buffer_is_host(l.ffn_up_exps_cold->buffer);
        if ((int64_t) x->n_hot + x->n_cold != l.ffn_exps_slot->ne[0] || l.ffn_exps_slot->type != GGML_TYPE_I32 ||
                l.ffn_exps_id->type != GGML_TYPE_I32 || l.ffn_exps_id->ne[0] != l.ffn_exps_slot->ne[0]) {
            throw std::runtime_error(format("split layer %zu: %d hot + %d cold rows do not match the slot tables (%" PRId64 " experts)",
                    il, x->n_hot, x->n_cold, l.ffn_exps_slot->ne[0]));
        }
        own->layers.push_back(std::move(x));
    }
    if (own->layers.empty()) {
        return;
    }

    // the id tables next to the hot half (the router's device: the get_rows run there); a hot half in host memory (also a repack
    // buffer of the CPU backend) gets the plain CPU buffer type. The skip tables are in host memory.
    std::map<ggml_backend_buffer_type_t, std::vector<llama_expert_split_layer *>> groups;
    std::vector<llama_expert_split_layer *> all;
    for (auto & x : own->layers) {
        const ggml_backend_buffer_t b = x->up_hot->buffer;
        groups[on_cpu(b) ? ggml_backend_cpu_buffer_type() : ggml_backend_buffer_get_type(b)].push_back(x.get());
        all.push_back(x.get());
    }
    auto alloc = [&](ggml_backend_buffer_type_t bt, const std::vector<llama_expert_split_layer *> & xs, bool host) {
        ggml_init_params ip = { ggml_tensor_overhead() * (xs.size() * 2 + 4), nullptr, true };
        ggml_context * ctx = ggml_init(ip);
        if (!ctx) {
            throw std::runtime_error("split experts: ggml_init failed");
        }
        own->ctxs.push_back(ctx);
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
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, bt);
        if (!buf) {
            throw std::runtime_error(format("split experts: no memory for the tables on %s", ggml_backend_buft_name(bt)));
        }
        own->bufs.push_back(buf);
    };
    for (auto & g : groups) {
        alloc(g.first, g.second, false);
    }
    alloc(ggml_backend_cpu_buffer_type(), all, true);

    size_t n_hot_sum = 0, n_cold_sum = 0, n_placeholder = 0;
    std::vector<uint8_t> zrow;
    for (auto * x : all) {
        const auto & l = model.layers[(size_t) x->il];
        const int64_t n_expert = (int64_t) x->n_hot + x->n_cold;
        std::vector<int32_t> slot((size_t) n_expert, 0), id((size_t) n_expert, 0);
        std::vector<int32_t> hot_of((size_t) n_expert, x->n_hot), cold_of((size_t) n_expert, x->n_cold);
        // a memory-estimation model has no weights: every id maps to the zero rows (only the graph shape matters there)
        const bool have = l.ffn_exps_slot->buffer && l.ffn_exps_slot->data && l.ffn_exps_id->buffer && l.ffn_exps_id->data &&
                          x->up_hot->buffer && x->up_hot->data;
        x->placeholder = !have;
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
                if (sl >= x->n_hot) {
                    if (x->cold_ids.empty()) { x->cold_ids.assign((size_t) x->n_cold, -1); }
                    x->cold_ids[(size_t) (sl - x->n_hot)] = (int32_t) e;
                }
            }
            // the hot zero row is computed for every cold expert on the GPU: it must really be zero
            for (const ggml_tensor * t : { x->gate_hot, x->up_hot, x->down_hot }) {
                zrow.assign(t->nb[2], 0xff);
                ggml_backend_tensor_get(t, zrow.data(), (size_t) x->n_hot * t->nb[2], t->nb[2]);
                for (const uint8_t c : zrow) {
                    if (c != 0) {
                        throw std::runtime_error(format("%s: the trailing row %d is not all zero (not a split file of format v1?)", t->name, x->n_hot));
                    }
                }
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
    }

    // publish only after everything succeeded (a throw above frees the owner and leaves the layers untouched)
    for (auto * x : all) {
        model.layers[(size_t) x->il].fle_split = x;
    }
    const llama_expert_split_layer * x0 = all[0];
    LLAMA_LOG_INFO("%s: FLE split experts: %zu layers, %zu hot experts on %s, %zu cold experts on %s%s\n", __func__, all.size(),
            n_hot_sum, x0->up_hot->buffer ? ggml_backend_buffer_name(x0->up_hot->buffer) : "?",
            n_cold_sum, x0->up_cold->buffer ? ggml_backend_buffer_name(x0->up_cold->buffer) : "?",
            n_placeholder ? " (no weights loaded: placeholder tables)" : "");
    model.fle_expert_split = std::shared_ptr<void>(own.release(), [](void * p) { delete (split_owner *) p; });
}

std::vector<const llama_expert_split_layer *> llama_expert_split_layers(const llama_model & model) {
    std::vector<const llama_expert_split_layer *> out;
    for (const auto & l : model.layers) {
        if (l.fle_split) {
            out.push_back(l.fle_split);
        }
    }
    return out;
}
