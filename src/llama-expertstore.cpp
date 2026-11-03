#include "llama-expertstore.h"

#include "llama-impl.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

#ifndef _WIN32
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

static constexpr size_t ES_ALIGN = 4096;

static size_t es_round_up(size_t x) { return (x + ES_ALIGN - 1) & ~(ES_ALIGN - 1); }

struct es_job { int fd; uint64_t off; size_t len; char * dst; std::atomic<int> * ctr; };   // ctr: prefetch (async), nullptr: demand

struct llama_expert_store::impl {
    struct role_t { int fd = -1; uint64_t off0 = 0; size_t nb2 = 0; size_t region = 0; size_t region_off = 0; };
    struct layer_t {
        int il = -1; int64_t n_expert = 0; role_t r[3]; int cls = 0; size_t blob = 0;
        std::vector<int32_t>      slot_of;   // expert -> slot, -1
        std::vector<const char *> ptr[3];    // expert -> weights in the arena (nullptr: not resident)
    };
    struct slot_t { int32_t li = -1; int32_t e = -1; uint32_t cnt = 0; uint8_t pre = 0; const char * staged[3] = { nullptr, nullptr, nullptr }; };
    std::unique_ptr<std::atomic<int>[]> inflight;   // per slot: prefetch reads still running
    std::unique_ptr<std::atomic<int>[]> pins;       // per slot: GPU-cache uploads reading it
    std::mutex smtx;                                 // ownership changes vs acquire() from the upload thread

    std::vector<layer_t> layers;
    std::unordered_map<const void *, std::pair<int32_t, int32_t>> by_data;   // tensor data -> (layer index, role)
    std::vector<int> fds;
    char *  arena = nullptr;
    size_t  arena_bytes = 0;
    size_t  blob = 0;
    std::vector<slot_t> slots;
    size_t  next_free = 0;     // (unused with size classes, kept for the stats)
    size_t  cursor = 0;        // eviction sampling position
    // size classes: layers whose experts have the same slot size share a contiguous slot range [lo, hi); every slot of a class
    // is exactly that class's size, so a 9 MB IQ1_M expert no longer takes a slot sized for a 14 MB IQ2_XS one
    std::vector<size_t> cls_blob, cls_lo, cls_hi, cls_next;
    std::vector<size_t> slot_off;   // byte offset of each slot in the arena
    // a free slot of class c, or a victim sampled from class c (the caller's filter decides); SIZE_MAX if the class is empty
    size_t cls_free(int c) { return cls_next[(size_t) c] < cls_hi[(size_t) c] ? cls_next[(size_t) c]++ : SIZE_MAX; }
    size_t cls_sample(int c, size_t k) const {
        const size_t lo = cls_lo[(size_t) c], n = cls_hi[(size_t) c] - lo;
        return n ? lo + (cursor + k * 131) % n : SIZE_MAX;
    }
    uint64_t n_ensure = 0;
    int32_t  pin_li = -1; std::vector<int32_t> pin;   // the experts the running layer computes (not given up by a prefetch)

    // reader pool
    std::vector<std::thread> pool;
    std::mutex               mtx;
    std::condition_variable  cv, done_cv;
    std::deque<es_job>       q;
    size_t                   pending = 0;
    bool                     stop = false;
    uint64_t                 read_err = 0;

    void reader() {
        for (;;) {
            es_job j;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&] { return stop || !q.empty(); });
                if (stop && q.empty()) { return; }
                j = q.front(); q.pop_front();
            }
            bool ok = true;
#ifndef _WIN32
            size_t got = 0;
            while (got < j.len) {
                const ssize_t r = pread(j.fd, j.dst + got, j.len - got, (off_t) (j.off + got));
                if (r <= 0) { ok = r == 0 && got > 0; break; }   // EOF after data: the tail past the file end is not needed
                got += (size_t) r;
            }
#endif
            if (j.ctr) {
                if (!ok) { std::lock_guard<std::mutex> lk(mtx); read_err++; }
                j.ctr->fetch_sub(1, std::memory_order_release);
                continue;
            }
            {
                std::lock_guard<std::mutex> lk(mtx);
                if (!ok) { read_err++; }
                if (--pending == 0) { done_cv.notify_all(); }
            }
        }
    }

    void run(std::vector<es_job> & jobs) {
        if (jobs.empty()) { return; }
        {
            std::lock_guard<std::mutex> lk(mtx);
            for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) { q.push_front(*it); }   // demand before prefetch
            pending += jobs.size();
        }
        cv.notify_all();
        std::unique_lock<std::mutex> lk(mtx);
        done_cv.wait(lk, [&] { return pending == 0; });
        jobs.clear();
    }

    // queue the three reads of expert e of layer li into slot s; returns the weight pointers via ptr_out
    void queue_reads(size_t li, int32_t e, size_t s, std::vector<es_job> & jobs, const char * ptr_out[3]) {
        layer_t & L = layers[li];
        char * base = arena + slot_off[s];
        for (int r = 0; r < 3; ++r) {
            const role_t & R = L.r[r];
            const uint64_t off = R.off0 + (uint64_t) e * R.nb2;
            const uint64_t a   = off & ~(uint64_t) (ES_ALIGN - 1);
            const size_t   len = es_round_up((size_t) (off + R.nb2 - a));
            char * dst = base + R.region_off;
            jobs.push_back({ R.fd, a, len, dst, nullptr });
            ptr_out[r] = dst + (off - a);
        }
    }

    void assign(size_t li, int32_t e, size_t s, const char * p[3]) {
        slot_t & S = slots[s];
        if (S.li >= 0) {   // give up the old owner
            layer_t & O = layers[(size_t) S.li];
            O.slot_of[(size_t) S.e] = -1;
            for (int r = 0; r < 3; ++r) { O.ptr[r][(size_t) S.e] = nullptr; }
        }
        S.li = (int32_t) li; S.e = e; S.cnt = 1; S.pre = 0;
        layer_t & L = layers[li];
        L.slot_of[(size_t) e] = (int32_t) s;
        for (int r = 0; r < 3; ++r) { L.ptr[r][(size_t) e] = p[r]; }
    }
};

#ifndef _WIN32
// file and offset behind a mapped address, from /proc/self/maps
static bool es_file_of(const void * addr, std::string & path, uint64_t & file_off) {
    FILE * f = std::fopen("/proc/self/maps", "r");
    if (!f) { return false; }
    char line[4096];
    const uintptr_t x = (uintptr_t) addr;
    bool found = false;
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long lo = 0, hi = 0, off = 0; char perms[8] = {0}; char dev[32] = {0}; unsigned long ino = 0; int n = 0;
        if (std::sscanf(line, "%lx-%lx %7s %lx %31s %lu %n", &lo, &hi, perms, &off, dev, &ino, &n) < 6) { continue; }
        if (x < lo || x >= hi || ino == 0) { continue; }
        std::string p = line + n;
        while (!p.empty() && (p.back() == '\n' || p.back() == ' ')) { p.pop_back(); }
        if (p.empty() || p[0] != '/') { continue; }
        path = p; file_off = off + (x - lo); found = true; break;
    }
    std::fclose(f);
    return found;
}
#endif

std::unique_ptr<llama_expert_store> llama_expert_store::create(const std::vector<llama_expert_store_layer> & in, size_t budget_bytes, int threads) {
#ifdef _WIN32
    (void) in; (void) budget_bytes; (void) threads;
    return nullptr;
#else
    auto * d = new impl();
    std::map<std::string, int> fd_of;
    for (const auto & il : in) {
        impl::layer_t L;
        L.il = il.il;
        size_t region_off = 0;
        bool ok = true;
        for (int r = 0; r < 3; ++r) {
            const ggml_tensor * t = il.src[r];
            if (!t || !t->data) { ok = false; break; }
            std::string path; uint64_t foff = 0;
            if (!es_file_of(t->data, path, foff)) { ok = false; break; }
            auto it = fd_of.find(path);
            if (it == fd_of.end()) {
                #ifdef O_DIRECT
                const int fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
#else
                const int fd = ::open(path.c_str(), O_RDONLY);   // (no O_DIRECT here: buffered reads)
#endif
                if (fd < 0) { ok = false; break; }
                it = fd_of.emplace(path, fd).first;
                d->fds.push_back(fd);
            }
            L.r[r].fd = it->second;
            L.r[r].off0 = foff;
            L.r[r].nb2 = t->nb[2];
            L.r[r].region = es_round_up(t->nb[2]) + 2 * ES_ALIGN;
            L.r[r].region_off = region_off;
            region_off += L.r[r].region;
            L.n_expert = t->ne[2];
            d->by_data[t->data] = { (int32_t) d->layers.size(), r };
        }
        if (!ok) {
            LLAMA_LOG_WARN("%s: layer %d: expert tensors not mapped from a file, store off\n", __func__, il.il);
            for (int fd : d->fds) { ::close(fd); }
            delete d;
            return nullptr;
        }
        L.slot_of.assign((size_t) L.n_expert, -1);
        for (int r = 0; r < 3; ++r) { L.ptr[r].assign((size_t) L.n_expert, nullptr); }
        d->blob = std::max(d->blob, region_off);
        L.blob = region_off;
        int c = -1;
        for (size_t k = 0; k < d->cls_blob.size(); ++k) { if (d->cls_blob[k] == region_off) { c = (int) k; break; } }
        if (c < 0) { c = (int) d->cls_blob.size(); d->cls_blob.push_back(region_off); }
        L.cls = c;
        d->layers.push_back(std::move(L));
    }
    if (d->layers.empty() || d->blob == 0) { delete d; return nullptr; }
    // the same number of slots for every layer: class c gets budget * n_layers(c) / sum_j(n_layers(j) * blob(j)) slots
    std::vector<size_t> n_lay(d->cls_blob.size(), 0);
    for (const auto & L : d->layers) { n_lay[(size_t) L.cls]++; }
    double denom = 0.0;
    for (size_t c = 0; c < n_lay.size(); ++c) { denom += (double) n_lay[c] * (double) d->cls_blob[c]; }
    size_t n_slots = 0;
    d->arena_bytes = 0;
    for (size_t c = 0; c < n_lay.size(); ++c) {
        const size_t nc = (size_t) ((double) budget_bytes * (double) n_lay[c] / denom);
        d->cls_lo.push_back(n_slots); d->cls_next.push_back(n_slots);
        for (size_t k = 0; k < nc; ++k) { d->slot_off.push_back(d->arena_bytes); d->arena_bytes += d->cls_blob[c]; }
        n_slots += nc;
        d->cls_hi.push_back(n_slots);
    }
    if (n_slots == 0) { delete d; return nullptr; }
    void * a = mmap(nullptr, d->arena_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (a == MAP_FAILED) {
        LLAMA_LOG_WARN("%s: arena of %.1f GB failed, store off\n", __func__, d->arena_bytes / 1e9);
        for (int fd : d->fds) { ::close(fd); }
        delete d;
        return nullptr;
    }
#ifdef MADV_HUGEPAGE
    if (!std::getenv("FLE_MOE_STORE_NOHUGE")) { madvise(a, d->arena_bytes, MADV_HUGEPAGE); }
#endif
    if (std::getenv("FLE_MOE_STORE_POPULATE")) {   // fault the whole arena in now (first-touch on this thread), not under the readers' DMA
        for (size_t o = 0; o < d->arena_bytes; o += 4096) { ((volatile char *) a)[o] = 0; }
    }
    d->arena = (char *) a;
    d->slots.resize(n_slots);
    d->inflight.reset(new std::atomic<int>[n_slots]);
    d->pins.reset(new std::atomic<int>[n_slots]);
    for (size_t i = 0; i < n_slots; ++i) { d->inflight[i].store(0); d->pins[i].store(0); }
    for (int i = 0; i < std::max(1, threads); ++i) { d->pool.emplace_back([d] { d->reader(); }); }
#if defined(__linux__)
    // FLE_MOE_STORE_READER_CPUS="16-31": the readers wait on the SSD most of the time; keep them off the cores the compute threads use
    if (const char * rc = std::getenv("FLE_MOE_STORE_READER_CPUS")) {
        int lo = -1, hi = -1;
        if (std::sscanf(rc, "%d-%d", &lo, &hi) == 2 && lo >= 0 && hi >= lo) {
            cpu_set_t set; CPU_ZERO(&set);
            for (int c = lo; c <= hi && c < CPU_SETSIZE; ++c) { CPU_SET(c, &set); }
            for (auto & t : d->pool) { pthread_setaffinity_np(t.native_handle(), sizeof(set), &set); }
        }
    }
#endif
    LLAMA_LOG_INFO("%s: FLE_MOE_STORE: %zu slots in %zu size classes (%.1f GB, largest %.1f MB), %d unbuffered readers, %zu layers\n", __func__,
        n_slots, d->cls_blob.size(), d->arena_bytes / 1e9, d->blob / 1e6, std::max(1, threads), d->layers.size());
    for (size_t c = 0; c < d->cls_blob.size(); ++c) {
        LLAMA_LOG_INFO("%s: FLE_MOE_STORE: class %zu: %zu slots of %.2f MB\n", __func__, c, d->cls_hi[c] - d->cls_lo[c], d->cls_blob[c] / 1e6);
    }
    return std::unique_ptr<llama_expert_store>(new llama_expert_store(d));
#endif
}

llama_expert_store::~llama_expert_store() {
    if (!d) { return; }
    std::fprintf(stderr, "[fle-store] hits %llu, misses %llu, evictions %llu, read %.1f GB, wait %.0f ms, preload %llu in %.0f ms, prefetch %llu (used %llu, wasted %llu), read errors %llu\n",
        (unsigned long long) st.hits, (unsigned long long) st.misses, (unsigned long long) st.evictions, st.read_bytes / 1e9, st.wait_ms,
        (unsigned long long) st.preloaded, st.preload_ms, (unsigned long long) st.pre_issued, (unsigned long long) st.pre_used,
        (unsigned long long) st.pre_wasted, (unsigned long long) d->read_err);
    {
        std::lock_guard<std::mutex> lk(d->mtx);
        d->stop = true;
    }
    d->cv.notify_all();
    for (auto & t : d->pool) { t.join(); }
#ifndef _WIN32
    if (d->arena) { munmap(d->arena, d->arena_bytes); }
    for (int fd : d->fds) { ::close(fd); }
#endif
    delete d;
}

size_t llama_expert_store::expert_bytes(size_t li) const {
    const auto & L = d->layers[li];
    return L.r[0].nb2 + L.r[1].nb2 + L.r[2].nb2;
}

size_t llama_expert_store::n_slots() const { return d->slots.size(); }

bool llama_expert_store::resident(size_t li, int32_t e) const { return d->layers[li].slot_of[(size_t) e] >= 0; }

size_t llama_expert_store::preload(const std::vector<std::pair<int32_t, int32_t>> & li_expert) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<es_job> jobs;
    struct pend { size_t li; int32_t e; size_t s; const char * p[3]; };
    std::vector<pend> ps;
    size_t n = 0;
    for (const auto & x : li_expert) {
        const size_t li = (size_t) x.first;
        if (li >= d->layers.size() || x.second < 0 || x.second >= d->layers[li].n_expert || resident(li, x.second)) { continue; }
        const size_t fs = d->cls_free(d->layers[li].cls);
        if (fs == SIZE_MAX) { continue; }   // this class is full; others may still take their hot experts
        pend p{ li, x.second, fs, { nullptr, nullptr, nullptr } };
        d->queue_reads(li, x.second, p.s, jobs, p.p);
        ps.push_back(p);
        st.read_bytes += expert_bytes(li);
        if (ps.size() >= 256) {
            d->run(jobs);
            for (auto & q : ps) { d->assign(q.li, q.e, q.s, q.p); }
            n += ps.size(); ps.clear();
        }
    }
    d->run(jobs);
    for (auto & q : ps) { d->assign(q.li, q.e, q.s, q.p); }
    n += ps.size();
    st.preloaded += n;
    st.preload_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return n;
}

void llama_expert_store::ensure(size_t li, const int32_t * experts, size_t n, std::vector<std::pair<int32_t, int32_t>> * evicted) {
    if (li >= d->layers.size()) { return; }
    std::lock_guard<std::mutex> slk(d->smtx);
    impl::layer_t & L = d->layers[li];
    d->pin_li = (int32_t) li;
    d->pin.assign(experts, experts + n);
    std::vector<es_job> jobs;
    struct pend { int32_t e; size_t s; const char * p[3]; };
    pend ps[64]; size_t np = 0;
    for (size_t i = 0; i < n && np < 64; ++i) {
        const int32_t e = experts[i];
        if (e < 0 || e >= L.n_expert) { continue; }
        const int32_t s = L.slot_of[(size_t) e];
        if (s >= 0) {
            auto & S = d->slots[(size_t) s];
            if (!L.ptr[0][(size_t) e]) {   // prefetched: wait for its reads, then publish
                const auto t0 = std::chrono::steady_clock::now();
                while (d->inflight[(size_t) s].load(std::memory_order_acquire) > 0) { std::this_thread::yield(); }
                st.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                for (int r = 0; r < 3; ++r) { L.ptr[r][(size_t) e] = S.staged[r]; }
            }
            if (S.pre == 1) { S.pre = 2; st.pre_used++; }
            st.hits++;
            if (S.cnt < 0x7fffffff) { S.cnt++; }
            continue;
        }
        bool dup = false;
        for (size_t k = 0; k < np; ++k) { if (ps[k].e == e) { dup = true; break; } }
        if (dup) { continue; }
        // a slot: a never-used one first, then the least used of a sample, never one this call needs
        size_t victim = d->cls_free(L.cls);
        if (victim == SIZE_MAX) {
            uint32_t best = UINT32_MAX;
            const size_t ns = d->cls_hi[(size_t) L.cls] - d->cls_lo[(size_t) L.cls];
            for (size_t k = 0; k < 64 && ns; ++k) {
                const size_t c = d->cls_sample(L.cls, k);
                const auto & S = d->slots[c];
                if (d->inflight[c].load(std::memory_order_acquire) > 0 || d->pins[c].load(std::memory_order_acquire) > 0) { continue; }
                if (S.li == (int32_t) li) {
                    bool needed = false;
                    for (size_t j = 0; j < n; ++j) { if (experts[j] == S.e) { needed = true; break; } }
                    for (size_t j = 0; j < np && !needed; ++j) { if (ps[j].s == c) { needed = true; } }
                    if (needed) { continue; }
                }
                bool taken = false;
                for (size_t j = 0; j < np; ++j) { if (ps[j].s == c) { taken = true; break; } }
                if (taken) { continue; }
                if (S.cnt < best) { best = S.cnt; victim = c; }
            }
            d->cursor = (d->cursor + 61) % std::max<size_t>(1, d->slots.size());
            if (victim == SIZE_MAX) { continue; }   // no slot: this expert stays on the mmap path
            st.evictions++;
        }
        pend p{ e, victim, { nullptr, nullptr, nullptr } };
        d->queue_reads(li, e, victim, jobs, p.p);
        ps[np++] = p;
        st.misses++;
        st.read_bytes += expert_bytes(li);
    }
    if (np) {
        // the victims stop being visible before their bytes change
        for (size_t k = 0; k < np; ++k) {
            auto & S = d->slots[ps[k].s];
            if (S.li >= 0) {
                if (S.pre == 1) { st.pre_wasted++; }
                if (evicted) { evicted->push_back({ S.li, S.e }); }
                auto & O = d->layers[(size_t) S.li];
                O.slot_of[(size_t) S.e] = -1;
                for (int r = 0; r < 3; ++r) { O.ptr[r][(size_t) S.e] = nullptr; }
                S.li = -1; S.e = -1;
            }
        }
        static const bool async = [] { const char * e = std::getenv("FLE_MOE_STORE_ASYNC"); return e && atoi(e) != 0; }();
        if (async) {
            // FLE_MOE_STORE_ASYNC: the reads go to the front of the queue and ensure returns; mul_mat_id computes the resident
            // experts first and data_cb waits for each missing one (Strata's order), the GPU-cache upload sees it once published
            for (size_t pi = 0; pi < np; ++pi) {
                auto & S = d->slots[ps[pi].s];
                S.li = (int32_t) li; S.e = ps[pi].e; S.cnt = 1; S.pre = 0;
                for (int r = 0; r < 3; ++r) { S.staged[r] = ps[pi].p[r]; }
                L.slot_of[(size_t) ps[pi].e] = (int32_t) ps[pi].s;
            }
            // queue_reads appends three reads (gate, up, down) per expert, in ps order
            GGML_ASSERT(jobs.size() == 3*np);
            for (size_t pi = 0; pi < np; ++pi) {
                for (int r = 0; r < 3; ++r) { jobs[3*pi + r].ctr = &d->inflight[ps[pi].s]; }
                d->inflight[ps[pi].s].store(3, std::memory_order_release);
            }
            {
                std::lock_guard<std::mutex> lk(d->mtx);
                for (auto it = jobs.rbegin(); it != jobs.rend(); ++it) { d->q.push_front(*it); }   // demand before prefetch
            }
            d->cv.notify_all();
        } else {
            const auto t0 = std::chrono::steady_clock::now();
            d->run(jobs);
            st.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            for (size_t k = 0; k < np; ++k) { d->assign(li, ps[k].e, ps[k].s, ps[k].p); }
        }
    }
    if (++d->n_ensure % 8192 == 0) {   // age: old popularity must not hold slots forever
        for (auto & S : d->slots) { S.cnt >>= 1; }
    }
    if (d->n_ensure % 40960 == 0) {
        std::fprintf(stderr, "[fle-store] hits %llu, misses %llu (%.1f%% hit), evictions %llu, read %.1f GB, wait %.0f ms, prefetch %llu used %llu wasted %llu\n",
            (unsigned long long) st.hits, (unsigned long long) st.misses,
            100.0 * st.hits / std::max<uint64_t>(1, st.hits + st.misses), (unsigned long long) st.evictions, st.read_bytes / 1e9, st.wait_ms,
            (unsigned long long) st.pre_issued, (unsigned long long) st.pre_used, (unsigned long long) st.pre_wasted);
    }
}

size_t llama_expert_store::queue_len() const {
    std::lock_guard<std::mutex> lk(d->mtx);
    return d->q.size();
}

bool llama_expert_store::prefetch(size_t li, int32_t e, std::vector<std::pair<int32_t, int32_t>> * evicted) {
    if (li >= d->layers.size()) { return false; }
    std::lock_guard<std::mutex> slk(d->smtx);
    impl::layer_t & L = d->layers[li];
    if (e < 0 || e >= L.n_expert || L.slot_of[(size_t) e] >= 0) { return false; }
    // a slot: a never-used one, else the least used of a sample that is neither being read nor needed by the running layer
    size_t victim = d->cls_free(L.cls);
    if (victim == SIZE_MAX) {
        uint32_t best = UINT32_MAX;
        const size_t ns = d->cls_hi[(size_t) L.cls] - d->cls_lo[(size_t) L.cls];
        for (size_t k = 0; k < 64 && ns; ++k) {
            const size_t c = d->cls_sample(L.cls, k);
            const auto & S = d->slots[c];
            if (d->inflight[c].load(std::memory_order_acquire) > 0 || d->pins[c].load(std::memory_order_acquire) > 0) { continue; }
            if (S.li == d->pin_li && std::find(d->pin.begin(), d->pin.end(), S.e) != d->pin.end()) { continue; }
            if (S.cnt < best) { best = S.cnt; victim = c; }
        }
        d->cursor = (d->cursor + 61) % std::max<size_t>(1, d->slots.size());
        if (victim == SIZE_MAX || best > 1) { return false; }   // only cold slots: a prefetch never pushes out a used expert
        st.evictions++;
    }
    auto & S = d->slots[victim];
    if (S.li >= 0) {
        if (S.pre == 1) { st.pre_wasted++; }
        if (evicted) { evicted->push_back({ S.li, S.e }); }
        auto & O = d->layers[(size_t) S.li];
        O.slot_of[(size_t) S.e] = -1;
        for (int r = 0; r < 3; ++r) { O.ptr[r][(size_t) S.e] = nullptr; }
    }
    std::vector<es_job> jobs;
    const char * p[3];
    d->queue_reads(li, e, victim, jobs, p);
    S.li = (int32_t) li; S.e = e; S.cnt = 0; S.pre = 1;
    for (int r = 0; r < 3; ++r) { S.staged[r] = p[r]; }
    L.slot_of[(size_t) e] = (int32_t) victim;   // known, not yet visible (ptr stays nullptr until ensure publishes it)
    d->inflight[victim].store((int) jobs.size(), std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(d->mtx);
        for (auto & j : jobs) { j.ctr = &d->inflight[victim]; d->q.push_back(j); }
    }
    d->cv.notify_all();
    st.pre_issued++;
    st.read_bytes += expert_bytes(li);
    return true;
}

const char * llama_expert_store::acquire(const ggml_tensor * src, int32_t e, int32_t * slot) {
    const auto it = d->by_data.find(src->data);
    if (it == d->by_data.end()) { return nullptr; }
    std::lock_guard<std::mutex> slk(d->smtx);
    const auto & L = d->layers[(size_t) it->second.first];
    if (e < 0 || e >= L.n_expert) { return nullptr; }
    const int32_t s = L.slot_of[(size_t) e];
    const char * p = L.ptr[it->second.second][(size_t) e];
    if (s < 0 || !p) { return nullptr; }
    d->pins[(size_t) s].fetch_add(1, std::memory_order_acq_rel);
    *slot = s;
    return p;
}

void llama_expert_store::release(int32_t slot) {
    if (slot >= 0) { d->pins[(size_t) slot].fetch_sub(1, std::memory_order_acq_rel); }
}

const void * llama_expert_store::data_cb(const ggml_tensor * src0, int32_t expert, void * ud) {
    const auto * s = (const llama_expert_store *) ud;
    const auto it = s->d->by_data.find(src0->data);
    if (it == s->d->by_data.end()) { return nullptr; }
    const auto & L = s->d->layers[(size_t) it->second.first];
    if (expert < 0 || expert >= L.n_expert) { return nullptr; }
    const char * p = L.ptr[it->second.second][(size_t) expert];
    if (p) { return p; }
    const int32_t sl = L.slot_of[(size_t) expert];
    if (sl < 0) { return nullptr; }
    // FLE_MOE_STORE_ASYNC: read still running (or a prefetch not yet published): wait for this expert only, then publish it
    auto & S = s->d->slots[(size_t) sl];
    if (S.li != (int32_t) it->second.first || S.e != expert) { return nullptr; }
    const auto t0 = std::chrono::steady_clock::now();
    while (s->d->inflight[(size_t) sl].load(std::memory_order_acquire) > 0) { std::this_thread::yield(); }
    auto & st = const_cast<llama_expert_store *>(s)->st;
    st.wait_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    auto & LW = const_cast<impl::layer_t &>(L);
    for (int r = 0; r < 3; ++r) { LW.ptr[r][(size_t) expert] = S.staged[r]; }
    return S.staged[it->second.second];
}

bool llama_expert_store::ready_cb(const ggml_tensor * src0, int32_t expert, void * ud) {
    const auto * s = (const llama_expert_store *) ud;
    const auto it = s->d->by_data.find(src0->data);
    if (it == s->d->by_data.end()) { return true; }
    const auto & L = s->d->layers[(size_t) it->second.first];
    if (expert < 0 || expert >= L.n_expert) { return true; }
    if (L.ptr[it->second.second][(size_t) expert]) { return true; }
    const int32_t sl = L.slot_of[(size_t) expert];
    return sl < 0 || s->d->inflight[(size_t) sl].load(std::memory_order_acquire) == 0;
}
