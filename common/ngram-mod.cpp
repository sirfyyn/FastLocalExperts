#include "ngram-mod.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

//
// common_ngram_mod
//

static uint64_t mix64(uint64_t x) { // splitmix64 finalizer: the old multiplicative hash alone leaves the low bits weak
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

common_ngram_mod::common_ngram_mod(uint16_t n, size_t size) : n(n), used(0), clock(0) {
    const size_t buckets = std::max<size_t>(1, size / WAYS);
    slots.resize(buckets * WAYS);

    reset();
}

void common_ngram_mod::locate(const entry_t * tokens, size_t & bucket, uint32_t & tag) const {
    uint64_t res = 0;

    for (size_t i = 0; i < n; ++i) {
        res = res*6364136223846793005ULL + (uint64_t) (uint32_t) tokens[i];
    }

    res    = mix64(res);
    bucket = (size_t) (res % (slots.size() / WAYS));
    tag    = (uint32_t) (res >> 32) | 1u; // never 0
}

void common_ngram_mod::add(const entry_t * tokens) {
    size_t bucket; uint32_t tag;
    locate(tokens, bucket, tag);

    slot_t * b = &slots[bucket * WAYS];
    clock++;

    slot_t * victim = &b[0];
    for (size_t w = 0; w < WAYS; ++w) {
        if (b[w].tag == tag) { // same context: the newest continuation wins
            b[w].token = tokens[n];
            b[w].stamp = clock;
            return;
        }
        if (b[w].tag == 0) {
            victim = &b[w];
            break;
        }
        if (b[w].stamp < victim->stamp) {
            victim = &b[w];
        }
    }

    if (victim->tag == 0) {
        used++;
    }
    victim->tag   = tag;
    victim->token = tokens[n];
    victim->stamp = clock;
}

common_ngram_mod::entry_t common_ngram_mod::get(const entry_t * tokens) const {
    size_t bucket; uint32_t tag;
    locate(tokens, bucket, tag);

    const slot_t * b = &slots[bucket * WAYS];
    for (size_t w = 0; w < WAYS; ++w) {
        if (b[w].tag == tag) {
            return b[w].token;
        }
    }

    return EMPTY;
}

void common_ngram_mod::reset() {
    std::fill(slots.begin(), slots.end(), slot_t{0, EMPTY, 0});
    used  = 0;
    clock = 0;
}

size_t common_ngram_mod::get_n() const {
    return n;
}

size_t common_ngram_mod::get_used() const {
    return used;
}

size_t common_ngram_mod::size() const {
    return slots.size();
}

size_t common_ngram_mod::size_bytes() const {
    return slots.size() * sizeof(slot_t);
}

static const char NGRAM_MAGIC[8] = {'F','L','E','N','G','M','2','\0'};

static uint64_t fnv1a(const void * p, size_t len, uint64_t h = 1469598103934665603ULL) {
    const uint8_t * b = (const uint8_t *) p;
    for (size_t i = 0; i < len; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

bool common_ngram_mod::save(const std::string & path) const {
    const std::string tmp = path + ".tmp";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        return false;
    }
    const uint64_t hdr[3] = { (uint64_t) n, (uint64_t) slots.size(), (uint64_t) clock };
    uint64_t sum = fnv1a(hdr, sizeof(hdr));
    sum = fnv1a(slots.data(), slots.size() * sizeof(slot_t), sum);
    bool ok = std::fwrite(NGRAM_MAGIC, 1, sizeof(NGRAM_MAGIC), f) == sizeof(NGRAM_MAGIC)
           && std::fwrite(hdr, sizeof(hdr), 1, f) == 1
           && std::fwrite(slots.data(), sizeof(slot_t), slots.size(), f) == slots.size()
           && std::fwrite(&sum, sizeof(sum), 1, f) == 1;
    ok = (std::fclose(f) == 0) && ok;
    if (!ok || std::rename(tmp.c_str(), path.c_str()) != 0) {
        std::remove(tmp.c_str());
        return false;
    }
    return true;
}

bool common_ngram_mod::load(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    char magic[sizeof(NGRAM_MAGIC)];
    uint64_t hdr[3] = {0, 0, 0};
    bool ok = std::fread(magic, 1, sizeof(magic), f) == sizeof(magic) && std::memcmp(magic, NGRAM_MAGIC, sizeof(magic)) == 0
           && std::fread(hdr, sizeof(hdr), 1, f) == 1 && hdr[0] == n && hdr[1] == slots.size();
    std::vector<slot_t> tmp;
    uint64_t sum = 0;
    if (ok) {
        tmp.resize(slots.size());
        ok = std::fread(tmp.data(), sizeof(slot_t), tmp.size(), f) == tmp.size() && std::fread(&sum, sizeof(sum), 1, f) == 1;
    }
    std::fclose(f);
    if (ok) {
        uint64_t want = fnv1a(hdr, sizeof(hdr));
        want = fnv1a(tmp.data(), tmp.size() * sizeof(slot_t), want);
        ok = want == sum;
    }
    if (!ok) {
        return false;
    }
    slots = std::move(tmp);
    clock = (uint32_t) hdr[2];
    used  = 0;
    for (const slot_t & s : slots) {
        used += s.tag != 0;
    }
    return true;
}
