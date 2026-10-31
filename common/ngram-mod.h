#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

//
// common_ngram_mod
// ref: https://github.com/ggml-org/llama.cpp/pull/19164
//
// FLE v2 (2026-10-31): the table keeps the same API but no longer overwrites blindly.
//   - Buckets of WAYS slots, each slot carries a 32-bit tag of the hashed context. A lookup that hits another context's
//     bucket no longer returns that context's token (the old table returned it, and the verifier had to reject it).
//   - Replacement is by age (oldest slot of the bucket), so the table forgets slowly and never has to be reset when it
//     fills up. Knowledge from earlier prompts survives (the old policy wiped everything above 25 % occupancy).
//   - save()/load(): the table can live in a file and be warm-started.
//
struct common_ngram_mod {
    using entry_t = int32_t;

    static constexpr entry_t EMPTY = -1;
    static constexpr size_t  WAYS  = 4;

    common_ngram_mod(uint16_t n, size_t size);

    void    add(const entry_t * tokens);
    entry_t get(const entry_t * tokens) const; // return -1 if not found

    void reset();

    size_t get_n()    const;
    size_t get_used() const;

    size_t size()       const; // number of slots
    size_t size_bytes() const;

    // File format: magic, n, number of slots, clock, data, FNV-1a checksum. Written to path+".tmp", then renamed.
    bool save(const std::string & path) const;
    // Returns false (and leaves the table untouched) if the file is missing, belongs to another n/size or is damaged.
    bool load(const std::string & path);

private:
    struct slot_t {
        uint32_t tag;   // 0 = empty
        int32_t  token;
        uint32_t stamp; // insertion/refresh clock, lower = older
    };

    void     locate(const entry_t * tokens, size_t & bucket, uint32_t & tag) const;

    size_t   n; // ngram size to hash
    size_t   used;
    uint32_t clock;

    std::vector<slot_t> slots;
};
