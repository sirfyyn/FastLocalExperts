// Standalone test of the FLE v2 ngram table: g++ -std=c++17 -O2 -I common tests/test-ngram-mod-v2.cpp common/ngram-mod.cpp
#include "ngram-mod.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { std::printf("FAIL  %s\n", msg); fails++; } else { std::printf("ok    %s\n", msg); } } while (0)

static std::vector<int32_t> ctx(uint64_t seed, size_t n) {
    std::mt19937_64 g(seed);
    std::vector<int32_t> v(n + 1);
    for (auto & x : v) { x = (int32_t) (g() % 150000); }
    return v;
}

int main() {
    const size_t N = 16;
    {   // 1: add / get / newest wins
        common_ngram_mod m(N, 1 << 16);
        auto a = ctx(1, N); m.add(a.data());
        CHECK(m.get(a.data()) == a[N], "add then get returns the continuation");
        auto a2 = a; a2[N] = 4242; m.add(a2.data());
        CHECK(m.get(a.data()) == 4242, "the newest continuation of a context wins");
        auto b = ctx(2, N);
        CHECK(m.get(b.data()) == common_ngram_mod::EMPTY, "unknown context -> EMPTY");
    }
    {   // 2: tags: with a tiny table (one bucket of 4) other contexts must not return a wrong token
        common_ngram_mod m(N, 4);
        for (uint64_t s = 10; s < 14; ++s) { auto c = ctx(s, N); m.add(c.data()); }
        int wrong = 0, hits = 0;
        for (uint64_t s = 100; s < 2100; ++s) { auto c = ctx(s, N); auto r = m.get(c.data()); if (r != common_ngram_mod::EMPTY) { wrong++; } }
        for (uint64_t s = 10; s < 14; ++s) { auto c = ctx(s, N); hits += m.get(c.data()) == c[N]; }
        CHECK(wrong == 0, "foreign contexts in a full bucket are not answered (old table: every one was)");
        CHECK(hits == 4, "all four resident contexts are found");
    }
    {   // 3: age replacement: after 3x capacity of inserts the most recent ones are still there
        const size_t cap = 1 << 14;
        common_ngram_mod m(N, cap);
        const size_t total = cap * 3;
        for (uint64_t s = 0; s < total; ++s) { auto c = ctx(1000 + s, N); m.add(c.data()); }
        size_t recent = 0, tail = cap / 4;
        for (uint64_t s = total - tail; s < total; ++s) { auto c = ctx(1000 + s, N); recent += m.get(c.data()) == c[N]; }
        double frac = (double) recent / tail;
        std::printf("      newest quarter of the capacity still resident: %.1f %%\n", 100 * frac);
        CHECK(frac > 0.90, "the newest quarter of the capacity survives 3x overfill");
        size_t old = 0;
        for (uint64_t s = 0; s < tail; ++s) { auto c = ctx(1000 + s, N); old += m.get(c.data()) == c[N]; }
        CHECK(old < tail / 4, "the oldest entries have been forgotten");
    }
    {   // 4: persistence round trip, other size rejected, damage rejected
        const std::string path = "/tmp/ngram-v2-test.bin";
        common_ngram_mod m(N, 1 << 12);
        for (uint64_t s = 0; s < 1000; ++s) { auto c = ctx(5000 + s, N); m.add(c.data()); }
        CHECK(m.save(path), "save");
        common_ngram_mod m2(N, 1 << 12);
        CHECK(m2.load(path), "load into an equal table");
        size_t same = 0; for (uint64_t s = 0; s < 1000; ++s) { auto c = ctx(5000 + s, N); same += m2.get(c.data()) == m.get(c.data()); }
        CHECK(same == 1000 && m2.get_used() == m.get_used(), "loaded table answers like the saved one");
        common_ngram_mod m3(N, 1 << 13);
        CHECK(!m3.load(path), "a table of another size does not take the file");
        common_ngram_mod m4(N + 1, 1 << 12);
        CHECK(!m4.load(path), "a table of another n does not take the file");
        FILE * f = std::fopen(path.c_str(), "r+b"); std::fseek(f, 100, SEEK_SET); std::fputc(0x7f, f); std::fclose(f);
        common_ngram_mod m5(N, 1 << 12);
        CHECK(!m5.load(path) && m5.get_used() == 0, "a damaged file is rejected and leaves the table untouched");
        CHECK(!m5.load("/tmp/does-not-exist-ngram"), "a missing file is not an error");
    }
    {   // 5: the policy change on a synthetic agent workload. Sessions share a fixed block (system prompt, tool schema)
        // and add fresh text. The old policy reset the whole table above 25 % occupancy.
        const size_t cap = 1 << 16, session = 60000, shared = 12000;
        auto run = [&](bool old_policy) {
            common_ngram_mod m(N, cap);
            std::vector<int32_t> block; { std::mt19937_64 g(7); for (size_t i = 0; i < shared + N; ++i) block.push_back((int32_t) (g() % 150000)); }
            double sum = 0; int cnt = 0;
            for (int sess = 0; sess < 12; ++sess) {
                std::mt19937_64 g(900 + sess);
                std::vector<int32_t> text;
                for (size_t i = 0; i < session; ++i) { text.push_back((int32_t) (g() % 150000)); }
                // the shared block first: how much of it does the table already know?
                size_t known = 0; for (size_t i = 0; i + N < block.size(); ++i) { known += m.get(block.data() + i) == block[i + N]; }
                if (sess > 0) { sum += (double) known / (block.size() - N); cnt++; }
                for (size_t i = 0; i + N < block.size(); ++i) { m.add(block.data() + i); }
                for (size_t i = 0; i + N < text.size(); ++i) { m.add(text.data() + i); if (old_policy && m.get_used() > cap / 4) { m.reset(); } }
            }
            return sum / cnt;
        };
        double old_k = run(true), new_k = run(false);
        std::printf("      shared block known at the start of a session: old policy %.1f %%, new policy %.1f %%\n", 100 * old_k, 100 * new_k);
        CHECK(new_k > old_k + 0.2, "knowledge of the shared block survives sessions (reset-at-25%% loses it)");
    }
    std::printf(fails ? "\n%d FAILED\n" : "\nalle bestanden\n", fails);
    return fails ? 1 : 0;
}
