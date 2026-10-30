// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
// scene_probe: a conjunction asked of the frame vectors by superposition.
//
//   scene_probe <out_dir> word [word...]   e.g.  scene_probe out people fuel
//
// Every word is grounded to EVERY token kind the index holds that it can
// name (all beam variants): the detector class as spelled, its singular
// (people -> person, trucks -> truck), the OCR word (fuel -> ocr:FUEL), the
// scene tag (ground -> scene:ground). The grounded tokens are superposed
// into ONE probe vector (bsc::superpose, the operation the encoder used to
// fuse the frame's own tokens into its vector) and the frame LSH is asked
// for its nearest frames. Each hit is then checked against the frame's own
// tag list: a frame that carries every grounded word is a confirmed hit.
// Prints the ranking with the check, and the recall against the tag ground
// truth, so the superposition's recall is measured, not assumed.
#include "ehc/core/bsc.hpp"
#include "ehc/core/sparse_vector.hpp"
#include "ehc/index/bsc_lsh_index.hpp"
#include "ehc/media/feature_hash.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {
std::string lower(std::string s) { for (char& c : s) c = (char)std::tolower((unsigned char)c); return s; }
std::string upper(std::string s) { for (char& c : s) c = (char)std::toupper((unsigned char)c); return s; }
std::string singular(const std::string& w) {
    static const std::map<std::string, std::string> irregular{{"people", "person"}, {"persons", "person"}};
    if (const auto it = irregular.find(w); it != irregular.end()) return it->second;
    if (w.size() > 3 && w.compare(w.size() - 3, 3, "ies") == 0) return w.substr(0, w.size() - 3) + "y";
    if (w.size() > 2 && w.back() == 's' && w[w.size() - 2] != 's') return w.substr(0, w.size() - 1);
    return w;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: scene_probe <out_dir> word [word...]\n"); return 2; }
    const std::string out = argv[1];
    json cfg;
    { std::ifstream f(out + "/config.json"); f >> cfg; }
    const int dim = cfg.at("dim").get<int>(), k = cfg.at("k").get<int>();
    const uint64_t tok_seed = cfg.value("detector_token_seed", uint64_t{2400});

    // The frames and their tags, by LSH row.
    struct Row { int frame_idx; std::vector<std::string> tags; };
    std::vector<Row> rows;
    std::set<std::string> vocab;
    {
        std::ifstream f(out + "/frames.jsonl");
        for (std::string line; std::getline(f, line);) {
            if (line.empty()) continue;
            const json j = json::parse(line);
            Row r{j.value("frame_idx", -1), {}};
            for (const auto& t : j.value("tags", json::array())) { r.tags.push_back(t.get<std::string>()); vocab.insert(r.tags.back()); }
            const size_t id = j.value("row_id", rows.size());
            if (rows.size() <= id) rows.resize(id + 1);
            rows[id] = std::move(r);
        }
    }

    // Ground every word to every token kind it can name in this index.
    std::vector<std::vector<std::string>> groundings;   // per word
    std::vector<std::string> tokens;                    // the union, superposed
    for (int i = 2; i < argc; ++i) {
        const std::string w = lower(argv[i]), s = singular(w);
        std::vector<std::string> g;
        for (const std::string& cand : {w, s, "ocr:" + upper(w), "ocr:" + upper(s), "scene:" + w, "scene:" + s})
            if (vocab.count(cand) && std::find(g.begin(), g.end(), cand) == g.end()) g.push_back(cand);
        std::printf("word %-12s -> %s\n", argv[i], g.empty() ? "(nothing in this index)" : "");
        for (const auto& t : g) std::printf("    %s\n", t.c_str());
        groundings.push_back(g);
        for (const auto& t : g) if (std::find(tokens.begin(), tokens.end(), t) == tokens.end()) tokens.push_back(t);
    }
    if (tokens.empty()) { std::printf("no token to ask\n"); return 1; }

    // Ground truth from the tags: frames carrying one grounding of EVERY word.
    auto carries_all = [&](const Row& r) {
        for (const auto& g : groundings) {
            if (g.empty()) return false;
            bool any = false;
            for (const auto& t : g) any = any || std::find(r.tags.begin(), r.tags.end(), t) != r.tags.end();
            if (!any) return false;
        }
        return true;
    };
    std::set<int> truth;
    for (const auto& r : rows) if (carries_all(r)) truth.insert(r.frame_idx);

    // One probe: the superposition of the grounded tokens.
    std::vector<ehc::core::SparseVector> items;
    for (const auto& t : tokens) items.push_back(ehc::media::feature_to_sparse(t, dim, k, tok_seed));
    const ehc::core::SparseVector probe = ehc::bsc::superpose(items, k);
    auto lsh = ehc::index::BSCLSHIndex::load(out + "/frames.lsh");
    const auto hits = lsh.knn_query(probe, (int)rows.size(), 1, 100000);

    std::printf("\nprobe = superpose{");
    for (size_t i = 0; i < tokens.size(); ++i) std::printf("%s%s", i ? ", " : "", tokens[i].c_str());
    std::printf("}  over %lld frame vectors; %zu frames carry every word by their tags\n\n", (long long)lsh.size(), truth.size());
    size_t confirmed = 0, shown = 0;
    std::set<int> found;
    for (size_t i = 0; i < hits.ids.size(); ++i) {
        const int64_t id = hits.ids[i];
        if (id < 0 || (size_t)id >= rows.size()) continue;
        const Row& r = rows[(size_t)id];
        const bool ok = carries_all(r);
        if (ok) { ++confirmed; found.insert(r.frame_idx); }
        if (shown < 25 || ok) {
            if (shown < 25 || (ok && i < 60)) {
                std::printf("  #%-3zu frame %-4d score %.4f %s  [", i + 1, r.frame_idx, (double)hits.scores[i], ok ? "CONFIRMED" : "         ");
                for (size_t t = 0; t < r.tags.size() && t < 6; ++t) std::printf("%s%s", t ? " " : "", r.tags[t].c_str());
                std::printf("]\n");
            }
            ++shown;
        }
    }
    size_t first_miss = 0;
    for (size_t i = 0; i < hits.ids.size(); ++i) {
        const int64_t id = hits.ids[i];
        if (id >= 0 && (size_t)id < rows.size() && !carries_all(rows[(size_t)id])) { first_miss = i + 1; break; }
    }
    std::printf("\nconfirmed by tags among the LSH hits: %zu of %zu truth frames (%zu hits returned); "
                "first non-matching rank: %zu\n", confirmed, truth.size(), hits.ids.size(), first_miss);

    // The same probe against EVERY frame vector, no hashing: does the
    // superposition signal exist in the vectors at all? (326 frames is a
    // scan; the LSH is a shortcut, not the ground truth.)
    struct Scored { double s; size_t row; };
    std::vector<Scored> all;
    for (size_t id = 0; id < rows.size() && (int64_t)id < lsh.size(); ++id)
        all.push_back({(double)ehc::bsc::similarity(probe, lsh.get_vector_by_id((int64_t)id)), id});
    std::stable_sort(all.begin(), all.end(), [](const Scored& a, const Scored& b) { return a.s > b.s; });
    std::printf("\nbrute-force similarity of the probe to every frame vector (top 20, then where the truth frames rank):\n");
    for (size_t i = 0; i < all.size() && i < 20; ++i) {
        const Row& r = rows[all[i].row];
        std::printf("  #%-3zu frame %-4d sim %.4f %s  [", i + 1, r.frame_idx, all[i].s, carries_all(r) ? "CONFIRMED" : "         ");
        for (size_t t = 0; t < r.tags.size() && t < 6; ++t) std::printf("%s%s", t ? " " : "", r.tags[t].c_str());
        std::printf("]\n");
    }
    size_t in_top_n = 0, worst = 0;
    for (size_t i = 0; i < all.size(); ++i)
        if (carries_all(rows[all[i].row])) { if (i < truth.size()) ++in_top_n; worst = i + 1; }
    std::printf("  truth frames in the top %zu: %zu; the last truth frame ranks %zu of %zu\n", truth.size(), in_top_n, worst, all.size());
    return 0;
}
