// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#include "rules.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <numeric>
#include <regex>
#include <sstream>

namespace eworld {
namespace {

const std::set<std::string> kArmor  = {"tank", "apc", "humvee", "artillery"};
const std::set<std::string> kGround = {"car", "truck", "bus", "motorcycle", "crate", "bridge",
                                       "water-tower", "storage-tank"};
const std::set<std::string> kAir    = {"airplane", "helicopter"};
const std::set<std::string> kEcho   = {"TEXT", "IN", "THE", "FIGURE", "OF", "AS"};

std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string alnum_upper(const std::string& s) {
    std::string out;
    for (unsigned char c : s)
        if (std::isalnum(c)) out.push_back(static_cast<char>(std::toupper(c)));
    return out;
}

std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream in(s);
    std::vector<std::string> out;
    for (std::string w; in >> w;) out.push_back(w);
    return out;
}

std::string bare_name(const std::string& label) {
    if (label.rfind("weapon:", 0) == 0) return label.substr(7);
    return lower(label);
}

bool intersects(const std::set<std::string>& a, const std::set<std::string>& b) {
    for (const auto& x : a)
        if (b.count(x)) return true;
    return false;
}

}  // namespace

std::optional<std::string> ocr_token(const std::string& text) {
    const std::string chars = alnum_upper(text);
    if (chars.size() < 2 || chars.size() > 16) return std::nullopt;
    return "ocr:" + chars;
}

double area(const double xyxy[4]) {
    return std::max(0.0, xyxy[2] - xyxy[0]) * std::max(0.0, xyxy[3] - xyxy[1]);
}

double iou(const double a[4], const double b[4]) {
    const double iw = std::max(0.0, std::min(a[2], b[2]) - std::max(a[0], b[0]));
    const double ih = std::max(0.0, std::min(a[3], b[3]) - std::max(a[1], b[1]));
    const double inter = iw * ih;
    const double u = area(a) + area(b) - inter;
    return u > 0 ? inter / u : 0.0;
}

std::string scene_tag(const std::set<std::string>& class_names) {
    std::set<std::string> names;
    for (const auto& label : class_names) {
        if (label.empty() || label.rfind("ocr:", 0) == 0) continue;
        names.insert(bare_name(label));
    }
    if (names.empty()) return "";
    std::set<std::string> ground_or_armor = kGround;
    ground_or_armor.insert(kArmor.begin(), kArmor.end());
    const bool air    = intersects(names, kAir);
    const bool water  = names.count("boat") > 0;
    const bool ground = intersects(names, ground_or_armor);
    const bool person = names.count("person") > 0;
    if (intersects(names, kArmor)) return "scene:armor";
    if (air && !ground) return "scene:air";
    if (water && !ground && !air) return "scene:water";
    if (ground && person) return "scene:ground-with-person";
    if (ground || intersects(names, {"rifle", "machine-gun", "pistol"})) return "scene:ground";
    if (person) return "scene:person";
    return "scene:other";
}

bool keep_stencil(const std::string& token) {
    if (token.empty() || kEcho.count(token)) return false;
    if (token.size() < 2 || token.size() > 16) return false;
    // A decoder stuck repeating itself ("KKKWKWWKKHKW") is not a stencil, and
    // neither is a letters-only read of blur ("OOCO", "CKKK"): a word has
    // more than half its letters distinct. Digits repeat in real plates.
    const size_t distinct = std::set<char>(token.begin(), token.end()).size();
    if (token.size() >= 6 && distinct * 3 < token.size()) return false;
    if (std::any_of(token.begin(), token.end(), [](unsigned char c) { return std::isdigit(c); })) return true;
    return token.size() >= 4 && distinct * 2 > token.size();
}

std::vector<std::string> figure_words(const std::string& text) {
    // Python: re.search(r"Text in the figure:\s*(.*?)\s*$", text, re.I). `.`
    // stops at a newline and `$` is the end of the text, so this is the line
    // only when it is the last one.
    static const std::regex re(R"(Text in the figure:\s*(.*?)\s*$)", std::regex::icase);
    std::smatch m;
    std::vector<std::string> words;
    if (!std::regex_search(text, m, re)) return words;
    for (const auto& raw : split_ws(m[1].str())) {
        const std::string token = alnum_upper(raw);
        if (!keep_stencil(token) || std::find(words.begin(), words.end(), token) != words.end()) continue;
        words.push_back(token);
        if (words.size() >= 8) break;
    }
    return words;
}

namespace {
size_t edit_distance(const std::string& a, const std::string& b) {
    std::vector<size_t> row(b.size() + 1);
    std::iota(row.begin(), row.end(), size_t{0});
    for (size_t i = 1; i <= a.size(); ++i) {
        size_t diag = row[0];
        row[0] = i;
        for (size_t j = 1; j <= b.size(); ++j) {
            const size_t up = row[j];
            row[j] = std::min({row[j] + 1, row[j - 1] + 1, diag + (a[i - 1] == b[j - 1] ? 0 : 1)});
            diag = up;
        }
    }
    return row[b.size()];
}
}  // namespace

std::vector<std::string> corroborate(const std::vector<Read>& reads) {
    struct Cluster {
        std::vector<std::string> spellings;   // every member seen
        std::set<size_t>         reads;       // which reads support it
        std::map<std::string, int> florence;  // Florence spelling -> count
        std::map<std::string, int> any;       // any spelling -> count
    };
    std::vector<Cluster> clusters;
    for (size_t r = 0; r < reads.size(); ++r)
        for (const auto& w : reads[r].words) {
            Cluster* home = nullptr;
            for (auto& c : clusters) {
                for (const auto& s : c.spellings)
                    if (edit_distance(w, s) * 2 <= std::max(w.size(), s.size())) { home = &c; break; }
                if (home) break;
            }
            if (!home) home = &clusters.emplace_back();
            if (std::find(home->spellings.begin(), home->spellings.end(), w) == home->spellings.end())
                home->spellings.push_back(w);
            home->reads.insert(r);
            ++home->any[w];
            if (reads[r].florence) ++home->florence[w];
        }
    // Only reads that saw text can disagree: a view of the side panel does
    // not veto the tailgate's AMMO.
    const auto seeing = std::count_if(reads.begin(), reads.end(), [](const Read& r) { return !r.words.empty(); });
    const size_t need = seeing > 1 ? 2 : 1;
    std::vector<std::pair<size_t, std::string>> kept;   // support, spelling
    for (const auto& c : clusters) {
        if (c.reads.size() < need) continue;
        const auto& pool = c.florence.empty() ? c.any : c.florence;
        const auto top = std::max_element(pool.begin(), pool.end(), [&](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second < b.second;
            return a.first.size() < b.first.size();   // tie: the fuller spelling
        });
        kept.emplace_back(c.reads.size(), top->first);
    }
    std::stable_sort(kept.begin(), kept.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    std::vector<std::string> out;
    for (const auto& [n, w] : kept)
        if (static_cast<int>(out.size()) < kMaxOcrTags) out.push_back(w);
    return out;
}

}  // namespace eworld
