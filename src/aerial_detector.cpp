// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#include "aerial_detector.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

namespace eworld {
namespace fs = std::filesystem;
using ehc::media::DetectedTag;
using ehc::media::FrameDetections;

namespace {

constexpr const char* kText = "text";

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

/// How much of `inner` lies inside `outer` (0..1).
double containment(const double inner[4], const double outer[4]) {
    const double iw = std::max(0.0, std::min(inner[2], outer[2]) - std::max(inner[0], outer[0]));
    const double ih = std::max(0.0, std::min(inner[3], outer[3]) - std::max(inner[1], outer[1]));
    const double a = area(inner);
    return a > 0 ? iw * ih / a : 0.0;
}

/// The object's crop with `pad` margin, upscaled so its short side is at
/// least 256 px (the door drops small figures; Tesseract reads small text
/// badly).
cv::Mat object_crop(const cv::Mat& bgr, const double xyxy[4], double pad) {
    const double w = xyxy[2] - xyxy[0], h = xyxy[3] - xyxy[1];
    const int x0 = std::max(0, static_cast<int>(std::floor(xyxy[0] - pad * w)));
    const int y0 = std::max(0, static_cast<int>(std::floor(xyxy[1] - pad * h)));
    const int x1 = std::min(bgr.cols, static_cast<int>(std::ceil(xyxy[2] + pad * w)));
    const int y1 = std::min(bgr.rows, static_cast<int>(std::ceil(xyxy[3] + pad * h)));
    if (x1 - x0 < 2 || y1 - y0 < 2) return {};
    cv::Mat crop = bgr(cv::Range(y0, y1), cv::Range(x0, x1)).clone();
    const int short_side = std::min(crop.cols, crop.rows);
    if (short_side < 256) {
        const double s = 256.0 / short_side;
        cv::resize(crop, crop, cv::Size(), s, s, cv::INTER_CUBIC);
    }
    return crop;
}

/// What a box looks like: the normalised hue-saturation histogram of its
/// crop (30 x 32 bins). Empty for a degenerate box.
cv::Mat appearance(const cv::Mat& bgr, const double xyxy[4]) {
    const int x0 = std::max(0, static_cast<int>(xyxy[0])), y0 = std::max(0, static_cast<int>(xyxy[1]));
    const int x1 = std::min(bgr.cols, static_cast<int>(xyxy[2])), y1 = std::min(bgr.rows, static_cast<int>(xyxy[3]));
    if (x1 - x0 < 2 || y1 - y0 < 2) return {};
    cv::Mat hsv, hist;
    cv::cvtColor(bgr(cv::Range(y0, y1), cv::Range(x0, x1)), hsv, cv::COLOR_BGR2HSV);
    const int channels[] = {0, 1}, bins[] = {30, 32};
    const float h_range[] = {0, 180}, s_range[] = {0, 256};
    const float* ranges[] = {h_range, s_range};
    cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, bins, ranges);
    cv::normalize(hist, hist, 1, 0, cv::NORM_L1);
    return hist;
}

/// Up to `n` survivors of an object, evenly spread over its frames (first
/// and last included), as indexes into its survivors / boxes.
std::vector<size_t> spread(size_t count, int n) {
    std::vector<size_t> out;
    if (count == 0 || n <= 0) return out;
    if (count <= static_cast<size_t>(n)) {
        for (size_t k = 0; k < count; ++k) out.push_back(k);
        return out;
    }
    if (n == 1) return {count / 2};
    for (int k = 0; k < n; ++k) {
        const size_t idx = static_cast<size_t>(std::lround(static_cast<double>(k) * (count - 1) / (n - 1)));
        if (out.empty() || out.back() != idx) out.push_back(idx);
    }
    return out;
}

}  // namespace

ObjectLayerConfig ObjectLayerConfig::load(const std::string& path, const std::string& root) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("object layer config not found: " + path);
    const auto j = nlohmann::json::parse(in);
    auto resolve = [&](const std::string& p) {
        return p.empty() || fs::path(p).is_absolute() ? p : (fs::path(root) / p).string();
    };
    ObjectLayerConfig c;
    const auto& coco = j.at("coco");
    c.coco_model = resolve(coco.at("model").get<std::string>());
    c.coco_conf  = coco.value("conf", 0.25f);
    for (const auto& k : coco.at("keep")) c.coco_keep.insert(k.get<std::string>());
    if (j.contains("world") && !j["world"].is_null()) {
        const auto& world = j["world"];
        c.world_model = resolve(world.at("model").get<std::string>());
        c.world_conf  = world.value("conf", 0.20f);
        for (const auto& [prompt, token] : world.at("prompts").items()) c.world_tokens[prompt] = token.get<std::string>();
    }
    c.merge_iou = j.value("merge_iou", 0.5);
    if (j.contains("track")) {
        c.track_iou            = j["track"].value("iou", 0.3);
        c.track_max_gap        = j["track"].value("max_gap", 2);
        c.track_min_appearance = j["track"].value("min_appearance", 0.5);
        int g = 0;
        for (const auto& group : j["track"].value("groups", nlohmann::json::array())) {
            for (const auto& label : group) c.track_group[label.get<std::string>()] = g;
            ++g;
        }
    }
    if (j.contains("text")) {
        const auto& t = j["text"];
        c.text_enabled          = t.value("enabled", true);
        c.text_psm              = t.value("psm", 11);
        c.text_min_conf         = t.value("min_conf", 40.0);
        c.text_min_height       = t.value("min_height", 10);
        c.text_pad              = t.value("pad", 0.3);
        c.text_host_containment = t.value("host_containment", 0.6);
    }
    if (j.contains("ocr")) {
        c.ocr_min_side = j["ocr"].value("min_side", 48);
        c.ocr_pad      = j["ocr"].value("pad", 0.1);
        c.ocr_views    = j["ocr"].value("views", 3);
        for (const auto& k : j["ocr"].value("skip", nlohmann::json::array())) c.ocr_skip.insert(k.get<std::string>());
    }
    return c;
}

bool ObjectLayerConfig::same_group(const std::string& a, const std::string& b) const {
    if (a == b) return true;
    const auto ia = track_group.find(a), ib = track_group.find(b);
    return ia != track_group.end() && ib != track_group.end() && ia->second == ib->second;
}

nlohmann::ordered_json ObjectLayerConfig::to_json() const {
    return {{"coco", {{"model", coco_model}, {"conf", coco_conf}, {"keep", coco_keep}}},
            {"world", {{"model", world_model}, {"conf", world_conf}, {"prompts", world_tokens}}},
            {"merge_iou", merge_iou},
            {"track", {{"iou", track_iou}, {"max_gap", track_max_gap}, {"min_appearance", track_min_appearance},
                       {"groups", track_group}}},
            {"text", {{"enabled", text_enabled}, {"psm", text_psm}, {"min_conf", text_min_conf},
                      {"min_height", text_min_height}, {"pad", text_pad},
                      {"host_containment", text_host_containment}}},
            {"ocr", {{"min_side", ocr_min_side}, {"pad", ocr_pad}, {"views", ocr_views}, {"skip", ocr_skip}}}};
}

AerialDetector::AerialDetector(DetectorOptions opt) : opt_(std::move(opt)) {
    const auto& L = opt_.layer;
    coco_ = std::make_unique<YoloHead>(L.coco_model, opt_.imgsz);
    for (const auto& k : L.coco_keep)
        if (std::find(coco_->names().begin(), coco_->names().end(), k) == coco_->names().end())
            throw std::runtime_error("coco.keep names '" + k + "', which " + L.coco_model + " does not have");
    if (!L.world_model.empty()) {
        world_ = std::make_unique<YoloHead>(L.world_model, opt_.imgsz);
        // The graph answers the prompts it was exported with; they must be the config's.
        std::set<std::string> have(world_->names().begin(), world_->names().end()), want;
        for (const auto& [p, t] : L.world_tokens) want.insert(p);
        if (have != want)
            throw std::runtime_error(L.world_model + " was exported with other prompts than the config's "
                                     "world.prompts; export it again (run.sh does when the file is missing)");
    }
    if (opt_.ocr == "florence") {
        std::cout << "[aerial] Florence-2 is the HaloCore door (OCR per object, " << L.ocr_views << " view(s))"
                  << std::endl;
        florence_ = std::make_unique<HaloFlorence>();
    } else if (opt_.ocr == "tesseract") {
        tess_ = std::make_unique<Tesseract>();
    }
    if (L.text_enabled && opt_.ocr != "none") scout_ = std::make_unique<Tesseract>();
}

std::vector<Box> AerialDetector::object_layer(const cv::Mat& bgr) {
    const auto& L = opt_.layer;
    std::vector<Box> all;
    for (const auto& yb : coco_->predict(bgr, L.coco_conf)) {
        const std::string& name = coco_->names()[static_cast<size_t>(yb.cls)];
        if (!L.coco_keep.count(name)) continue;
        Box b;
        b.cls = b.detected = name;
        b.confidence = yb.conf;
        std::copy(yb.xyxy, yb.xyxy + 4, b.xyxy);
        b.source = "coco";
        all.push_back(b);
    }
    if (world_) {
        for (const auto& yb : world_->predict(bgr, L.world_conf)) {
            Box b;
            b.cls = b.detected = L.world_tokens.at(world_->names()[static_cast<size_t>(yb.cls)]);
            b.confidence = yb.conf;
            std::copy(yb.xyxy, yb.xyxy + 4, b.xyxy);
            b.source = "world";
            all.push_back(b);
        }
    }
    // Class-agnostic merge across both heads: most confident first, and a box
    // overlapping a kept one is the same object.
    std::stable_sort(all.begin(), all.end(), [](const Box& a, const Box& b) { return a.confidence > b.confidence; });
    std::vector<Box> kept;
    for (const auto& b : all) {
        bool dup = false;
        for (const auto& k : kept)
            if (iou(b.xyxy, k.xyxy) > L.merge_iou) { dup = true; break; }
        if (!dup) kept.push_back(b);
    }
    return kept;
}

std::vector<Box> AerialDetector::text_boxes(const cv::Mat& bgr) {
    const auto& L = opt_.layer;
    std::vector<Box> out;
    if (!scout_ || bgr.empty()) return out;
    // Words that look like a stencil. The scout's spelling is not kept:
    // Florence reads the line.
    struct W { double b[4]; double conf; std::string token; };
    std::vector<W> words;
    for (const auto& w : scout_->words(bgr, L.text_psm)) {
        const auto tok = ocr_token(w.text);
        if (!tok || !keep_stencil(tok->substr(4)) || w.conf < L.text_min_conf) continue;
        if (w.xyxy[3] - w.xyxy[1] < L.text_min_height) continue;
        words.push_back({{static_cast<double>(w.xyxy[0]), static_cast<double>(w.xyxy[1]),
                          static_cast<double>(w.xyxy[2]), static_cast<double>(w.xyxy[3])}, w.conf, tok->substr(4)});
    }
    // Words on one line, close together, are one text box ("NORTH GATE").
    std::vector<int> parent(words.size());
    std::iota(parent.begin(), parent.end(), 0);
    std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
    for (size_t a = 0; a < words.size(); ++a)
        for (size_t b = a + 1; b < words.size(); ++b) {
            const double* p = words[a].b;
            const double* q = words[b].b;
            const double ha = p[3] - p[1], hb = q[3] - q[1];
            const double v_overlap = std::min(p[3], q[3]) - std::max(p[1], q[1]);
            const double gap = std::max(p[0], q[0]) - std::min(p[2], q[2]);
            if (v_overlap >= 0.5 * std::min(ha, hb) && gap <= 1.5 * std::max(ha, hb))
                parent[find(static_cast<int>(a))] = find(static_cast<int>(b));
        }
    std::map<int, Box> lines;
    std::map<int, int> n;
    for (size_t k = 0; k < words.size(); ++k) {
        const int r = find(static_cast<int>(k));
        auto [it, fresh] = lines.try_emplace(r);
        Box& b = it->second;
        if (fresh) {
            b.cls = b.detected = kText;
            b.source = "tesseract";
            std::copy(words[k].b, words[k].b + 4, b.xyxy);
        } else {
            b.xyxy[0] = std::min(b.xyxy[0], words[k].b[0]);
            b.xyxy[1] = std::min(b.xyxy[1], words[k].b[1]);
            b.xyxy[2] = std::max(b.xyxy[2], words[k].b[2]);
            b.xyxy[3] = std::max(b.xyxy[3], words[k].b[3]);
        }
        b.confidence += words[k].conf / 100.0;
        b.scout.push_back(words[k].token);
        ++n[r];
    }
    for (auto& [r, b] : lines) {
        b.confidence /= n[r];
        const double pad = L.text_pad * (b.xyxy[3] - b.xyxy[1]);
        b.xyxy[0] = std::max(0.0, b.xyxy[0] - pad);
        b.xyxy[1] = std::max(0.0, b.xyxy[1] - pad);
        b.xyxy[2] = std::min(static_cast<double>(bgr.cols), b.xyxy[2] + pad);
        b.xyxy[3] = std::min(static_cast<double>(bgr.rows), b.xyxy[3] + pad);
        out.push_back(b);
    }
    return out;
}

void AerialDetector::read_objects(const std::vector<cv::Mat>& images) {
    const auto& L = opt_.layer;
    struct View { size_t object; std::string name; cv::Mat crop; };
    std::vector<View> views;
    size_t n_objects = 0;
    for (size_t i = 0; i < objects_.size(); ++i) {
        const auto& o = objects_[i];
        if (L.ocr_skip.count(o.cls)) continue;
        const bool text = o.cls == kText;
        bool any = false;
        for (size_t k : spread(o.boxes.size(), L.ocr_views)) {
            const double* xy = o.boxes[k].data();
            // A text box is already a tight line with its own margin.
            if (!text && std::min(xy[2] - xy[0], xy[3] - xy[1]) < L.ocr_min_side) continue;
            cv::Mat crop = object_crop(images[static_cast<size_t>(o.survivors[k])], xy, text ? 0.0 : L.ocr_pad);
            if (crop.empty()) continue;
            views.push_back({i, "object_" + std::to_string(o.object_id) + "_v" + std::to_string(k), std::move(crop)});
            any = true;
        }
        n_objects += any;
    }
    std::cout << "[aerial] ocr " << n_objects << " of " << objects_.size() << " object(s), " << views.size()
              << " view(s)" << std::endl;
    if (views.empty()) return;

    auto take = [&](ObjectEntry& o, const std::vector<std::string>& words) {
        o.reads.push_back(Read{words, florence_ != nullptr});
    };
    if (florence_) {
        std::error_code ec;
        const fs::path dir = fs::temp_directory_path(ec) / ("eworld_objects_" + std::to_string(::getpid()));
        fs::create_directories(dir, ec);
        std::vector<std::string> paths;
        for (const auto& v : views) {
            const fs::path p = dir / (v.name + ".png");
            cv::imwrite(p.string(), v.crop);
            paths.push_back(p.string());
        }
        const auto texts = florence_->read_many(paths);
        for (const auto& v : views) {
            const auto it = texts.find(v.name);
            if (it != texts.end()) take(objects_[v.object], figure_words(it->second));
        }
        fs::remove_all(dir, ec);
    } else if (tess_) {
        for (const auto& v : views) {
            std::vector<std::string> words;
            for (const auto& word : tess_->words(v.crop, 6)) {
                const auto tok = ocr_token(word.text);
                if (tok && keep_stencil(tok->substr(4))) words.push_back(tok->substr(4));
            }
            take(objects_[v.object], words);
        }
    }
}

void AerialDetector::attach_text(std::vector<std::vector<Box>>& frames) {
    const auto& L = opt_.layer;
    std::map<long, size_t> at;
    for (size_t i = 0; i < objects_.size(); ++i) at[objects_[i].object_id] = i;
    // Text first: Florence's reads and the scout's must agree.
    for (auto& t : objects_)
        if (t.cls == kText) t.ocr = corroborate(t.reads);
    // A text object is written on the object whose box holds it in most of
    // its frames; that object takes its reads.
    for (auto& t : objects_) {
        if (t.cls != kText || t.ocr.empty()) continue;
        std::map<long, int> votes;
        for (size_t k = 0; k < t.survivors.size(); ++k)
            for (const auto& b : frames[static_cast<size_t>(t.survivors[k])])
                if (b.cls != kText && !L.ocr_skip.count(b.cls) &&
                    containment(t.boxes[k].data(), b.xyxy) >= L.text_host_containment)
                    ++votes[b.object_id];
        if (votes.empty()) continue;
        const auto top = std::max_element(votes.begin(), votes.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; });
        t.host_id = top->first;
        auto& host = objects_[at.at(top->first)];
        host.reads.insert(host.reads.end(), t.reads.begin(), t.reads.end());
    }
    for (auto& o : objects_)
        if (o.cls != kText) o.ocr = corroborate(o.reads);
    // Text drawn on its host, or read as nothing, is not a box of its own.
    std::set<long> hidden;
    for (const auto& o : objects_)
        if (o.cls == kText && (o.ocr.empty() || o.host_id >= 0)) hidden.insert(o.object_id);
    for (auto& f : frames)
        f.erase(std::remove_if(f.begin(), f.end(), [&](const Box& b) { return hidden.count(b.object_id) > 0; }),
                f.end());
    objects_.erase(std::remove_if(objects_.begin(), objects_.end(),
                                  [](const ObjectEntry& o) { return o.cls == kText && o.ocr.empty(); }),
                   objects_.end());
}

std::vector<FrameDetections> AerialDetector::detect(const std::vector<std::string>& image_paths) {
    records_.clear();
    objects_.clear();
    std::vector<FrameDetections> out(image_paths.size());
    if (image_paths.empty()) return out;
    const auto& L = opt_.layer;

    // 1. Object layer on every survivor, scene text included.
    std::vector<cv::Mat> images;
    std::vector<std::vector<Box>> frames(image_paths.size());
    try {
        for (const auto& p : image_paths) images.push_back(cv::imread(p, cv::IMREAD_COLOR));
        for (size_t i = 0; i < images.size(); ++i) {
            frames[i] = object_layer(images[i]);
            for (auto& t : text_boxes(images[i])) frames[i].push_back(std::move(t));
        }
    } catch (const std::exception& e) {   // detector contract: do not throw
        std::cerr << "[aerial] object layer failed: " << e.what() << "\n";
        records_.assign(image_paths.size(), FrameRecord{});
        return out;
    }

    // 2. Dedup across survivors: greedy IoU match against objects seen in the
    //    last max_gap + 1 frames. Labels need only share a group (heads swap
    //    labels inside one: truck/tank/boat), and the vote below names the
    //    object. The crops must look alike, or a box would follow the frame
    //    across a cut into a different shot.
    struct Live { size_t entry; int last; double xyxy[4]; std::string label; cv::Mat hist; };
    std::vector<Live> live;
    for (int i = 0; i < static_cast<int>(frames.size()); ++i) {
        live.erase(std::remove_if(live.begin(), live.end(),
                                  [&](const Live& l) { return i - l.last > L.track_max_gap + 1; }),
                   live.end());
        std::vector<cv::Mat> hists;
        for (const auto& b : frames[i]) hists.push_back(appearance(images[static_cast<size_t>(i)], b.xyxy));
        struct Pair { double v; size_t box; size_t live; };
        std::vector<Pair> pairs;
        for (size_t b = 0; b < frames[i].size(); ++b)
            for (size_t l = 0; l < live.size(); ++l) {
                if (!L.same_group(frames[i][b].detected, live[l].label)) continue;
                const double v = iou(frames[i][b].xyxy, live[l].xyxy);
                if (v < L.track_iou) continue;
                if (hists[b].empty() || live[l].hist.empty() ||
                    cv::compareHist(hists[b], live[l].hist, cv::HISTCMP_CORREL) < L.track_min_appearance)
                    continue;
                pairs.push_back({v, b, l});
            }
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.v > b.v; });
        std::vector<bool> box_done(frames[i].size()), live_done(live.size());
        auto observe = [&](ObjectEntry& o, const Box& b) {
            o.survivors.push_back(i);
            o.boxes.push_back({b.xyxy[0], b.xyxy[1], b.xyxy[2], b.xyxy[3]});
            o.confidence = std::max(o.confidence, b.confidence);
            o.votes[b.detected] += b.confidence;
            o.sources.emplace(b.detected, b.source);
            if (!b.scout.empty()) o.reads.push_back(Read{b.scout, false});
            if (o.best_survivor < 0 || area(b.xyxy) > area(o.best_xyxy)) {
                o.best_survivor = i;
                std::copy(b.xyxy, b.xyxy + 4, o.best_xyxy);
            }
        };
        for (const auto& p : pairs) {
            if (box_done[p.box] || live_done[p.live]) continue;
            box_done[p.box] = live_done[p.live] = true;
            auto& lv = live[p.live];
            Box& b = frames[i][p.box];
            b.object_id = objects_[lv.entry].object_id;
            observe(objects_[lv.entry], b);
            lv.last = i;
            std::copy(b.xyxy, b.xyxy + 4, lv.xyxy);
            lv.label = b.detected;
            lv.hist = hists[p.box];
        }
        for (size_t b = 0; b < frames[i].size(); ++b) {
            if (box_done[b]) continue;
            ObjectEntry o;
            o.object_id = next_object_id_++;
            observe(o, frames[i][b]);
            frames[i][b].object_id = o.object_id;
            objects_.push_back(o);
            Live lv{objects_.size() - 1, i, {}, frames[i][b].detected, hists[b]};
            std::copy(frames[i][b].xyxy, frames[i][b].xyxy + 4, lv.xyxy);
            live.push_back(lv);
        }
    }
    for (auto& o : objects_) {
        const auto top = std::max_element(o.votes.begin(), o.votes.end(),
                                          [](const auto& a, const auto& b) { return a.second < b.second; });
        o.cls = top->first;
        o.source = o.sources.at(top->first);
    }
    std::cout << "[aerial] " << image_paths.size() << " survivor(s): "
              << std::accumulate(frames.begin(), frames.end(), size_t{0},
                                 [](size_t s, const std::vector<Box>& f) { return s + f.size(); })
              << " box(es) -> " << objects_.size() << " object(s)" << std::endl;

    // 3. Read each object on a few views; text goes to the object it is on.
    if (florence_ || tess_) read_objects(images);
    attach_text(frames);

    // 4. Every box looks its object up in the index; tags follow.
    std::map<long, const ObjectEntry*> index;
    for (const auto& o : objects_) index[o.object_id] = &o;
    for (size_t i = 0; i < frames.size(); ++i) {
        std::vector<std::pair<std::string, double>> best;   // label -> best confidence, first-seen order
        auto bump = [&](const std::string& label, double conf) {
            for (auto& [l, c] : best)
                if (l == label) { c = std::max(c, conf); return; }
            best.emplace_back(label, conf);
        };
        std::set<std::string> classes;
        for (auto& b : frames[i]) {
            const ObjectEntry& o = *index.at(b.object_id);
            b.cls = o.cls;
            b.ocr = o.ocr;
            if (b.cls != kText) {
                bump(b.cls, b.confidence);
                classes.insert(b.cls);
            }
            for (const auto& w : b.ocr) bump("ocr:" + w, std::max(b.confidence, 0.5));
        }
        const std::string scene = scene_tag(classes);
        if (!scene.empty()) bump(scene, 0.55);
        std::stable_sort(best.begin(), best.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        int ocr_added = 0;
        for (const auto& [label, conf] : best) {
            if (starts_with(label, "ocr:") && ocr_added++ >= kMaxOcrTags) continue;
            out[i].tags.push_back(DetectedTag{label, static_cast<float>(conf)});
        }
        records_.push_back(FrameRecord{std::move(frames[i]), scene});
    }
    return out;
}

}  // namespace eworld
