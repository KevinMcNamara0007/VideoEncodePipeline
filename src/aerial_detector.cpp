// ------------------------------------------------------------------------
// Copyright (c) 2026 Mjolnir Photonics Labs. All rights reserved.
// Architect: Kevin McNamara, CTO and Chief Science Officer.
// PROPRIETARY AND CONFIDENTIAL. Protected by patents and pending patent
// applications. No permission to copy, modify or reuse; use is strictly
// governed by the EULA (EULA.md at the repository root).
// ------------------------------------------------------------------------
#include "aerial_detector.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace eworld {
namespace fs = std::filesystem;
using ehc::media::DetectedTag;
using ehc::media::FrameDetections;

namespace {

constexpr const char* kText = "text";

bool starts_with(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Adds the seconds since construction to `slot` at stop() or when it dies.
struct Stopwatch {
    double& slot;
    double  t0{now_s()};
    bool    live{true};
    explicit Stopwatch(double& s) : slot(s) {}
    void stop() {
        if (live) slot += now_s() - t0;
        live = false;
    }
    ~Stopwatch() { stop(); }
};

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

/// The object's class: the detected label with the most summed confidence.
void vote(ObjectEntry& o) {
    const auto top = std::max_element(o.votes.begin(), o.votes.end(),
                                      [](const auto& a, const auto& b) { return a.second < b.second; });
    o.cls = top->first;
    o.source = o.sources.at(top->first);
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
        c.ocr_batch    = std::max(1, j["ocr"].value("batch", 32));
        for (const auto& k : j["ocr"].value("skip", nlohmann::json::array())) c.ocr_skip.insert(k.get<std::string>());
    }
    if (j.contains("objects")) {
        c.fact_min_confidence = j["objects"].value("min_confidence", 0.4);
        c.fact_min_frames     = std::max(1, j["objects"].value("min_frames", 2));
        c.near_reach          = std::max(0.0, j["objects"].value("near_reach", 0.5));
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
            {"ocr", {{"min_side", ocr_min_side}, {"pad", ocr_pad}, {"views", ocr_views}, {"batch", ocr_batch},
                     {"skip", ocr_skip}}},
            {"objects", {{"min_confidence", fact_min_confidence}, {"min_frames", fact_min_frames},
                         {"near_reach", near_reach}}}};
}

// ── the reader thread ─────────────────────────────────────────────────────

/// Reads closed objects' views as they arrive, in batches, on its own
/// thread. Results are keyed by object index and merged by the caller after
/// join(): objects_ grows on the tracking thread meanwhile.
class AerialDetector::Reader {
public:
    Reader(HaloFlorence* florence, Tesseract* tess, int batch)
        : florence_(florence), tess_(tess), batch_(batch), thread_([this] { loop(); }) {}
    ~Reader() {
        if (thread_.joinable()) join();
    }
    void submit(std::vector<View> views) {
        {
            std::lock_guard<std::mutex> g(m_);
            for (auto& v : views) queue_.push_back(std::move(v));
        }
        cv_.notify_one();
    }
    /// Closes the queue and waits for the last batch.
    void join() {
        {
            std::lock_guard<std::mutex> g(m_);
            done_ = true;
        }
        cv_.notify_one();
        thread_.join();
    }
    std::map<size_t, std::vector<Read>>& reads() { return reads_; }
    double busy_s() const { return busy_s_; }
    size_t views_read() const { return n_views_; }
    size_t batches() const { return n_batches_; }

private:
    void loop() {
        for (;;) {
            std::vector<View> batch;
            {
                std::unique_lock<std::mutex> g(m_);
                // A full batch, or whatever is left once the queue closes:
                // a door call loads the model, so a call should carry many
                // views, and objects close one or two at a time.
                cv_.wait(g, [this] { return done_ || queue_.size() >= static_cast<size_t>(batch_); });
                if (queue_.empty()) return;
                while (!queue_.empty() && batch.size() < static_cast<size_t>(batch_)) {
                    batch.push_back(std::move(queue_.front()));
                    queue_.pop_front();
                }
            }
            const double t0 = now_s();
            read(batch);
            busy_s_ += now_s() - t0;
            n_views_ += batch.size();
            ++n_batches_;
        }
    }
    void read(const std::vector<View>& views) {
        auto take = [&](size_t object, std::vector<std::string> words) {
            reads_[object].push_back(Read{std::move(words), florence_ != nullptr});
        };
        if (florence_) {
            std::error_code ec;
            const fs::path dir = fs::temp_directory_path(ec) /
                                 ("eworld_objects_" + std::to_string(::getpid()) + "_" + std::to_string(n_batches_));
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
                if (it != texts.end()) take(v.object, figure_words(it->second));
            }
            fs::remove_all(dir, ec);
        } else if (tess_) {
            for (const auto& v : views) {
                std::vector<std::string> words;
                for (const auto& word : tess_->words(v.crop, 6)) {
                    const auto tok = ocr_token(word.text);
                    if (tok && keep_stencil(tok->substr(4))) words.push_back(tok->substr(4));
                }
                take(v.object, std::move(words));
            }
        }
    }

    HaloFlorence* florence_;
    Tesseract*    tess_;
    int           batch_;
    std::mutex              m_;
    std::condition_variable cv_;
    std::deque<View>        queue_;
    bool                    done_{false};
    std::map<size_t, std::vector<Read>> reads_;
    double busy_s_{0};
    size_t n_views_{0}, n_batches_{0};
    std::thread thread_;
};

// ── the detector ──────────────────────────────────────────────────────────

AerialDetector::AerialDetector(DetectorOptions opt) : opt_(std::move(opt)) {
    const auto& L = opt_.layer;
    const int cores = std::max(1u, std::thread::hardware_concurrency());
    // A YOLO forward spreads over OpenCV's thread pool already; a second
    // frame in flight fills the cores the pool leaves idle between layers
    // (16 cores, 326 frames: 1 worker 153 s, 2 workers 126 s, 4 workers
    // 136 s), and the reader thread needs cores too.
    const int workers = opt_.workers > 0 ? opt_.workers : (cores >= 8 ? 2 : 1);
    // The reader shares the cores with the workers: a quarter of them keeps
    // the layer at its solo pace, and the reads still finish under it.
    if (const char* f = std::getenv("HALOCORE_FIGURES_THREADS"); !f || !*f)
        ::setenv("HALOCORE_FIGURES_THREADS", std::to_string(std::clamp(cores / 4, 1, 8)).c_str(), 0);
    // The door's figure pass describes a picture three ways (object labels,
    // text, a long caption); only the text is used here, and the caption is
    // half the pass. Needs a door that knows HALOCORE_FIGURES_TASKS; an
    // older one runs all three.
    ::setenv("HALOCORE_FIGURES_TASKS", "ocr", 0);
    heads_.resize(static_cast<size_t>(workers));
    for (auto& h : heads_) {
        h.coco = std::make_unique<YoloHead>(L.coco_model, opt_.imgsz);
        if (!L.world_model.empty()) h.world = std::make_unique<YoloHead>(L.world_model, opt_.imgsz);
        if (L.text_enabled && opt_.ocr != "none") h.scout = std::make_unique<Tesseract>();
    }
    const Heads& h0 = heads_.front();
    for (const auto& k : L.coco_keep)
        if (std::find(h0.coco->names().begin(), h0.coco->names().end(), k) == h0.coco->names().end())
            throw std::runtime_error("coco.keep names '" + k + "', which " + L.coco_model + " does not have");
    if (h0.world) {
        // The graph answers the prompts it was exported with; they must be the config's.
        std::set<std::string> have(h0.world->names().begin(), h0.world->names().end()), want;
        for (const auto& [p, t] : L.world_tokens) want.insert(p);
        if (have != want)
            throw std::runtime_error(L.world_model + " was exported with other prompts than the config's "
                                     "world.prompts; export it again (run.sh does when the file is missing)");
    }
    if (opt_.ocr == "florence") {
        std::cout << "[aerial] Florence-2 is the HaloCore door (OCR per object, " << L.ocr_views
                  << " view(s), batches of " << L.ocr_batch << ", tasks "
                  << (std::getenv("HALOCORE_FIGURES_TASKS") ? std::getenv("HALOCORE_FIGURES_TASKS") : "od,ocr,caption")
                  << ", " << std::getenv("HALOCORE_FIGURES_THREADS") << " threads)" << std::endl;
        florence_ = std::make_unique<HaloFlorence>();
    } else if (opt_.ocr == "tesseract") {
        tess_ = std::make_unique<Tesseract>();
    }
    std::cout << "[aerial] object layer: " << workers << " worker(s) on " << cores << " cores" << std::endl;
}

AerialDetector::~AerialDetector() = default;

std::vector<Box> AerialDetector::object_layer(Heads& h, const cv::Mat& bgr) const {
    const auto& L = opt_.layer;
    std::vector<Box> all;
    std::vector<YoloBox> coco_boxes, world_boxes;
    {
        Stopwatch sw(h.coco_s);
        coco_boxes = h.coco->predict(bgr, L.coco_conf);
    }
    if (h.world) {
        Stopwatch sw(h.world_s);
        world_boxes = h.world->predict(bgr, L.world_conf);
    }
    for (const auto& yb : coco_boxes) {
        const std::string& name = h.coco->names()[static_cast<size_t>(yb.cls)];
        if (!L.coco_keep.count(name)) continue;
        Box b;
        b.cls = b.detected = name;
        b.confidence = yb.conf;
        std::copy(yb.xyxy, yb.xyxy + 4, b.xyxy);
        b.source = "coco";
        all.push_back(b);
    }
    for (const auto& yb : world_boxes) {
        Box b;
        b.cls = b.detected = L.world_tokens.at(h.world->names()[static_cast<size_t>(yb.cls)]);
        b.confidence = yb.conf;
        std::copy(yb.xyxy, yb.xyxy + 4, b.xyxy);
        b.source = "world";
        all.push_back(b);
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

std::vector<Box> AerialDetector::text_boxes(Heads& h, const cv::Mat& bgr) const {
    const auto& L = opt_.layer;
    std::vector<Box> out;
    if (!h.scout || bgr.empty()) return out;
    // Words that look like a stencil. The scout's spelling is not kept:
    // Florence reads the line.
    struct W { double b[4]; double conf; std::string token; };
    std::vector<W> words;
    std::vector<OcrWord> scouted;
    {
        Stopwatch sw(h.scout_s);
        scouted = h.scout->words(bgr, L.text_psm);
    }
    for (const auto& w : scouted) {
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

std::vector<AerialDetector::View> AerialDetector::views_of(size_t object, const std::vector<cv::Mat>& images) const {
    const auto& L = opt_.layer;
    const ObjectEntry& o = objects_[object];
    std::vector<View> views;
    if (L.ocr_skip.count(o.cls)) return views;
    const bool text = o.cls == kText;
    for (size_t k : spread(o.boxes.size(), L.ocr_views)) {
        const double* xy = o.boxes[k].data();
        // A text box is already a tight line with its own margin.
        if (!text && std::min(xy[2] - xy[0], xy[3] - xy[1]) < L.ocr_min_side) continue;
        cv::Mat crop = object_crop(images[static_cast<size_t>(o.survivors[k])], xy, text ? 0.0 : L.ocr_pad);
        if (crop.empty()) continue;
        views.push_back({object, "object_" + std::to_string(o.object_id) + "_v" + std::to_string(k), std::move(crop)});
    }
    // EWORLD_KEEP_VIEWS=<dir>: the crops are kept there too, for reader benches.
    static const char* keep = std::getenv("EWORLD_KEEP_VIEWS");
    if (keep && *keep && !views.empty()) {
        std::error_code ec;
        fs::create_directories(keep, ec);
        for (const auto& v : views) cv::imwrite((fs::path(keep) / (v.name + ".png")).string(), v.crop);
    }
    return views;
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
    timing_.clear();
    for (auto& h : heads_) h.coco_s = h.world_s = h.scout_s = 0;
    std::vector<FrameDetections> out(image_paths.size());
    if (image_paths.empty()) return out;
    const auto& L = opt_.layer;
    const size_t n = image_paths.size();
    Stopwatch sw_total(timing_["total"]);

    // 1. Object layer on every survivor, scene text included: the workers
    //    take frames in order, each delivering its boxes through a promise
    //    the tracker below waits on. A worker that throws fails the frame
    //    (detector contract: detect() does not throw).
    std::vector<cv::Mat> images(n);
    std::vector<std::vector<Box>> frames(n);
    std::vector<std::promise<std::string>> ready(n);   // "" ok, else the error
    std::vector<std::future<std::string>> done;
    for (auto& p : ready) done.push_back(p.get_future());
    std::atomic<size_t> next{0};
    const int cores = std::max(1u, std::thread::hardware_concurrency());
    const int saved_threads = cv::getNumThreads();
    cv::setNumThreads(std::max(1, cores / static_cast<int>(heads_.size())));
    double layer_s = 0;
    std::vector<std::thread> pool;
    const double t_layer = now_s();
    for (auto& h : heads_)
        pool.emplace_back([&, hp = &h] {
            for (size_t i = next++; i < n; i = next++) {
                std::string err;
                try {
                    images[i] = cv::imread(image_paths[i], cv::IMREAD_COLOR);
                    frames[i] = object_layer(*hp, images[i]);
                    for (auto& t : text_boxes(*hp, images[i])) frames[i].push_back(std::move(t));
                } catch (const std::exception& e) {
                    err = e.what();
                }
                ready[i].set_value(err);
            }
        });

    // 3 (started first, fed by 2): the reader takes each object the moment
    //    its track closes.
    std::unique_ptr<Reader> reader;
    if (florence_ || tess_) reader = std::make_unique<Reader>(florence_.get(), tess_.get(), L.ocr_batch);
    auto close = [&](size_t entry) {
        vote(objects_[entry]);
        if (reader) {
            auto views = views_of(entry, images);
            if (!views.empty()) reader->submit(std::move(views));
        }
    };

    // 2. Dedup across survivors, in frame order as the layer delivers them:
    //    greedy IoU match against objects seen in the last max_gap + 1
    //    frames. Labels need only share a group (heads swap labels inside
    //    one: truck/tank/boat), and the vote names the object. The crops
    //    must look alike, or a box would follow the frame across a cut into
    //    a different shot.
    struct Live { size_t entry; int last; double xyxy[4]; std::string label; cv::Mat hist; };
    std::vector<Live> live;
    double track_s = 0;
    std::string failure;
    for (int i = 0; i < static_cast<int>(n); ++i) {
        const std::string err = done[static_cast<size_t>(i)].get();
        if (!err.empty() && failure.empty()) failure = err;
        if (!failure.empty()) continue;
        const double t0 = now_s();
        for (auto& l : live)
            if (i - l.last > L.track_max_gap + 1) close(l.entry);
        live.erase(std::remove_if(live.begin(), live.end(),
                                  [&](const Live& l) { return i - l.last > L.track_max_gap + 1; }),
                   live.end());
        std::vector<cv::Mat> hists;
        for (const auto& b : frames[static_cast<size_t>(i)]) hists.push_back(appearance(images[static_cast<size_t>(i)], b.xyxy));
        struct Pair { double v; size_t box; size_t live; };
        std::vector<Pair> pairs;
        auto& fr = frames[static_cast<size_t>(i)];
        for (size_t b = 0; b < fr.size(); ++b)
            for (size_t l = 0; l < live.size(); ++l) {
                if (!L.same_group(fr[b].detected, live[l].label)) continue;
                const double v = iou(fr[b].xyxy, live[l].xyxy);
                if (v < L.track_iou) continue;
                if (hists[b].empty() || live[l].hist.empty() ||
                    cv::compareHist(hists[b], live[l].hist, cv::HISTCMP_CORREL) < L.track_min_appearance)
                    continue;
                pairs.push_back({v, b, l});
            }
        std::stable_sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.v > b.v; });
        std::vector<bool> box_done(fr.size()), live_done(live.size());
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
            Box& b = fr[p.box];
            b.object_id = objects_[lv.entry].object_id;
            observe(objects_[lv.entry], b);
            lv.last = i;
            std::copy(b.xyxy, b.xyxy + 4, lv.xyxy);
            lv.label = b.detected;
            lv.hist = hists[p.box];
        }
        for (size_t b = 0; b < fr.size(); ++b) {
            if (box_done[b]) continue;
            ObjectEntry o;
            o.object_id = next_object_id_++;
            observe(o, fr[b]);
            fr[b].object_id = o.object_id;
            objects_.push_back(o);
            Live lv{objects_.size() - 1, i, {}, fr[b].detected, hists[b]};
            std::copy(fr[b].xyxy, fr[b].xyxy + 4, lv.xyxy);
            live.push_back(lv);
        }
        track_s += now_s() - t0;
    }
    for (auto& t : pool) t.join();
    layer_s = now_s() - t_layer;
    cv::setNumThreads(saved_threads);
    if (!failure.empty()) {
        std::cerr << "[aerial] object layer failed: " << failure << "\n";
        if (reader) reader->join();
        records_.assign(n, FrameRecord{});
        return out;
    }
    {
        const double t0 = now_s();
        for (auto& l : live) close(l.entry);
        live.clear();
        track_s += now_s() - t0;
    }
    timing_["layer"] = layer_s;
    timing_["track"] = track_s;
    for (const auto& h : heads_) {
        timing_["coco"] += h.coco_s;
        timing_["world"] += h.world_s;
        timing_["scout"] += h.scout_s;
    }
    std::cout << "[aerial] " << n << " survivor(s): "
              << std::accumulate(frames.begin(), frames.end(), size_t{0},
                                 [](size_t s, const std::vector<Box>& f) { return s + f.size(); })
              << " box(es) -> " << objects_.size() << " object(s)" << std::endl;

    // 3. The last batch; then every read lands on its object.
    if (reader) {
        const double t0 = now_s();
        reader->join();
        timing_["read_wait"] = now_s() - t0;   // the tail the layer did not cover
        timing_["read"] = reader->busy_s();
        for (auto& [object, reads] : reader->reads())
            objects_[object].reads.insert(objects_[object].reads.end(), reads.begin(), reads.end());
        size_t n_objects = 0;
        for (const auto& [object, reads] : reader->reads()) n_objects += !reads.empty();
        std::cout << "[aerial] ocr " << n_objects << " of " << objects_.size() << " object(s), "
                  << reader->views_read() << " view(s) in " << reader->batches() << " batch(es)" << std::endl;
    }
    {
        Stopwatch sw(timing_["attach"]);
        attach_text(frames);
    }

    // 4. Every box looks its object up in the index; tags follow.
    Stopwatch sw_index(timing_["index"]);
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
    sw_index.stop();
    sw_total.stop();
    std::string line = "[aerial] detect seconds:";
    for (const auto& [k, v] : timing_) {
        char buf[64];
        std::snprintf(buf, sizeof buf, " %s=%.1f", k.c_str(), v);
        line += buf;
    }
    std::cout << line << std::endl;
    return out;
}

}  // namespace eworld
