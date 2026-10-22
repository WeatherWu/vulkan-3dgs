#pragma once
#include <algorithm>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>

namespace optimizer_r1 {
struct Statistics {
    double mean = 0, minimum = 0, maximum = 0;
};
inline Statistics statistics(std::span<const double> samples) {
    if (samples.empty()) throw std::runtime_error("S4-1R empty timing phase");
    Statistics s{0, samples.front(), samples.front()};
    for (double value : samples) {
        if (!std::isfinite(value) || value <= 0)
            throw std::runtime_error("S4-1R non-finite/non-positive timing sample");
        s.mean += value / double(samples.size());
        s.minimum = std::min(s.minimum, value);
        s.maximum = std::max(s.maximum, value);
    }
    if (!std::isfinite(s.mean)) throw std::runtime_error("S4-1R timing mean overflow");
    return s;
}
struct Decision {
    Statistics a1, b, a2;
    double baselineMean = 0, gainMs = 0, gainPercent = 0, baselineDriftPercent = 0;
    std::string verdict;
};
inline Decision evaluate(std::span<const double> a1, std::span<const double> b,
                         std::span<const double> a2) {
    Decision d;
    d.a1 = statistics(a1);
    d.b = statistics(b);
    d.a2 = statistics(a2);
    d.baselineMean = (d.a1.mean + d.a2.mean) / 2;
    const double driftMs = std::abs(d.a1.mean - d.a2.mean);
    d.baselineDriftPercent = 100 * driftMs / d.baselineMean;
    d.gainMs = d.baselineMean - d.b.mean;
    d.gainPercent = 100 * d.gainMs / d.baselineMean;
    // ALL samples are retained. Never silently drop a first-use/paging anomaly.
    if (d.a1.maximum > d.a1.minimum * 1.2 || d.b.maximum > d.b.minimum * 1.2 ||
        d.a2.maximum > d.a2.minimum * 1.2)
        d.verdict = "INCONCLUSIVE_SAMPLE_OUTLIER";
    else if (d.baselineDriftPercent > 2)
        d.verdict = "INCONCLUSIVE_BASELINE_DRIFT";
    else if (d.gainPercent < 5 || d.gainMs <= 2 * driftMs)
        d.verdict = "STOP_NO_SHORT_GAIN";
    else
        d.verdict = "SHORT_GAIN_REQUIRES_NATIVE_REVIEW";
    return d;
}
} // namespace optimizer_r1
