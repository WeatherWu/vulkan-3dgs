#include "device_selection.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <limits>

namespace vulkan3DGS {

namespace {

std::string trim(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }).base();
    return first < last ? std::string(first, last) : std::string{};
}

std::string lowercase(const std::string& value) {
    std::string result = value;
    std::transform(result.begin(), result.end(), result.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return result;
}

std::string normalizeUuid(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (unsigned char ch : value) {
        if (ch == '-') {
            continue;
        }
        if (!std::isxdigit(ch)) {
            return {};
        }
        result.push_back(static_cast<char>(std::tolower(ch)));
    }
    return result.size() == 32 ? result : std::string{};
}

GpuSelectionResult uniqueMatch(const std::vector<size_t>& matches,
                               const std::string& selector,
                               const char* matchType) {
    if (matches.size() == 1) {
        return {matches.front(), {}};
    }
    if (matches.size() > 1) {
        return {std::nullopt,
                "GPU selector '" + selector + "' matches multiple " + matchType +
                    "; use a Vulkan index or UUID"};
    }
    return {};
}

} // namespace

GpuSelectionResult selectGpuCandidate(const std::vector<GpuSelectionCandidate>& candidates,
                                      const std::string& selector) {
    const std::string requested = trim(selector);
    if (requested.empty()) {
        return {std::nullopt, "GPU selector must not be empty"};
    }

    uint32_t requestedIndex = 0;
    const char* indexBegin = requested.data();
    const char* indexEnd = indexBegin + requested.size();
    const auto indexResult = std::from_chars(indexBegin, indexEnd, requestedIndex);
    if (indexResult.ec == std::errc{} && indexResult.ptr == indexEnd) {
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            if (candidates[candidateIndex].vulkanIndex == requestedIndex) {
                return {candidateIndex, {}};
            }
        }
        return {std::nullopt,
                "GPU selector '" + requested + "' is not a suitable Vulkan device index"};
    }

    const std::string requestedUuid = normalizeUuid(requested);
    if (!requestedUuid.empty()) {
        for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
            if (normalizeUuid(candidates[candidateIndex].uuid) == requestedUuid) {
                return {candidateIndex, {}};
            }
        }
        return {std::nullopt, "GPU UUID '" + requested + "' is not available"};
    }

    const std::string requestedName = lowercase(requested);
    std::vector<size_t> exactMatches;
    std::vector<size_t> substringMatches;
    for (size_t candidateIndex = 0; candidateIndex < candidates.size(); ++candidateIndex) {
        const std::string candidateName = lowercase(candidates[candidateIndex].name);
        if (candidateName == requestedName) {
            exactMatches.push_back(candidateIndex);
        } else if (candidateName.find(requestedName) != std::string::npos) {
            substringMatches.push_back(candidateIndex);
        }
    }

    GpuSelectionResult result = uniqueMatch(exactMatches, requested, "device names");
    if (result || !result.error.empty()) {
        return result;
    }
    result = uniqueMatch(substringMatches, requested, "device names");
    if (result || !result.error.empty()) {
        return result;
    }
    return {std::nullopt, "GPU selector '" + requested + "' did not match a suitable device"};
}

} // namespace vulkan3DGS
