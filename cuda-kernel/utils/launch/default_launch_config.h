#ifndef __RAPID_DEFAULT_LAUNCH_CONFIG_H__
#define __RAPID_DEFAULT_LAUNCH_CONFIG_H__

#include <array>
#include <cstddef>

namespace rapid_launch_config {

inline constexpr std::array<unsigned int, 3> kGrid{{1u, 1u, 1u}};
inline constexpr std::array<unsigned int, 7> kBlockCandidates{
    {1024u, 512u, 256u, 128u, 64u, 32u, 16u}};
inline constexpr unsigned int kPhysicalBlockMax = 1024u;
inline constexpr std::array<unsigned int, 3> kLogicalGrid{{1u, 1u, 1u}};
inline constexpr std::array<unsigned int, 3> kLogicalBlock{{1u, 1u, 1u}};
inline constexpr bool kHasLogicalVConfigBounds = false;
inline constexpr size_t kTargetDynamicSharedBytes = 0u;
inline constexpr const char *kCoverageMemory = "global";
inline constexpr bool kVConfigReserved = true;
inline constexpr bool kVConfigEnabled = true;
inline constexpr bool kVConfigWarpAligned = false;
inline constexpr bool kVConfigMutation = true;

} // namespace rapid_launch_config

#endif // __RAPID_DEFAULT_LAUNCH_CONFIG_H__
