// Speaker-owned lifecycle boundary for the pinned parakeet.cpp C API.
//
// Build this source against parakeet.cpp v0.5.0/ABI v6 and link it to the
// receipt-bound libparakeet beside this bridge.  The exported ABI deliberately
// exposes no upstream error string: model paths and transcript fragments must
// never escape through worker diagnostics.

#include "backend.hpp"
#include "ggml_graph.hpp"
#include "parakeet_capi.h"

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <string>

#if defined(_WIN32)
#define SPEAKER_PARAKEET_EXPORT extern "C" __declspec(dllexport)
#else
#define SPEAKER_PARAKEET_EXPORT                                             \
    extern "C" __attribute__((visibility("default")))
#endif

namespace {

constexpr int kBridgeAbiVersion = 1;
constexpr int kRequiredParakeetCapiVersion = 6;
constexpr int kNativeChunkSamples = 1280;
constexpr int kMaximumThreads = 256;
constexpr std::size_t kMaximumDeviceBytes = 63;
constexpr std::size_t kMaximumModelPathBytes = 4096;
constexpr std::size_t kMaximumJsonBytes = 64 * 1024;
constexpr std::uint64_t kHandleMagic = UINT64_C(0x53504b4252494447);

enum Status : int {
    kOk = 0,
    kInvalidArgument = 1,
    kAbiMismatch = 2,
    kDeviceMismatch = 3,
    kLoadFailure = 4,
    kLifecycleFailure = 5,
    kNativeFailure = 6,
    kOutputFailure = 7,
    kInternalFailure = 8,
};

enum class Lifecycle {
    kReady,
    kStreaming,
    kPoisoned,
    kClosed,
};

struct EnvironmentSnapshot {
    bool present = false;
    bool changed = false;
    std::string value;
};

struct speaker_parakeet_bridge {
    std::uint64_t magic = kHandleMagic;
    parakeet_ctx* context = nullptr;
    parakeet_stream* stream = nullptr;
    Lifecycle lifecycle = Lifecycle::kReady;
    std::string requested_device;
    std::string actual_device;
    EnvironmentSnapshot prior_device;
    std::mutex mutex;
};

std::mutex g_lifecycle_mutex;
bool g_handle_open = false;

bool valid_device(const char* raw) {
    if (!raw) return false;
    const std::size_t length = std::strlen(raw);
    if (length == 0 || length > kMaximumDeviceBytes) return false;
    for (std::size_t index = 0; index < length; ++index) {
        const unsigned char byte = static_cast<unsigned char>(raw[index]);
        if (!(std::isalnum(byte) || byte == '_' || byte == '-' || byte == '.'))
            return false;
    }
    return true;
}

bool equal_ascii_casefold(const std::string& left, const std::string& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto a = static_cast<unsigned char>(left[index]);
        const auto b = static_cast<unsigned char>(right[index]);
        if (std::tolower(a) != std::tolower(b)) return false;
    }
    return true;
}

bool set_device_environment(const std::string& value,
                            EnvironmentSnapshot& prior) {
    const char* existing = std::getenv("PARAKEET_DEVICE");
    if (existing) {
        prior.present = true;
        prior.value = existing;
    }
#if defined(_WIN32)
    if (_putenv_s("PARAKEET_DEVICE", value.c_str()) != 0) return false;
#else
    if (setenv("PARAKEET_DEVICE", value.c_str(), 1) != 0) return false;
#endif
    prior.changed = true;
    return true;
}

bool restore_device_environment(EnvironmentSnapshot& prior) noexcept {
    if (!prior.changed) return true;
    int result = 0;
#if defined(_WIN32)
    result = _putenv_s("PARAKEET_DEVICE",
                       prior.present ? prior.value.c_str() : "");
#else
    result = prior.present
                 ? setenv("PARAKEET_DEVICE", prior.value.c_str(), 1)
                 : unsetenv("PARAKEET_DEVICE");
#endif
    prior.changed = false;
    return result == 0;
}

bool is_valid_handle(const speaker_parakeet_bridge* handle) {
    return handle && handle->magic == kHandleMagic &&
           handle->lifecycle != Lifecycle::kClosed;
}

void discard_stream(speaker_parakeet_bridge* handle) noexcept {
    if (handle && handle->stream) {
        parakeet_capi_stream_free(handle->stream);
        handle->stream = nullptr;
    }
}

void poison(speaker_parakeet_bridge* handle) noexcept {
    discard_stream(handle);
    if (handle && handle->lifecycle != Lifecycle::kClosed)
        handle->lifecycle = Lifecycle::kPoisoned;
}

bool destroy_native(speaker_parakeet_bridge* handle) noexcept {
    if (!handle) return true;
    discard_stream(handle);
    if (handle->context) {
        parakeet_capi_free(handle->context);
        handle->context = nullptr;
    }
    bool clean = true;
    try {
        // The model and its device weight buffers are gone before the
        // process-global backend.  This ordering is required on GPU builds.
        pk::shutdown_backend();
    } catch (...) {
        clean = false;
    }
    try {
        pk::set_num_threads(0);
    } catch (...) {
        clean = false;
    }
    if (!restore_device_environment(handle->prior_device)) clean = false;
    handle->lifecycle = Lifecycle::kClosed;
    handle->magic = 0;
    return clean;
}

std::size_t bounded_json_length(const char* value) {
    if (!value) return kMaximumJsonBytes + 1;
    for (std::size_t length = 0; length <= kMaximumJsonBytes; ++length) {
        if (value[length] == '\0') return length;
    }
    return kMaximumJsonBytes + 1;
}

int transfer_json(speaker_parakeet_bridge* handle, char* raw,
                  char** output, std::size_t* output_bytes) noexcept {
    if (!raw) {
        poison(handle);
        return kNativeFailure;
    }
    const std::size_t length = bounded_json_length(raw);
    if (length == 0 || length > kMaximumJsonBytes) {
        parakeet_capi_free_string(raw);
        poison(handle);
        return kOutputFailure;
    }
    *output = raw;
    *output_bytes = length;
    return kOk;
}

}  // namespace

SPEAKER_PARAKEET_EXPORT int speaker_parakeet_bridge_abi_version(void) noexcept {
    return kBridgeAbiVersion;
}

SPEAKER_PARAKEET_EXPORT int
speaker_parakeet_bridge_upstream_abi_version(void) noexcept {
    try {
        return parakeet_capi_abi_version();
    } catch (...) {
        return -1;
    }
}

SPEAKER_PARAKEET_EXPORT int speaker_parakeet_bridge_open(
    const char* model_path, const char* requested_device, int num_threads,
    speaker_parakeet_bridge** output) noexcept {
    if (!output) return kInvalidArgument;
    *output = nullptr;
    try {
        const std::size_t model_path_bytes = model_path ? std::strlen(model_path) : 0;
        if (model_path_bytes == 0 || model_path_bytes > kMaximumModelPathBytes ||
            !valid_device(requested_device) || num_threads <= 0 ||
            num_threads > kMaximumThreads)
            return kInvalidArgument;
        if (parakeet_capi_abi_version() != kRequiredParakeetCapiVersion)
            return kAbiMismatch;

        std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
        if (g_handle_open) return kLifecycleFailure;

        auto handle = std::unique_ptr<speaker_parakeet_bridge>(
            new (std::nothrow) speaker_parakeet_bridge());
        if (!handle) return kInternalFailure;
        try {
            handle->requested_device = requested_device;
            if (!set_device_environment(handle->requested_device,
                                        handle->prior_device))
                return kInternalFailure;

            pk::set_num_threads(num_threads);
            pk::Backend& backend = pk::global_backend();
            const char* actual = backend.device_name();
            if (!valid_device(actual)) {
                destroy_native(handle.get());
                return kDeviceMismatch;
            }
            handle->actual_device = actual;
            if (!equal_ascii_casefold(handle->requested_device,
                                      handle->actual_device) ||
                pk::num_threads() != num_threads ||
                backend.n_threads() != num_threads) {
                destroy_native(handle.get());
                return kDeviceMismatch;
            }

            handle->context = parakeet_capi_load(model_path);
            if (!handle->context) {
                destroy_native(handle.get());
                return kLoadFailure;
            }
            const char* actual_after_load = pk::global_backend().device_name();
            if (!valid_device(actual_after_load) ||
                !equal_ascii_casefold(handle->actual_device,
                                      actual_after_load)) {
                destroy_native(handle.get());
                return kDeviceMismatch;
            }
        } catch (...) {
            destroy_native(handle.get());
            return kInternalFailure;
        }

        g_handle_open = true;
        *output = handle.release();
        return kOk;
    } catch (...) {
        return kInternalFailure;
    }
}

SPEAKER_PARAKEET_EXPORT int speaker_parakeet_bridge_actual_device(
    speaker_parakeet_bridge* handle, char* output,
    std::size_t capacity) noexcept {
    try {
        if (!is_valid_handle(handle) || !output || capacity == 0)
            return kInvalidArgument;
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (!is_valid_handle(handle) ||
            handle->actual_device.size() + 1 > capacity)
            return kLifecycleFailure;
        std::memcpy(output, handle->actual_device.data(),
                    handle->actual_device.size());
        output[handle->actual_device.size()] = '\0';
        return kOk;
    } catch (...) {
        return kInternalFailure;
    }
}

SPEAKER_PARAKEET_EXPORT int
speaker_parakeet_bridge_begin(speaker_parakeet_bridge* handle) noexcept {
    try {
        if (!is_valid_handle(handle)) return kInvalidArgument;
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (!is_valid_handle(handle) || handle->lifecycle != Lifecycle::kReady ||
            !handle->context || handle->stream)
            return kLifecycleFailure;
        handle->stream = parakeet_capi_stream_begin(handle->context);
        if (!handle->stream) {
            poison(handle);
            return kNativeFailure;
        }
        handle->lifecycle = Lifecycle::kStreaming;
        return kOk;
    } catch (...) {
        poison(handle);
        return kInternalFailure;
    }
}

SPEAKER_PARAKEET_EXPORT int speaker_parakeet_bridge_feed_json(
    speaker_parakeet_bridge* handle, const float* pcm, int n_samples,
    char** output, std::size_t* output_bytes) noexcept {
    if (output) *output = nullptr;
    if (output_bytes) *output_bytes = 0;
    try {
        if (!is_valid_handle(handle) || !output || !output_bytes || !pcm ||
            n_samples <= 0 || n_samples > kNativeChunkSamples)
            return kInvalidArgument;
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (!is_valid_handle(handle) ||
            handle->lifecycle != Lifecycle::kStreaming || !handle->stream)
            return kLifecycleFailure;
        char* raw = parakeet_capi_stream_feed_json(handle->stream, pcm, n_samples);
        return transfer_json(handle, raw, output, output_bytes);
    } catch (...) {
        poison(handle);
        return kInternalFailure;
    }
}

SPEAKER_PARAKEET_EXPORT int speaker_parakeet_bridge_finalize_json(
    speaker_parakeet_bridge* handle, char** output,
    std::size_t* output_bytes) noexcept {
    if (output) *output = nullptr;
    if (output_bytes) *output_bytes = 0;
    try {
        if (!is_valid_handle(handle) || !output || !output_bytes)
            return kInvalidArgument;
        std::lock_guard<std::mutex> lock(handle->mutex);
        if (!is_valid_handle(handle) ||
            handle->lifecycle != Lifecycle::kStreaming || !handle->stream)
            return kLifecycleFailure;
        char* raw = parakeet_capi_stream_finalize_json(handle->stream);
        if (!raw) {
            poison(handle);
            return kNativeFailure;
        }
        // Finalize exactly once, then destroy the utterance-bounded stream.
        // The returned JSON allocation is independent of the stream object.
        discard_stream(handle);
        handle->lifecycle = Lifecycle::kReady;
        return transfer_json(handle, raw, output, output_bytes);
    } catch (...) {
        poison(handle);
        return kInternalFailure;
    }
}

SPEAKER_PARAKEET_EXPORT void
speaker_parakeet_bridge_free_json(char* value) noexcept {
    try {
        parakeet_capi_free_string(value);
    } catch (...) {
    }
}

SPEAKER_PARAKEET_EXPORT int speaker_parakeet_bridge_close(
    speaker_parakeet_bridge** inout) noexcept {
    if (!inout) return kInvalidArgument;
    speaker_parakeet_bridge* handle = *inout;
    *inout = nullptr;
    if (!handle) return kOk;
    try {
        std::lock_guard<std::mutex> lifecycle_lock(g_lifecycle_mutex);
        if (handle->magic != kHandleMagic) return kLifecycleFailure;
        bool clean = false;
        {
            std::lock_guard<std::mutex> handle_lock(handle->mutex);
            clean = destroy_native(handle);
        }
        delete handle;
        g_handle_open = false;
        return clean ? kOk : kInternalFailure;
    } catch (...) {
        return kInternalFailure;
    }
}
