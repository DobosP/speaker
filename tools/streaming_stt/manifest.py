"""Strict, hash-bound worker manifests for the streaming-STT harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .bounded_io import (
    BoundedReadError,
    hash_regular_bounded,
    read_regular_bounded,
)
from .runtime_receipt import (
    RuntimeTreeReceiptError,
    verify_isolated_venv_marker,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ID_RE = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}\Z")
_ELF_LIBRARY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+~-]{0,127}\Z")
_MAX_MANIFEST_BYTES = 64 * 1024
MAX_PYTHON_BYTES = 512 * 1024 * 1024
MAX_WORKER_BYTES = 4 * 1024 * 1024
MAX_FAKE_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_MOONSHINE_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MOONSHINE_TOTAL_ARTIFACT_BYTES = 1024 * 1024 * 1024
MAX_NEMOTRON_ARTIFACT_BYTES = 3 * 1024 * 1024 * 1024
MAX_NEMOTRON_TOTAL_ARTIFACT_BYTES = 4 * 1024 * 1024 * 1024
MAX_SHERPA_ZIPFORMER_ARTIFACT_BYTES = 384 * 1024 * 1024
MAX_SHERPA_ZIPFORMER_TOTAL_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_MOBILE_ZIPFORMER_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_MOBILE_ZIPFORMER_TOTAL_ARTIFACT_BYTES = 128 * 1024 * 1024
MAX_MOBILE_ZIPFORMER_SOURCE_LOCK_BYTES = 1024 * 1024
MAX_PARAKEET_ARTIFACT_BYTES = 512 * 1024 * 1024
MAX_PARAKEET_TOTAL_ARTIFACT_BYTES = 640 * 1024 * 1024
MAX_FASTER_WHISPER_CONTROL_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_FASTER_WHISPER_TOTAL_ARTIFACT_BYTES = 40 * 1024 * 1024
MAX_PARAKEET_CPP_CONTROL_ARTIFACT_BYTES = 1024 * 1024
MAX_PARAKEET_CPP_BRIDGE_SOURCE_BYTES = 1024 * 1024
MAX_PARAKEET_CPP_LIBRARY_BYTES = 256 * 1024 * 1024
MAX_PARAKEET_CPP_MODEL_BYTES = 320 * 1024 * 1024
MAX_PARAKEET_CPP_TOTAL_ARTIFACT_BYTES = 640 * 1024 * 1024
MAX_KYUTAI_ARTIFACT_BYTES = 2 * 1024 * 1024 * 1024
MAX_KYUTAI_TOTAL_ARTIFACT_BYTES = 3 * 1024 * 1024 * 1024
_MANIFEST_V1_FIELDS = {
    "schema_version",
    "model_id",
    "adapter",
    "python",
    "worker",
    "artifacts",
    "limits",
}
_MANIFEST_V2_FIELDS = {*_MANIFEST_V1_FIELDS, "adapter_config"}
_MANIFEST_V3_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V4_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V5_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V6_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V7_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V8_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V9_FIELDS = _MANIFEST_V2_FIELDS
_MANIFEST_V10_FIELDS = {
    "schema_version",
    "kind",
    "model_id",
    "adapter",
    "python",
    "worker",
    "artifacts",
    "limits",
    "mobile_config",
    "source",
    "runtime",
    "artifact_set_sha256",
    "total_size_bytes",
    "evidence_scope",
}
_FILE_FIELDS = {"path", "sha256", "size_bytes"}
_ARTIFACT_FIELDS = {"name", *_FILE_FIELDS}
_LIMIT_FIELDS = {"startup_timeout_sec", "case_timeout_sec"}
_MOONSHINE_CONFIG_FIELDS = {
    "package_version",
    "api_version",
    "model_arch",
    "provider",
    "language",
}
_MOONSHINE_EXTERNAL_ENDPOINT_CONFIG_FIELDS = {
    *_MOONSHINE_CONFIG_FIELDS,
    "segmentation_mode",
    "endpoint_owner",
    "vad_threshold",
    "vad_max_segment_duration_sec",
    "vad_hop_size_samples",
    "streaming_chunk_samples",
    "online_partial_interval_ms",
    "authoritative_alignment_samples",
    "tail_alignment_policy",
    "finalization_policy",
    "maximum_source_samples",
}
_NEMOTRON_CONFIG_FIELDS = {
    "python_version",
    "transformers_version",
    "librosa_version",
    "torch_version",
    "cuda_version",
    "model_revision",
    "lookahead_tokens",
    "language",
    "device",
    "dtype",
    "native_sample_rate_hz",
    "native_hop_length_samples",
    "native_n_fft_samples",
    "native_win_length_samples",
    "native_first_chunk_frames",
    "native_chunk_frames",
    "native_first_window_samples",
    "native_window_samples",
    "native_stride_samples",
    "streaming_latency_ms",
    "wheel_lock_sha256",
    "runtime_content_sha256",
    "runtime_file_count",
    "runtime_total_size_bytes",
    "runtime_maximum_file_bytes",
}
_SHERPA_ZIPFORMER_CONFIG_FIELDS = {
    "package_version",
    "numpy_version",
    "source_repo_id",
    "variant",
    "language",
    "sample_rate",
    "feature_dim",
    "production_device_profile",
    "production_num_threads",
    "benchmark_profile",
    "num_threads",
    "provider",
    "enable_endpoint_detection",
    "decoding_method",
    "max_active_paths",
    "rule1_min_trailing_silence",
    "rule2_min_trailing_silence",
    "rule3_min_utterance_length",
}
_MOBILE_ZIPFORMER_CONFIG_FIELDS = {
    "package_version",
    "core_package_version",
    "numpy_version",
    "source_repo_id",
    "source_revision",
    "variant",
    "language",
    "sample_rate",
    "feature_dim",
    "num_threads",
    "provider",
    "debug",
    "decoding_method",
    "max_active_paths",
    "model_type",
    "enable_endpoint_detection",
    "rule1_min_trailing_silence",
    "rule2_min_trailing_silence",
    "rule3_min_utterance_length",
    "native_chunk_samples",
    "maximum_tail_padding_samples",
}
_MOBILE_ZIPFORMER_SOURCE_FIELDS = {
    "repo_id",
    "revision",
    "lock",
    "lock_recipe_sha256",
}
_MOBILE_ZIPFORMER_RUNTIME_FIELDS = {
    "python",
    "worker",
    "distributions",
    "metadata_only_verified",
    "packages_imported",
    "model_loaded",
}
_MOBILE_ZIPFORMER_DISTRIBUTION_FIELDS = {
    "sherpa-onnx",
    "sherpa-onnx-core",
    "numpy",
}
_MOBILE_ZIPFORMER_EVIDENCE_FIELDS = {
    "downloaded",
    "packages_imported",
    "model_loaded",
    "model_executed",
    "gpu_used",
    "audio_device_opened",
    "evaluation_result_authority",
    "mobile_device_evidence",
}
_PARAKEET_CONFIG_FIELDS = {
    "python_version",
    "nemo_version",
    "torch_version",
    "cuda_version",
    "numpy_version",
    "model_repo_id",
    "model_revision",
    "model_filename",
    "language",
    "device",
    "dtype",
    "sample_rate",
    "native_chunk_samples",
    "maximum_tail_padding_samples",
    "attention_context_left",
    "attention_context_right",
    "batch_size",
    "eou_token",
    "eob_token",
    "use_amp",
    "wheel_lock_sha256",
    "runtime_content_sha256",
    "runtime_file_count",
    "runtime_total_size_bytes",
    "runtime_maximum_file_bytes",
}
_FASTER_WHISPER_CONFIG_FIELDS = {
    "python_version",
    "faster_whisper_version",
    "ctranslate2_version",
    "numpy_version",
    "cublas_version",
    "cudnn_version",
    "cuda_nvrtc_version",
    "language",
    "task",
    "device",
    "device_index",
    "compute_type",
    "cpu_threads",
    "num_workers",
    "sample_rate",
    "execution_mode",
    "partial_hypotheses",
    "tail_padding_policy",
    "beam_size",
    "patience",
    "temperature",
    "compression_ratio_threshold",
    "log_prob_threshold",
    "no_speech_threshold",
    "vad_filter",
    "condition_on_previous_text",
    "without_timestamps",
    "word_timestamps",
    "runtime_content_sha256",
    "runtime_file_count",
    "runtime_total_size_bytes",
    "runtime_maximum_file_bytes",
    "model_content_sha256",
    "model_file_count",
    "model_total_size_bytes",
    "model_maximum_file_bytes",
}
_PARAKEET_CPP_CONFIG_FIELDS = {
    "upstream_repo_id",
    "upstream_commit",
    "upstream_tag",
    "ggml_commit",
    "c_api_version",
    "bridge_abi_version",
    "upstream_license",
    "upstream_license_sha256",
    "model_repo_id",
    "model_revision",
    "source_model_repo_id",
    "source_model_revision",
    "model_filename",
    "model_sha256",
    "model_size_bytes",
    "model_dtype",
    "model_license",
    "requested_device",
    "actual_device",
    "num_threads",
    "ggml_native",
    "ggml_cuda",
    "ggml_metal",
    "ggml_vulkan",
    "ggml_hip",
    "sample_rate",
    "native_chunk_samples",
    "maximum_tail_padding_samples",
    "frame_sec",
}
_KYUTAI_CONFIG_FIELDS = {
    "python_version",
    "moshi_version",
    "torch_version",
    "cuda_version",
    "julius_version",
    "wheel_lock_sha256",
    "runtime_content_sha256",
    "runtime_file_count",
    "runtime_total_size_bytes",
    "runtime_maximum_file_bytes",
    "model_repo_id",
    "model_revision",
    "model_config_filename",
    "model_weights_filename",
    "mimi_weights_filename",
    "tokenizer_filename",
    "model_license",
    "language",
    "device",
    "dtype",
    "input_sample_rate_hz",
    "mimi_sample_rate_hz",
    "input_chunk_samples",
    "mimi_frame_samples",
    "resampling_mode",
    "initial_frame_policy",
    "initial_frame_prime_steps",
    "terminal_tail_samples",
    "partial_interval_ms",
    "maximum_source_samples",
    "batch_size",
    "temperature",
    "text_temperature",
    "use_sampling",
    "text_padding_token_id",
    "end_of_padding_token_id",
    "audio_delay_seconds",
    "audio_silence_prefix_seconds",
    "semantic_head_count",
    "semantic_head_dim",
    "semantic_head_policy",
    "endpoint_owner",
    "early_stop",
    "num_threads",
    "maximum_vram_fraction",
    "minimum_free_vram_mb",
    "minimum_host_available_bytes",
    "torch_compile",
    "no_torch_compile_env",
    "cuda_graph",
    "no_cuda_graph_env",
    "local_files_only",
}
MOONSHINE_ADAPTER = "moonshine-voice-stream-v1"
MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER = (
    "moonshine-voice-external-endpoint-v1"
)
NEMOTRON_ADAPTER = "transformers-nemotron-3.5-stream-v1"
SHERPA_ZIPFORMER_ADAPTER = "sherpa-onnx-gigaspeech-zipformer-stream-v1"
MOBILE_ZIPFORMER_ADAPTER = "sherpa-onnx-mobile-zipformer-endpoint-v1"
PARAKEET_REALTIME_EOU_ADAPTER = "nemo-parakeet-realtime-eou-v1"
FASTER_WHISPER_ENDPOINT_ADAPTER = "faster-whisper-endpoint-v1"
PARAKEET_CPP_ADAPTER = "parakeet-cpp-realtime-eou-v1"
KYUTAI_ADAPTER = "kyutai-stt-1b-semantic-stream-v1"
NEMOTRON_WHEEL_LOCK_SHA256 = (
    "df268a2e268221428256b3ec525a3ad49da65b526b2e09b88df3802533b5af01"
)
NEMOTRON_RUNTIME_CONTENT_SHA256 = (
    "ebcf9cab82c93ab5503bb438fbc9585475fa138c4e18aaea73d9ab34fc48bb8f"
)
NEMOTRON_RUNTIME_FILE_COUNT = 24_273
NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES = 6_752_927_292
NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES = 1_007_735_593
PARAKEET_REALTIME_EOU_PYTHON_VERSION = "3.12.3"
PARAKEET_REALTIME_EOU_NEMO_VERSION = "2.7.3"
PARAKEET_REALTIME_EOU_TORCH_VERSION = "2.7.1+cu126"
PARAKEET_REALTIME_EOU_CUDA_VERSION = "12.6"
PARAKEET_REALTIME_EOU_NUMPY_VERSION = "2.2.6"
PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256 = (
    "b58bc18fda01e91fc92d70fdb7d69451faa392bf846245feecea8f24be4f7069"
)
PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES = 77_368
PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256 = (
    "36224564c7b0c1895e91fae76b060382dd4376e9d4e93e575b99ca043475df39"
)
PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT = 49_600
PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES = 6_430_910_098
PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES = 984_633_129
KYUTAI_RUNTIME_WHEEL_LOCK_SHA256 = (
    "8d0e41563bb5e91500af42a912c5e6f825f16a67537ddc350556303e88609e25"
)
KYUTAI_RUNTIME_WHEEL_LOCK_SIZE_BYTES = 25_818
KYUTAI_RUNTIME_CONTENT_SHA256 = (
    "9edc9c42b8c718d0e4b17d917c04c05acc0b5ecca4ec9504866ad34f1f62dbf0"
)
KYUTAI_RUNTIME_FILE_COUNT = 16_547
KYUTAI_RUNTIME_TOTAL_SIZE_BYTES = 5_694_993_765
KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES = 984_633_129
MOONSHINE_ARTIFACT_NAMES = (
    "runtime-receipt",
    "venv-marker",
    "release-wheel",
    "model-adapter",
    "model-cross-kv",
    "model-decoder-kv",
    "model-encoder",
    "model-frontend",
    "model-config",
    "model-tokenizer",
)
_MOONSHINE_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "venv-marker": "pyvenv.cfg",
    "release-wheel": ("moonshine_voice-0.1.0-py3-none-manylinux_2_34_x86_64.whl"),
    "model-adapter": "adapter.ort",
    "model-cross-kv": "cross_kv.ort",
    "model-decoder-kv": "decoder_kv.ort",
    "model-encoder": "encoder.ort",
    "model-frontend": "frontend.ort",
    "model-config": "streaming_config.json",
    "model-tokenizer": "tokenizer.bin",
}
_MOONSHINE_SMALL_ARTIFACTS = {
    "runtime-receipt": 4 * 1024 * 1024,
    "venv-marker": 64 * 1024,
    "release-wheel": 128 * 1024 * 1024,
}
_MOONSHINE_RELEASE_RECEIPT = (
    "0f833deb43bad5dcfb4cfd3257b6df83ef9abd3f27be3199622fe41932e8d916",
    56_369_045,
)
_MOONSHINE_MODEL_RECEIPTS = {
    "tiny-streaming": {
        "model-adapter": (
            "df13e655b29d279911fcb42d8b91b0e655b8fe32b7ba1f463ece663ce55ae6eb",
            1_319_440,
        ),
        "model-cross-kv": (
            "5acfca68f7bb068c68c1960b54e215995ba07ee46b61645b78bff010a14e5a92",
            1_264_384,
        ),
        "model-decoder-kv": (
            "6e3828f1db4b634bc525cb8ba1f0b628ec56059168f0336ad060891c7c1c9154",
            32_403_688,
        ),
        "model-encoder": (
            "96dde726be90c4429f3bc458d04e3ea5bd1818a5fdcd0152edf4c07b8e405c07",
            7_569_200,
        ),
        "model-frontend": (
            "bbdf5edb120cb3df1adf9ebc07c35136539b007a7047fd148c6f2960fc56fcf1",
            8_324_600,
        ),
        "model-config": (
            "74fe5ddebd63b17caf59e8a3b18c17547ff7bce1642050edbb1c3962674f8950",
            509,
        ),
        "model-tokenizer": (
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            249_974,
        ),
    },
    "small-streaming": {
        "model-adapter": (
            "d8493e0ac76a198b309a8be6f74b3101e235f773ffe5d6b378278cd7e4177992",
            2_867_424,
        ),
        "model-cross-kv": (
            "6e57d1361717e00d73336a0c3beafedae784b1e537905ad253dee33db4007466",
            5_298_736,
        ),
        "model-decoder-kv": (
            "d5adfcfaa6e582144791f1568bd0f683852c7bfbb8c79acad97499da05e4ffcf",
            81_435_904,
        ),
        "model-encoder": (
            "3b21d02eff6aa5651524ada4271d37c1d7bba4eb3d256415074f2cfdbaeb526a",
            43_853_224,
        ),
        "model-frontend": (
            "e086451043c1c8652a9614e4a4a81d5807221b611584a3cf31f73779d5900003",
            30_984_200,
        ),
        "model-config": (
            "26f02b6afb22d60871a5efd85c3d38e569cc0ddb6c5eb6e93d3260152ae8a47a",
            512,
        ),
        "model-tokenizer": (
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            249_974,
        ),
    },
    "medium-streaming": {
        "model-adapter": (
            "16307442b7f4229f2f1511fc51b545cec9616e55872c588f3a297bbc6f4762ea",
            3_647_712,
        ),
        "model-cross-kv": (
            "354b9a955caeb768b528f447f0a36ce4b850ca7b4531900165df304d97904fba",
            11_544_952,
        ),
        "model-decoder-kv": (
            "fa67aa87521247f5bf44d3e44d4e4978e58c1f114249c3c6909c882624056715",
            146_216_448,
        ),
        "model-encoder": (
            "a5f11167a62eef61787fe8410453257d6ddb8eba90af461a9604e5f2e93d5322",
            94_202_872,
        ),
        "model-frontend": (
            "378fe8a5d7090a1b9ab88bbb1fc95bde010cdd64ec23419350d2d23c675636e9",
            47_467_256,
        ),
        "model-config": (
            "28e83b7a28e91472692a035e0dae3116422ae43aeb2bef5ed822c44ce89b88af",
            513,
        ),
        "model-tokenizer": (
            "6884b35fd6377d4c4d32336a0bc152f36b64d1e45b6503683cdc238250a8472d",
            249_974,
        ),
    },
}
NEMOTRON_ARTIFACT_NAMES = (
    "runtime-receipt",
    "runtime-wheel-lock",
    "venv-marker",
    "model-config",
    "model-generation-config",
    "model-weights",
    "model-processor-config",
    "model-tokenizer",
    "model-tokenizer-config",
)
_NEMOTRON_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "runtime-wheel-lock": "nemotron-runtime-wheels.lock.json",
    "venv-marker": "pyvenv.cfg",
    "model-config": "config.json",
    "model-generation-config": "generation_config.json",
    "model-weights": "model.safetensors",
    "model-processor-config": "processor_config.json",
    "model-tokenizer": "tokenizer.json",
    "model-tokenizer-config": "tokenizer_config.json",
}
_NEMOTRON_SMALL_ARTIFACTS = {
    "runtime-receipt": 32 * 1024 * 1024,
    "runtime-wheel-lock": 256 * 1024,
    "venv-marker": 64 * 1024,
    "model-config": 4 * 1024 * 1024,
    "model-generation-config": 4 * 1024 * 1024,
    "model-processor-config": 4 * 1024 * 1024,
    "model-tokenizer": 16 * 1024 * 1024,
    "model-tokenizer-config": 4 * 1024 * 1024,
}
_NEMOTRON_CONTROL_RECEIPTS = {
    "runtime-wheel-lock": (NEMOTRON_WHEEL_LOCK_SHA256, 32_099),
}
_NEMOTRON_MODEL_RECEIPTS = {
    "model-config": (
        "62d186fd91f518e00e7867500f1f5819225e8ee95ea3e21b546514bf2048e845",
        1_376,
    ),
    "model-generation-config": (
        "993e5d4cb74a6fe9d6e7084a76b3313c1446740679be4676570c23b664fdc07e",
        193,
    ),
    "model-weights": (
        "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174",
        2_552_062_944,
    ),
    "model-processor-config": (
        "ec47870f1091ea4f25539208387b45b902c92d0e3f997a30061ef88f73437ab0",
        2_519,
    ),
    "model-tokenizer": (
        "3f3d481deb073b64c2082e8c7860d487a3a62774bf4e9e4faac83007e181f246",
        752_051,
    ),
    "model-tokenizer-config": (
        "5c641c5b3f50702a60082690d27c1ce7fcb5a92c4a624793bcae0f21eda3d6e0",
        881,
    ),
}
_NEMOTRON_STREAM_GEOMETRY = {
    0: {
        "native_first_chunk_frames": 1,
        "native_chunk_frames": 8,
        "native_first_window_samples": 200,
        "native_window_samples": 1_680,
        "native_stride_samples": 1_280,
        "streaming_latency_ms": 80,
    },
    3: {
        "native_first_chunk_frames": 25,
        "native_chunk_frames": 32,
        "native_first_window_samples": 4_040,
        "native_window_samples": 5_520,
        "native_stride_samples": 5_120,
        "streaming_latency_ms": 320,
    },
    6: {
        "native_first_chunk_frames": 49,
        "native_chunk_frames": 56,
        "native_first_window_samples": 7_880,
        "native_window_samples": 9_360,
        "native_stride_samples": 8_960,
        "streaming_latency_ms": 560,
    },
    13: {
        "native_first_chunk_frames": 105,
        "native_chunk_frames": 112,
        "native_first_window_samples": 16_840,
        "native_window_samples": 18_320,
        "native_stride_samples": 17_920,
        "streaming_latency_ms": 1_120,
    },
}
SHERPA_ZIPFORMER_ARTIFACT_SPECS = (
    (
        "model-encoder",
        "encoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "a423883ce5754507fd941755ab0b5bc426a84ac670cbe21cf060e9e2c66dc660",
        262_127_043,
    ),
    (
        "model-decoder",
        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
        2_092_621,
    ),
    (
        "model-joiner",
        "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
        1_026_405,
    ),
    (
        "model-tokens",
        "tokens.txt",
        "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
        5_048,
    ),
)
SHERPA_ZIPFORMER_ARTIFACT_NAMES = tuple(
    name for name, _basename, _sha256, _size_bytes in SHERPA_ZIPFORMER_ARTIFACT_SPECS
)
_SHERPA_ZIPFORMER_ARTIFACT_BASENAMES = {
    name: basename
    for name, basename, _sha256, _size_bytes in SHERPA_ZIPFORMER_ARTIFACT_SPECS
}
_SHERPA_ZIPFORMER_MODEL_RECEIPTS = {
    name: (sha256, size_bytes)
    for name, _basename, sha256, size_bytes in SHERPA_ZIPFORMER_ARTIFACT_SPECS
}
MOBILE_ZIPFORMER_ARTIFACT_SPECS = (
    (
        "model-encoder",
        "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
        "int8",
        "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1",
        71_083_163,
    ),
    (
        "model-decoder",
        "decoder-epoch-99-avg-1-chunk-16-left-128.onnx",
        "fp32",
        "7bf787f90b194b307e5a4ad6a34fadb4e748304c35f78a8d66358a05b13ee6ef",
        2_092_621,
    ),
    (
        "model-joiner",
        "joiner-epoch-99-avg-1-chunk-16-left-128.onnx",
        "fp32",
        "210591f72b3c56b8364f85f345dca240bc2b4c00632848f4aa923630d5639d3b",
        1_026_405,
    ),
    (
        "model-tokens",
        "tokens.txt",
        "text",
        "49e3c2646595fd907228b3c6787069658f67b17377c60aeb8619c4551b2316fb",
        5_048,
    ),
)
MOBILE_ZIPFORMER_ARTIFACT_NAMES = tuple(
    name
    for name, _basename, _precision, _sha256, _size_bytes in (
        MOBILE_ZIPFORMER_ARTIFACT_SPECS
    )
)
_MOBILE_ZIPFORMER_ARTIFACT_BASENAMES = {
    name: basename
    for name, basename, _precision, _sha256, _size_bytes in (
        MOBILE_ZIPFORMER_ARTIFACT_SPECS
    )
}
_MOBILE_ZIPFORMER_MODEL_RECEIPTS = {
    name: (sha256, size_bytes)
    for name, _basename, _precision, sha256, size_bytes in (
        MOBILE_ZIPFORMER_ARTIFACT_SPECS
    )
}
MOBILE_ZIPFORMER_TOTAL_SIZE_BYTES = sum(
    size_bytes
    for _name, _basename, _precision, _sha256, size_bytes in (
        MOBILE_ZIPFORMER_ARTIFACT_SPECS
    )
)
_MOBILE_ZIPFORMER_ARTIFACT_SET_ROWS = [
    {
        "name": name,
        "filename": basename,
        "precision": precision,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }
    for name, basename, precision, sha256, size_bytes in (
        MOBILE_ZIPFORMER_ARTIFACT_SPECS
    )
]
MOBILE_ZIPFORMER_ARTIFACT_SET_SHA256 = hashlib.sha256(
    b"speaker-mobile-zipformer-artifact-set-v1\0"
    + json.dumps(
        _MOBILE_ZIPFORMER_ARTIFACT_SET_ROWS,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
MOBILE_ZIPFORMER_SOURCE_LOCK_RECIPE_SHA256 = (
    "736d516ceada4bc8864ca0b5de9f03167a415cb93ad66252ab99351028a15c49"
)
MOBILE_ZIPFORMER_SOURCE_LOCK_SHA256 = (
    "3b928f709a1c50f426291968bedff67811f7eed0e1c48e0773df475558520efc"
)
MOBILE_ZIPFORMER_SOURCE_LOCK_SIZE_BYTES = 2_335
PARAKEET_REALTIME_EOU_ARTIFACT_NAMES = (
    "runtime-receipt",
    "runtime-wheel-lock",
    "venv-marker",
    "model-nemo",
)
_PARAKEET_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "runtime-wheel-lock": "parakeet-realtime-eou-runtime-wheels.lock.json",
    "venv-marker": "pyvenv.cfg",
    "model-nemo": "parakeet_realtime_eou_120m-v1.nemo",
}
_PARAKEET_SMALL_ARTIFACTS = {
    "runtime-receipt": 32 * 1024 * 1024,
    "runtime-wheel-lock": 512 * 1024,
    "venv-marker": 64 * 1024,
}
PARAKEET_REALTIME_EOU_MODEL_RECEIPT = (
    "6603a22a53b7c1a4bac4736cb24628fb568a7102ba931a28c799e2e72f109893",
    460_062_720,
)
FASTER_WHISPER_ARTIFACT_NAMES = (
    "runtime-receipt",
    "model-receipt",
    "venv-marker",
)
_FASTER_WHISPER_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "model-receipt": "model-receipt.json",
    "venv-marker": "pyvenv.cfg",
}
_FASTER_WHISPER_SMALL_ARTIFACTS = {
    "runtime-receipt": 32 * 1024 * 1024,
    "model-receipt": 4 * 1024 * 1024,
    "venv-marker": 64 * 1024,
}
FASTER_WHISPER_REQUIRED_MODEL_FILES = frozenset(
    {"config.json", "model.bin", "tokenizer.json"}
)
FASTER_WHISPER_VOCABULARY_FILES = frozenset({"vocabulary.json", "vocabulary.txt"})
PARAKEET_CPP_ARTIFACT_NAMES = (
    "source-receipt",
    "build-receipt",
    "model-receipt",
    "libparakeet",
    "bridge-library",
    "model-gguf",
)
_PARAKEET_CPP_ARTIFACT_BASENAMES = {
    "source-receipt": "source-receipt.json",
    "build-receipt": "build-receipt.json",
    "model-receipt": "model-receipt.json",
    "libparakeet": "libparakeet.so",
    "bridge-library": "libspeaker_parakeet_bridge.so",
    "model-gguf": "realtime_eou_120m-v1-f16.gguf",
}
PARAKEET_CPP_MODEL_RECEIPT = (
    "d1a2b12f12b8a096a57499c9111ed13b442a2b786e17a292c168be45088f0edc",
    266_517_952,
)
PARAKEET_CPP_LIBPARAKEET_RECEIPT = (
    "eae6fc236de170164193ec3c0b326cfc2327d586161c1d727d1109bbc8d5659d",
    2_601_992,
)
PARAKEET_CPP_LIBPARAKEET_NEEDED = (
    "libstdc++.so.6",
    "libm.so.6",
    "libgcc_s.so.1",
    "libc.so.6",
    "ld-linux-x86-64.so.2",
)
PARAKEET_CPP_BRIDGE_RECEIPT = (
    "eb57b8b8db463ebb287997495699bfee9cf72df3099de21bef8d343d4c510f5e",
    27_168,
)
PARAKEET_CPP_BRIDGE_NEEDED = (
    "libparakeet.so",
    "libstdc++.so.6",
    "libc.so.6",
)
PARAKEET_CPP_BRIDGE_RUNPATH = "$ORIGIN"
PARAKEET_CPP_PARENT_GIT_TREE = "7e1b2eeb55caf1c0f6e0a15da0bc769ac7f2524c"
PARAKEET_CPP_GGML_GIT_TREE = "fb778cbfe12baef2e69f7f6132f2cf8b243269a2"
PARAKEET_CPP_PATCH_SPECS = (
    (
        "0001-ggml-cpu-fold-broadcast-iterations-in-llamafile_sgem.patch",
        "779bb7c37d38c7a007e9f6e874db7040fafe7e9d93d4152efcbdae8fa560963b",
    ),
    (
        "0002-metal-conv-2d-dw.patch",
        "55bad8241fd355fbecb63516a5778dd92b916fdd97b6e780ff2979b74f4a8fe3",
    ),
    (
        "0003-metal-pad-leading.patch",
        "61e7e7be0a2b22afa8aa05afb1d90d89e75f67a2ecdc0fa7d8f51bd705002609",
    ),
    (
        "0004-cuda-pad-grid-stride.patch",
        "6a57d875c16a0f9aee42f88b9f07ba44fdf05718f54009697299495d8fb02424",
    ),
)
PARAKEET_CPP_PATCHED_DIFF_SHA256 = (
    "e62f5e880cde081d478927b62f304f60c93e92a8996b4e82f2e3b6a9205e9926"
)
PARAKEET_CPP_BRIDGE_SOURCE_SHA256 = (
    "1fdf9d194afadc986cb956e419c75cf68e7c154eee8098b591c4bcb340eef2e4"
)
_PARAKEET_CPP_RECEIPT_NAMES = frozenset(
    {"source-receipt", "build-receipt", "model-receipt"}
)
_PARAKEET_CPP_LIBRARY_NAMES = frozenset({"libparakeet", "bridge-library"})
KYUTAI_ARTIFACT_NAMES = (
    "runtime-receipt",
    "runtime-wheel-lock",
    "venv-marker",
    "model-config",
    "model-weights",
    "model-mimi",
    "model-tokenizer",
)
_KYUTAI_ARTIFACT_BASENAMES = {
    "runtime-receipt": "runtime-receipt.json",
    "runtime-wheel-lock": "kyutai-stt-runtime-wheels.lock.json",
    "venv-marker": "pyvenv.cfg",
    "model-config": "config.json",
    "model-weights": "model.safetensors",
    "model-mimi": "mimi-pytorch-e351c8d8@125.safetensors",
    "model-tokenizer": "tokenizer_en_fr_audio_8000.model",
}
_KYUTAI_SMALL_ARTIFACTS = {
    "runtime-receipt": 32 * 1024 * 1024,
    "runtime-wheel-lock": 512 * 1024,
    "venv-marker": 64 * 1024,
    "model-config": 64 * 1024,
    "model-tokenizer": 4 * 1024 * 1024,
}
KYUTAI_MODEL_RECEIPTS = {
    "model-config": (
        "a3f1c6f7a39fca1fb1bbff68eaabc560b8037d2cdc68aa1f489859949a4223de",
        1_315,
    ),
    "model-weights": (
        "b9e97c53229dce728d65c76bfa892f7b563c69d671899f0ebc6518582dddec6f",
        1_978_522_200,
    ),
    "model-mimi": (
        "09b782f0629851a271227fb9d36db65c041790365f11bbe5d3d59369cf863f50",
        384_644_900,
    ),
    "model-tokenizer": (
        "cd87dd5d17169151782ac700280ec057e5d658a9afbe238a048ea5ff318cce69",
        120_378,
    ),
}


class ManifestError(RuntimeError):
    """A detail-free manifest or local-artifact validation failure."""


@dataclass(frozen=True)
class BoundFile:
    path: Path
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class BoundArtifact(BoundFile):
    name: str


@dataclass(frozen=True)
class ElfInspection:
    dependencies: tuple[str, ...]
    runpath: str | None
    relro: bool
    bind_now: bool
    noexecstack: bool


@dataclass(frozen=True)
class WorkerLimits:
    startup_timeout_sec: float
    case_timeout_sec: float


@dataclass(frozen=True)
class MoonshineConfig:
    package_version: str = "0.1.0"
    api_version: int = 30000
    model_arch: str = "tiny-streaming"
    provider: str = "cpu"
    language: str = "en"

    def __post_init__(self) -> None:
        if (
            self.package_version != "0.1.0"
            or type(self.api_version) is not int
            or self.api_version != 30000
            or self.model_arch
            not in {"tiny-streaming", "small-streaming", "medium-streaming"}
            or self.provider != "cpu"
            or self.language != "en"
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "package_version": self.package_version,
            "api_version": self.api_version,
            "model_arch": self.model_arch,
            "provider": self.provider,
            "language": self.language,
        }


@dataclass(frozen=True)
class MoonshineExternalEndpointConfig(MoonshineConfig):
    """Moonshine decode bound to a controlled external-input boundary."""

    segmentation_mode: str = "external-presegmented"
    endpoint_owner: str = "external-input-boundary"
    vad_threshold: float = 0.0
    vad_max_segment_duration_sec: float = 136.0
    vad_hop_size_samples: int = 512
    streaming_chunk_samples: int = 1280
    online_partial_interval_ms: int = 500
    authoritative_alignment_samples: int = 2560
    tail_alignment_policy: str = "zero-pad-to-vad-model-lcm"
    finalization_policy: str = "verified-native-free-authoritative-batch-v2"
    maximum_source_samples: int = 2_097_152

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.segmentation_mode) is not str
            or self.segmentation_mode != "external-presegmented"
            or type(self.endpoint_owner) is not str
            or self.endpoint_owner != "external-input-boundary"
            or type(self.vad_threshold) is not float
            or self.vad_threshold != 0.0
            or math.copysign(1.0, self.vad_threshold) != 1.0
            or type(self.vad_max_segment_duration_sec) is not float
            or self.vad_max_segment_duration_sec != 136.0
            or type(self.vad_hop_size_samples) is not int
            or self.vad_hop_size_samples != 512
            or type(self.streaming_chunk_samples) is not int
            or self.streaming_chunk_samples != 1280
            or type(self.online_partial_interval_ms) is not int
            or self.online_partial_interval_ms != 500
            or type(self.authoritative_alignment_samples) is not int
            or self.authoritative_alignment_samples != 2560
            or type(self.tail_alignment_policy) is not str
            or self.tail_alignment_policy != "zero-pad-to-vad-model-lcm"
            or type(self.finalization_policy) is not str
            or self.finalization_policy
            != "verified-native-free-authoritative-batch-v2"
            or type(self.maximum_source_samples) is not int
            or self.maximum_source_samples != 2_097_152
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        value = super().as_dict()
        value.update(
            {
                "segmentation_mode": self.segmentation_mode,
                "endpoint_owner": self.endpoint_owner,
                "vad_threshold": self.vad_threshold,
                "vad_max_segment_duration_sec": self.vad_max_segment_duration_sec,
                "vad_hop_size_samples": self.vad_hop_size_samples,
                "streaming_chunk_samples": self.streaming_chunk_samples,
                "online_partial_interval_ms": self.online_partial_interval_ms,
                "authoritative_alignment_samples": (
                    self.authoritative_alignment_samples
                ),
                "tail_alignment_policy": self.tail_alignment_policy,
                "finalization_policy": self.finalization_policy,
                "maximum_source_samples": self.maximum_source_samples,
            }
        )
        return value


@dataclass(frozen=True)
class NemotronConfig:
    python_version: str = "3.12.3"
    transformers_version: str = "5.13.1"
    librosa_version: str = "0.11.0"
    torch_version: str = "2.12.1+cu126"
    cuda_version: str = "12.6"
    model_revision: str = "f3d333391852ba876df169dcc9ba902d25b6ab0b"
    lookahead_tokens: int = 3
    language: str = "en-US"
    device: str = "cuda:0"
    dtype: str = "float32"
    native_sample_rate_hz: int | None = None
    native_hop_length_samples: int | None = None
    native_n_fft_samples: int | None = None
    native_win_length_samples: int | None = None
    native_first_chunk_frames: int | None = None
    native_chunk_frames: int | None = None
    native_first_window_samples: int | None = None
    native_window_samples: int | None = None
    native_stride_samples: int | None = None
    streaming_latency_ms: int | None = None
    wheel_lock_sha256: str = NEMOTRON_WHEEL_LOCK_SHA256
    runtime_content_sha256: str = NEMOTRON_RUNTIME_CONTENT_SHA256
    runtime_file_count: int = NEMOTRON_RUNTIME_FILE_COUNT
    runtime_total_size_bytes: int = NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES
    runtime_maximum_file_bytes: int = NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES

    def __post_init__(self) -> None:
        if (
            self.python_version != "3.12.3"
            or self.transformers_version != "5.13.1"
            or self.librosa_version != "0.11.0"
            or self.torch_version != "2.12.1+cu126"
            or self.cuda_version != "12.6"
            or self.model_revision != "f3d333391852ba876df169dcc9ba902d25b6ab0b"
            or type(self.lookahead_tokens) is not int
            or self.lookahead_tokens not in {0, 3, 6, 13}
            or self.language not in {"en-US", "en-GB", "ro-RO"}
            or self.device != "cuda:0"
            or self.dtype != "float32"
            or self.wheel_lock_sha256 != NEMOTRON_WHEEL_LOCK_SHA256
            or self.runtime_content_sha256 != NEMOTRON_RUNTIME_CONTENT_SHA256
            or self.runtime_file_count != NEMOTRON_RUNTIME_FILE_COUNT
            or self.runtime_total_size_bytes != NEMOTRON_RUNTIME_TOTAL_SIZE_BYTES
            or self.runtime_maximum_file_bytes != NEMOTRON_RUNTIME_MAXIMUM_FILE_BYTES
        ):
            raise ManifestError()
        expected_geometry = {
            "native_sample_rate_hz": 16_000,
            "native_hop_length_samples": 160,
            "native_n_fft_samples": 512,
            "native_win_length_samples": 400,
            **_NEMOTRON_STREAM_GEOMETRY[self.lookahead_tokens],
        }
        for field_name, expected in expected_geometry.items():
            actual = getattr(self, field_name)
            if actual is None:
                object.__setattr__(self, field_name, expected)
            elif type(actual) is not int or actual != expected:
                raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "python_version": self.python_version,
            "transformers_version": self.transformers_version,
            "librosa_version": self.librosa_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "model_revision": self.model_revision,
            "lookahead_tokens": self.lookahead_tokens,
            "language": self.language,
            "device": self.device,
            "dtype": self.dtype,
            "native_sample_rate_hz": self.native_sample_rate_hz,
            "native_hop_length_samples": self.native_hop_length_samples,
            "native_n_fft_samples": self.native_n_fft_samples,
            "native_win_length_samples": self.native_win_length_samples,
            "native_first_chunk_frames": self.native_first_chunk_frames,
            "native_chunk_frames": self.native_chunk_frames,
            "native_first_window_samples": self.native_first_window_samples,
            "native_window_samples": self.native_window_samples,
            "native_stride_samples": self.native_stride_samples,
            "streaming_latency_ms": self.streaming_latency_ms,
            "wheel_lock_sha256": self.wheel_lock_sha256,
            "runtime_content_sha256": self.runtime_content_sha256,
            "runtime_file_count": self.runtime_file_count,
            "runtime_total_size_bytes": self.runtime_total_size_bytes,
            "runtime_maximum_file_bytes": self.runtime_maximum_file_bytes,
        }


@dataclass(frozen=True)
class SherpaZipformerConfig:
    """Production model/config with an explicit one-thread benchmark override."""

    package_version: str = "1.13.3"
    numpy_version: str = "2.4.6"
    source_repo_id: str = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    variant: str = "epoch-99-avg-1-chunk-16-left-128"
    language: str = "en"
    sample_rate: int = 16_000
    feature_dim: int = 80
    production_device_profile: str = "desktop_gpu_4090"
    production_num_threads: int = 4
    benchmark_profile: str = "resource-controlled-one-thread"
    num_threads: int = 1
    provider: str = "cpu"
    enable_endpoint_detection: bool = True
    decoding_method: str = "modified_beam_search"
    max_active_paths: int = 4
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 0.8
    rule3_min_utterance_length: float = 20.0

    def __post_init__(self) -> None:
        timing = (
            self.rule1_min_trailing_silence,
            self.rule2_min_trailing_silence,
            self.rule3_min_utterance_length,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in timing
        ):
            raise ManifestError()
        if (
            self.package_version != "1.13.3"
            or self.numpy_version != "2.4.6"
            or self.source_repo_id
            != "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
            or self.variant != "epoch-99-avg-1-chunk-16-left-128"
            or self.language != "en"
            or type(self.sample_rate) is not int
            or self.sample_rate != 16_000
            or type(self.feature_dim) is not int
            or self.feature_dim != 80
            or self.production_device_profile != "desktop_gpu_4090"
            or type(self.production_num_threads) is not int
            or self.production_num_threads != 4
            or self.benchmark_profile != "resource-controlled-one-thread"
            or type(self.num_threads) is not int
            or self.num_threads != 1
            or self.provider != "cpu"
            or type(self.enable_endpoint_detection) is not bool
            or not self.enable_endpoint_detection
            or self.decoding_method != "modified_beam_search"
            or type(self.max_active_paths) is not int
            or self.max_active_paths != 4
            or tuple(float(value) for value in timing) != (2.4, 0.8, 20.0)
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "package_version": self.package_version,
            "numpy_version": self.numpy_version,
            "source_repo_id": self.source_repo_id,
            "variant": self.variant,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "feature_dim": self.feature_dim,
            "production_device_profile": self.production_device_profile,
            "production_num_threads": self.production_num_threads,
            "benchmark_profile": self.benchmark_profile,
            "num_threads": self.num_threads,
            "provider": self.provider,
            "enable_endpoint_detection": self.enable_endpoint_detection,
            "decoding_method": self.decoding_method,
            "max_active_paths": self.max_active_paths,
            "rule1_min_trailing_silence": self.rule1_min_trailing_silence,
            "rule2_min_trailing_silence": self.rule2_min_trailing_silence,
            "rule3_min_utterance_length": self.rule3_min_utterance_length,
        }


@dataclass(frozen=True)
class MobileZipformerConfig:
    """Immutable CPU contract matching the shipped mobile recognizer."""

    package_version: str = "1.13.3"
    core_package_version: str = "1.13.3"
    numpy_version: str = "2.4.6"
    source_repo_id: str = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
    source_revision: str = "672fbf1b30579d6585301139bb363f42a0ad4a24"
    variant: str = "epoch-99-avg-1-chunk-16-left-128-mobile-hybrid"
    language: str = "en"
    sample_rate: int = 16_000
    feature_dim: int = 80
    num_threads: int = 1
    provider: str = "cpu"
    debug: bool = True
    decoding_method: str = "greedy_search"
    max_active_paths: int = 4
    model_type: str = "zipformer2"
    enable_endpoint_detection: bool = True
    rule1_min_trailing_silence: float = 2.4
    rule2_min_trailing_silence: float = 0.8
    rule3_min_utterance_length: float = 20.0
    native_chunk_samples: int = 1_600
    maximum_tail_padding_samples: int = 48_000

    def __post_init__(self) -> None:
        timing = (
            self.rule1_min_trailing_silence,
            self.rule2_min_trailing_silence,
            self.rule3_min_utterance_length,
        )
        if (
            any(type(value) not in {int, float} for value in timing)
            or self.package_version != "1.13.3"
            or self.core_package_version != "1.13.3"
            or self.numpy_version != "2.4.6"
            or self.source_repo_id
            != "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
            or self.source_revision != "672fbf1b30579d6585301139bb363f42a0ad4a24"
            or self.variant != "epoch-99-avg-1-chunk-16-left-128-mobile-hybrid"
            or self.language != "en"
            or type(self.sample_rate) is not int
            or self.sample_rate != 16_000
            or type(self.feature_dim) is not int
            or self.feature_dim != 80
            or type(self.num_threads) is not int
            or self.num_threads != 1
            or self.provider != "cpu"
            or type(self.debug) is not bool
            or not self.debug
            or self.decoding_method != "greedy_search"
            or type(self.max_active_paths) is not int
            or self.max_active_paths != 4
            or self.model_type != "zipformer2"
            or type(self.enable_endpoint_detection) is not bool
            or not self.enable_endpoint_detection
            or tuple(float(value) for value in timing) != (2.4, 0.8, 20.0)
            or type(self.native_chunk_samples) is not int
            or self.native_chunk_samples != 1_600
            or type(self.maximum_tail_padding_samples) is not int
            or self.maximum_tail_padding_samples != 48_000
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _MOBILE_ZIPFORMER_CONFIG_FIELDS}


@dataclass(frozen=True)
class ParakeetRealtimeEouConfig:
    """Exact model semantics plus a receipt-bound disposable NeMo runtime."""

    python_version: str = PARAKEET_REALTIME_EOU_PYTHON_VERSION
    nemo_version: str = PARAKEET_REALTIME_EOU_NEMO_VERSION
    torch_version: str = PARAKEET_REALTIME_EOU_TORCH_VERSION
    cuda_version: str = PARAKEET_REALTIME_EOU_CUDA_VERSION
    numpy_version: str = PARAKEET_REALTIME_EOU_NUMPY_VERSION
    wheel_lock_sha256: str = PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
    runtime_content_sha256: str = PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256
    runtime_file_count: int = PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT
    runtime_total_size_bytes: int = PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES
    runtime_maximum_file_bytes: int = PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES
    model_repo_id: str = "nvidia/parakeet_realtime_eou_120m-v1"
    model_revision: str = "a7e2b4629593dce0ec19f600e00e9904353fda2d"
    model_filename: str = "parakeet_realtime_eou_120m-v1.nemo"
    language: str = "en"
    device: str = "cuda:0"
    dtype: str = "float32"
    sample_rate: int = 16_000
    native_chunk_samples: int = 1_280
    maximum_tail_padding_samples: int = 48_000
    attention_context_left: int = 70
    attention_context_right: int = 1
    batch_size: int = 1
    eou_token: str = "<EOU>"
    eob_token: str = "<EOB>"
    use_amp: bool = False

    def __post_init__(self) -> None:
        if (
            self.python_version != PARAKEET_REALTIME_EOU_PYTHON_VERSION
            or self.nemo_version != PARAKEET_REALTIME_EOU_NEMO_VERSION
            or self.torch_version != PARAKEET_REALTIME_EOU_TORCH_VERSION
            or self.cuda_version != PARAKEET_REALTIME_EOU_CUDA_VERSION
            or self.numpy_version != PARAKEET_REALTIME_EOU_NUMPY_VERSION
            or self.model_repo_id != "nvidia/parakeet_realtime_eou_120m-v1"
            or self.model_revision != "a7e2b4629593dce0ec19f600e00e9904353fda2d"
            or self.model_filename != "parakeet_realtime_eou_120m-v1.nemo"
            or self.language != "en"
            or self.device != "cuda:0"
            or self.dtype != "float32"
            or type(self.sample_rate) is not int
            or self.sample_rate != 16_000
            or type(self.native_chunk_samples) is not int
            or self.native_chunk_samples != 1_280
            or type(self.maximum_tail_padding_samples) is not int
            or self.maximum_tail_padding_samples != 48_000
            or type(self.attention_context_left) is not int
            or self.attention_context_left != 70
            or type(self.attention_context_right) is not int
            or self.attention_context_right != 1
            or type(self.batch_size) is not int
            or self.batch_size != 1
            or self.eou_token != "<EOU>"
            or self.eob_token != "<EOB>"
            or type(self.use_amp) is not bool
            or self.use_amp
            or self.wheel_lock_sha256 != PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256
            or self.runtime_content_sha256
            != PARAKEET_REALTIME_EOU_RUNTIME_CONTENT_SHA256
            or self.runtime_file_count != PARAKEET_REALTIME_EOU_RUNTIME_FILE_COUNT
            or self.runtime_total_size_bytes
            != PARAKEET_REALTIME_EOU_RUNTIME_TOTAL_SIZE_BYTES
            or self.runtime_maximum_file_bytes
            != PARAKEET_REALTIME_EOU_RUNTIME_MAXIMUM_FILE_BYTES
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "python_version": self.python_version,
            "nemo_version": self.nemo_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
            "numpy_version": self.numpy_version,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "model_filename": self.model_filename,
            "language": self.language,
            "device": self.device,
            "dtype": self.dtype,
            "sample_rate": self.sample_rate,
            "native_chunk_samples": self.native_chunk_samples,
            "maximum_tail_padding_samples": self.maximum_tail_padding_samples,
            "attention_context_left": self.attention_context_left,
            "attention_context_right": self.attention_context_right,
            "batch_size": self.batch_size,
            "eou_token": self.eou_token,
            "eob_token": self.eob_token,
            "use_amp": self.use_amp,
            "wheel_lock_sha256": self.wheel_lock_sha256,
            "runtime_content_sha256": self.runtime_content_sha256,
            "runtime_file_count": self.runtime_file_count,
            "runtime_total_size_bytes": self.runtime_total_size_bytes,
            "runtime_maximum_file_bytes": self.runtime_maximum_file_bytes,
        }


@dataclass(frozen=True)
class ParakeetCppConfig:
    """Exact portable parakeet.cpp source, model, and CPU decode contract."""

    upstream_repo_id: str = "mudler/parakeet.cpp"
    upstream_commit: str = "1bfbebfaaf493866f49597cd3b7901959d395c60"
    upstream_tag: str = "v0.5.0"
    ggml_commit: str = "e705c5fed490514458bdd2eaddc43bd098fcce9b"
    c_api_version: int = 6
    bridge_abi_version: int = 1
    upstream_license: str = "MIT"
    upstream_license_sha256: str = (
        "396cb1a512310cb4fabd73118114c8cb53c7352955a37b583e165f15af64f095"
    )
    model_repo_id: str = "mudler/parakeet-cpp-gguf"
    model_revision: str = "bf0af9f425fa01809cadec671b3cb672709d13e9"
    source_model_repo_id: str = "nvidia/parakeet_realtime_eou_120m-v1"
    source_model_revision: str = "a7e2b4629593dce0ec19f600e00e9904353fda2d"
    model_filename: str = "realtime_eou_120m-v1-f16.gguf"
    model_sha256: str = PARAKEET_CPP_MODEL_RECEIPT[0]
    model_size_bytes: int = PARAKEET_CPP_MODEL_RECEIPT[1]
    model_dtype: str = "F16"
    model_license: str = "CC-BY-4.0"
    requested_device: str = "cpu"
    actual_device: str = "cpu"
    num_threads: int = 1
    ggml_native: bool = False
    ggml_cuda: bool = False
    ggml_metal: bool = False
    ggml_vulkan: bool = False
    ggml_hip: bool = False
    sample_rate: int = 16_000
    native_chunk_samples: int = 1_280
    maximum_tail_padding_samples: int = 48_000
    frame_sec: float = 0.08

    def __post_init__(self) -> None:
        exact = {
            "upstream_repo_id": "mudler/parakeet.cpp",
            "upstream_commit": "1bfbebfaaf493866f49597cd3b7901959d395c60",
            "upstream_tag": "v0.5.0",
            "ggml_commit": "e705c5fed490514458bdd2eaddc43bd098fcce9b",
            "c_api_version": 6,
            "bridge_abi_version": 1,
            "upstream_license": "MIT",
            "upstream_license_sha256": (
                "396cb1a512310cb4fabd73118114c8cb53c7352955a37b583e165f15af64f095"
            ),
            "model_repo_id": "mudler/parakeet-cpp-gguf",
            "model_revision": "bf0af9f425fa01809cadec671b3cb672709d13e9",
            "source_model_repo_id": "nvidia/parakeet_realtime_eou_120m-v1",
            "source_model_revision": (
                "a7e2b4629593dce0ec19f600e00e9904353fda2d"
            ),
            "model_filename": "realtime_eou_120m-v1-f16.gguf",
            "model_sha256": PARAKEET_CPP_MODEL_RECEIPT[0],
            "model_size_bytes": PARAKEET_CPP_MODEL_RECEIPT[1],
            "model_dtype": "F16",
            "model_license": "CC-BY-4.0",
            "requested_device": "cpu",
            "actual_device": "cpu",
            "num_threads": 1,
            "ggml_native": False,
            "ggml_cuda": False,
            "ggml_metal": False,
            "ggml_vulkan": False,
            "ggml_hip": False,
            "sample_rate": 16_000,
            "native_chunk_samples": 1_280,
            "maximum_tail_padding_samples": 48_000,
            "frame_sec": 0.08,
        }
        if any(
            type(getattr(self, name)) is not type(expected)
            or getattr(self, name) != expected
            for name, expected in exact.items()
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in _PARAKEET_CPP_CONFIG_FIELDS
        }


@dataclass(frozen=True)
class FasterWhisperEndpointConfig:
    """Exact final-only decode semantics plus two receipt-bound local trees."""

    python_version: str = "3.12.3"
    faster_whisper_version: str = "1.2.1"
    ctranslate2_version: str = "4.8.1"
    numpy_version: str = "2.4.6"
    cublas_version: str = "12.9.2.10"
    cudnn_version: str = "9.24.0.43"
    cuda_nvrtc_version: str = "12.9.86"
    language: str = "en"
    task: str = "transcribe"
    device: str = "cuda"
    device_index: int = 0
    compute_type: str = "float16"
    cpu_threads: int = 1
    num_workers: int = 1
    sample_rate: int = 16_000
    execution_mode: str = "endpoint-final-only"
    partial_hypotheses: bool = False
    tail_padding_policy: str = "pace-only-not-decoded"
    beam_size: int = 5
    patience: float = 1.0
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    vad_filter: bool = False
    condition_on_previous_text: bool = False
    without_timestamps: bool = True
    word_timestamps: bool = False
    runtime_content_sha256: str = "0" * 64
    runtime_file_count: int = 1
    runtime_total_size_bytes: int = 1
    runtime_maximum_file_bytes: int = 1
    model_content_sha256: str = "0" * 64
    model_file_count: int = 4
    model_total_size_bytes: int = 4
    model_maximum_file_bytes: int = 1

    def __post_init__(self) -> None:
        exact_numbers = (
            self.patience,
            self.temperature,
            self.compression_ratio_threshold,
            self.log_prob_threshold,
            self.no_speech_threshold,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in exact_numbers
        ):
            raise ManifestError()
        tree_integers = (
            self.runtime_file_count,
            self.runtime_total_size_bytes,
            self.runtime_maximum_file_bytes,
            self.model_file_count,
            self.model_total_size_bytes,
            self.model_maximum_file_bytes,
        )
        if (
            self.python_version != "3.12.3"
            or self.faster_whisper_version != "1.2.1"
            or self.ctranslate2_version != "4.8.1"
            or self.numpy_version != "2.4.6"
            or self.cublas_version != "12.9.2.10"
            or self.cudnn_version != "9.24.0.43"
            or self.cuda_nvrtc_version != "12.9.86"
            or self.language not in {"en", "ro"}
            or self.task != "transcribe"
            or self.device != "cuda"
            or type(self.device_index) is not int
            or self.device_index != 0
            or self.compute_type != "float16"
            or type(self.cpu_threads) is not int
            or self.cpu_threads != 1
            or type(self.num_workers) is not int
            or self.num_workers != 1
            or type(self.sample_rate) is not int
            or self.sample_rate != 16_000
            or self.execution_mode != "endpoint-final-only"
            or type(self.partial_hypotheses) is not bool
            or self.partial_hypotheses
            or self.tail_padding_policy != "pace-only-not-decoded"
            or type(self.beam_size) is not int
            or self.beam_size != 5
            or tuple(float(value) for value in exact_numbers)
            != (1.0, 0.0, 2.4, -1.0, 0.6)
            or type(self.vad_filter) is not bool
            or self.vad_filter
            or type(self.condition_on_previous_text) is not bool
            or self.condition_on_previous_text
            or type(self.without_timestamps) is not bool
            or not self.without_timestamps
            or type(self.word_timestamps) is not bool
            or self.word_timestamps
            or _SHA256_RE.fullmatch(self.runtime_content_sha256) is None
            or _SHA256_RE.fullmatch(self.model_content_sha256) is None
            or any(type(value) is not int or value <= 0 for value in tree_integers)
            or self.runtime_file_count > 100_000
            or self.runtime_total_size_bytes > 20 * 1024 * 1024 * 1024
            or self.runtime_maximum_file_bytes > 4 * 1024 * 1024 * 1024
            or self.runtime_maximum_file_bytes > self.runtime_total_size_bytes
            or not 4 <= self.model_file_count <= 4096
            or self.model_total_size_bytes > 8 * 1024 * 1024 * 1024
            or self.model_maximum_file_bytes > 4 * 1024 * 1024 * 1024
            or self.model_maximum_file_bytes > self.model_total_size_bytes
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {
            "python_version": self.python_version,
            "faster_whisper_version": self.faster_whisper_version,
            "ctranslate2_version": self.ctranslate2_version,
            "numpy_version": self.numpy_version,
            "cublas_version": self.cublas_version,
            "cudnn_version": self.cudnn_version,
            "cuda_nvrtc_version": self.cuda_nvrtc_version,
            "language": self.language,
            "task": self.task,
            "device": self.device,
            "device_index": self.device_index,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.num_workers,
            "sample_rate": self.sample_rate,
            "execution_mode": self.execution_mode,
            "partial_hypotheses": self.partial_hypotheses,
            "tail_padding_policy": self.tail_padding_policy,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "temperature": self.temperature,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "vad_filter": self.vad_filter,
            "condition_on_previous_text": self.condition_on_previous_text,
            "without_timestamps": self.without_timestamps,
            "word_timestamps": self.word_timestamps,
            "runtime_content_sha256": self.runtime_content_sha256,
            "runtime_file_count": self.runtime_file_count,
            "runtime_total_size_bytes": self.runtime_total_size_bytes,
            "runtime_maximum_file_bytes": self.runtime_maximum_file_bytes,
            "model_content_sha256": self.model_content_sha256,
            "model_file_count": self.model_file_count,
            "model_total_size_bytes": self.model_total_size_bytes,
            "model_maximum_file_bytes": self.model_maximum_file_bytes,
        }


@dataclass(frozen=True)
class KyutaiConfig:
    """One local, diagnostic-only Moshi 0.2.11 semantic-STT cell."""

    python_version: str = "3.12.3"
    moshi_version: str = "0.2.11"
    torch_version: str = "2.7.1+cu126"
    cuda_version: str = "12.6"
    julius_version: str = "0.2.8"
    wheel_lock_sha256: str = KYUTAI_RUNTIME_WHEEL_LOCK_SHA256
    runtime_content_sha256: str = KYUTAI_RUNTIME_CONTENT_SHA256
    runtime_file_count: int = KYUTAI_RUNTIME_FILE_COUNT
    runtime_total_size_bytes: int = KYUTAI_RUNTIME_TOTAL_SIZE_BYTES
    runtime_maximum_file_bytes: int = KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES
    model_repo_id: str = "kyutai/stt-1b-en_fr-candle"
    model_revision: str = "095e38f6242006a93c2541149b181988397f5c7c"
    model_config_filename: str = "config.json"
    model_weights_filename: str = "model.safetensors"
    mimi_weights_filename: str = "mimi-pytorch-e351c8d8@125.safetensors"
    tokenizer_filename: str = "tokenizer_en_fr_audio_8000.model"
    model_license: str = "CC-BY-4.0"
    language: str = "en_fr"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    input_sample_rate_hz: int = 16_000
    mimi_sample_rate_hz: int = 24_000
    input_chunk_samples: int = 1_280
    mimi_frame_samples: int = 1_920
    resampling_mode: str = "whole-buffer-noncausal"
    initial_frame_policy: str = "duplicate-first-frame-prime"
    initial_frame_prime_steps: int = 1
    terminal_tail_samples: int = 16_000
    partial_interval_ms: int = 160
    maximum_source_samples: int = 2_097_152
    batch_size: int = 1
    temperature: float = 0.0
    text_temperature: float = 0.0
    use_sampling: bool = False
    text_padding_token_id: int = 3
    end_of_padding_token_id: int = 0
    audio_delay_seconds: float = 0.5
    audio_silence_prefix_seconds: float = 0.0
    semantic_head_count: int = 4
    semantic_head_dim: int = 6
    semantic_head_policy: str = "diagnostic-finite-only"
    endpoint_owner: str = "none"
    early_stop: bool = False
    num_threads: int = 1
    maximum_vram_fraction: float = 0.5
    minimum_free_vram_mb: int = 8_192
    minimum_host_available_bytes: int = 12 * 1024**3
    torch_compile: bool = False
    no_torch_compile_env: str = "1"
    cuda_graph: bool = False
    no_cuda_graph_env: str = "1"
    local_files_only: bool = True

    def __post_init__(self) -> None:
        exact = {
            "python_version": "3.12.3",
            "moshi_version": "0.2.11",
            "torch_version": "2.7.1+cu126",
            "cuda_version": "12.6",
            "julius_version": "0.2.8",
            "model_repo_id": "kyutai/stt-1b-en_fr-candle",
            "model_revision": "095e38f6242006a93c2541149b181988397f5c7c",
            "model_config_filename": "config.json",
            "model_weights_filename": "model.safetensors",
            "mimi_weights_filename": "mimi-pytorch-e351c8d8@125.safetensors",
            "tokenizer_filename": "tokenizer_en_fr_audio_8000.model",
            "model_license": "CC-BY-4.0",
            "language": "en_fr",
            "device": "cuda:0",
            "dtype": "bfloat16",
            "input_sample_rate_hz": 16_000,
            "mimi_sample_rate_hz": 24_000,
            "input_chunk_samples": 1_280,
            "mimi_frame_samples": 1_920,
            "resampling_mode": "whole-buffer-noncausal",
            "initial_frame_policy": "duplicate-first-frame-prime",
            "initial_frame_prime_steps": 1,
            "terminal_tail_samples": 16_000,
            "partial_interval_ms": 160,
            "maximum_source_samples": 2_097_152,
            "batch_size": 1,
            "temperature": 0.0,
            "text_temperature": 0.0,
            "use_sampling": False,
            "text_padding_token_id": 3,
            "end_of_padding_token_id": 0,
            "audio_delay_seconds": 0.5,
            "audio_silence_prefix_seconds": 0.0,
            "semantic_head_count": 4,
            "semantic_head_dim": 6,
            "semantic_head_policy": "diagnostic-finite-only",
            "endpoint_owner": "none",
            "early_stop": False,
            "num_threads": 1,
            "maximum_vram_fraction": 0.5,
            "minimum_free_vram_mb": 8_192,
            "minimum_host_available_bytes": 12 * 1024**3,
            "torch_compile": False,
            "no_torch_compile_env": "1",
            "cuda_graph": False,
            "no_cuda_graph_env": "1",
            "local_files_only": True,
        }
        receipt_integers = (
            self.runtime_file_count,
            self.runtime_total_size_bytes,
            self.runtime_maximum_file_bytes,
        )
        if (
            any(
                type(getattr(self, name)) is not type(expected)
                or getattr(self, name) != expected
                for name, expected in exact.items()
            )
            or math.copysign(1.0, self.temperature) != 1.0
            or math.copysign(1.0, self.text_temperature) != 1.0
            or math.copysign(1.0, self.audio_silence_prefix_seconds) != 1.0
            or self.wheel_lock_sha256 != KYUTAI_RUNTIME_WHEEL_LOCK_SHA256
            or self.runtime_content_sha256 != KYUTAI_RUNTIME_CONTENT_SHA256
            or self.runtime_file_count != KYUTAI_RUNTIME_FILE_COUNT
            or self.runtime_total_size_bytes != KYUTAI_RUNTIME_TOTAL_SIZE_BYTES
            or self.runtime_maximum_file_bytes
            != KYUTAI_RUNTIME_MAXIMUM_FILE_BYTES
            or any(type(value) is not int for value in receipt_integers)
        ):
            raise ManifestError()

    def as_dict(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in _KYUTAI_CONFIG_FIELDS}


@dataclass(frozen=True)
class WorkerManifest:
    path: Path
    digest: str
    schema_version: int
    model_id: str
    adapter: str
    python: BoundFile
    worker: BoundFile
    artifacts: tuple[BoundArtifact, ...]
    limits: WorkerLimits
    adapter_config: (
        MoonshineConfig
        | MoonshineExternalEndpointConfig
        | NemotronConfig
        | SherpaZipformerConfig
        | MobileZipformerConfig
        | ParakeetRealtimeEouConfig
        | ParakeetCppConfig
        | FasterWhisperEndpointConfig
        | KyutaiConfig
        | None
    ) = None
    control_files: tuple[BoundFile, ...] = ()

    @property
    def artifact_by_name(self) -> Mapping[str, BoundArtifact]:
        return {artifact.name: artifact for artifact in self.artifacts}


def _bad() -> object:
    raise ManifestError()


def _strict_json(raw: bytes) -> object:
    def pairs(values: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in values:
            if key in result:
                raise ManifestError()
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: _bad())
    except (UnicodeError, ValueError, OverflowError, ManifestError):
        raise ManifestError() from None


def _safe_id(value: object) -> str:
    if not isinstance(value, str) or _SAFE_ID_RE.fullmatch(value) is None:
        raise ManifestError()
    return value


def _sha256(value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ManifestError()
    return value


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ManifestError()
    return value


def _bounded_seconds(value: object, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ManifestError()
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ManifestError() from None
    if not math.isfinite(number) or number <= 0.0 or number > maximum:
        raise ManifestError()
    return number


def artifact_maximum_bytes(adapter: str, artifact_name: str) -> int:
    """Return the closed per-artifact read budget for one adapter."""

    if adapter == "fake-json-v1" and artifact_name == "fake-script":
        return MAX_FAKE_ARTIFACT_BYTES
    if (
        adapter in {MOONSHINE_ADAPTER, MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER}
        and artifact_name in MOONSHINE_ARTIFACT_NAMES
    ):
        return _MOONSHINE_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_MOONSHINE_ARTIFACT_BYTES,
        )
    if adapter == NEMOTRON_ADAPTER and artifact_name in NEMOTRON_ARTIFACT_NAMES:
        return _NEMOTRON_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_NEMOTRON_ARTIFACT_BYTES,
        )
    if (
        adapter == SHERPA_ZIPFORMER_ADAPTER
        and artifact_name in SHERPA_ZIPFORMER_ARTIFACT_NAMES
    ):
        return MAX_SHERPA_ZIPFORMER_ARTIFACT_BYTES
    if (
        adapter == MOBILE_ZIPFORMER_ADAPTER
        and artifact_name in MOBILE_ZIPFORMER_ARTIFACT_NAMES
    ):
        return MAX_MOBILE_ZIPFORMER_ARTIFACT_BYTES
    if (
        adapter == PARAKEET_REALTIME_EOU_ADAPTER
        and artifact_name in PARAKEET_REALTIME_EOU_ARTIFACT_NAMES
    ):
        return _PARAKEET_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_PARAKEET_ARTIFACT_BYTES,
        )
    if (
        adapter == FASTER_WHISPER_ENDPOINT_ADAPTER
        and artifact_name in FASTER_WHISPER_ARTIFACT_NAMES
    ):
        return _FASTER_WHISPER_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_FASTER_WHISPER_CONTROL_ARTIFACT_BYTES,
        )
    if adapter == PARAKEET_CPP_ADAPTER and artifact_name in PARAKEET_CPP_ARTIFACT_NAMES:
        if artifact_name in _PARAKEET_CPP_RECEIPT_NAMES:
            return MAX_PARAKEET_CPP_CONTROL_ARTIFACT_BYTES
        if artifact_name in _PARAKEET_CPP_LIBRARY_NAMES:
            return MAX_PARAKEET_CPP_LIBRARY_BYTES
        return MAX_PARAKEET_CPP_MODEL_BYTES
    if adapter == KYUTAI_ADAPTER and artifact_name in KYUTAI_ARTIFACT_NAMES:
        return _KYUTAI_SMALL_ARTIFACTS.get(
            artifact_name,
            MAX_KYUTAI_ARTIFACT_BYTES,
        )
    raise ManifestError()


def _bound_file(
    value: object,
    *,
    expected_fields: set[str],
    allow_symlink: bool,
    maximum_bytes: int,
    retain_lexical_path: bool = False,
) -> BoundFile:
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ManifestError()
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
        raise ManifestError()
    candidate = Path(raw_path)
    if not candidate.is_absolute() or (candidate.is_symlink() and not allow_symlink):
        raise ManifestError()
    size = _positive_int(value.get("size_bytes"))
    expected_hash = _sha256(value.get("sha256"))
    try:
        digest = hash_regular_bounded(
            candidate,
            maximum_bytes=maximum_bytes,
            expected_bytes=size,
            allow_final_symlink=allow_symlink,
        )
    except BoundedReadError:
        raise ManifestError() from None
    if digest.sha256 != expected_hash:
        raise ManifestError()
    return BoundFile(
        path=candidate if retain_lexical_path else digest.path,
        sha256=expected_hash,
        size_bytes=size,
    )


def _artifact(value: object, *, adapter: str) -> BoundArtifact:
    if not isinstance(value, dict):
        raise ManifestError()
    name = _safe_id(value.get("name"))
    bound = _bound_file(
        value,
        expected_fields=_ARTIFACT_FIELDS,
        allow_symlink=False,
        maximum_bytes=artifact_maximum_bytes(adapter, name),
    )
    return BoundArtifact(
        path=bound.path,
        sha256=bound.sha256,
        size_bytes=bound.size_bytes,
        name=name,
    )


def _moonshine_config(value: object) -> MoonshineConfig:
    if not isinstance(value, dict) or set(value) != _MOONSHINE_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return MoonshineConfig(
            package_version=value.get("package_version"),  # type: ignore[arg-type]
            api_version=value.get("api_version"),  # type: ignore[arg-type]
            model_arch=value.get("model_arch"),  # type: ignore[arg-type]
            provider=value.get("provider"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _moonshine_external_endpoint_config(
    value: object,
) -> MoonshineExternalEndpointConfig:
    if (
        not isinstance(value, dict)
        or set(value) != _MOONSHINE_EXTERNAL_ENDPOINT_CONFIG_FIELDS
    ):
        raise ManifestError()
    try:
        return MoonshineExternalEndpointConfig(
            package_version=value.get("package_version"),  # type: ignore[arg-type]
            api_version=value.get("api_version"),  # type: ignore[arg-type]
            model_arch=value.get("model_arch"),  # type: ignore[arg-type]
            provider=value.get("provider"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            segmentation_mode=value.get(  # type: ignore[arg-type]
                "segmentation_mode"
            ),
            endpoint_owner=value.get("endpoint_owner"),  # type: ignore[arg-type]
            vad_threshold=value.get("vad_threshold"),  # type: ignore[arg-type]
            vad_max_segment_duration_sec=value.get(  # type: ignore[arg-type]
                "vad_max_segment_duration_sec"
            ),
            vad_hop_size_samples=value.get(  # type: ignore[arg-type]
                "vad_hop_size_samples"
            ),
            streaming_chunk_samples=value.get(  # type: ignore[arg-type]
                "streaming_chunk_samples"
            ),
            online_partial_interval_ms=value.get(  # type: ignore[arg-type]
                "online_partial_interval_ms"
            ),
            authoritative_alignment_samples=value.get(  # type: ignore[arg-type]
                "authoritative_alignment_samples"
            ),
            tail_alignment_policy=value.get(  # type: ignore[arg-type]
                "tail_alignment_policy"
            ),
            finalization_policy=value.get(  # type: ignore[arg-type]
                "finalization_policy"
            ),
            maximum_source_samples=value.get(  # type: ignore[arg-type]
                "maximum_source_samples"
            ),
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _nemotron_config(value: object) -> NemotronConfig:
    if not isinstance(value, dict) or set(value) != _NEMOTRON_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return NemotronConfig(
            python_version=value.get("python_version"),  # type: ignore[arg-type]
            transformers_version=value.get("transformers_version"),  # type: ignore[arg-type]
            librosa_version=value.get("librosa_version"),  # type: ignore[arg-type]
            torch_version=value.get("torch_version"),  # type: ignore[arg-type]
            cuda_version=value.get("cuda_version"),  # type: ignore[arg-type]
            model_revision=value.get("model_revision"),  # type: ignore[arg-type]
            lookahead_tokens=value.get("lookahead_tokens"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            device=value.get("device"),  # type: ignore[arg-type]
            dtype=value.get("dtype"),  # type: ignore[arg-type]
            native_sample_rate_hz=value.get("native_sample_rate_hz"),  # type: ignore[arg-type]
            native_hop_length_samples=value.get("native_hop_length_samples"),  # type: ignore[arg-type]
            native_n_fft_samples=value.get("native_n_fft_samples"),  # type: ignore[arg-type]
            native_win_length_samples=value.get("native_win_length_samples"),  # type: ignore[arg-type]
            native_first_chunk_frames=value.get("native_first_chunk_frames"),  # type: ignore[arg-type]
            native_chunk_frames=value.get("native_chunk_frames"),  # type: ignore[arg-type]
            native_first_window_samples=value.get("native_first_window_samples"),  # type: ignore[arg-type]
            native_window_samples=value.get("native_window_samples"),  # type: ignore[arg-type]
            native_stride_samples=value.get("native_stride_samples"),  # type: ignore[arg-type]
            streaming_latency_ms=value.get("streaming_latency_ms"),  # type: ignore[arg-type]
            wheel_lock_sha256=value.get("wheel_lock_sha256"),  # type: ignore[arg-type]
            runtime_content_sha256=value.get("runtime_content_sha256"),  # type: ignore[arg-type]
            runtime_file_count=value.get("runtime_file_count"),  # type: ignore[arg-type]
            runtime_total_size_bytes=value.get("runtime_total_size_bytes"),  # type: ignore[arg-type]
            runtime_maximum_file_bytes=value.get("runtime_maximum_file_bytes"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _sherpa_zipformer_config(value: object) -> SherpaZipformerConfig:
    if not isinstance(value, dict) or set(value) != _SHERPA_ZIPFORMER_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return SherpaZipformerConfig(
            package_version=value.get("package_version"),  # type: ignore[arg-type]
            numpy_version=value.get("numpy_version"),  # type: ignore[arg-type]
            source_repo_id=value.get("source_repo_id"),  # type: ignore[arg-type]
            variant=value.get("variant"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            sample_rate=value.get("sample_rate"),  # type: ignore[arg-type]
            feature_dim=value.get("feature_dim"),  # type: ignore[arg-type]
            production_device_profile=value.get(  # type: ignore[arg-type]
                "production_device_profile"
            ),
            production_num_threads=value.get(  # type: ignore[arg-type]
                "production_num_threads"
            ),
            benchmark_profile=value.get("benchmark_profile"),  # type: ignore[arg-type]
            num_threads=value.get("num_threads"),  # type: ignore[arg-type]
            provider=value.get("provider"),  # type: ignore[arg-type]
            enable_endpoint_detection=value.get(  # type: ignore[arg-type]
                "enable_endpoint_detection"
            ),
            decoding_method=value.get("decoding_method"),  # type: ignore[arg-type]
            max_active_paths=value.get("max_active_paths"),  # type: ignore[arg-type]
            rule1_min_trailing_silence=value.get(  # type: ignore[arg-type]
                "rule1_min_trailing_silence"
            ),
            rule2_min_trailing_silence=value.get(  # type: ignore[arg-type]
                "rule2_min_trailing_silence"
            ),
            rule3_min_utterance_length=value.get(  # type: ignore[arg-type]
                "rule3_min_utterance_length"
            ),
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _mobile_zipformer_config(value: object) -> MobileZipformerConfig:
    if not isinstance(value, dict) or set(value) != _MOBILE_ZIPFORMER_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return MobileZipformerConfig(
            **{name: value.get(name) for name in _MOBILE_ZIPFORMER_CONFIG_FIELDS}
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _parakeet_config(value: object) -> ParakeetRealtimeEouConfig:
    if not isinstance(value, dict) or set(value) != _PARAKEET_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return ParakeetRealtimeEouConfig(
            python_version=value.get("python_version"),  # type: ignore[arg-type]
            nemo_version=value.get("nemo_version"),  # type: ignore[arg-type]
            torch_version=value.get("torch_version"),  # type: ignore[arg-type]
            cuda_version=value.get("cuda_version"),  # type: ignore[arg-type]
            numpy_version=value.get("numpy_version"),  # type: ignore[arg-type]
            wheel_lock_sha256=value.get("wheel_lock_sha256"),  # type: ignore[arg-type]
            runtime_content_sha256=value.get(  # type: ignore[arg-type]
                "runtime_content_sha256"
            ),
            runtime_file_count=value.get("runtime_file_count"),  # type: ignore[arg-type]
            runtime_total_size_bytes=value.get(  # type: ignore[arg-type]
                "runtime_total_size_bytes"
            ),
            runtime_maximum_file_bytes=value.get(  # type: ignore[arg-type]
                "runtime_maximum_file_bytes"
            ),
            model_repo_id=value.get("model_repo_id"),  # type: ignore[arg-type]
            model_revision=value.get("model_revision"),  # type: ignore[arg-type]
            model_filename=value.get("model_filename"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            device=value.get("device"),  # type: ignore[arg-type]
            dtype=value.get("dtype"),  # type: ignore[arg-type]
            sample_rate=value.get("sample_rate"),  # type: ignore[arg-type]
            native_chunk_samples=value.get(  # type: ignore[arg-type]
                "native_chunk_samples"
            ),
            maximum_tail_padding_samples=value.get(  # type: ignore[arg-type]
                "maximum_tail_padding_samples"
            ),
            attention_context_left=value.get(  # type: ignore[arg-type]
                "attention_context_left"
            ),
            attention_context_right=value.get(  # type: ignore[arg-type]
                "attention_context_right"
            ),
            batch_size=value.get("batch_size"),  # type: ignore[arg-type]
            eou_token=value.get("eou_token"),  # type: ignore[arg-type]
            eob_token=value.get("eob_token"),  # type: ignore[arg-type]
            use_amp=value.get("use_amp"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _parakeet_cpp_config(value: object) -> ParakeetCppConfig:
    if not isinstance(value, dict) or set(value) != _PARAKEET_CPP_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return ParakeetCppConfig(
            upstream_repo_id=value.get("upstream_repo_id"),  # type: ignore[arg-type]
            upstream_commit=value.get("upstream_commit"),  # type: ignore[arg-type]
            upstream_tag=value.get("upstream_tag"),  # type: ignore[arg-type]
            ggml_commit=value.get("ggml_commit"),  # type: ignore[arg-type]
            c_api_version=value.get("c_api_version"),  # type: ignore[arg-type]
            bridge_abi_version=value.get("bridge_abi_version"),  # type: ignore[arg-type]
            upstream_license=value.get("upstream_license"),  # type: ignore[arg-type]
            upstream_license_sha256=value.get(  # type: ignore[arg-type]
                "upstream_license_sha256"
            ),
            model_repo_id=value.get("model_repo_id"),  # type: ignore[arg-type]
            model_revision=value.get("model_revision"),  # type: ignore[arg-type]
            source_model_repo_id=value.get(  # type: ignore[arg-type]
                "source_model_repo_id"
            ),
            source_model_revision=value.get(  # type: ignore[arg-type]
                "source_model_revision"
            ),
            model_filename=value.get("model_filename"),  # type: ignore[arg-type]
            model_sha256=value.get("model_sha256"),  # type: ignore[arg-type]
            model_size_bytes=value.get("model_size_bytes"),  # type: ignore[arg-type]
            model_dtype=value.get("model_dtype"),  # type: ignore[arg-type]
            model_license=value.get("model_license"),  # type: ignore[arg-type]
            requested_device=value.get("requested_device"),  # type: ignore[arg-type]
            actual_device=value.get("actual_device"),  # type: ignore[arg-type]
            num_threads=value.get("num_threads"),  # type: ignore[arg-type]
            ggml_native=value.get("ggml_native"),  # type: ignore[arg-type]
            ggml_cuda=value.get("ggml_cuda"),  # type: ignore[arg-type]
            ggml_metal=value.get("ggml_metal"),  # type: ignore[arg-type]
            ggml_vulkan=value.get("ggml_vulkan"),  # type: ignore[arg-type]
            ggml_hip=value.get("ggml_hip"),  # type: ignore[arg-type]
            sample_rate=value.get("sample_rate"),  # type: ignore[arg-type]
            native_chunk_samples=value.get(  # type: ignore[arg-type]
                "native_chunk_samples"
            ),
            maximum_tail_padding_samples=value.get(  # type: ignore[arg-type]
                "maximum_tail_padding_samples"
            ),
            frame_sec=value.get("frame_sec"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _faster_whisper_config(value: object) -> FasterWhisperEndpointConfig:
    if not isinstance(value, dict) or set(value) != _FASTER_WHISPER_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return FasterWhisperEndpointConfig(
            python_version=value.get("python_version"),  # type: ignore[arg-type]
            faster_whisper_version=value.get("faster_whisper_version"),  # type: ignore[arg-type]
            ctranslate2_version=value.get("ctranslate2_version"),  # type: ignore[arg-type]
            numpy_version=value.get("numpy_version"),  # type: ignore[arg-type]
            cublas_version=value.get("cublas_version"),  # type: ignore[arg-type]
            cudnn_version=value.get("cudnn_version"),  # type: ignore[arg-type]
            cuda_nvrtc_version=value.get("cuda_nvrtc_version"),  # type: ignore[arg-type]
            language=value.get("language"),  # type: ignore[arg-type]
            task=value.get("task"),  # type: ignore[arg-type]
            device=value.get("device"),  # type: ignore[arg-type]
            device_index=value.get("device_index"),  # type: ignore[arg-type]
            compute_type=value.get("compute_type"),  # type: ignore[arg-type]
            cpu_threads=value.get("cpu_threads"),  # type: ignore[arg-type]
            num_workers=value.get("num_workers"),  # type: ignore[arg-type]
            sample_rate=value.get("sample_rate"),  # type: ignore[arg-type]
            execution_mode=value.get("execution_mode"),  # type: ignore[arg-type]
            partial_hypotheses=value.get("partial_hypotheses"),  # type: ignore[arg-type]
            tail_padding_policy=value.get("tail_padding_policy"),  # type: ignore[arg-type]
            beam_size=value.get("beam_size"),  # type: ignore[arg-type]
            patience=value.get("patience"),  # type: ignore[arg-type]
            temperature=value.get("temperature"),  # type: ignore[arg-type]
            compression_ratio_threshold=value.get("compression_ratio_threshold"),  # type: ignore[arg-type]
            log_prob_threshold=value.get("log_prob_threshold"),  # type: ignore[arg-type]
            no_speech_threshold=value.get("no_speech_threshold"),  # type: ignore[arg-type]
            vad_filter=value.get("vad_filter"),  # type: ignore[arg-type]
            condition_on_previous_text=value.get("condition_on_previous_text"),  # type: ignore[arg-type]
            without_timestamps=value.get("without_timestamps"),  # type: ignore[arg-type]
            word_timestamps=value.get("word_timestamps"),  # type: ignore[arg-type]
            runtime_content_sha256=value.get("runtime_content_sha256"),  # type: ignore[arg-type]
            runtime_file_count=value.get("runtime_file_count"),  # type: ignore[arg-type]
            runtime_total_size_bytes=value.get("runtime_total_size_bytes"),  # type: ignore[arg-type]
            runtime_maximum_file_bytes=value.get("runtime_maximum_file_bytes"),  # type: ignore[arg-type]
            model_content_sha256=value.get("model_content_sha256"),  # type: ignore[arg-type]
            model_file_count=value.get("model_file_count"),  # type: ignore[arg-type]
            model_total_size_bytes=value.get("model_total_size_bytes"),  # type: ignore[arg-type]
            model_maximum_file_bytes=value.get("model_maximum_file_bytes"),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _kyutai_config(value: object) -> KyutaiConfig:
    if not isinstance(value, dict) or set(value) != _KYUTAI_CONFIG_FIELDS:
        raise ManifestError()
    try:
        return KyutaiConfig(**value)  # type: ignore[arg-type]
    except (TypeError, ValueError, ManifestError):
        raise ManifestError() from None


def _validate_venv_layout(
    python: BoundFile,
    marker: BoundArtifact,
) -> None:
    lexical_python = Path(os.path.abspath(python.path))
    venv_root = lexical_python.parent.parent
    if (
        lexical_python != python.path
        or lexical_python.parent.name != "bin"
        or marker.path.parent != venv_root
        or venv_root.resolve(strict=True) != venv_root
    ):
        raise ManifestError()
    try:
        verify_isolated_venv_marker(
            marker.path,
            expected_digest=marker.sha256,
            expected_bytes=marker.size_bytes,
        )
    except RuntimeTreeReceiptError:
        raise ManifestError() from None


def _validate_moonshine_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
    config: MoonshineConfig,
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    model_receipts = _MOONSHINE_MODEL_RECEIPTS.get(config.model_arch)
    if (
        tuple(by_name) != MOONSHINE_ARTIFACT_NAMES
        or model_receipts is None
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_MOONSHINE_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _MOONSHINE_ARTIFACT_BASENAMES.items()
        )
        or (
            by_name["release-wheel"].sha256,
            by_name["release-wheel"].size_bytes,
        )
        != _MOONSHINE_RELEASE_RECEIPT
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in model_receipts.items()
        )
    ):
        raise ManifestError()

    model_names = tuple(
        name for name in MOONSHINE_ARTIFACT_NAMES if name.startswith("model-")
    )
    model_parents = {by_name[name].path.parent for name in model_names}
    if len(model_parents) != 1:
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def _validate_nemotron_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    if (
        tuple(by_name) != NEMOTRON_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_NEMOTRON_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _NEMOTRON_ARTIFACT_BASENAMES.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in _NEMOTRON_MODEL_RECEIPTS.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in _NEMOTRON_CONTROL_RECEIPTS.items()
        )
    ):
        raise ManifestError()
    model_names = tuple(
        name for name in NEMOTRON_ARTIFACT_NAMES if name.startswith("model-")
    )
    model_parents = {by_name[name].path.parent for name in model_names}
    if len(model_parents) != 1:
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def _validate_sherpa_zipformer_layout(
    artifacts: tuple[BoundArtifact, ...],
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    if (
        tuple(by_name) != SHERPA_ZIPFORMER_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_SHERPA_ZIPFORMER_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _SHERPA_ZIPFORMER_ARTIFACT_BASENAMES.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in _SHERPA_ZIPFORMER_MODEL_RECEIPTS.items()
        )
        or len({artifact.path.parent for artifact in artifacts}) != 1
    ):
        raise ManifestError()


def _validate_mobile_zipformer_layout(
    artifacts: tuple[BoundArtifact, ...],
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    if (
        tuple(by_name) != MOBILE_ZIPFORMER_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        != MOBILE_ZIPFORMER_TOTAL_SIZE_BYTES
        or MOBILE_ZIPFORMER_TOTAL_SIZE_BYTES > MAX_MOBILE_ZIPFORMER_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _MOBILE_ZIPFORMER_ARTIFACT_BASENAMES.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in _MOBILE_ZIPFORMER_MODEL_RECEIPTS.items()
        )
        or len({artifact.path.parent for artifact in artifacts}) != 1
    ):
        raise ManifestError()


def _mobile_zipformer_control_files(
    value: Mapping[str, object],
    *,
    python: BoundFile,
    worker: BoundFile,
    config: MobileZipformerConfig,
) -> tuple[BoundFile, ...]:
    if (
        value.get("kind") != "mobile-zipformer-provision-v1"
        or value.get("model_id")
        != "sherpa-onnx-streaming-zipformer-en-2023-06-26-mobile-hybrid-v1"
        or value.get("artifact_set_sha256") != MOBILE_ZIPFORMER_ARTIFACT_SET_SHA256
        or type(value.get("total_size_bytes")) is not int
        or value.get("total_size_bytes") != MOBILE_ZIPFORMER_TOTAL_SIZE_BYTES
    ):
        raise ManifestError()
    source = value.get("source")
    if not isinstance(source, dict) or set(source) != _MOBILE_ZIPFORMER_SOURCE_FIELDS:
        raise ManifestError()
    if (
        source.get("repo_id") != config.source_repo_id
        or source.get("revision") != config.source_revision
    ):
        raise ManifestError()
    if (
        _sha256(source.get("lock_recipe_sha256"))
        != MOBILE_ZIPFORMER_SOURCE_LOCK_RECIPE_SHA256
    ):
        raise ManifestError()
    source_lock = _bound_file(
        source.get("lock"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=False,
        maximum_bytes=MAX_MOBILE_ZIPFORMER_SOURCE_LOCK_BYTES,
    )
    if (
        source_lock.path.name != "source-lock.json"
        or source_lock.sha256 != MOBILE_ZIPFORMER_SOURCE_LOCK_SHA256
        or source_lock.size_bytes != MOBILE_ZIPFORMER_SOURCE_LOCK_SIZE_BYTES
    ):
        raise ManifestError()

    runtime = value.get("runtime")
    if (
        not isinstance(runtime, dict)
        or set(runtime) != _MOBILE_ZIPFORMER_RUNTIME_FIELDS
        or runtime.get("python") != value.get("python")
        or runtime.get("worker") != value.get("worker")
        or runtime.get("metadata_only_verified") is not True
        or runtime.get("packages_imported") is not False
        or runtime.get("model_loaded") is not False
    ):
        raise ManifestError()
    distributions = runtime.get("distributions")
    if (
        not isinstance(distributions, dict)
        or set(distributions) != _MOBILE_ZIPFORMER_DISTRIBUTION_FIELDS
        or distributions
        != {
            "sherpa-onnx": config.package_version,
            "sherpa-onnx-core": config.core_package_version,
            "numpy": config.numpy_version,
        }
    ):
        raise ManifestError()
    evidence_scope = value.get("evidence_scope")
    if (
        not isinstance(evidence_scope, dict)
        or set(evidence_scope) != _MOBILE_ZIPFORMER_EVIDENCE_FIELDS
        or any(type(item) is not bool or item for item in evidence_scope.values())
    ):
        raise ManifestError()
    if runtime["python"] != {
        "path": str(python.path),
        "sha256": python.sha256,
        "size_bytes": python.size_bytes,
    } or runtime["worker"] != {
        "path": str(worker.path),
        "sha256": worker.sha256,
        "size_bytes": worker.size_bytes,
    }:
        raise ManifestError()
    return (source_lock,)


def _validate_parakeet_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
    config: ParakeetRealtimeEouConfig,
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    model = by_name.get("model-nemo")
    wheel_lock = by_name.get("runtime-wheel-lock")
    if (
        tuple(by_name) != PARAKEET_REALTIME_EOU_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_PARAKEET_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _PARAKEET_ARTIFACT_BASENAMES.items()
        )
        or model is None
        or (model.sha256, model.size_bytes) != PARAKEET_REALTIME_EOU_MODEL_RECEIPT
        or wheel_lock is None
        or (wheel_lock.sha256, wheel_lock.size_bytes)
        != (
            PARAKEET_REALTIME_EOU_WHEEL_LOCK_SHA256,
            PARAKEET_REALTIME_EOU_WHEEL_LOCK_SIZE_BYTES,
        )
    ):
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def _exact_receipt_payload(
    artifact: BoundArtifact,
    expected: Mapping[str, object],
) -> dict[str, object]:
    try:
        snapshot = read_regular_bounded(
            artifact.path,
            maximum_bytes=MAX_PARAKEET_CPP_CONTROL_ARTIFACT_BYTES,
            expected_bytes=artifact.size_bytes,
        )
    except BoundedReadError:
        raise ManifestError() from None
    value = _strict_json(snapshot.data)
    if (
        snapshot.path != artifact.path
        or hashlib.sha256(snapshot.data).hexdigest() != artifact.sha256
        or not isinstance(value, dict)
        or set(value) != set(expected)
        or not _exact_json_value(value, expected)
    ):
        raise ManifestError()
    return value


def _exact_json_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        return set(actual) == set(expected) and all(
            _exact_json_value(actual[name], expected_value)
            for name, expected_value in expected.items()
        )
    if isinstance(expected, list):
        assert isinstance(actual, list)
        return len(actual) == len(expected) and all(
            _exact_json_value(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return actual == expected


def _manifest_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _elf_integer(data: bytes, offset: int, width: int) -> int:
    end = offset + width
    if offset < 0 or width <= 0 or end > len(data):
        raise ManifestError()
    return int.from_bytes(data[offset:end], "little")


def inspect_elf(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    maximum_bytes: int,
) -> ElfInspection:
    """Inspect one bounded x86-64 shared ELF without loading it."""

    try:
        snapshot = read_regular_bounded(
            path,
            maximum_bytes=maximum_bytes,
            expected_bytes=expected_size_bytes,
        )
    except BoundedReadError:
        raise ManifestError() from None
    data = snapshot.data
    if (
        snapshot.path != path
        or hashlib.sha256(data).hexdigest() != expected_sha256
        or len(data) < 64
        or data[:4] != b"\x7fELF"
        or data[4:7] != b"\x02\x01\x01"
        or _elf_integer(data, 16, 2) != 3
        or _elf_integer(data, 18, 2) != 62
        or _elf_integer(data, 20, 4) != 1
        or _elf_integer(data, 52, 2) != 64
    ):
        raise ManifestError()
    program_offset = _elf_integer(data, 32, 8)
    program_size = _elf_integer(data, 54, 2)
    program_count = _elf_integer(data, 56, 2)
    if (
        program_size != 56
        or not 1 <= program_count <= 64
        or program_offset < 64
        or program_offset + (program_size * program_count) > len(data)
    ):
        raise ManifestError()

    load_segments: list[tuple[int, int, int]] = []
    dynamic_segments: list[tuple[int, int]] = []
    relro = False
    stack_flags: list[int] = []
    for index in range(program_count):
        entry = program_offset + (index * program_size)
        segment_type = _elf_integer(data, entry, 4)
        segment_flags = _elf_integer(data, entry + 4, 4)
        file_offset = _elf_integer(data, entry + 8, 8)
        virtual_address = _elf_integer(data, entry + 16, 8)
        file_size = _elf_integer(data, entry + 32, 8)
        if file_offset + file_size > len(data):
            raise ManifestError()
        if segment_type == 1:
            load_segments.append((file_offset, virtual_address, file_size))
        elif segment_type == 2:
            dynamic_segments.append((file_offset, file_size))
        elif segment_type == 0x6474E551:
            stack_flags.append(segment_flags)
        elif segment_type == 0x6474E552:
            relro = True
    if len(dynamic_segments) != 1 or not load_segments:
        raise ManifestError()
    dynamic_offset, dynamic_size = dynamic_segments[0]
    if dynamic_size < 16 or dynamic_size % 16:
        raise ManifestError()

    needed_offsets: list[int] = []
    string_table_addresses: list[int] = []
    string_table_sizes: list[int] = []
    runpath_offsets: list[int] = []
    rpath_offsets: list[int] = []
    bind_now = False
    terminated = False
    for offset in range(dynamic_offset, dynamic_offset + dynamic_size, 16):
        tag = _elf_integer(data, offset, 8)
        value = _elf_integer(data, offset + 8, 8)
        if tag == 0:
            terminated = True
            break
        if tag == 1:
            needed_offsets.append(value)
        elif tag == 5:
            string_table_addresses.append(value)
        elif tag == 10:
            string_table_sizes.append(value)
        elif tag == 15:
            rpath_offsets.append(value)
        elif tag == 24:
            bind_now = True
        elif tag == 29:
            runpath_offsets.append(value)
        elif tag == 30 and value & 0x8:
            bind_now = True
        elif tag == 0x6FFFFFFB and value & 0x1:
            bind_now = True
    if (
        not terminated
        or not needed_offsets
        or len(string_table_addresses) != 1
        or len(string_table_sizes) != 1
        or string_table_sizes[0] <= 1
        or string_table_sizes[0] > maximum_bytes
        or rpath_offsets
        or len(runpath_offsets) > 1
    ):
        raise ManifestError()

    string_address = string_table_addresses[0]
    string_size = string_table_sizes[0]
    table_offsets = []
    for file_offset, virtual_address, file_size in load_segments:
        if virtual_address <= string_address:
            relative = string_address - virtual_address
            if relative + string_size <= file_size:
                table_offsets.append(file_offset + relative)
    if len(table_offsets) != 1:
        raise ManifestError()
    table_offset = table_offsets[0]

    def string_at(relative: int) -> str:
        if relative >= string_size:
            raise ManifestError()
        start = table_offset + relative
        end = data.find(b"\x00", start, table_offset + string_size)
        if end < 0 or end == start:
            raise ManifestError()
        try:
            return data[start:end].decode("ascii", errors="strict")
        except UnicodeError:
            raise ManifestError() from None

    names: list[str] = []
    for relative in needed_offsets:
        name = string_at(relative)
        if _ELF_LIBRARY_RE.fullmatch(name) is None or name in names:
            raise ManifestError()
        names.append(name)
    runpath = string_at(runpath_offsets[0]) if runpath_offsets else None
    return ElfInspection(
        dependencies=tuple(names),
        runpath=runpath,
        relro=relro,
        bind_now=bind_now,
        noexecstack=len(stack_flags) == 1 and stack_flags[0] & 0x1 == 0,
    )


def elf_dynamic_dependencies(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    maximum_bytes: int,
) -> tuple[str, ...]:
    """Return exact DT_NEEDED order from one bounded x86-64 shared ELF."""

    return inspect_elf(
        path,
        expected_sha256=expected_sha256,
        expected_size_bytes=expected_size_bytes,
        maximum_bytes=maximum_bytes,
    ).dependencies


def _validate_shared_elf(
    artifact: BoundArtifact,
    *,
    expected_dependencies: tuple[str, ...],
) -> ElfInspection:
    inspection = inspect_elf(
        artifact.path,
        expected_sha256=artifact.sha256,
        expected_size_bytes=artifact.size_bytes,
        maximum_bytes=artifact_maximum_bytes(PARAKEET_CPP_ADAPTER, artifact.name),
    )
    if inspection.dependencies != expected_dependencies:
        raise ManifestError()
    return inspection


def _closed_directory(path: Path, expected_names: set[str]) -> None:
    try:
        metadata = path.lstat()
        names = set(os.listdir(path))
        current = path.lstat()
    except (OSError, RuntimeError, ValueError):
        raise ManifestError() from None
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or _manifest_file_identity(current) != _manifest_file_identity(metadata)
        or names != expected_names
    ):
        raise ManifestError()


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _validate_parakeet_cpp_layout(
    artifacts: tuple[BoundArtifact, ...],
    config: ParakeetCppConfig,
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    if (
        tuple(by_name) != PARAKEET_CPP_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_PARAKEET_CPP_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _PARAKEET_CPP_ARTIFACT_BASENAMES.items()
        )
        or (
            by_name["model-gguf"].sha256,
            by_name["model-gguf"].size_bytes,
        )
        != PARAKEET_CPP_MODEL_RECEIPT
        or (
            by_name["libparakeet"].sha256,
            by_name["libparakeet"].size_bytes,
        )
        != PARAKEET_CPP_LIBPARAKEET_RECEIPT
        or (
            by_name["bridge-library"].sha256,
            by_name["bridge-library"].size_bytes,
        )
        != PARAKEET_CPP_BRIDGE_RECEIPT
    ):
        raise ManifestError()

    native_names = (
        "source-receipt",
        "build-receipt",
        "libparakeet",
        "bridge-library",
    )
    model_names = ("model-receipt", "model-gguf")
    native_roots = {by_name[name].path.parent for name in native_names}
    model_roots = {by_name[name].path.parent for name in model_names}
    if len(native_roots) != 1 or len(model_roots) != 1:
        raise ManifestError()
    native_root = next(iter(native_roots))
    model_root = next(iter(model_roots))
    if _paths_overlap(native_root, model_root):
        raise ManifestError()
    _closed_directory(
        native_root,
        {_PARAKEET_CPP_ARTIFACT_BASENAMES[name] for name in native_names},
    )
    _closed_directory(
        model_root,
        {_PARAKEET_CPP_ARTIFACT_BASENAMES[name] for name in model_names},
    )
    _validate_shared_elf(
        by_name["libparakeet"],
        expected_dependencies=PARAKEET_CPP_LIBPARAKEET_NEEDED,
    )
    bridge_inspection = _validate_shared_elf(
        by_name["bridge-library"],
        expected_dependencies=PARAKEET_CPP_BRIDGE_NEEDED,
    )
    if (
        bridge_inspection.runpath != PARAKEET_CPP_BRIDGE_RUNPATH
        or not bridge_inspection.relro
        or not bridge_inspection.bind_now
        or not bridge_inspection.noexecstack
    ):
        raise ManifestError()

    _exact_receipt_payload(
        by_name["source-receipt"],
        {
            "schema_version": 1,
            "kind": "parakeet-cpp-source-receipt-v1",
            "repo_id": config.upstream_repo_id,
            "commit": config.upstream_commit,
            "tag": config.upstream_tag,
            "ggml_commit": config.ggml_commit,
            "parent_pristine_git_tree": PARAKEET_CPP_PARENT_GIT_TREE,
            "ggml_pristine_git_tree": PARAKEET_CPP_GGML_GIT_TREE,
            "ordered_patches": [
                {
                    "order": order,
                    "filename": filename,
                    "sha256": sha256,
                }
                for order, (filename, sha256) in enumerate(
                    PARAKEET_CPP_PATCH_SPECS,
                    start=1,
                )
            ],
            "patched_diff_sha256": PARAKEET_CPP_PATCHED_DIFF_SHA256,
            "license": config.upstream_license,
            "license_sha256": config.upstream_license_sha256,
        },
    )
    _exact_receipt_payload(
        by_name["build-receipt"],
        {
            "schema_version": 1,
            "kind": "parakeet-cpp-build-receipt-v1",
            "source_receipt_sha256": by_name["source-receipt"].sha256,
            "c_api_version": config.c_api_version,
            "bridge_abi_version": config.bridge_abi_version,
            "requested_device": config.requested_device,
            "actual_device": config.actual_device,
            "num_threads": config.num_threads,
            "bridge_source_sha256": PARAKEET_CPP_BRIDGE_SOURCE_SHA256,
            "compiler_id": "GNU",
            "compiler_version": "13.3.0",
            "compiler_package": "Ubuntu 13.3.0-6ubuntu2~24.04.1",
            "cmake_version": "3.31.10",
            "ninja_version": "1.13.0.git.kitware.jobserver-pipe-1",
            "system_processor": "x86_64",
            "ggml_system_arch": "x86",
            "cpu_variant_flags": [
                "-msse4.2",
                "-mf16c",
                "-mfma",
                "-mbmi2",
                "-mavx",
                "-mavx2",
            ],
            "cmake": {
                "PARAKEET_SHARED": "ON",
                "BUILD_SHARED_LIBS": "OFF",
                "GGML_STATIC": "ON",
                "GGML_NATIVE": "OFF",
                "GGML_OPENMP": "OFF",
                "PARAKEET_GGML_CUDA": "OFF",
                "PARAKEET_GGML_METAL": "OFF",
                "PARAKEET_GGML_VULKAN": "OFF",
                "PARAKEET_GGML_HIP": "OFF",
            },
            "libparakeet_sha256": by_name["libparakeet"].sha256,
            "libparakeet_size_bytes": by_name["libparakeet"].size_bytes,
            "bridge_library_sha256": by_name["bridge-library"].sha256,
            "bridge_library_size_bytes": by_name["bridge-library"].size_bytes,
            "libparakeet_needed": list(PARAKEET_CPP_LIBPARAKEET_NEEDED),
            "bridge_needed": list(PARAKEET_CPP_BRIDGE_NEEDED),
            "bridge_runpath": PARAKEET_CPP_BRIDGE_RUNPATH,
            "bridge_relro": True,
            "bridge_bind_now": True,
            "bridge_noexecstack": True,
        },
    )
    _exact_receipt_payload(
        by_name["model-receipt"],
        {
            "schema_version": 1,
            "kind": "parakeet-cpp-model-receipt-v1",
            "model_repo_id": config.model_repo_id,
            "model_revision": config.model_revision,
            "source_model_repo_id": config.source_model_repo_id,
            "source_model_revision": config.source_model_revision,
            "filename": config.model_filename,
            "sha256": config.model_sha256,
            "size_bytes": config.model_size_bytes,
            "dtype": config.model_dtype,
            "license": config.model_license,
        },
    )


def _validate_faster_whisper_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    receipt_parents = {
        by_name[name].path.parent
        for name in ("runtime-receipt", "model-receipt")
        if name in by_name
    }
    if (
        tuple(by_name) != FASTER_WHISPER_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_FASTER_WHISPER_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _FASTER_WHISPER_ARTIFACT_BASENAMES.items()
        )
        or len(receipt_parents) != 1
    ):
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def _validate_kyutai_layout(
    python: BoundFile,
    artifacts: tuple[BoundArtifact, ...],
    config: KyutaiConfig,
) -> None:
    by_name = {artifact.name: artifact for artifact in artifacts}
    model_names = tuple(
        name for name in KYUTAI_ARTIFACT_NAMES if name.startswith("model-")
    )
    if (
        tuple(by_name) != KYUTAI_ARTIFACT_NAMES
        or sum(artifact.size_bytes for artifact in artifacts)
        > MAX_KYUTAI_TOTAL_ARTIFACT_BYTES
        or any(
            by_name[name].path.name != basename
            for name, basename in _KYUTAI_ARTIFACT_BASENAMES.items()
        )
        or any(
            (by_name[name].sha256, by_name[name].size_bytes) != receipt
            for name, receipt in KYUTAI_MODEL_RECEIPTS.items()
        )
        or by_name["runtime-wheel-lock"].sha256 != config.wheel_lock_sha256
        or (
            by_name["runtime-wheel-lock"].sha256,
            by_name["runtime-wheel-lock"].size_bytes,
        )
        != (
            KYUTAI_RUNTIME_WHEEL_LOCK_SHA256,
            KYUTAI_RUNTIME_WHEEL_LOCK_SIZE_BYTES,
        )
        or len({by_name[name].path.parent for name in model_names}) != 1
        or len(
            {
                by_name[name].path.parent
                for name in ("runtime-receipt", "runtime-wheel-lock")
            }
        )
        != 1
    ):
        raise ManifestError()
    _validate_venv_layout(python, by_name["venv-marker"])


def load_worker_manifest(path: Path | str) -> WorkerManifest:
    """Load and verify one immutable, machine-local worker receipt."""

    candidate = Path(path).expanduser()
    try:
        snapshot = read_regular_bounded(
            candidate,
            maximum_bytes=_MAX_MANIFEST_BYTES,
        )
    except BoundedReadError:
        raise ManifestError() from None
    resolved = snapshot.path
    raw = snapshot.data
    value = _strict_json(raw)
    if not isinstance(value, dict):
        raise ManifestError()
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version not in {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
    }:
        raise ManifestError()
    expected_fields = {
        1: _MANIFEST_V1_FIELDS,
        2: _MANIFEST_V2_FIELDS,
        3: _MANIFEST_V3_FIELDS,
        4: _MANIFEST_V4_FIELDS,
        5: _MANIFEST_V5_FIELDS,
        6: _MANIFEST_V6_FIELDS,
        7: _MANIFEST_V7_FIELDS,
        8: _MANIFEST_V8_FIELDS,
        9: _MANIFEST_V9_FIELDS,
        10: _MANIFEST_V10_FIELDS,
    }[schema_version]
    if set(value) != expected_fields:
        raise ManifestError()

    adapter = _safe_id(value.get("adapter"))
    if (
        (schema_version == 1 and adapter != "fake-json-v1")
        or (schema_version == 2 and adapter != MOONSHINE_ADAPTER)
        or (schema_version == 3 and adapter != NEMOTRON_ADAPTER)
        or (schema_version == 4 and adapter != SHERPA_ZIPFORMER_ADAPTER)
        or (schema_version == 5 and adapter != PARAKEET_REALTIME_EOU_ADAPTER)
        or (schema_version == 6 and adapter != FASTER_WHISPER_ENDPOINT_ADAPTER)
        or (
            schema_version == 7
            and adapter != MOONSHINE_EXTERNAL_ENDPOINT_ADAPTER
        )
        or (schema_version == 8 and adapter != PARAKEET_CPP_ADAPTER)
        or (schema_version == 9 and adapter != KYUTAI_ADAPTER)
        or (schema_version == 10 and adapter != MOBILE_ZIPFORMER_ADAPTER)
    ):
        raise ManifestError()

    python = _bound_file(
        value.get("python"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=True,
        maximum_bytes=MAX_PYTHON_BYTES,
        retain_lexical_path=True,
    )
    worker = _bound_file(
        value.get("worker"),
        expected_fields=_FILE_FIELDS,
        allow_symlink=False,
        maximum_bytes=MAX_WORKER_BYTES,
    )
    raw_artifacts = value.get("artifacts")
    expected_artifact_names = {
        1: ("fake-script",),
        2: MOONSHINE_ARTIFACT_NAMES,
        3: NEMOTRON_ARTIFACT_NAMES,
        4: SHERPA_ZIPFORMER_ARTIFACT_NAMES,
        5: PARAKEET_REALTIME_EOU_ARTIFACT_NAMES,
        6: FASTER_WHISPER_ARTIFACT_NAMES,
        7: MOONSHINE_ARTIFACT_NAMES,
        8: PARAKEET_CPP_ARTIFACT_NAMES,
        9: KYUTAI_ARTIFACT_NAMES,
        10: MOBILE_ZIPFORMER_ARTIFACT_NAMES,
    }[schema_version]
    if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(
        expected_artifact_names
    ):
        raise ManifestError()
    artifacts = tuple(_artifact(item, adapter=adapter) for item in raw_artifacts)
    artifact_names = [artifact.name for artifact in artifacts]
    if artifact_names != list(expected_artifact_names):
        raise ManifestError()
    adapter_config = {
        1: lambda: None,
        2: lambda: _moonshine_config(value.get("adapter_config")),
        3: lambda: _nemotron_config(value.get("adapter_config")),
        4: lambda: _sherpa_zipformer_config(value.get("adapter_config")),
        5: lambda: _parakeet_config(value.get("adapter_config")),
        6: lambda: _faster_whisper_config(value.get("adapter_config")),
        7: lambda: _moonshine_external_endpoint_config(
            value.get("adapter_config")
        ),
        8: lambda: _parakeet_cpp_config(value.get("adapter_config")),
        9: lambda: _kyutai_config(value.get("adapter_config")),
        10: lambda: _mobile_zipformer_config(value.get("mobile_config")),
    }[schema_version]()
    if schema_version == 2:
        assert isinstance(adapter_config, MoonshineConfig)
        _validate_moonshine_layout(python, artifacts, adapter_config)
    elif schema_version == 3:
        assert isinstance(adapter_config, NemotronConfig)
        _validate_nemotron_layout(python, artifacts)
    elif schema_version == 4:
        assert isinstance(adapter_config, SherpaZipformerConfig)
        _validate_sherpa_zipformer_layout(artifacts)
    elif schema_version == 5:
        assert isinstance(adapter_config, ParakeetRealtimeEouConfig)
        _validate_parakeet_layout(python, artifacts, adapter_config)
    elif schema_version == 6:
        assert isinstance(adapter_config, FasterWhisperEndpointConfig)
        _validate_faster_whisper_layout(python, artifacts)
    elif schema_version == 7:
        assert isinstance(adapter_config, MoonshineExternalEndpointConfig)
        _validate_moonshine_layout(python, artifacts, adapter_config)
    elif schema_version == 8:
        assert isinstance(adapter_config, ParakeetCppConfig)
        _validate_parakeet_cpp_layout(artifacts, adapter_config)
    elif schema_version == 9:
        assert isinstance(adapter_config, KyutaiConfig)
        _validate_kyutai_layout(python, artifacts, adapter_config)
    elif schema_version == 10:
        assert isinstance(adapter_config, MobileZipformerConfig)
        _validate_mobile_zipformer_layout(artifacts)

    control_files: tuple[BoundFile, ...] = ()
    if schema_version == 10:
        assert isinstance(adapter_config, MobileZipformerConfig)
        control_files = _mobile_zipformer_control_files(
            value,
            python=python,
            worker=worker,
            config=adapter_config,
        )

    raw_limits = value.get("limits")
    if not isinstance(raw_limits, dict) or set(raw_limits) != _LIMIT_FIELDS:
        raise ManifestError()
    limits = WorkerLimits(
        startup_timeout_sec=_bounded_seconds(
            raw_limits.get("startup_timeout_sec"),
            maximum=600.0,
        ),
        case_timeout_sec=_bounded_seconds(
            raw_limits.get("case_timeout_sec"),
            maximum=3600.0,
        ),
    )
    if schema_version == 10 and limits != WorkerLimits(
        startup_timeout_sec=120.0,
        case_timeout_sec=300.0,
    ):
        raise ManifestError()
    return WorkerManifest(
        path=resolved,
        digest=hashlib.sha256(raw).hexdigest(),
        schema_version=schema_version,
        model_id=_safe_id(value.get("model_id")),
        adapter=adapter,
        python=python,
        worker=worker,
        artifacts=artifacts,
        limits=limits,
        control_files=control_files,
        adapter_config=adapter_config,
    )
