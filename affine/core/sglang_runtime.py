"""Shared SGLang runtime defaults for validator-managed deployments."""

DEFAULT_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.14"
DEFAULT_SGLANG_SERVER_ARGS = (
    "--num-continuous-decode-steps",
    "4",
    "--enable-mixed-chunk",
)
