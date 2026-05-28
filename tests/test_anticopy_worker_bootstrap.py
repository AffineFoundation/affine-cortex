"""Mid-flight bootstrap helpers: error signature detection +
script formatting. The actual SSH round-trip is exercised in
deployment smoke tests; these unit tests pin the pure-Python
classification logic that decides *when* to bootstrap.
"""

from affine.src.anticopy.worker import (
    _REMOTE_BOOTSTRAP_SCRIPT,
    _is_remote_env_missing_error,
)


def test_targon_rescheduled_signature_matches():
    # Exact log line shape observed when the targon GPU pod was
    # re-scheduled and the wrapper python was gone — worker burned
    # 36h looping on this until the operator manually redeployed.
    err = RuntimeError(
        "remote snapshot_download rc=127: "
        "Connection to ssh.deployments.targon.com closed."
    )
    assert _is_remote_env_missing_error(err)


def test_command_not_found_signature_matches():
    err = RuntimeError(
        "remote helper rc=127: bash: line 1: /root/.venv/bin/python: "
        "command not found"
    )
    assert _is_remote_env_missing_error(err)


def test_no_such_file_signature_matches():
    err = RuntimeError(
        "remote rc=127: bash: /workspace/anticopy/.venv/bin/python: "
        "No such file or directory"
    )
    assert _is_remote_env_missing_error(err)


def test_network_timeout_does_not_match():
    # Tunnel drop / sglang slow → not the same recovery path. We
    # rely on autossh + the existing per-job retry to absorb these.
    err = RuntimeError(
        "ClientConnectorError: Cannot connect to host localhost:33000 "
        "ssl:default [Connect call failed]"
    )
    assert not _is_remote_env_missing_error(err)


def test_sglang_launch_timeout_does_not_match():
    err = RuntimeError(
        "sglang launch on http://localhost:33000 never became ready "
        "after 15min"
    )
    assert not _is_remote_env_missing_error(err)


def test_rc127_alone_does_not_match():
    # rc=127 from some other command without the connection-closed
    # marker is ambiguous; don't trigger an expensive bootstrap.
    err = RuntimeError("some helper exited rc=127")
    assert not _is_remote_env_missing_error(err)


def test_bootstrap_script_is_idempotent_shape():
    # Cheap regression on the shell logic: the script must short-circuit
    # when the venv + packages are already present (otherwise every
    # worker restart pays the multi-minute pip install).
    s = _REMOTE_BOOTSTRAP_SCRIPT.format(
        hf_cache="'/root/hf-cache'",
        remote_python="'/root/.venv/bin/python'",
    )
    assert "import sglang, huggingface_hub, hf_transfer" in s
    assert "BOOTSTRAP_OK" in s
    assert "set -e" in s


def test_bootstrap_script_handles_pipless_venv():
    # Regression: a previous version of the script only probed
    # ``-x python`` and walked away when the targon node had a venv
    # whose ensurepip step had failed silently. The script must also
    # probe ``pip --version`` and reset the venv if it's broken.
    s = _REMOTE_BOOTSTRAP_SCRIPT.format(
        hf_cache="'/root/hf-cache'",
        remote_python="'/root/.venv/bin/python'",
    )
    assert "pip --version" in s
    assert "--clear" in s  # python3 -m venv --clear forces re-init
    assert "ensurepip" in s


def test_bootstrap_script_includes_ninja():
    # Regression: sglang's CUDA graph warmup JIT-builds kernels via
    # tvm_ffi -> ninja; a cold-start node without ninja installs the
    # engine fine, loads weights, then SIGQUITs partway through
    # ``Capture cuda graph`` with ``FileNotFoundError: 'ninja'``.
    # Probe + bootstrap must include it so a ninja-less host gets
    # repaired before the worker tries to launch sglang.
    s = _REMOTE_BOOTSTRAP_SCRIPT.format(
        hf_cache="'/root/hf-cache'",
        remote_python="'/root/.venv/bin/python'",
    )
    assert "ninja" in s


def test_bootstrap_script_installs_venv_package():
    # Regression: targon's Ubuntu 24.04 image ships python3 but not
    # python3-venv, so ``python3 -m venv`` fails outright. Bootstrap
    # must apt-install it (best-effort) before attempting the venv.
    s = _REMOTE_BOOTSTRAP_SCRIPT.format(
        hf_cache="'/root/hf-cache'",
        remote_python="'/root/.venv/bin/python'",
    )
    assert "apt-get install" in s
    assert "python3-venv" in s
    assert "DEBIAN_FRONTEND=noninteractive" in s
