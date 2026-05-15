"""sglang ``/generate`` logprob shape → CEAC score format.

sglang protocol: when ``logprob_start_len = prompt_len - 1`` the
returned ``input_token_logprobs`` has shape
``[boundary_None, response_0, response_1, ...]``. The parser slices
``[1:1+n_response]`` to drop the boundary entry and keep just the
response positions.
"""

from affine.src.anticopy.worker import _parse_sglang_logprobs


def _payload(lp_entries, top_entries):
    return {
        "meta_info": {
            "input_token_logprobs": lp_entries,
            "input_top_logprobs": top_entries,
        }
    }


def test_parse_strips_boundary_and_returns_response_logprobs():
    payload = _payload(
        # index 0 = boundary token at start_len, lp=None per sglang contract
        [
            [None, 999, None],
            [-0.1, 7, None],
            [-0.5, 13, None],
        ],
        [
            [],                                # boundary slot
            [[-0.1, 7, None], [-1.2, 99, None]],
            [[-0.5, 13, None], [-1.4, 11, None]],
        ],
    )
    lp, top = _parse_sglang_logprobs(payload, n_response=2)
    assert lp == [-0.1, -0.5]
    assert top == [[[-0.1, 7], [-1.2, 99]], [[-0.5, 13], [-1.4, 11]]]


def test_parse_handles_batched_first_element():
    inner = _payload(
        [[None, 0, None], [-0.2, 1, None]],
        [[], [[-0.2, 1, None]]],
    )
    lp, top = _parse_sglang_logprobs([inner], n_response=1)
    assert lp == [-0.2]
    assert top == [[[-0.2, 1]]]


def test_parse_drops_malformed_entries_per_position():
    payload = _payload(
        [
            [None, 0, None],            # boundary
            [-0.1, 7, None],
            None,                        # masked position
            "garbage",                   # invalid entry
        ],
        [
            [],
            [[-0.1, 7, None]],
            [],
            [["x", "y", None]],
        ],
    )
    lp, top = _parse_sglang_logprobs(payload, n_response=3)
    assert lp == [-0.1, None, None]
    assert top[0] == [[-0.1, 7]]
    assert top[1] == []
    assert top[2] == []


def test_parse_empty_response_is_empty_lists():
    lp, top = _parse_sglang_logprobs({}, n_response=0)
    assert lp == []
    assert top == []


def test_parse_slices_n_response_exactly():
    """If sglang returned extra trailing positions (e.g. the one
    sampled token), n_response slicing must not include them."""
    payload = _payload(
        [
            [None, 0, None],
            [-0.1, 1, None],
            [-0.2, 2, None],
            [-0.3, 3, None],
        ],
        [
            [], [[-0.1, 1]], [[-0.2, 2]], [[-0.3, 3]],
        ],
    )
    lp, top = _parse_sglang_logprobs(payload, n_response=2)
    assert lp == [-0.1, -0.2]
    assert top == [[[-0.1, 1]], [[-0.2, 2]]]


def test_parse_legacy_path_returns_raw_when_no_n_response():
    """Backward-compat: no n_response → return all entries verbatim,
    leaving the caller to slice (used by debug / tests)."""
    payload = _payload(
        [[None, 0, None], [-0.5, 1, None]],
        [[], [[-0.5, 1]]],
    )
    lp, top = _parse_sglang_logprobs(payload)
    assert lp == [None, -0.5]
    assert top[0] == []
    assert top[1] == [[-0.5, 1]]
