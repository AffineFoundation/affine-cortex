# SGLang 4P4D deployment

`serving_mode=pd` turns one 8-GPU SSH endpoint into this fixed topology:

| Role | GPU | HTTP ports | Bootstrap ports | Exposure |
| --- | --- | --- | --- | --- |
| Prefill replicas 0–3 | 0–3 | 11000–11003 | 12000–12003 | loopback |
| Decode replicas 0–3 | 4–7 | 13000–13003 | — | loopback |
| Model Gateway | CPU only | 10001 | — | public inference URL |

Every worker is DP1. The gateway uses `cache_aware` prefill routing and
`power_of_two` decode routing. It sends coordinated request legs to one
prefill and one decode worker; Mooncake transfers KV cache directly between
them over intra-node NVLink. KV payloads do not pass through the gateway.

Only the gateway URL is stored in scheduler/scorer deployment records.
Terminal is not told the DP size or a rank. Multi-turn locality is the
gateway's cache-aware routing responsibility rather than a client-side fixed
rank contract.

## Prerequisites

- Exactly eight visible NVIDIA GPUs. The current implementation intentionally
  rejects any topology other than 4P4D.
- Docker with NVIDIA Container Toolkit and host networking.
- The champion model must fit on one B200 because each worker is an
  independent single-GPU model replica.
- Worker and gateway images must be pinned by full `sha256` digest and must
  contain compatible SGLang releases. The gateway image must provide
  `python -m sglang_router.launch_router`.
- Ports 10001, 11000–11003, 12000–12003, 13000–13003, and local Prometheus
  port 14000 must be free. Only port 10001 should be reachable externally.
- The worker image must include the selected Mooncake or NIXL transfer
  backend. Mooncake is the default.

SGLang's relevant upstream interfaces are documented under
[PD disaggregation](https://docs.sglang.ai/advanced_features/pd_disaggregation.html)
and the
[Model Gateway](https://docs.sglang.ai/advanced_features/sgl_model_gateway.html).

## Static endpoint registration

Replace both example digests with verified release digests. Keep the worker
and gateway versions aligned.

```bash
af db register-static-endpoint \
  --name b200-pd-1 \
  --kind ssh \
  --ssh-url ssh://root@B200_HOST \
  --public-inference-url http://B200_PUBLIC_IP:10001/v1 \
  --serving-mode pd \
  --sglang-image 'lmsysorg/sglang:v0.5.14@sha256:<64-hex-digest>' \
  --sglang-pd-gateway-image 'lmsysorg/sgl-model-gateway:v0.5.14@sha256:<64-hex-digest>' \
  --sglang-pd-prefill-replicas 4 \
  --sglang-pd-decode-replicas 4 \
  --sglang-chunked-prefill 8192 \
  --sglang-pd-transfer-backend mooncake \
  --sglang-pd-prefill-policy cache_aware \
  --sglang-pd-decode-policy power_of_two
```

Registration first stages the row as inactive, verifies SSH, Docker, and at
least eight GPUs, then activates it. PD-specific validation also rejects
unpinned images, overlapping ports, unsupported policies, and non-4P4D
replica counts before the row is staged.

Autoscaler-managed endpoint slots accept the same fields inside their
`endpoint` mapping:

```json
{
  "name": "lium-b200-pd-1",
  "provider": "lium",
  "endpoint": {
    "serving_mode": "pd",
    "sglang_image": "lmsysorg/sglang:v0.5.14@sha256:<64-hex-digest>",
    "sglang_pd_gateway_image": "lmsysorg/sgl-model-gateway:v0.5.14@sha256:<64-hex-digest>",
    "sglang_pd_prefill_replicas": 4,
    "sglang_pd_decode_replicas": 4,
    "sglang_pd_transfer_backend": "mooncake"
  }
}
```

## Deployment lifecycle

For each champion or challenger assignment, the scheduler:

1. validates the fixed 4P4D configuration and image digests;
2. removes the nine stable PD names and the exact legacy unified container
   name, so a drained mode switch cannot leave conflicting GPU processes;
3. starts all eight workers, each restricted to one GPU and loopback HTTP;
4. fails immediately if a worker exits while becoming ready;
5. starts the CPU-only gateway after every worker serves `/v1/models`;
6. fails immediately if the gateway exits during readiness;
7. probes the public gateway and sends a one-token chat completion through the
   complete Prefill → KV transfer → Decode path;
8. publishes the gateway as the assignment's only deployment URL.

If any step fails, those exact managed containers are removed. Periodic health
checks require all nine containers to be running with the expected model,
revision, role, replica, and GPU labels, plus a healthy public gateway.

The workers include `--enable-mixed-chunk` and
`--num-continuous-decode-steps 8`. Gateway upstream request timeout is 3720
seconds, matching the long Terminal request envelope.

## Rollout and rollback

Start with one drained endpoint and one fixed champion revision. Before
allowing sampling traffic, verify:

- `af db list-endpoints` reports `mode=pd`, `prefill=4`, and `decode=4`;
- all nine containers have the expected labels and GPU mapping;
- only the gateway port is externally reachable;
- gateway `/v1/models`, one non-streaming completion, and one streaming
  Terminal canary all succeed;
- per-role request counts and GPU utilization are non-zero on all eight GPUs;
- Mooncake logs show successful bootstrap and KV transfer without TCP reset or
  timeout errors.

This single-host topology cannot blue/green swap models without another
8-GPU host. Drain the endpoint before changing its mode or image. Rollback is
to register the drained endpoint again with `--serving-mode unified` and the
previous digest-pinned worker configuration, then reactivate it after its
preflight succeeds.
