#!/usr/bin/env python3
"""Pin the process to ONE GPU, before torch is imported.

WHY THIS EXISTS. With more than one GPU visible and no distributed launcher,
several training paths auto-wrap the model in nn.DataParallel across all of
them. On the Pascal-era cards this was written against that stalls rather than
failing: one run sat 45 minutes at step 0 of 39, a single PID pinned at 99% on
all four cards, DataParallel scatter/gather thrashing. It looks like a hang, but
the GPUs are busy, so the usual "is it dead?" checks say everything is fine.

A DistilBERT cross-encoder fits one 12 GB card with room to spare, so there is
nothing to gain from multi-GPU here and a documented pathology to lose. Every
entry point calls pin_single_gpu() as its first statement.

    from gpu import pin_single_gpu
    pin_single_gpu()          # MUST precede `import torch`
    import torch

DEFAULT_GPU below is machine-specific: on the original host, device 0 was a
display adapter and the compute cards were 1 through 4. Change it for yours.
Override per-run with IN_CLS_GPU, e.g. IN_CLS_GPU=3, or IN_CLS_GPU=cpu.
"""

from __future__ import annotations

import os
import sys

DEFAULT_GPU = "1"          # machine-specific; see the module docstring

_PINNED = None             # what this process already pinned, if anything


def pin_single_gpu(default: str = DEFAULT_GPU, verbose: bool = True) -> str:
    """Set CUDA_VISIBLE_DEVICES to exactly one device. Returns what was set.

    Idempotent. Importing one entry point from another re-runs its module-level
    pin, and that second call necessarily happens after torch is loaded. Warning
    about it would be crying wolf, since the first call already did the work, so
    a repeat pin to the same device returns quietly. The warning below stays for
    the case that actually matters: torch imported before ANY pin.
    """
    global _PINNED
    want = os.environ.get("IN_CLS_GPU", default).strip()
    if _PINNED is not None:
        if _PINNED != want and verbose:
            print(f"  WARNING  already pinned to {_PINNED!r}; ignoring request "
                  f"for {want!r} (a process pins once).", file=sys.stderr)
        return _PINNED

    if "torch" in sys.modules and verbose:
        print("  WARNING  torch was already imported; the pin may not take "
              "effect. Call pin_single_gpu() before importing torch.",
              file=sys.stderr)
    if want.lower() in ("cpu", "none", "-1"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _PINNED = ""
        if verbose:
            print("  device: CPU (IN_CLS_GPU=cpu)")
        return ""

    if "," in want:
        raise ValueError(
            f"IN_CLS_GPU={want!r} names several devices. This trains on exactly "
            f"one by design; see the module docstring.")

    prior = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    os.environ["CUDA_VISIBLE_DEVICES"] = want
    _PINNED = want
    if verbose:
        print(f"  device: pinned CUDA_VISIBLE_DEVICES {prior} -> {want} "
              f"(single-GPU by design)")
    return want


def describe_device():
    """Import torch (after pinning) and report what we actually got."""
    import torch
    if not torch.cuda.is_available():
        return {"device": "cpu", "cuda": False}
    return {
        "device": "cuda:0",
        "cuda": True,
        "device_count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "total_memory_gb": round(
            torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1),
    }
