"""Device-level operations for Cambricon MLU hardware.

Provides :class:`MLUDeviceMixin`, a :class:`DeviceMixin` subclass that implements
device query, memory management, seeding, and synchronization for MLU hardware
via the ``torch.mlu`` API. This mixin is composed with :class:`MLUPlatform`
(defined in ``platform.py``) via multiple inheritance.

All ``torch.mlu`` calls are wrapped in ``try/except`` blocks so the module
imports cleanly when ``torch_mlu`` or MLU drivers are not installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Tuple

import torch

# Lazy base-class imports so the module imports cleanly without sglang.
try:
    from sglang.srt.platforms.device_mixin import (
        CpuArchEnum,
        DeviceCapability,
        DeviceMixin,
        PlatformEnum,
    )
except ImportError:
    if not TYPE_CHECKING:

        class DeviceMixin:
            def __init_subclass__(cls, **kwargs):
                pass

        class DeviceCapability(  # type: ignore[no-redef]
            tuple,
        ):
            def __new__(cls, major=0, minor=0):
                return super().__new__(cls, (major, minor))

            @property
            def major(self):
                return self[0]

            @property
            def minor(self):
                return self[1]

            @staticmethod
            def as_version_str():
                return ""

            def __repr__(self):
                return f"DeviceCapability(major={self[0]}, minor={self[1]})"

        class PlatformEnum:  # type: ignore[no-redef]
            OOT = "oot"
            UNSPECIFIED = "unspecified"
            CUDA = "cuda"
            ROCM = "rocm"
            CPU = "cpu"
            NPU = "npu"
            XPU = "xpu"
            MUSA = "musa"
            TPU = "tpu"
            MPS = "mps"

        class CpuArchEnum:  # type: ignore[no-redef]
            X86 = "x86"
            ARM = "arm"
            UNSPECIFIED = "unspecified"

        import typing

        if not hasattr(PlatformEnum, "__members__"):
            PlatformEnum.__members__ = {}


if TYPE_CHECKING:
    from sglang.srt.platforms.device_mixin import (
        CpuArchEnum,
        DeviceCapability,
        DeviceMixin,
        PlatformEnum,
    )

logger = logging.getLogger(__name__)


class MLUDeviceMixin(DeviceMixin):
    """DeviceMixin implementation for Cambricon MLU hardware.

    Overrides every abstract device-level hook in :class:`DeviceMixin` with
    ``torch.mlu.*`` equivalents. MLU-specific imports (``torch_mlu``) are
    performed inside methods so the module remains importable without MLU
    drivers installed.

    Class attributes:
        _enum: Platform enum identifying this as an out-of-tree plugin.
        device_name: Short device identifier string.
        device_type: PyTorch device type string passed to ``torch.device()``.
    """

    _enum: PlatformEnum = PlatformEnum.OOT
    device_name: str = "mlu"
    device_type: str = "mlu"

    # ------------------------------------------------------------------
    # Required / abstract methods from DeviceMixin
    # ------------------------------------------------------------------

    def get_device_total_memory(self, device_id: int = 0) -> int:
        """Return total MLU device memory in bytes.

        Args:
            device_id: Index of the MLU device to query.

        Returns:
            Total memory in bytes, or 0 on failure.
        """
        try:
            import torch_mlu  # noqa: F401

            return torch.mlu.get_device_properties(device_id).total_memory
        except (AttributeError, ImportError):
            logger.warning(
                "torch.mlu.get_device_properties not available; "
                "returning total_memory=0 for device %d",
                device_id,
            )
            return 0

    def get_current_memory_usage(
        self, device: Optional[torch.device] = None
    ) -> float:
        """Return peak memory allocated on the MLU device in bytes.

        Args:
            device: Specific MLU device. If ``None``, the current device is used.

        Returns:
            Peak allocated bytes, or 0.0 on failure.
        """

        def _default() -> float:
            logger.warning(
                "torch.mlu.max_memory_allocated not available; "
                "returning current_memory_usage=0.0 for device %s",
            )
            return 0.0

        try:
            import torch_mlu  # noqa: F401

            if device is not None:
                return torch.mlu.max_memory_allocated(device)
            return torch.mlu.max_memory_allocated()
        except (AttributeError, ImportError):
            return _default()

    # ------------------------------------------------------------------
    # Recommended methods (needed for runtime performance)
    # ------------------------------------------------------------------

    def set_device(self, device: torch.device) -> None:
        """Set the current MLU device for subsequent operations.

        Args:
            device: Target MLU device (e.g. ``torch.device("mlu", 0)``).
        """
        try:
            import torch_mlu  # noqa: F401

            torch.mlu.set_device(device)
        except (AttributeError, ImportError):
            logger.warning(
                "torch.mlu.set_device not available; cannot set device to %s",
                device,
            )

    def get_device(self, local_rank: int = 0) -> torch.device:
        """Return a ``torch.device`` for the given MLU *local_rank*.

        When ``torch_mlu`` is not installed the "mlu" device type is unknown
        to PyTorch; fall back to ``privateuse1`` (the slot torch_mlu
        renames to "mlu" at runtime).

        Args:
            local_rank: Rank index of the desired MLU device.

        Returns:
            A ``torch.device("mlu", local_rank)`` or
            ``torch.device("privateuse1", local_rank)`` depending on
            whether torch_mlu is available.
        """
        try:
            import torch_mlu  # noqa: F401

            return torch.device(type="mlu", index=local_rank)
        except ImportError:
            # torch_mlu is the component that registers "mlu" with PyTorch;
            # without it the device type is unknown. This method is only
            # called from inside the SGLang runtime on MLU-equipped machines,
            # so reaching here indicates a broken installation rather than
            # a normal dev-machine fallback.
            raise RuntimeError(
                'MLU device requested but "mlu" device type is not registered. '
                "Is torch_mlu installed and importable?"
            ) from None

    def get_device_name(self, device_id: int = 0) -> str:
        """Return the human-readable product name of the MLU at *device_id*.

        Args:
            device_id: Index of the MLU device.

        Returns:
            Device name string (e.g. ``"MLU370-S4"``) or ``"MLU-unknown"`` on failure.
        """
        try:
            import torch_mlu  # noqa: F401

            return torch.mlu.get_device_name(device_id)
        except (AttributeError, ImportError):
            logger.warning(
                "torch.mlu.get_device_name not available; returning 'MLU-unknown'"
            )
            return "MLU-unknown"

    def get_device_capability(
        self, device_id: int = 0
    ) -> Optional[DeviceCapability]:
        """Return the compute capability *(major, minor)* of the MLU.

        MLU hardware does not expose a CUDA-style compute capability, so this
        method returns ``None``.

        Args:
            device_id: Index of the MLU device.

        Returns:
            Always ``None``.
        """
        return None

    def empty_cache(self) -> None:
        """Release all cached MLU memory back to the runtime/driver."""
        try:
            import torch_mlu  # noqa: F401

            torch.mlu.empty_cache()
        except (AttributeError, ImportError):
            logger.warning("torch.mlu.empty_cache not available; skipping cache clear")

    def synchronize(self) -> None:
        """Block until all queued MLU operations on the current device complete."""
        try:
            import torch_mlu  # noqa: F401

            torch.mlu.synchronize()
        except (AttributeError, ImportError):
            logger.warning("torch.mlu.synchronize not available; skipping sync")

    def get_available_memory(self, device_id: int = 0) -> Tuple[int, int]:
        """Return free and total memory on the MLU device.

        Args:
            device_id: Index of the MLU device.

        Returns:
            A ``(free_bytes, total_bytes)`` tuple, or ``(0, 0)`` on failure.
        """
        try:
            import torch_mlu  # noqa: F401

            return torch.mlu.mem_get_info(device_id)
        except (AttributeError, ImportError, RuntimeError):
            total = self.get_device_total_memory(device_id)
            logger.warning(
                "torch.mlu.mem_get_info not available; "
                "returning (%d, %d) — assuming all memory free",
                total,
                total,
            )
            return (total, total)

    def get_torch_distributed_backend_str(self) -> str:
        """Return the PyTorch distributed backend string for MLU collectives.

        Returns:
            ``"cncl"`` — Cambricon's collective communication library.
        """
        return "cncl"

    @classmethod
    def seed_everything(cls, seed: Optional[int] = None) -> None:
        """Seed all random generators for reproducible MLU execution.

        Sets seeds for Python ``random``, NumPy, CPU torch, and all MLU devices.

        Args:
            seed: Integer seed value. If ``None``, no seeding is performed.

        Raises:
            ValueError: If *seed* is negative.
        """
        if seed is None:
            return
        if seed < 0:
            raise ValueError(f"seed must be non-negative, got {seed}")

        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        try:
            import torch_mlu  # noqa: F401

            torch.mlu.manual_seed_all(seed)
        except (AttributeError, ImportError):
            logger.warning(
                "torch.mlu.manual_seed_all not available; "
                "skipping MLU-specific seeding"
            )

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def get_device_uuid(self, device_id: int = 0) -> str:
        """Return a unique identifier (UUID) for the MLU at *device_id*.

        Args:
            device_id: Index of the MLU device.

        Returns:
            UUID string, or an empty string if the device does not expose a
            UUID or ``torch_mlu`` is unavailable.
        """
        try:
            import torch_mlu  # noqa: F401

            return torch.mlu.get_device_properties(device_id).uuid
        except (AttributeError, ImportError):
            logger.warning(
                "torch.mlu.get_device_properties.uuid not available; "
                "returning empty UUID for device %d",
                device_id,
            )
            return ""
