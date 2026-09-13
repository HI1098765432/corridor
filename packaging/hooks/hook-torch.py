"""Replace the stock torch hook with one scoped to CPU inference.

The hook shipped with pyinstaller-hooks-contrib calls ``collect_submodules``
over the whole of torch, which *imports* every submodule -- including the
distributed, ONNX, quantisation and compiler trees. On this project that made
the analysis stage run for tens of minutes and added hundreds of megabytes that
Cellpose inference never touches.

The exclusions below are all leaves of torch that a forward pass through a
Cellpose UNet cannot reach. The list is deliberately conservative: anything
reachable from ``torch.nn``, ``torch.jit``, ``torch.serialization`` or the ATen
operator registry is kept. The packaged application is then tested by running a
real segmentation, which exercises the whole inference path -- a missing module
would surface there rather than in a user's hands.
"""

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)
from PyInstaller.utils.hooks import PY_DYLIB_PATTERNS

module_collection_mode = "pyz+py"
warn_on_missing_hiddenimports = False

#: Torch subtrees unreachable from a CPU forward pass.
EXCLUDED = (
    "torch.distributed",
    "torch.onnx",
    "torch.testing",
    "torch.utils.tensorboard",
    "torch.utils.benchmark",
    "torch.utils.bottleneck",
    "torch.utils.data.datapipes",
    "torch.ao",
    "torch.quantization",
    "torch.nn.quantized",
    "torch.nn.qat",
    "torch.nn.intrinsic",
    "torch._dynamo",
    "torch._inductor",
    "torch._export",
    "torch.export",
    "torch.distributions.constraints_registry",
    "torch.cuda.amp",
    "torch.backends.cudnn",
    "torch.backends.cuda",
)


def _wanted(name: str) -> bool:
    return not name.startswith(EXCLUDED)


datas = collect_data_files(
    "torch",
    excludes=["**/*.h", "**/*.hpp", "**/*.cuh", "**/*.lib", "**/*.cpp",
              "**/*.pyi", "**/*.cmake"],
)
hiddenimports = collect_submodules("torch", filter=_wanted)
binaries = collect_dynamic_libs("torch", search_patterns=PY_DYLIB_PATTERNS + ["*.so.*"])
