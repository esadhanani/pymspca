import os
import sys
from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


def eigen_include():
    # Pinned Eigen 3.4.0 (the version bundled by RcppEigen) first, for bit-identical results with R.
    candidates = [os.environ.get("EIGEN_INCLUDE_DIR"), str(Path("third_party/eigen").resolve()),
                  "/opt/homebrew/include/eigen3", "/usr/local/include/eigen3", "/usr/include/eigen3"]
    for c in candidates:
        if c and (Path(c) / "Eigen" / "Dense").exists():
            return c
    sys.exit("Eigen 3 headers not found: set EIGEN_INCLUDE_DIR (brew install eigen / apt install libeigen3-dev)")


ext = Pybind11Extension(
    "pymspca._core",
    ["src/pymspca/_core/bindings.cpp"],
    include_dirs=[eigen_include(), "src/pymspca/_core"],
    cxx_std=17,
    define_macros=[("EIGEN_MPL2_ONLY", None)],
)
if sys.platform == "win32":
    ext.extra_compile_args += ["/O2", "/bigobj"]
else:
    ext.extra_compile_args += ["-O2"]  # same optimisation level as R CMD INSTALL

setup(ext_modules=[ext], cmdclass={"build_ext": build_ext})
