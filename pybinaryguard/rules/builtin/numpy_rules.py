"""NumPy ABI compatibility rules.

NumPy exposes a C API for packages that build compiled extensions against
it (e.g. SciPy, pandas, scikit-learn).  The API version is baked into
the extension at compile time; if the installed NumPy provides a different
API version the extension may crash or produce wrong results.
"""

from __future__ import annotations

from typing import List

from pybinaryguard.models.enums import Severity
from pybinaryguard.models.finding import Finding
from pybinaryguard.models.package import PackageBinaryInfo
from pybinaryguard.models.system import SystemProfile
from pybinaryguard.rules.base import Rule


def _find_numpy(packages: List[PackageBinaryInfo]) -> PackageBinaryInfo | None:
    """Return the numpy PackageBinaryInfo from the package list, if present."""
    for pkg in packages:
        if pkg.package_name.lower() == "numpy":
            return pkg
    return None


class NumpyABIMismatchRule(Rule):
    """Detects packages compiled against a different NumPy C API version.

    When a package is compiled against NumPy's C API (``numpy/arrayobject.h``
    etc.), the ``NPY_VERSION`` / ``numpy_api_version`` constant is recorded.
    If the installed NumPy has a different API version, the struct layouts
    may differ and the extension may segfault or produce corrupt data.

    The typical symptom is a ``RuntimeWarning: numpy.dtype size changed,
    may indicate binary incompatibility`` or a hard crash.
    """

    rule_id = "NUMPY_ABI_MISMATCH"
    description = (
        "Check that packages compiled against the NumPy C API match the "
        "installed NumPy's API version."
    )

    def evaluate(
        self,
        profile: SystemProfile,
        packages: List[PackageBinaryInfo],
    ) -> List[Finding]:
        findings: List[Finding] = []

        numpy_pkg = _find_numpy(packages)
        if numpy_pkg is None:
            # NumPy is not installed; nothing to check.
            return findings

        numpy_api = numpy_pkg.numpy_api_version
        if numpy_api is None:
            # We do not know NumPy's own API version; skip.
            return findings

        for pkg in packages:
            if pkg.package_name.lower() == "numpy":
                continue
            if pkg.numpy_api_version is None:
                continue
            if pkg.numpy_api_version != numpy_api:
                findings.append(
                    Finding(
                        rule_id=self.rule_id,
                        severity=Severity.CRITICAL,
                        title=(
                            f"{pkg.package_name} was built against a "
                            f"different NumPy ABI"
                        ),
                        explanation=(
                            f"Package {pkg.package_name} "
                            f"{pkg.package_version} was compiled against "
                            f"NumPy C API version {pkg.numpy_api_version:#x} "
                            f"but the installed NumPy "
                            f"({numpy_pkg.package_version}) provides API "
                            f"version {numpy_api:#x}.  This mismatch means "
                            f"the internal array and dtype struct layouts "
                            f"may differ, which can cause segmentation "
                            f"faults, data corruption, or RuntimeWarnings "
                            f"about 'dtype size changed'."
                        ),
                        technical_detail=(
                            f"Package numpy_api_version: "
                            f"{pkg.numpy_api_version:#x}, "
                            f"Installed NumPy API version: "
                            f"{numpy_api:#x}"
                        ),
                        suggestion=(
                            f"Rebuild {pkg.package_name} against the "
                            f"currently installed NumPy:\n"
                            f"  pip install --no-binary :all: --force-reinstall "
                            f"{pkg.package_name}\n\n"
                            f"Or install a version of {pkg.package_name} "
                            f"that was built for NumPy "
                            f"{numpy_pkg.package_version}:\n"
                            f"  pip install --force-reinstall "
                            f"{pkg.package_name}"
                        ),
                        package=pkg.package_name,
                        package_version=pkg.package_version,
                    )
                )
        return findings
