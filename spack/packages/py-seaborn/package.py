# Copyright 2026 Edric Matwiejew
# SPDX-License-Identifier: GPL-3.0-only

from spack.package import *


class PySeaborn(PythonPackage):
    """Statistical data visualization."""

    homepage = "https://seaborn.pydata.org/"
    pypi = "seaborn/seaborn-0.13.2.tar.gz"

    license("BSD-3-Clause")

    version("0.13.2", sha256="93e60a40988f4f9e9ee5e63ee7c9c802c3b35eaffc6a2f7c3dc7e4fcb39c0e2e")
    version("0.11.2", sha256="cf45e9286d40826864be0e3c066f98536982baf701a7caa386511792d61ff4f6")

    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-flit-core@3.2:3", type="build")
    depends_on("py-numpy@1.20:1", when="@0.13:", type=("build", "run"))
    depends_on("py-numpy@1.15:", when="@:0.12", type=("build", "run"))
    depends_on("py-pandas@1.2:", type=("build", "run"))
    depends_on("py-matplotlib@3.4:3", when="@0.13:", type=("build", "run"))
    depends_on("py-matplotlib@2.2:", when="@:0.12", type=("build", "run"))
