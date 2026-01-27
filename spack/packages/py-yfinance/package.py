# Copyright 2026 Edric Matwiejew
# SPDX-License-Identifier: GPL-3.0-only

from spack.package import *


class PyYfinance(PythonPackage):
    """Download market data from Yahoo! Finance's API."""

    homepage = "https://github.com/ranaroussi/yfinance"
    pypi = "yfinance/yfinance-0.2.54.tar.gz"

    license("Apache-2.0")

    version("0.2.54", sha256="c8bca85c26a0db857593fce9c2e90c79d96f3da0e97dcb9165e47c5e12152bb8")

    depends_on("python@3.8:", type=("build", "run"))
    depends_on("py-setuptools", type="build")
    depends_on("py-pandas@1.3:", type=("build", "run"))
    depends_on("py-numpy@1.16:", type=("build", "run"))
    depends_on("py-requests@2.31:", type=("build", "run"))
    depends_on("py-multitasking@0.0.7:", type=("build", "run"))
    depends_on("py-lxml@4.9:", type=("build", "run"))
    depends_on("py-platformdirs@2.0:", type=("build", "run"))
    depends_on("py-pytz@2022.5:", type=("build", "run"))
    depends_on("py-frozendict@2.3.4:", type=("build", "run"))
    depends_on("py-peewee@3.16:", type=("build", "run"))
    depends_on("py-beautifulsoup4@4.11:", type=("build", "run"))
    depends_on("py-html5lib@1.1:", type=("build", "run"))
