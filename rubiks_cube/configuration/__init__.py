from __future__ import annotations

from typing import Final
from typing import Literal
from typing import TypeAlias

import attrs

from rubiks_cube.configuration.enumeration import Metric

DEFAULT_METRIC: Final[Metric] = Metric.HTM
DEFAULT_GENERATOR: Final[str] = "<U, D, L, R, F, B>"

LogLevel: TypeAlias = Literal["debug", "info", "warning", "error", "critical"]


@attrs.frozen()
class AppConfig:
    cube_size: int = 2
    metric: Metric = Metric.HTM
    layout: Literal["centered", "wide"] = "wide"
    log_level: LogLevel = "debug"


APP_CFG = AppConfig()
