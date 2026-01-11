import sys
from importlib.metadata import version

from loguru import logger

__version__ = version("vectormesh")

logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("logs/dataset.log", rotation="10 MB", level="DEBUG")
