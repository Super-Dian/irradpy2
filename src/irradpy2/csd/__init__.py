import functools
import warnings

from .AliaMartinez2016CSD import AliaMartinez2016CSD
from .Batlles1999CSD import Batlles1999CSD
from .BrightSun2020CSDc import BrightSun2020CSDc
from .BrightSun2020CSDs import BrightSun2020CSDs
from .Ellis2018CSD import Ellis2018CSD
from .Garcia2014CSD import Garcia2014CSD
from .Gueymard2013CSD import Gueymard2013CSD
from .Ineichen2006CSD import Ineichen2006CSD
from .Ineichen2009CSD import Ineichen2009CSD
from .Ineichen2016CSD import Ineichen2016CSD
from .Inman2015CSD import Inman2015CSD
from .Larraneta2017CSD import Larraneta2017CSD
from .Lefevre2013CSD import Lefevre2013CSD
from .LongAckerman2000CSD import LongAckerman2000CSD
from .Perez1990CSD import Perez1990CSD
from .Polo2009CSD import Polo2009CSD
from .Quesadaruiz2015CSD import Quesadaruiz2015CSD
from .Reno2016CSD import Reno2016CSD
from .RuizArias2018CSD import RuizArias2018CSD
from .Shen2018CSD import Shen2018CSD
from .Xie2013CSD import Xie2013CSD
from .Zhang2018CSD import Zhang2018CSD
from .Zhao2018CSD import Zhao2018CSD


__all__ = [
    "AliaMartinez2016CSD",
    "Batlles1999CSD",
    "BrightSun2020CSDc",
    "BrightSun2020CSDs",
    "Ellis2018CSD",
    "Garcia2014CSD",
    "Gueymard2013CSD",
    "Ineichen2006CSD",
    "Ineichen2009CSD",
    "Ineichen2016CSD",
    "Inman2015CSD",
    "Larraneta2017CSD",
    "Lefevre2013CSD",
    "LongAckerman2000CSD",
    "Perez1990CSD",
    "Polo2009CSD",
    "Quesadaruiz2015CSD",
    "Reno2016CSD",
    "RuizArias2018CSD",
    "Shen2018CSD",
    "Xie2013CSD",
    "Zhang2018CSD",
    "Zhao2018CSD",
]


def _ignore_runtime_warnings(func):
    """
    Wrap a CSD function so RuntimeWarning messages raised during
    calculation are ignored without changing the calculated values.
    """

    if getattr(func, "_runtime_warning_wrapped", False):
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return func(*args, **kwargs)

    wrapper._runtime_warning_wrapped = True
    return wrapper


for _name in __all__:
    _func = globals().get(_name)

    if callable(_func):
        globals()[_name] = _ignore_runtime_warnings(_func)


del _name
del _func