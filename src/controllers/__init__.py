REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
