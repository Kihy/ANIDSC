"""Custom type markers for generic types"""
from typing import List, Dict, Any

class RecordList(list):
    """Marker type for List[Dict] (list of records)"""
    pass

class StringList(list):
    """Marker type for List[str]"""
    pass

class IntList(list):
    """Marker type for List[int]"""
    pass