"""Best-effort WDDM per-process shared GPU memory, all adapters.

Missing/unsupported PDH counters are not zero. Sampled shared usage is not
proof of CUDA paging; WDDM also reports intentionally shared allocations.
"""
import ctypes as C
from ctypes import wintypes as W
import os


class _Value(C.Structure):
    _fields_ = [("status", W.DWORD), ("value", C.c_longlong)]


class _Item(C.Structure):
    _fields_ = [("name", W.LPWSTR), ("value", _Value)]


class SharedGpuMemory:
    def __init__(self):
        self.dll = C.WinDLL("pdh")
        self.query = C.c_void_p()
        self.counter = C.c_void_p()
        self.prefix = f"pid_{os.getpid()}_"
        dll = self.dll
        dll.PdhOpenQueryW.argtypes = [W.LPCWSTR, C.c_size_t, C.POINTER(C.c_void_p)]
        dll.PdhAddEnglishCounterW.argtypes = [C.c_void_p, W.LPCWSTR, C.c_size_t, C.POINTER(C.c_void_p)]
        dll.PdhCollectQueryData.argtypes = [C.c_void_p]
        dll.PdhGetFormattedCounterArrayW.argtypes = [C.c_void_p, W.DWORD, C.POINTER(W.DWORD),
                                                      C.POINTER(W.DWORD), C.c_void_p]
        dll.PdhCloseQuery.argtypes = [C.c_void_p]
        for fn in ("PdhOpenQueryW", "PdhAddEnglishCounterW", "PdhCollectQueryData",
                   "PdhGetFormattedCounterArrayW", "PdhCloseQuery"):
            getattr(dll, fn).restype = W.DWORD
        if dll.PdhOpenQueryW(None, 0, C.byref(self.query)):
            raise OSError("PDH query unavailable")
        if dll.PdhAddEnglishCounterW(self.query, r"\GPU Process Memory(*)\Shared Usage", 0, C.byref(self.counter)):
            self.close()
            raise OSError("WDDM shared GPU counter unavailable")

    def read(self):
        if not self.query or self.dll.PdhCollectQueryData(self.query):
            return None
        size, count = W.DWORD(), W.DWORD()
        status = self.dll.PdhGetFormattedCounterArrayW(self.counter, 0x400,
                                                        C.byref(size), C.byref(count), None)
        if status != 0x800007D2 or not size.value or size.value > 32 * 2**20:
            return None
        buf = C.create_string_buffer(size.value)
        if self.dll.PdhGetFormattedCounterArrayW(self.counter, 0x400, C.byref(size),
                                                 C.byref(count), buf):
            return None
        items = C.cast(buf, C.POINTER(_Item))
        values = [items[i].value.value for i in range(count.value)
                  if items[i].name and items[i].name.lower().startswith(self.prefix)
                  and items[i].value.status in (0, 1) and items[i].value.value >= 0]
        return sum(values) if values else None

    def close(self):
        if self.query:
            self.dll.PdhCloseQuery(self.query)
            self.query = C.c_void_p()
