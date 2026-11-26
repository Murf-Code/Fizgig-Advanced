"""Best-effort WDDM per-process GPU memory sampling; never substitute zero for missing."""
import csv
import os
import re
import subprocess
import sys
import threading


def counters(header,pid):
    pattern=re.compile(r"\(pid_"+str(pid)+r"_",re.I)
    return {i:("dedicated" if p.lower().endswith("\\dedicated usage") else "shared")
            for i,p in enumerate(header) if pattern.search(p) and
            p.lower().endswith(("\\dedicated usage","\\shared usage"))}


def values(row,cols):
    out={}
    for i,kind in cols.items():
        try: x=float(row[i])
        except (IndexError,ValueError): continue
        if x>=0 and x<float("inf"):
            out[kind]=out.get(kind,0)+int(x)
    return out


class WddmMemory:
    def __init__(self,pid=None):
        self.pid=pid or os.getpid()
        self.proc=None; self.thread=None
        self.result={"status":"unavailable","samples":0,"peak_dedicated_bytes":None,"peak_shared_bytes":None}
        if sys.platform!="win32":
            self.result["reason"]="Windows-only GPU Process Memory counters"
            return
        try:
            self.proc=subprocess.Popen(["typeperf",r"\GPU Process Memory(*)\Dedicated Usage",
                       r"\GPU Process Memory(*)\Shared Usage","-si","1"],
                       stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,errors="replace",
                       creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            self.thread=threading.Thread(target=self.read,daemon=True)
            self.thread.start()
        except OSError as exc:
            self.result["reason"]=str(exc)

    def read(self):
        cols={}
        try:
            for row in csv.reader(self.proc.stdout):
                if not row: continue
                if any("gpu process memory" in x.lower() for x in row):
                    cols=counters(row,self.pid)
                    continue
                v=values(row,cols)
                if not v: continue
                self.result["status"]="sampled"
                self.result["samples"]+=1
                for kind,num in v.items():
                    key=f"peak_{kind}_bytes"
                    self.result[key]=max(self.result[key] or 0,num)
        except Exception as exc:
            self.result["reason"]=str(exc)

    def stop(self):
        if self.proc:
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait()
        if self.thread: self.thread.join(timeout=3)
        if not self.result["samples"]:
            self.result.setdefault("reason","counter unavailable, localized, or process too short")
        return dict(self.result)
