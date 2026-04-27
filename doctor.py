import os
import subprocess
import time
import shutil
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class SystemDoctor:
    """
    Anthropic-Grade Diagnostic Engine (F-44).
    Ported from utils/doctorDiagnostic.ts.
    
    Audits the environment for binary dependencies, network health, 
    and path misconfigurations.
    """
    
    @staticmethod
    def audit() -> Dict[str, Any]:
        results = {
            "binaries": {},
            "environment": {},
            "workspace": [],
            "network": {},
            "status": "PASS"
        }
        
        # 🛡️ 1. Binary Dependency Probe
        dependencies = ["git", "rg", "python", "npm"]
        for dep in dependencies:
            path = shutil.which(dep)
            results["binaries"][dep] = {
                "available": path is not None,
                "path": path or "NOT FOUND"
            }
            if not path:
                results["status"] = "WARNING"

        # 🚀 2. PATH & Windows OS Check (doctorDiagnostic.ts:374)
        path_var = os.environ.get("PATH", "")
        results["environment"] = {
            "os": os.name,
            "path_length": len(path_var),
            "cwd": os.getcwd(),
            "temp_dir": os.environ.get("TEMP", "C:\\Temp")
        }

        # 🌊 3. Network Latency Probe
        # Simple probe to identify connection bottlenecks
        try:
            start = time.time()
            # Probing a stable endpoint
            subprocess.run(["ping", "-n", "1", "8.8.8.8"], 
                          capture_output=True, timeout=2)
            results["network"]["latency_ms"] = round((time.time() - start) * 1000, 2)
        except:
            results["network"]["latency_ms"] = "TIMEOUT"
            results["status"] = "WARNING"

        # ☣️ 4. Toxic Workspace Pattern Detector
        # Scans for huge files that might accidentally bloat context
        root = os.getcwd()
        for f in os.listdir(root):
            fpath = os.path.join(root, f)
            if os.path.isfile(fpath) and os.path.getsize(fpath) > 10 * 1024 * 1024:
                results["workspace"].append(f"WARNING: Large file '{f}' detected (>10MB). Ensure it is in .gitignore.")
                results["status"] = "WARNING"

        return results
