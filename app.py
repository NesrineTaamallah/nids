#!/usr/bin/env python3
"""
NetGuard - Network Intrusion Detection Platform
Backend API — Dual Pipeline:
  Pipeline A (Supervised XGBoost):
    - best_binary_model.pkl         (Normal vs Attack)
    - scaler_binary.pkl
    - powertransformer_binary.pkl   (optional)
    - xgb_hierarchical_multiclass.pkl
    - scaler_hierarchical.pkl
    - powertransformer_hierarchical.pkl (optional)
    - label_encoder_hierarchical.pkl
    - metadata.json

  Pipeline B (Unsupervised KitNET):
    - kitsune_mirai_model.pkl       (pre-trained on Mirai dataset)
    - KitNET-py repo must be cloned alongside this file

FIXES APPLIED:
  1. /api/interfaces — parsing robuste + stderr fallback + tshark_found flag
  2. _process_batch  — kitnet_phase + kitnet_progress envoyés dans chaque event SSE
  3. kitnet_threshold — valeur partielle envoyée pendant warmup (non null)
"""

import os
import sys
import io
import json
import time
import pickle
import threading
import subprocess
import queue
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import deque
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request, Response, stream_with_context, send_from_directory
from flask_cors import CORS
import joblib

# ─── Flask App ────────────────────────────────────────────────────────────────
STATIC_DIR = Path(os.environ.get('STATIC_DIR', str(Path(__file__).parent)))
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/*": {"origins": "*"}})

# ─── Model Directory ──────────────────────────────────────────────────────────
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))

# ─── UNSW-NB15 Attack Categories ─────────────────────────────────────────────
ATTACK_CATEGORIES = [
    'Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS',
    'Exploits', 'Generic', 'Reconnaissance', 'Shellcode', 'Worms',
    'Backdoor', 'Rare_Attack_Worms_or_Shellcode'
]

ATTACK_COLORS = {
    'Normal':                       '#22c55e',
    'Fuzzers':                      '#f97316',
    'Analysis':                     '#3b82f6',
    'Backdoors':                    '#a855f7',
    'Backdoor':                     '#a855f7',
    'DoS':                          '#ef4444',
    'Exploits':                     '#ec4899',
    'Generic':                      '#eab308',
    'Reconnaissance':               '#06b6d4',
    'Shellcode':                    '#f43f5e',
    'Worms':                        '#84cc16',
    'Rare_Attack_Worms_or_Shellcode': '#ff6b35',
    'Zero-Day / Anomaly':           '#f59e0b',
}

# ─── UNSW-NB15 Raw CSV Columns ────────────────────────────────────────────────
UNSW_RAW_COLS = [
    'srcip','sport','dstip','dsport','proto','state','dur','sbytes','dbytes',
    'sttl','dttl','sloss','dloss','service','Sload','Dload','Spkts','Dpkts',
    'smeansz','dmeansz','trans_depth','res_bdy_len','Sjit','Djit','Sintpkt',
    'Dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl',
    'ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst',
    'ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm',
    'ct_dst_src_ltm','attack_cat','label'
]

LOG_COLS = [
    'dur','sbytes','dbytes','sttl','dttl','sloss','dloss','Sload','Dload',
    'Spkts','Dpkts','smeansz','dmeansz','trans_depth','res_bdy_len',
    'Sjit','Djit','Sintpkt','Dintpkt','tcprtt','synack','ackdat',
    'ct_state_ttl','ct_flw_http_mthd','ct_srv_src','ct_srv_dst',
    'ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm','ct_dst_sport_ltm','ct_dst_src_ltm',
]

STATE_CATS   = ['ACC','CLO','CON','ECO','ECR','FIN','INT','MAS','PAR','REQ','RST','TST','TXD','URH','URN','no']
SERVICE_CATS = ['Unknown','dhcp','dns','ftp','ftp-data','http','irc','pop3','radius','smtp','snmp','ssh','ssl']


# ═══════════════════════════════════════════════════════════════════════════════
# KITNET ENGINE  (Pipeline B — Unsupervised Zero-Day Detection)
# ═══════════════════════════════════════════════════════════════════════════════

# Numpy compat patch (KitNET was written for older numpy)
for _attr, _val in [('Inf', np.inf), ('Infinity', np.inf), ('NaN', np.nan),
                    ('bool', bool), ('int', int), ('float', float)]:
    if not hasattr(np, _attr):
        setattr(np, _attr, _val)

# Locate KitNET-py repo (cloned alongside app.py)
_PROJECT_ROOT = Path(__file__).parent
_KITNET_CANDIDATES = [
    _PROJECT_ROOT / "KitNET-py",
    _PROJECT_ROOT / "Kitsune-py",
    _PROJECT_ROOT / "kitsune-py",
    _PROJECT_ROOT / "kitnet-py",
]
for _c in _KITNET_CANDIDATES:
    if _c.exists():
        sys.path.insert(0, str(_c))
        print(f"[KitNET] Found repo at: {_c}", flush=True)
        break

KITNET_AVAILABLE = False
kit = None

def _try_import_kitnet():
    """Try every known way KitNET-py may be structured."""
    global kit, KITNET_AVAILABLE

    # ── Strategy 1: plain top-level import ──────────────────────────────────
    try:
        import KitNET as _kit
        kit = _kit
        KITNET_AVAILABLE = True
        print("[KitNET] ✓ KitNET loaded via 'import KitNET'", flush=True)
        return
    except ImportError:
        pass

    # ── Strategy 2: the repo may expose KitNET as a package with __init__ ──
    try:
        import importlib, types
        for _c in _KITNET_CANDIDATES:
            for sub in ['KitNET', 'KitNET_py', 'kitsune', 'Kitsune']:
                init_file = _c / sub / '__init__.py'
                mod_file  = _c / (sub + '.py')
                if init_file.exists():
                    spec = importlib.util.spec_from_file_location("KitNET", str(init_file))
                    _kit = importlib.util.module_from_spec(spec)
                    sys.modules['KitNET'] = _kit
                    spec.loader.exec_module(_kit)
                    kit = _kit
                    KITNET_AVAILABLE = True
                    print(f"[KitNET] ✓ KitNET loaded via package __init__ at {init_file}", flush=True)
                    return
                if mod_file.exists():
                    spec = importlib.util.spec_from_file_location("KitNET", str(mod_file))
                    _kit = importlib.util.module_from_spec(spec)
                    sys.modules['KitNET'] = _kit
                    spec.loader.exec_module(_kit)
                    kit = _kit
                    KITNET_AVAILABLE = True
                    print(f"[KitNET] ✓ KitNET loaded via file {mod_file}", flush=True)
                    return
    except Exception as _e2:
        print(f"[KitNET] Strategy 2 error: {_e2}", flush=True)

    # ── Strategy 3: look for KitNET.py directly in any candidate folder ────
    import importlib.util
    for _c in _KITNET_CANDIDATES:
        if not _c.exists():
            continue
        for fname in ['KitNET.py', 'kitnet.py', 'Kitsune.py', 'kitsune.py']:
            fpath = _c / fname
            if fpath.exists():
                try:
                    spec = importlib.util.spec_from_file_location("KitNET", str(fpath))
                    _kit = importlib.util.module_from_spec(spec)
                    sys.modules['KitNET'] = _kit
                    spec.loader.exec_module(_kit)
                    kit = _kit
                    KITNET_AVAILABLE = True
                    print(f"[KitNET] ✓ KitNET loaded via direct file {fpath}", flush=True)
                    return
                except Exception as _e3:
                    print(f"[KitNET] Strategy 3 error on {fpath}: {_e3}", flush=True)

_try_import_kitnet()

# Patch sigmoid overflow (KitNET uses exp(-x) which overflows for large x)
def _patch_kitnet_sigmoid():
    for mod_name in ['utils', 'KitNET.utils', 'kitnet.utils']:
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, 'sigmoid'):
            def _safe_sigmoid(x):
                x_clipped = np.clip(x, -500.0, 500.0)
                with np.errstate(over='ignore', invalid='ignore'):
                    result = 1.0 / (1.0 + np.exp(-x_clipped))
                return np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)
            mod.sigmoid = _safe_sigmoid
            print(f"[KitNET] ✓ Sigmoid overflow patch applied on '{mod_name}'", flush=True)
            return True
    return False

if KITNET_AVAILABLE:
    _patch_kitnet_sigmoid()

RMSE_MAX_SANE               = 100.0
KITNET_WARMUP_PKTS          = 500
KITNET_THRESHOLD_PERCENTILE = 99
KITNET_SAFETY_FACTOR        = 2.0
# Fixed threshold pre-computed from offline analysis (set None to use dynamic calibration)
KITNET_FIXED_THRESHOLD      = 0.046556
# ── KitsuneNetStat — AfterImage wrapper using the real netStat.py ────────────
# Requires the KitNET-py / Kitsune-py repo to be cloned alongside app.py.
# netStat.updateGetStats() produces 115 features per packet (5 lambdas × 23 stats)
# which is exactly what the pre-trained model was trained on.

class KitsuneNetStat:
    """
    Thin wrapper around netStat from the Kitsune repo.
    Call get_vector(pkt) for each raw packet → np.ndarray(115,).
    """

    def __init__(self):
        self._nstat     = None
        self._n_features = 0
        self._available  = False
        self._init()

    def _init(self):
        # netStat.py is importable because the KitNET-py folder was added to
        # sys.path above by _try_import_kitnet() / _KITNET_CANDIDATES loop.
        try:
            import netStat as _ns
            # Use unlimited host/session tables (same as FeatureExtractor.py)
            self._nstat = _ns.netStat(float('nan'), 100_000_000_000, 100_000_000_000)
            # Probe the feature count with a dummy packet
            import numpy as _np
            dummy = self._nstat.updateGetStats(
                0,
                "00:00:00:00:00:00", "00:00:00:00:00:00",
                "0.0.0.0", "0",
                "0.0.0.0", "0",
                0, 0.0
            )
            self._n_features = len(dummy)
            self._available  = True
            print(f"[AfterImage] ✓ netStat loaded — {self._n_features} features/packet", flush=True)
        except Exception as exc:
            print(f"[AfterImage] ✗ netStat not available: {exc}", flush=True)
            print("[AfterImage]   Make sure KitNET-py/Kitsune-py repo is cloned next to app.py", flush=True)

    # ------------------------------------------------------------------
    def get_vector(self, pkt: dict) -> "np.ndarray":
        """
        Convert one raw packet dict → AfterImage feature vector.

        Required pkt keys (from _extract_packet_fields / _parse_tshark_line):
            timestamp, src_ip, dst_ip, sport, dport, size, proto
        Optional:
            mac_src, mac_dst  (default to '00:…' if absent)
        """
        if not self._available:
            return np.zeros(max(self._n_features, 115), dtype=np.float64)

        ts      = float(pkt.get('timestamp', time.time()))
        size    = int(pkt.get('size', 0))
        src_ip  = str(pkt.get('src_ip',  '0.0.0.0'))
        dst_ip  = str(pkt.get('dst_ip',  '0.0.0.0'))
        sport   = str(pkt.get('sport',   0))
        dport   = str(pkt.get('dport',   0))
        proto   = str(pkt.get('proto',   'tcp')).lower()
        mac_src = str(pkt.get('mac_src', '00:00:00:00:00:00'))
        mac_dst = str(pkt.get('mac_dst', '00:00:00:00:00:00'))

        # IPtype: 0 = IPv4, 1 = IPv6, nan = L2 only
        ip_type = 1 if ':' in src_ip else 0

        # srcProtocol / dstProtocol — mirrors FeatureExtractor.py logic
        if proto in ('tcp', 'udp'):
            src_proto, dst_proto = sport, dport
        elif proto == 'arp':
            src_proto = dst_proto = 'arp'
        elif proto == 'icmp':
            src_proto = dst_proto = 'icmp'
        else:
            src_proto, dst_proto = sport, dport

        try:
            vec = self._nstat.updateGetStats(
                ip_type,
                mac_src, mac_dst,
                src_ip, src_proto,
                dst_ip, dst_proto,
                size, ts
            )
            arr = np.array(vec, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            return arr
        except Exception as exc:
            print(f"[AfterImage] get_vector error: {exc}", flush=True)
            return np.zeros(self._n_features or 115, dtype=np.float64)

    # ------------------------------------------------------------------
    def reset(self):
        """Reinitialise all incremental statistics (new capture session)."""
        self._nstat     = None
        self._available = False
        self._init()
        print("[AfterImage] Statistics reset for new capture session", flush=True)

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def available(self) -> bool:
        return self._available


# Singleton — one instance per process, reset on each capture session
after_image = KitsuneNetStat()


class _KitNETUnpickler(pickle.Unpickler):
    """
    Custom unpickler: resolves KitNET.KitNET → KitNET module at runtime.
    Also creates lightweight stub classes for any KitNET types that can't be
    resolved (so the PKL loads even when the full KitNET module is absent).
    """
    def find_class(self, module, name):
        if KITNET_AVAILABLE and kit is not None:
            if (module.startswith("KitNET") or module.startswith("kitsune")
                    or module.startswith("Kitsune")):
                if hasattr(kit, name):
                    return getattr(kit, name)
                for candidate in [module.split(".")[-1], "KitNET"]:
                    m = sys.modules.get(candidate)
                    if m and hasattr(m, name):
                        return getattr(m, name)
                try:
                    import importlib
                    m = importlib.import_module(module.split(".")[-1])
                    if hasattr(m, name):
                        return getattr(m, name)
                except Exception:
                    pass

        for alias in [module, module.split(".")[-1], "KitNET"]:
            m = sys.modules.get(alias)
            if m and hasattr(m, name):
                return getattr(m, name)

        if (module.startswith("KitNET") or module.startswith("kitsune")
                or module.startswith("Kitsune")):
            cache_key = f"__stub_{module}_{name}"
            if cache_key not in sys.modules:
                stub_cls = type(name, (object,), {
                    '__module__': module,
                    'process': lambda self, x: 0.0,
                    '__reduce__': lambda self: (object.__new__, (type(self),)),
                })
                stub_mod_name = module.split(".")[0]
                if stub_mod_name not in sys.modules:
                    import types as _types
                    stub_mod = _types.ModuleType(stub_mod_name)
                    sys.modules[stub_mod_name] = stub_mod
                setattr(sys.modules[stub_mod_name], name, stub_cls)
                sys.modules[cache_key] = stub_cls  # type: ignore
            return sys.modules[cache_key]  # type: ignore

        return super().find_class(module, name)


def _safe_pickle_load(path: str):
    with open(path, "rb") as f:
        data = f.read()
    return _KitNETUnpickler(io.BytesIO(data)).load()


def _sanitize_ae_weights(kitnet_obj):
    """Clip internal autoencoder weights to prevent inf/nan accumulation."""
    if kitnet_obj is None:
        return
    try:
        ae_list = None
        for attr in ('AD', 'ensembleLayer', 'outputAE', 'ADs'):
            if hasattr(kitnet_obj, attr):
                obj = getattr(kitnet_obj, attr)
                if isinstance(obj, list):
                    ae_list = obj
                    break

        def _clip_ae(ae):
            for w_attr in ('W', 'b', 'W_', 'b_', 'hbias', 'vbias'):
                if hasattr(ae, w_attr):
                    w = getattr(ae, w_attr)
                    if isinstance(w, np.ndarray):
                        clipped = np.clip(w, -10.0, 10.0)
                        clipped = np.nan_to_num(clipped, nan=0.0, posinf=10.0, neginf=-10.0)
                        setattr(ae, w_attr, clipped)

        if ae_list:
            for ae in ae_list:
                _clip_ae(ae)
            for out_attr in ('outputAE', 'output_ae'):
                if hasattr(kitnet_obj, out_attr):
                    _clip_ae(getattr(kitnet_obj, out_attr))
    except Exception:
        pass


class KitNETEngine:
    """
    Wraps the pre-trained KitNET/Kitsune model for online anomaly scoring.
    """

    def __init__(self, pretrained_path: str | None = None):
        self._kitnet       = None
        self._n            = None
        self._pretrained   = False
        self._mode         = "unavailable"
        # Use pre-computed fixed threshold if provided, skip warmup calibration
        if KITNET_FIXED_THRESHOLD is not None:
            self.threshold    = float(KITNET_FIXED_THRESHOLD)
            self.trained      = True
            self._warmup_done = True
        else:
            self.threshold    = float('inf')
            self.trained      = False
            self._warmup_done = False
        self.packet_count  = 0
        self.rmse_history: list[float] = []
        self._warmup_rmse: list[float] = []
        self._consecutive_errors = 0

        if pretrained_path and Path(pretrained_path).exists():
            ok = self._load_pretrained(pretrained_path)
            if ok:
                self._pretrained = True
                self._mode       = "pretrained"
                print(f"[KitNET] ✓ Pre-trained model loaded: {pretrained_path}", flush=True)
                print(f"[KitNET]   n_features={self._n}  warm-up={KITNET_WARMUP_PKTS} pkts", flush=True)
            else:
                print(f"[KitNET] ✗ Could not load pre-trained model from {pretrained_path}", flush=True)
        else:
            if pretrained_path:
                print(f"[KitNET] ✗ PKL not found: {pretrained_path}", flush=True)

    def _load_pretrained(self, path: str) -> bool:
        try:
            obj = _safe_pickle_load(path)
            print(f"[KitNET] PKL type: {type(obj).__name__}", flush=True)
        except Exception as e:
            print(f"[KitNET] PKL load error: {e}", flush=True)
            return False

        kitnet_obj = None
        n_features = None
        threshold  = None

        if isinstance(obj, dict):
            print(f"[KitNET] PKL keys: {list(obj.keys())}", flush=True)
            for k in ('model', 'kitnet', 'KitNET', 'kit', 'detector', 'engine'):
                if k in obj:
                    kitnet_obj = obj[k]
                    break
            for k in ('threshold', 'FPR', 'th', 'anomaly_threshold', 'rmse_threshold', 'thr'):
                if k in obj and isinstance(obj[k], (int, float)):
                    threshold = float(obj[k])
                    break
            for k in ('n_features', 'n', 'num_features', 'features'):
                if k in obj and isinstance(obj[k], int):
                    n_features = obj[k]
                    break
            if threshold is None:
                for k in ('RMSEs', 'rmse', 'rmse_history', 'benign_rmse'):
                    if k in obj:
                        arr = np.array(obj[k], dtype=float)
                        arr = arr[np.isfinite(arr) & (arr > 0) & (arr < RMSE_MAX_SANE)]
                        if len(arr) >= 10:
                            threshold = float(np.percentile(arr, 99)) * 2.0
                            print(f"[KitNET] Threshold recalculated P99×2: {threshold:.6f}", flush=True)
                            break
        elif KITNET_AVAILABLE and kit is not None and hasattr(kit, 'KitNET') and isinstance(obj, kit.KitNET):
            kitnet_obj = obj
        elif isinstance(obj, (list, tuple)) and len(obj) >= 2:
            kitnet_obj = obj[0]
            if isinstance(obj[1], (int, float)):
                threshold = float(obj[1])

        if kitnet_obj is None:
            if hasattr(obj, 'process'):
                kitnet_obj = obj
            else:
                return False

        if kitnet_obj is not None:
            if n_features is None:
                for attr in ('n', 'num_features', 'n_features', 'FM_n'):
                    if hasattr(kitnet_obj, attr):
                        v = getattr(kitnet_obj, attr)
                        if isinstance(v, int) and v > 0:
                            n_features = v
                            break
            if threshold is None:
                for attr in ('threshold', 'FPR', 'anomaly_threshold', '_threshold'):
                    if hasattr(kitnet_obj, attr):
                        v = getattr(kitnet_obj, attr)
                        if isinstance(v, (int, float)) and 0 < v < RMSE_MAX_SANE:
                            threshold = float(v)
                            break

        _sanitize_ae_weights(kitnet_obj)

        self._kitnet  = kitnet_obj
        self._n       = n_features
        # Use fixed threshold if configured, else warmup will calibrate
        if KITNET_FIXED_THRESHOLD is not None:
            self.threshold    = float(KITNET_FIXED_THRESHOLD)
            self.trained      = True
            self._warmup_done = True
        else:
            self.threshold = float('inf')   # will be re-calibrated during warm-up
            self.trained   = False
        print(f"[KitNET] n_features={n_features}  original_threshold={threshold}", flush=True)
        return True

    def _resize(self, vec: np.ndarray) -> np.ndarray:
        if self._n is None:
            return vec
        n = self._n
        if len(vec) == n:
            return vec
        if len(vec) > n:
            return vec[:n]
        out = np.zeros(n, dtype=np.float64)
        out[:len(vec)] = vec
        return out

    def process_vector(self, vec: np.ndarray) -> dict:
        """Score one feature vector through KitNET. Returns RMSE + anomaly flag."""
        if self._kitnet is None:
            return self._unavailable()
        if not KITNET_AVAILABLE and not hasattr(self._kitnet, 'process'):
            return self._unavailable()

        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        vec = np.clip(vec, -1e6, 1e6).astype(np.float64)
        if self._pretrained and self._n is not None:
            vec = self._resize(vec)

        with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
            try:
                rmse = float(self._kitnet.process(vec))
                self._consecutive_errors = 0
            except Exception as e:
                self._consecutive_errors += 1
                if self._consecutive_errors % 100 == 1:
                    print(f"[KitNET] process() error #{self._consecutive_errors}: {e}", flush=True)
                return self._unavailable()

        if not np.isfinite(rmse) or rmse < 0:
            rmse = 0.0
        rmse = min(rmse, RMSE_MAX_SANE)

        self.rmse_history.append(rmse)
        if len(self.rmse_history) > 5000:
            self.rmse_history = self.rmse_history[-5000:]
        self.packet_count += 1

        # Warm-up phase: calibrate threshold from live traffic
        if not self._warmup_done:
            if 0 < rmse < RMSE_MAX_SANE:
                self._warmup_rmse.append(rmse)
            progress = min(len(self._warmup_rmse) / KITNET_WARMUP_PKTS, 1.0)
            if len(self._warmup_rmse) >= KITNET_WARMUP_PKTS:
                arr = np.array(self._warmup_rmse)
                arr = arr[arr < np.percentile(arr, 99.5)]
                p_val = float(np.percentile(arr, KITNET_THRESHOLD_PERCENTILE))
                self.threshold  = p_val * KITNET_SAFETY_FACTOR
                self._warmup_done = True
                self.trained      = True
                print(f"[KitNET] ✓ Warm-up done — threshold={self.threshold:.6f} "
                      f"(P{KITNET_THRESHOLD_PERCENTILE}={p_val:.6f} × {KITNET_SAFETY_FACTOR})",
                      flush=True)
            # FIX: during warmup, send partial threshold estimate so frontend can display it
            partial_threshold = None
            if len(self._warmup_rmse) >= 10:
                arr_partial = np.array(self._warmup_rmse)
                partial_threshold = round(
                    float(np.percentile(arr_partial, KITNET_THRESHOLD_PERCENTILE)) * KITNET_SAFETY_FACTOR, 6
                )
            return {
                "rmse":              round(rmse, 6),
                "is_anomaly":        False,
                "phase":             "warmup",
                "progress":          round(progress, 4),
                "threshold":         partial_threshold,   # partial estimate, may be None
                "severity_score":    0.0,
                "trained":           False,
                "mode":              self._mode,
                "packet_count":      self.packet_count,
            }

        is_anomaly = rmse > self.threshold
        sev_score  = rmse / max(self.threshold, 1e-9)
        return {
            "rmse":           round(rmse, 6),
            "is_anomaly":     is_anomaly,
            "phase":          "monitoring",
            "progress":       1.0,
            "threshold":      round(self.threshold, 6),
            "severity_score": round(min(sev_score, 10.0), 3),
            "trained":        True,
            "mode":           self._mode,
            "packet_count":   self.packet_count,
        }

    def process_packet(self, pkt: dict) -> dict:
        """
        PRIMARY call path for live capture.
        pkt = raw packet dict from _extract_packet_fields() or _parse_tshark_line().
        Feeds through AfterImage (netStat) → KitNET.process().
        """
        vec = after_image.get_vector(pkt)
        return self.process_vector(vec)

    def process_features(self, features: dict) -> dict:
        """
        LEGACY path for CSV analysis (/api/analyze/csv).
        Reconstructs a minimal packet dict so AfterImage can still run.
        """
        pkt = {
            'timestamp': time.time(),
            'src_ip':    str(features.get('srcip',  features.get('_src_ip', '0.0.0.0'))),
            'dst_ip':    str(features.get('dstip',  features.get('_dst_ip', '0.0.0.0'))),
            'sport':     str(features.get('sport',  features.get('_sport',  0))),
            'dport':     str(features.get('dsport', features.get('_dport',  0))),
            'proto':     str(features.get('proto',  features.get('_proto',  'tcp'))).lower(),
            'size':      int(features.get('sbytes', 0)),
        }
        return self.process_packet(pkt)

    def _unavailable(self) -> dict:
        return {
            "rmse": 0.0, "is_anomaly": False, "phase": "unavailable",
            "progress": 0.0, "threshold": None, "severity_score": 0.0,
            "trained": False, "mode": "unavailable", "packet_count": self.packet_count,
        }

    @property
    def n_features(self):
        return self._n

    @property
    def status(self) -> dict:
        _functional = KITNET_AVAILABLE and self._kitnet is not None
        _stub_loaded = (not KITNET_AVAILABLE) and self._kitnet is not None and hasattr(self._kitnet, 'process')
        return {
            "available":    _functional or _stub_loaded,
            "pretrained":   self._pretrained,
            "mode":         self._mode,
            "trained":      self.trained,
            "warmup_done":  self._warmup_done,
            "threshold":    round(self.threshold, 6) if self.threshold != float('inf') else None,
            "packet_count": self.packet_count,
            "n_features":   self._n,
            "kitnet_module": KITNET_AVAILABLE,
            "stub_mode":    _stub_loaded,
            "progress":     1.0 if self._warmup_done else round(
                            len(self._warmup_rmse) / KITNET_WARMUP_PKTS, 4),
        }


# ─── Load KitNET ──────────────────────────────────────────────────────────────
_KITNET_PKL_PATH = str(MODEL_DIR / "kitsune_mirai_model.pkl")
_KITNET_ALT_PATH = r"C:\cybernew\models\kitsune_mirai_model.pkl"
if not Path(_KITNET_PKL_PATH).exists() and Path(_KITNET_ALT_PATH).exists():
    _KITNET_PKL_PATH = _KITNET_ALT_PATH

kitnet_engine = KitNETEngine(pretrained_path=_KITNET_PKL_PATH)
KITNET_READY  = (kitnet_engine._kitnet is not None)

_rmse_series: deque = deque(maxlen=300)
_kitnet_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# XGBOOST PIPELINE  (Pipeline A — Supervised Detection)
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_unsw_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    first_col = str(df.columns[0])
    looks_like_ip  = bool(__import__('re').match(r'\d+\.\d+\.\d+\.\d+', first_col))
    looks_like_num = first_col.replace('.','',1).isdigit()
    if looks_like_ip or (looks_like_num and len(df.columns) >= 40):
        vals = [first_col] + list(df.columns[1:])
        first_row = pd.DataFrame([vals], columns=range(len(vals)))
        rest = df.copy()
        rest.columns = range(len(rest.columns))
        df = pd.concat([first_row, rest], ignore_index=True)
        n = min(len(UNSW_RAW_COLS), len(df.columns))
        df = df.iloc[:, :n]
        df.columns = UNSW_RAW_COLS[:n]

    for c in df.columns:
        if c not in ('srcip','dstip','proto','state','service','attack_cat'):
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    if 'sport' in df.columns:
        freq = df['sport'].value_counts(normalize=True)
        df['sport_freq'] = df['sport'].map(freq).fillna(0)
    else:
        df['sport_freq'] = 0.0

    if 'dsport' in df.columns:
        freq = df['dsport'].value_counts(normalize=True)
        df['dsport_freq'] = df['dsport'].map(freq).fillna(0)
    else:
        df['dsport_freq'] = 0.0

    if 'proto' in df.columns:
        freq = df['proto'].value_counts(normalize=True)
        df['proto_freq'] = df['proto'].map(freq).fillna(0)
    else:
        df['proto_freq'] = 0.0

    for col in ('swin','dwin','stcpb','dtcpb'):
        if col not in df.columns:
            df[col] = 0

    for col in ('is_sm_ips_ports','is_ftp_login','ct_ftp_cmd'):
        if col not in df.columns:
            df[col] = 0

    for col in LOG_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
        else:
            df[f'log_{col}'] = 0.0

    if 'log_ct_src_ltm' in df.columns and 'log_ct_src_ ltm' not in df.columns:
        df['log_ct_src_ ltm'] = df['log_ct_src_ltm']

    state_col = df['state'].astype(str).str.strip() if 'state' in df.columns else pd.Series(['no']*len(df))
    for s in STATE_CATS:
        df[f'state_{s}'] = (state_col == s).astype(int)

    svc_col = df['service'].astype(str).str.strip() if 'service' in df.columns else pd.Series(['Unknown']*len(df))
    svc_col = svc_col.replace({'-': 'Unknown', '': 'Unknown', 'nan': 'Unknown'})
    for s in SERVICE_CATS:
        df[f'service_{s}'] = (svc_col == s).astype(int)

    return df


def _normalise_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    return _preprocess_unsw_csv(df)


print(f"🔍  Loading models from {MODEL_DIR} …")

def _load(name, required=True):
    p = MODEL_DIR / name
    if p.exists():
        obj = joblib.load(p)
        print(f"   ✓  {name}")
        return obj
    if required:
        print(f"   ✗  {name} NOT FOUND — will use stub")
    return None

meta_path = MODEL_DIR / "metadata.json"
if meta_path.exists():
    with open(meta_path) as f:
        META = json.load(f)
    FEATURE_COLS    = META.get("feature_cols", [])
    HIGH_SKEW_FEATS = META.get("high_skew_feats", [])
    ATTACK_CLASSES  = META.get("attack_classes", ATTACK_CATEGORIES[1:])
    print(f"   ✓  metadata.json  ({len(FEATURE_COLS)} features, {len(ATTACK_CLASSES)} attack classes)")
else:
    print("   ✗  metadata.json NOT FOUND — using default UNSW-NB15 columns")
    FEATURE_COLS = [
        'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
        'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
        'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len',
        'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat',
        'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst',
        'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
        'ct_dst_src_ltm'
    ]
    HIGH_SKEW_FEATS = []
    ATTACK_CLASSES  = ATTACK_CATEGORIES[1:]

bin_model  = _load("best_binary_model.pkl")
bin_scaler = _load("scaler_binary.pkl")
bin_pt     = _load("powertransformer_binary.pkl", required=False)
mul_model  = _load("xgb_hierarchical_multiclass.pkl")
mul_scaler = _load("scaler_hierarchical.pkl")
mul_pt     = _load("powertransformer_hierarchical.pkl", required=False)
mul_le     = _load("label_encoder_hierarchical.pkl")

MODELS_READY = bin_model is not None and mul_model is not None
print(f"\n{'✅  XGBoost models loaded' if MODELS_READY else '⚠️  Using stub detector (XGBoost models missing)'}")
print(f"{'✅  KitNET engine ready' if KITNET_READY else '⚠️  KitNET not available'}\n")


# ─── XGBoost Detector ─────────────────────────────────────────────────────────
class RealDetector:
    def _preprocess(self, df, scaler, pt, feat_cols) -> np.ndarray:
        X = df.reindex(columns=feat_cols, fill_value=0).fillna(0).astype(float)
        zero_cols = [c for c in feat_cols if X[c].sum() == 0]
        if len(zero_cols) > len(feat_cols) * 0.30:
            print(f"[WARN] {len(zero_cols)}/{len(feat_cols)} model features all-zero. Missing: {zero_cols[:8]}", flush=True)
        if pt is not None and HIGH_SKEW_FEATS:
            skew_cols = [c for c in HIGH_SKEW_FEATS if c in X.columns]
            if skew_cols:
                X[skew_cols] = pt.transform(X[skew_cols])
        if scaler is not None:
            X = pd.DataFrame(scaler.transform(X), columns=feat_cols)
        return X.values

    def predict(self, df):
        n = len(df)
        if bin_model is not None:
            X_bin  = self._preprocess(df, bin_scaler, bin_pt, FEATURE_COLS)
            labels = bin_model.predict(X_bin).astype(int)
            if hasattr(bin_model, 'predict_proba'):
                probas = bin_model.predict_proba(X_bin)
            else:
                score  = bin_model.decision_function(X_bin)
                p_atk  = 1 / (1 + np.exp(-score))
                probas = np.column_stack([1 - p_atk, p_atk])
        else:
            rng    = np.random.default_rng()
            labels = (rng.random(n) < 0.18).astype(int)
            p_atk  = rng.uniform(0.05, 0.95, n)
            probas = np.column_stack([1 - p_atk, p_atk])

        categories = ['Normal'] * n
        atk_idx    = np.where(labels == 1)[0]

        if len(atk_idx) > 0 and mul_model is not None:
            df_atk = df.iloc[atk_idx]
            X_mul  = self._preprocess(df_atk, mul_scaler, mul_pt, FEATURE_COLS)
            preds  = mul_model.predict(X_mul)
            cat_names = mul_le.inverse_transform(preds) if mul_le else [str(p) for p in preds]
            for i, idx in enumerate(atk_idx):
                categories[idx] = cat_names[i]
        elif len(atk_idx) > 0:
            rng = np.random.default_rng()
            for idx in atk_idx:
                categories[idx] = rng.choice(ATTACK_CLASSES)

        return labels, probas, categories


class StubDetector:
    def __init__(self):
        self.rng = np.random.default_rng()

    def _heuristic_label(self, row):
        dur      = float(row.get('dur', 1))
        sbytes   = float(row.get('sbytes', 0))
        spkts    = float(row.get('Spkts', row.get('spkts', 0)))
        dpkts    = float(row.get('Dpkts', row.get('dpkts', 0)))
        dport    = int(row.get('dsport', row.get('dport', 0)))
        sload    = float(row.get('Sload', 0))
        smeansz  = float(row.get('smeansz', 500))
        sintpkt  = float(row.get('Sintpkt', 100))
        sjit     = float(row.get('Sjit', 0))
        total_pkts = spkts + dpkts

        if sintpkt < 10 and smeansz < 150 and total_pkts > 15:
            return 1, min(0.95, 0.70 + total_pkts / 200), 'DoS'
        if sload > 500000 and smeansz < 200:
            return 1, 0.82, 'DoS'
        if dur < 0.1 and sbytes < 200 and dport not in (80, 443, 22, 53, 25, 21):
            return 1, 0.75, 'Reconnaissance'
        if sjit > 200 and total_pkts > 5:
            return 1, 0.68, 'Fuzzers'
        if dport not in (80, 443, 22, 53, 25, 21, 8080, 3306) and sbytes > 5000 and dur < 2:
            return 1, 0.71, 'Exploits'
        if spkts > 20 and dpkts == 0 and sbytes > 2000:
            return 1, min(0.90, 0.60 + spkts / 100), 'Generic'
        if self.rng.random() < 0.05:
            cat = str(self.rng.choice(['Backdoors', 'Analysis', 'Shellcode']))
            return 1, float(self.rng.uniform(0.55, 0.70)), cat
        return 0, float(self.rng.uniform(0.60, 0.95)), 'Normal'

    def predict(self, df):
        n = len(df)
        labels = np.zeros(n, dtype=int)
        cats   = ['Normal'] * n
        probas = np.zeros((n, 2))
        for i, (_, row) in enumerate(df.iterrows()):
            lbl, conf, cat = self._heuristic_label(row)
            labels[i] = lbl
            cats[i]   = cat
            probas[i] = [1 - conf, conf] if lbl else [conf, 1 - conf]
        return labels, probas, cats


detector = RealDetector() if MODELS_READY else StubDetector()


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════════

capture_state = {
    "running": False, "packets_captured": 0,
    "flows_analyzed": 0, "attacks_detected": 0,
    "anomalies_detected": 0,
    "interface": None, "start_time": None,
}
recent_alerts = []
stats_history = []
event_clients = []
capture_lock  = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════════
# FLOW FEATURE EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════════

class FlowFeatureExtractor:
    def __init__(self, window=100):
        self.flows        = {}
        self.window       = window
        self.recent_conns = []
        self._last_flush  = time.time()

    def _flow_key(self, pkt):
        src, dst = pkt.get('src_ip', '?'), pkt.get('dst_ip', '?')
        sp,  dp  = pkt.get('sport', 0),    pkt.get('dport', 0)
        proto    = pkt.get('proto', '?')
        if (src, sp) > (dst, dp):
            return (dst, src, dp, sp, proto)
        return (src, dst, sp, dp, proto)

    def _flush_stale(self):
        now = time.time()
        if now - self._last_flush < 1.0:
            return []
        self._last_flush = now
        ready  = []
        remove = []
        for key, f in list(self.flows.items()):
            if now - f['last_time'] > 1.0 and (f['spkts'] + f['dpkts']) >= 1:
                result = self._extract(key, f)
                if result: ready.append(result)
                remove.append(key)
        for key in remove:
            del self.flows[key]
        return ready

    def update(self, pkt):
        key = self._flow_key(pkt)
        now = pkt.get('timestamp', time.time())

        if key not in self.flows:
            canon_src, canon_dst, canon_sp, canon_dp = key[0], key[1], key[2], key[3]
            self.flows[key] = {
                'start_time': now, 'last_time': now,
                'sbytes': 0, 'dbytes': 0, 'spkts': 0, 'dpkts': 0,
                'sloss': 0, 'dloss': 0,
                'sttl': pkt.get('ttl', 64), 'dttl': 64,
                'swin': pkt.get('tcp_win', 0), 'dwin': 0,
                'stcpb': 0, 'dtcpb': 0,
                'sport_pkt_sizes': [], 'dport_pkt_sizes': [],
                'sport_iats': [], 'dport_iats': [],
                'last_sport_time': now, 'last_dport_time': now,
                'syn_time': None, 'synack_time': None,
                'proto': pkt.get('proto', '?'),
                'service': pkt.get('service', '-'),
                'src_ip': canon_src, 'dst_ip': canon_dst,
                'sport': canon_sp, 'dport': canon_dp,
            }

        f    = self.flows[key]
        size = pkt.get('size', 0)
        is_src = (pkt.get('src_ip', '?') == f['src_ip'])

        if is_src:
            f['sbytes'] += size; f['spkts'] += 1
            iat = now - f['last_sport_time']
            if iat > 0: f['sport_iats'].append(iat * 1000)
            f['last_sport_time'] = now
            f['sport_pkt_sizes'].append(size)
        else:
            f['dbytes'] += size; f['dpkts'] += 1
            iat = now - f['last_dport_time']
            if iat > 0: f['dport_iats'].append(iat * 1000)
            f['last_dport_time'] = now
            f['dport_pkt_sizes'].append(size)

        flags = pkt.get('tcp_flags', '')
        if 'SYN' in flags and 'ACK' not in flags: f['syn_time']    = now
        elif 'SYN' in flags and 'ACK' in flags:   f['synack_time'] = now
        f['last_time'] = now

        total_pkts = f['spkts'] + f['dpkts']
        last_emit  = f.get('_last_emit', 0)
        dur        = f['last_time'] - f['start_time']
        new_pkts   = total_pkts - last_emit
        bidir = f['spkts'] > 0 and f['dpkts'] > 0

        if bidir and total_pkts >= 2 and new_pkts >= 1:
            f['_last_emit'] = total_pkts
            return self._extract(key, f)
        if not bidir and total_pkts >= 2 and new_pkts >= 1:
            f['_last_emit'] = total_pkts
            return self._extract(key, f)
        if dur >= 0.8 and new_pkts >= 1:
            f['_last_emit'] = total_pkts
            return self._extract(key, f)
        return None

    def _sm(self, lst): return float(np.mean(lst)) if lst else 0.0
    def _sj(self, lst):
        if len(lst) < 2: return 0.0
        return float(np.mean([abs(lst[i]-lst[i-1]) for i in range(1, len(lst))]))

    def _ct(self, src, dst, sport, dport, svc):
        rc = self.recent_conns[-self.window:]
        return (
            sum(1 for r in rc if r['service'] == svc and r['src_ip'] == src),
            sum(1 for r in rc if r['service'] == svc and r['dst_ip'] == dst),
            sum(1 for r in rc if r['dst_ip'] == dst),
            sum(1 for r in rc if r['src_ip'] == src),
            sum(1 for r in rc if r['src_ip'] == src and r['dport'] == dport),
            sum(1 for r in rc if r['dst_ip'] == dst and r['sport'] == sport),
            sum(1 for r in rc if r['src_ip'] == src and r['dst_ip'] == dst),
        )

    def _extract(self, key, f):
        dur    = max(f['last_time'] - f['start_time'], 1e-6)
        sload  = (f['sbytes'] * 8) / dur
        dload  = (f['dbytes'] * 8) / dur
        synack = (f['synack_time'] - f['syn_time']) if (f['syn_time'] and f['synack_time']) else 0
        src_ip, dst_ip, sport, dport, proto = key
        svc  = f.get('service', '-')
        ct   = self._ct(src_ip, dst_ip, sport, dport, svc)

        self.recent_conns.append({'src_ip': src_ip, 'dst_ip': dst_ip,
                                   'sport': sport, 'dport': dport, 'service': svc})
        if len(self.recent_conns) > self.window * 2:
            self.recent_conns = self.recent_conns[-self.window:]

        if proto == 'tcp':
            state = 'CON' if (f.get('syn_time') and f.get('synack_time')) else \
                    ('REQ' if f.get('syn_time') else 'CON')
        elif proto == 'udp':
            state = 'INT'
        else:
            state = 'ECO'

        return {
            'dur': dur, 'sbytes': f['sbytes'], 'dbytes': f['dbytes'],
            'sttl': f['sttl'], 'dttl': f['dttl'], 'sloss': f['sloss'], 'dloss': f['dloss'],
            'Sload': sload, 'Dload': dload, 'Spkts': f['spkts'], 'Dpkts': f['dpkts'],
            'swin': f['swin'], 'dwin': f['dwin'], 'stcpb': f['stcpb'], 'dtcpb': f['dtcpb'],
            'smeansz': self._sm(f['sport_pkt_sizes']), 'dmeansz': self._sm(f['dport_pkt_sizes']),
            'trans_depth': 0, 'res_bdy_len': 0,
            'Sjit': self._sj(f['sport_iats']), 'Djit': self._sj(f['dport_iats']),
            'Sintpkt': self._sm(f['sport_iats']), 'Dintpkt': self._sm(f['dport_iats']),
            'tcprtt': synack, 'synack': synack, 'ackdat': 0,
            'ct_state_ttl': 0, 'ct_flw_http_mthd': 0,
            'ct_srv_src': ct[0], 'ct_srv_dst': ct[1], 'ct_dst_ltm': ct[2],
            'ct_src_ltm': ct[3], 'ct_src_dport_ltm': ct[4],
            'ct_dst_sport_ltm': ct[5], 'ct_dst_src_ltm': ct[6],
            'is_sm_ips_ports': int(src_ip == dst_ip and sport == dport),
            'is_ftp_login': int(dport == 21 or sport == 21),
            'ct_ftp_cmd': 0,
            'proto': proto, 'state': state,
            'service': svc if svc not in ('-', '') else 'Unknown',
            'sport': sport, 'dsport': dport,
            '_src_ip': src_ip, '_dst_ip': dst_ip,
            '_sport': sport, '_dport': dport, '_proto': proto, '_service': svc,
        }


extractor = FlowFeatureExtractor()


# ═══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING & BATCH PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess_live(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ('sport', 'dsport'):
        if col in df.columns:
            freq = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq).fillna(0)
        else:
            df[f'{col}_freq'] = 0.0

    if 'proto' in df.columns:
        freq = df['proto'].value_counts(normalize=True)
        df['proto_freq'] = df['proto'].map(freq).fillna(0)
    else:
        df['proto_freq'] = 0.0

    for col in LOG_COLS:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].clip(lower=0))
        else:
            df[f'log_{col}'] = 0.0

    if 'log_ct_src_ltm' in df.columns and 'log_ct_src_ ltm' not in df.columns:
        df['log_ct_src_ ltm'] = df['log_ct_src_ltm']

    state_col = df['state'].astype(str).str.strip() if 'state' in df.columns else pd.Series(['CON']*len(df))
    for s in STATE_CATS:
        df[f'state_{s}'] = (state_col == s).astype(int)

    svc_col = df['service'].astype(str).str.strip() if 'service' in df.columns else pd.Series(['Unknown']*len(df))
    svc_col = svc_col.replace({'-': 'Unknown', '': 'Unknown', 'nan': 'Unknown'})
    for s in SERVICE_CATS:
        df[f'service_{s}'] = (svc_col == s).astype(int)

    return df


def _push_stats():
    now_str = datetime.now().isoformat()
    stats_history.append({
        "timestamp": now_str,
        "flows":     capture_state["flows_analyzed"],
        "attacks":   capture_state["attacks_detected"],
        "anomalies": capture_state["anomalies_detected"],
        "packets":   capture_state["packets_captured"],
    })
    if len(stats_history) > 300:
        stats_history.pop(0)
    _broadcast_event({
        "type":    "stats",
        "stats":   capture_state.copy(),
        "history": stats_history[-30:],
        "kitnet":  kitnet_engine.status,
        "rmse_series": list(_rmse_series)[-60:],
    })


_feature_debug_done = False


def _process_batch(batch: list):
    global _feature_debug_done
    df = pd.DataFrame(batch)
    df = _preprocess_live(df)

    if not _feature_debug_done and FEATURE_COLS:
        _feature_debug_done = True
        produced  = set(df.columns)
        expected  = set(FEATURE_COLS)
        missing   = expected - produced
        match_pct = round(len(expected & produced) / len(expected) * 100, 1)
        print(f"[FEATURE CHECK] match: {match_pct}%", flush=True)
        if missing:
            print(f"[FEATURE CHECK] MISSING ({len(missing)}): {sorted(missing)[:15]}...", flush=True)

    labels, probas, categories = detector.predict(df)

    capture_state["flows_analyzed"]  += len(batch)
    capture_state["attacks_detected"] += int(np.sum(labels))

    now_str = datetime.now().isoformat()
    iface_val   = capture_state.get("interface") or "simulation"
    is_simulated = str(iface_val).lower() in ("simulation", "sim", "", "none")

    for i, (row, lbl, cat) in enumerate(zip(batch, labels, categories)):
        # ── KitNET anomaly score for this flow ──────────────────────────
        # Use process_packet: row is the raw packet dict → AfterImage → KitNET
        kitnet_result  = kitnet_engine.process_packet(row)
        rmse           = kitnet_result.get("rmse", 0.0)
        is_anomaly     = kitnet_result.get("is_anomaly", False)
        sev_score      = kitnet_result.get("severity_score", 0.0)
        kitnet_trained = kitnet_result.get("trained", False)
        # FIX: always forward phase + progress so frontend can show warmup bar
        kitnet_phase    = kitnet_result.get("phase", "unavailable")
        kitnet_progress = kitnet_result.get("progress", 0.0)
        # FIX: threshold may be partial estimate during warmup (not null)
        kitnet_thr_raw  = kitnet_result.get("threshold", None)
        kitnet_threshold = round(float(kitnet_thr_raw), 6) if kitnet_thr_raw is not None else None

        # Record RMSE for chart
        _rmse_series.append({"ts": time.time(), "rmse": rmse})

        # Escalate category if KitNET flags Zero-Day but XGBoost says Normal
        final_cat = cat
        final_lbl = int(lbl)
        if is_anomaly and kitnet_trained and lbl == 0:
            final_cat = "Zero-Day / Anomaly"
            final_lbl = 1
            capture_state["anomalies_detected"] += 1

        # Fuse severity
        if is_anomaly and lbl == 1 and sev_score >= 2.5:
            severity = "CRITICAL"
        elif is_anomaly or (lbl == 1 and float(probas[i][1]) >= 0.85):
            severity = "HIGH"
        elif lbl == 1:
            severity = "MEDIUM"
        elif is_anomaly and kitnet_trained:
            severity = "LOW"
        else:
            severity = "NORMAL"

        conf  = round(float(probas[i][lbl]) * 100, 1)
        event = {
            "type":             "detection",
            "id":               str(uuid.uuid4())[:8],
            "timestamp":        now_str,
            "src_ip":           row.get('_src_ip', '?'),
            "dst_ip":           row.get('_dst_ip', '?'),
            "sport":            row.get('_sport', 0),
            "dport":            row.get('_dport', 0),
            "proto":            row.get('_proto', '?'),
            "service":          row.get('_service', '-'),
            "label":            final_lbl,
            "label_str":        'Attack' if final_lbl else 'Normal',
            "category":         final_cat,
            "confidence":       conf,
            "color":            ATTACK_COLORS.get(final_cat, '#6b7280'),
            "simulated":        is_simulated,
            # KitNET fields — FIX: always send phase, progress, threshold (may be partial)
            "rmse":             round(rmse, 6),
            "is_anomaly":       is_anomaly,
            "severity":         severity,
            "severity_score":   round(sev_score, 3),
            "kitnet_trained":   kitnet_trained,
            "kitnet_phase":     kitnet_phase,
            "kitnet_progress":  round(kitnet_progress, 4),
            "kitnet_threshold": kitnet_threshold,
        }
        recent_alerts.insert(0, event)
        if len(recent_alerts) > 200:
            recent_alerts.pop()
        _broadcast_event(event)


# ═══════════════════════════════════════════════════════════════════════════════
# CAPTURE INFRASTRUCTURE (pyshark / tshark / simulation)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False

capture_stop_event = threading.Event()


def _detect_interface() -> str:
    env_iface = os.environ.get("NIDS_INTERFACE", "").strip()
    if env_iface:
        return env_iface
    if not PYSHARK_AVAILABLE:
        return "eth0"
    try:
        tmp = pyshark.LiveCapture()
        raw_ifaces = getattr(tmp, 'interfaces', [])
        if not raw_ifaces:
            return "eth0"

        def _friendly(guid):
            if sys.platform != "win32":
                return guid
            try:
                import winreg
                base = (r"SYSTEM\CurrentControlSet\Control\Network"
                        r"\{4D36E972-E325-11CE-BFC1-08002BE10318}")
                pure = guid.replace(r"\Device\NPF_", "").strip("{}")
                sub  = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{base}\\{{{pure}}}\\Connection")
                name, _ = winreg.QueryValueEx(sub, "Name")
                return str(name)
            except Exception:
                return guid

        pairs = [(g, _friendly(g)) for g in raw_ifaces]
        preferred = ["wi-fi", "wifi", "wireless", "wlan", "ethernet", "lan", "local"]
        skip      = ["loopback", "usbpcap", "etwdump", "bluetooth", "npcap loopback"]
        candidates = []
        for guid, name in pairs:
            nl = name.lower()
            if any(s in nl for s in skip):
                continue
            priority = next((i for i, kw in enumerate(preferred) if kw in nl), 99)
            candidates.append((priority, guid, name))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        for guid, name in pairs:
            if "loopback" not in name.lower():
                return guid
        return pairs[0][0]
    except Exception:
        return "eth0"


def _extract_packet_fields(pkt) -> dict | None:
    try:
        ts   = float(pkt.sniff_timestamp)
        size = int(pkt.length)
        try:
            ip_layer = pkt['ip']
            src_ip   = ip_layer.src
            dst_ip   = ip_layer.dst
            ttl      = int(ip_layer.ttl)
        except Exception:
            return None

        proto = 'other'; sport = 0; dport = 0; tcp_win = 0; tcp_flags = ''
        try:
            tcp      = pkt['tcp']
            proto    = 'tcp'
            sport    = int(tcp.srcport)
            dport    = int(tcp.dstport)
            tcp_win  = int(tcp.window_size_value) if hasattr(tcp, 'window_size_value') else \
                       int(tcp.window) if hasattr(tcp, 'window') else 0
            raw_flags = int(tcp.flags, 16) if hasattr(tcp, 'flags') else 0
            flag_parts = []
            if raw_flags & 0x02: flag_parts.append('SYN')
            if raw_flags & 0x10: flag_parts.append('ACK')
            if raw_flags & 0x04: flag_parts.append('RST')
            if raw_flags & 0x01: flag_parts.append('FIN')
            tcp_flags = ','.join(flag_parts)
        except Exception:
            pass
        if proto == 'other':
            try:
                udp = pkt['udp']; proto = 'udp'; sport = int(udp.srcport); dport = int(udp.dstport)
            except Exception:
                pass
        if proto == 'other':
            try: pkt['icmp']; proto = 'icmp'
            except Exception: pass

        service = '-'
        if   dport in (80,  8080) or sport in (80,  8080): service = 'http'
        elif dport == 443         or sport == 443:          service = 'https'
        elif dport == 21          or sport == 21:           service = 'ftp'
        elif dport == 22          or sport == 22:           service = 'ssh'
        elif dport == 53          or sport == 53:           service = 'dns'
        elif dport == 25          or sport == 25:           service = 'smtp'

        return {
            'timestamp': ts, 'src_ip': src_ip, 'dst_ip': dst_ip,
            'proto': proto, 'sport': sport, 'dport': dport,
            'size': size, 'ttl': ttl, 'tcp_win': tcp_win,
            'tcp_flags': tcp_flags, 'service': service, 'direction': 'src',
        }
    except Exception:
        return None


def _start_stats_timer():
    def _loop():
        while not capture_stop_event.is_set() and capture_state["running"]:
            time.sleep(2)
            if capture_state["running"] and not capture_stop_event.is_set():
                _push_stats()
    threading.Thread(target=_loop, daemon=True).start()


def run_capture(interface: str):
    global extractor
    capture_stop_event.clear()
    extractor = FlowFeatureExtractor()
    after_image.reset()   # reset AfterImage sliding-window statistics for this session
    capture_state.update({
        "running": True, "packets_captured": 0, "flows_analyzed": 0,
        "attacks_detected": 0, "anomalies_detected": 0,
        "interface": interface, "start_time": datetime.now().isoformat(),
    })
    _start_stats_timer()

    if interface.lower() in ('simulation', 'sim', ''):
        _run_simulation()
        return
    if interface.lower() == 'auto':
        interface = _detect_interface()
        capture_state["interface"] = interface
    if PYSHARK_AVAILABLE:
        _run_pyshark_capture(interface)
        return
    print("[CAPTURE] pyshark not available — falling back to tshark subprocess", flush=True)
    _run_tshark_subprocess(interface)


def _run_pyshark_capture(interface: str):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    batch = []; last_batch_time = time.time()

    def _timeout_checker():
        while not capture_stop_event.is_set() and capture_state["running"]:
            time.sleep(3)
            stale = extractor._flush_stale()
            if stale: _process_batch(stale)
    threading.Thread(target=_timeout_checker, daemon=True).start()

    cap = None
    try:
        cap = pyshark.LiveCapture(interface=interface, bpf_filter='ip', eventloop=loop)
        for raw_pkt in cap.sniff_continuously():
            if capture_stop_event.is_set():
                break
            pkt = _extract_packet_fields(raw_pkt)
            if not pkt:
                continue
            capture_state["packets_captured"] += 1
            feats = extractor.update(pkt)
            if feats:
                batch.append(feats)
            now = time.time()
            if len(batch) >= 1 and (len(batch) >= 3 or now - last_batch_time >= 1.0):
                _process_batch(batch); batch.clear(); last_batch_time = now
    except Exception as e:
        print(f"[PYSHARK] Capture error: {e}", flush=True)
        if not capture_stop_event.is_set():
            capture_state["interface"] = "simulation"
            _run_simulation(); return
    finally:
        if cap:
            try: cap.close()
            except Exception: pass
        try: loop.close()
        except Exception: pass

    stale = extractor._flush_stale()
    if stale: _process_batch(stale)
    capture_state["running"] = False
    _broadcast_event({"type": "capture_stopped"})


def _run_tshark_subprocess(interface: str):
    tshark_fields = ['frame.time_epoch','ip.src','ip.dst','ip.proto',
                     'tcp.srcport','tcp.dstport','udp.srcport','udp.dstport',
                     'frame.len','ip.ttl','tcp.window_size_value','tcp.flags.string']
    field_args = []
    for fld in tshark_fields:
        field_args += ['-e', fld]
    cmd = ['tshark', '-i', interface, '-T', 'fields', '-E', 'separator=\t',
           '-l', '-f', 'ip'] + field_args
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except FileNotFoundError:
        print("[TSHARK] not found — falling back to simulation", flush=True)
        _run_simulation(); return

    batch = []; last_batch_flush = time.time()
    try:
        for line in proc.stdout:
            if capture_stop_event.is_set():
                break
            pkt = _parse_tshark_line(line)
            if not pkt: continue
            capture_state["packets_captured"] += 1
            feats = extractor.update(pkt)
            if feats: batch.append(feats)
            stale = extractor._flush_stale()
            if stale: batch.extend(stale)
            now = time.time()
            if len(batch) >= 1 and (len(batch) >= 3 or now - last_batch_flush >= 1.0):
                _process_batch(batch); batch.clear(); last_batch_flush = now
    finally:
        if proc and proc.poll() is None:
            proc.kill()
            try: proc.wait(timeout=3)
            except Exception: pass
    capture_state["running"] = False
    _broadcast_event({"type": "capture_stopped"})


def _parse_tshark_line(line: str):
    parts = line.strip().split('\t')
    if len(parts) < 6: return None
    while len(parts) < 12: parts.append('')
    try:
        ts = float(parts[0]) if parts[0] else time.time()
        src_ip = parts[1] or '0.0.0.0'; dst_ip = parts[2] or '0.0.0.0'
        if src_ip == '0.0.0.0' or dst_ip == '0.0.0.0': return None
        def _p(a, b):
            for v in (a, b):
                try:
                    i = int(v)
                    if i > 0: return i
                except Exception: pass
            return 0
        sport = _p(parts[4], parts[6]); dport = _p(parts[5], parts[7])
        size  = int(parts[8] or 0);     ttl   = int(parts[9] or 64)
        tcp_win = int(parts[10] or 0);  flags = parts[11] if len(parts) > 11 else ''
        proto = {'6':'tcp','17':'udp','1':'icmp'}.get(parts[3] or '0', parts[3] or '?')
        service = '-'
        if dport in (80,8080) or sport in (80,8080): service = 'http'
        elif dport == 443 or sport == 443: service = 'https'
        elif dport == 21  or sport == 21:  service = 'ftp'
        elif dport == 22  or sport == 22:  service = 'ssh'
        elif dport == 53  or sport == 53:  service = 'dns'
        elif dport == 25  or sport == 25:  service = 'smtp'
        return {'timestamp':ts,'src_ip':src_ip,'dst_ip':dst_ip,'proto':proto,
                'sport':sport,'dport':dport,'size':size,'ttl':ttl,'tcp_win':tcp_win,
                'tcp_flags':flags,'service':service,'direction':'src'}
    except Exception:
        return None


def _run_simulation():
    rng     = np.random.default_rng()
    ip_pool = [f'192.168.{rng.integers(0,5)}.{i}' for i in range(1, 30)]
    SCENARIOS = [
        ('normal', (8,  20), 0.10, (300, 1400), 'tcp',  80,   'http',   True),
        ('normal', (3,  7),  0.06, (80,  300),  'udp',  53,   'dns',    False),
        ('normal', (10, 20), 0.15, (200, 900),  'tcp',  22,   'ssh',    True),
        ('normal', (5,  12), 0.10, (300, 1100), 'tcp',  443,  'https',  True),
        ('attack', (25, 60), 0.005,(40,  100),  'tcp',  80,   'http',   False),
        ('attack', (30, 70), 0.004,(40,  80),   'udp',  53,   'dns',    False),
        ('attack', (2,  5),  0.01, (40,  80),   'tcp',  4444, '-',      False),
        ('attack', (2,  4),  0.01, (40,  60),   'tcp',  8888, '-',      False),
        ('attack', (25, 45), 0.06, (900, 1500), 'tcp',  21,   'ftp',    False),
        ('attack', (20, 40), 0.05, (800, 1400), 'tcp',  6667, '-',      False),
    ]
    weights = [0.18, 0.15, 0.14, 0.13, 0.09, 0.08, 0.07, 0.07, 0.05, 0.04]
    t = time.time()
    while not capture_stop_event.is_set():
        idx  = int(rng.choice(len(SCENARIOS), p=weights))
        _lbl, (n_min, n_max), iat, (sz_min, sz_max), proto, dport, svc, do_handshake = SCENARIOS[idx]
        src_ip     = str(rng.choice(ip_pool))
        dst_ip     = str(rng.choice(ip_pool))
        flow_sport = int(rng.integers(1024, 65535))
        flow_ttl   = int(rng.choice([64, 128, 255]))
        n_pkts     = int(rng.integers(n_min, n_max))
        for pkt_i in range(n_pkts):
            if capture_stop_event.is_set(): break
            if do_handshake:
                flags = 'SYN' if pkt_i == 0 else ('SYN,ACK' if pkt_i == 1 else 'ACK')
            else:
                flags = ''
            pkt = {
                'timestamp': t, 'src_ip': src_ip, 'dst_ip': dst_ip,
                'proto': proto, 'sport': flow_sport, 'dport': dport,
                'size': int(rng.integers(sz_min, sz_max)), 'ttl': flow_ttl,
                'tcp_win': int(rng.integers(1024, 65535)), 'tcp_flags': flags,
                'service': svc, 'direction': 'src',
            }
            capture_state["packets_captured"] += 1
            feats = extractor.update(pkt)
            if feats: _process_batch([feats])
            jitter = float(rng.normal(iat, iat * 0.2))
            t += max(jitter, 0.001)
            time.sleep(max(min(iat, 0.05), 0.005))
        stale = extractor._flush_stale()
        for s in stale: _process_batch([s])
    capture_state["running"] = False
    _broadcast_event({"type": "capture_stopped"})


# ═══════════════════════════════════════════════════════════════════════════════
# SSE BROADCAST
# ═══════════════════════════════════════════════════════════════════════════════

def _broadcast_event(data: dict):
    dead    = []
    payload = f"data: {json.dumps(data)}\n\n"
    for q in event_clients:
        try:
            q.put_nowait(payload)
        except queue.Full:
            try: q.get_nowait(); q.put_nowait(payload)
            except Exception: dead.append(q)
        except Exception:
            dead.append(q)
    for q in dead:
        try: event_clients.remove(q)
        except ValueError: pass


@app.route('/api/events')
def sse_stream():
    q: queue.Queue = queue.Queue(maxsize=500)
    event_clients.append(q)

    def generate():
        yield 'data: {"type":"connected"}\n\n'
        try:
            while True:
                try:
                    msg = q.get(timeout=30)
                    yield msg
                except queue.Empty:
                    yield 'data: {"type":"ping"}\n\n'
        except GeneratorExit:
            pass
        finally:
            try: event_clients.remove(q)
            except ValueError: pass

    return Response(stream_with_context(generate()),
                    mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/ws/stream')
def ws_stream():
    return sse_stream()


# ═══════════════════════════════════════════════════════════════════════════════
# REST API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/status')
def status():
    return jsonify({
        "capture":           capture_state,
        "models_loaded":     MODELS_READY,
        "kitnet_ready":      KITNET_READY,
        "kitnet":            kitnet_engine.status,
        "feature_count":     len(FEATURE_COLS),
        "attack_classes":    ATTACK_CLASSES,
        "recent_alerts":     recent_alerts[:20],
        "stats_history":     stats_history[-30:],
        "attack_categories": ATTACK_CATEGORIES,
        "attack_colors":     ATTACK_COLORS,
        "rmse_series":       list(_rmse_series)[-60:],
    })


@app.route('/api/kitnet/status')
def kitnet_status():
    """Detailed KitNET engine status."""
    return jsonify({
        "status": kitnet_engine.status,
        "rmse_series": list(_rmse_series)[-100:],
        "rmse_history_tail": kitnet_engine.rmse_history[-50:],
    })


@app.route('/api/kitnet/reset', methods=['POST'])
def kitnet_reset():
    """Reset warm-up calibration (keeps pre-trained weights, recalibrates threshold)."""
    with _kitnet_lock:
        kitnet_engine.threshold    = float('inf')
        kitnet_engine.trained      = False
        kitnet_engine._warmup_done = False
        kitnet_engine._warmup_rmse = []
        kitnet_engine.packet_count = 0
        kitnet_engine.rmse_history = []
        _rmse_series.clear()
    return jsonify({"status": "reset", "message": "KitNET warm-up reset — will recalibrate threshold"})


# ── FIX: /api/interfaces — robuste tshark parsing + stderr fallback ──────────
@app.route('/api/interfaces')
def interfaces():
    ifaces = []
    tshark_found = False
    try:
        res = subprocess.run(
            ['tshark', '-D'],
            capture_output=True, text=True, timeout=5
        )
        # tshark peut écrire sur stdout ou stderr selon la version/OS
        output = res.stdout.strip() or res.stderr.strip()
        for line in output.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Format: "1. eth0 (desc)" ou "1. \Device\NPF_{GUID} (Wi-Fi)"
            parts = line.split('. ', 1)
            if len(parts) == 2:
                name = parts[1].split(' ')[0].strip()
                if name:
                    ifaces.append(name)
                    tshark_found = True
    except FileNotFoundError:
        print("[interfaces] tshark not found — using fallback list", flush=True)
    except subprocess.TimeoutExpired:
        print("[interfaces] tshark -D timed out", flush=True)
    except Exception as e:
        print(f"[interfaces] unexpected error: {e}", flush=True)

    return jsonify({
        "interfaces":   ifaces or ['eth0', 'wlan0', 'lo', 'any'],
        "tshark_found": tshark_found,
    })


@app.route('/api/capture/start', methods=['POST'])
def start_capture():
    with capture_lock:
        if capture_state["running"]:
            capture_stop_event.set()
            time.sleep(0.4)
            capture_stop_event.clear()
        data      = request.get_json() or {}
        interface = data.get('interface', 'simulation')
        capture_stop_event.clear()
        t = threading.Thread(target=run_capture, args=(interface,), daemon=True)
        t.start()
    return jsonify({"status": "started", "interface": interface,
                    "models_ready": MODELS_READY, "kitnet_ready": KITNET_READY})


@app.route('/api/capture/stop', methods=['POST'])
def stop_capture():
    capture_stop_event.set()
    capture_state["running"] = False
    _broadcast_event({"type": "capture_stopped"})
    return jsonify({"status": "stopped"})


@app.route('/api/analyze/csv', methods=['POST'])
def analyze_csv():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files['file']
    if not f.filename.endswith('.csv'):
        return jsonify({"error": "Only CSV files accepted"}), 400
    try:
        df = pd.read_csv(f)
        df = _normalise_csv_columns(df)
        n_total = len(df)

        missing    = [c for c in FEATURE_COLS if c not in df.columns]
        pct_missing = len(missing) / max(len(FEATURE_COLS), 1) * 100

        labels, probas, categories = detector.predict(df)

        kitnet_scores = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            kres = kitnet_engine.process_features(row_dict)
            kitnet_scores.append(kres)

        n_attacks    = int(np.sum(labels))
        n_anomalies  = sum(1 for k in kitnet_scores if k.get("is_anomaly", False))
        n_normal     = n_total - n_attacks
        cat_counts   = {}
        for c in categories:
            cat_counts[c] = cat_counts.get(c, 0) + 1

        results = []
        for i in range(min(200, n_total)):
            src   = str(df.get('srcip', df.get('src_ip', pd.Series(['?']*n_total))).iloc[i]) \
                    if ('srcip' in df.columns or 'src_ip' in df.columns) else '?'
            dst   = str(df.get('dstip', df.get('dst_ip', pd.Series(['?']*n_total))).iloc[i]) \
                    if ('dstip' in df.columns or 'dst_ip' in df.columns) else '?'
            proto = str(df['proto'].iloc[i]) if 'proto' in df.columns else '-'
            ks    = kitnet_scores[i] if i < len(kitnet_scores) else {}
            is_anomaly = ks.get("is_anomaly", False)
            final_cat  = categories[i]
            final_lbl  = int(labels[i])
            if is_anomaly and final_lbl == 0:
                final_cat = "Zero-Day / Anomaly"
                final_lbl = 1
            results.append({
                "id":             str(i),
                "src_ip":         src, "dst_ip": dst, "proto": proto,
                "label":          final_lbl,
                "label_str":      'Attack' if final_lbl else 'Normal',
                "category":       final_cat,
                "confidence":     round(float(probas[i][labels[i]]) * 100, 1),
                "color":          ATTACK_COLORS.get(final_cat, '#6b7280'),
                "rmse":           round(ks.get("rmse", 0.0), 6),
                "is_anomaly":     is_anomaly,
                "severity_score": round(ks.get("severity_score", 0.0), 3),
            })

        warning = None
        if pct_missing > 50:
            warning = (f"{len(missing)}/{len(FEATURE_COLS)} feature columns missing "
                       f"({pct_missing:.0f}%). Results may be unreliable. "
                       f"First missing: {missing[:5]}")

        return jsonify({
            "total":         n_total,
            "attacks":       n_attacks,
            "anomalies":     n_anomalies,
            "normal":        n_normal,
            "attack_rate":   round(n_attacks / n_total * 100, 2) if n_total else 0,
            "category_dist": cat_counts,
            "missing_cols":  missing,
            "warning":       warning,
            "models_used":   "real" if MODELS_READY else "stub",
            "kitnet_used":   KITNET_READY,
            "results":       results,
            "kitnet_status": kitnet_engine.status,
        })
    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route('/api/alerts')
def get_alerts():
    return jsonify(recent_alerts[:int(request.args.get('limit', 50))])


@app.route('/api/stats/history')
def get_history():
    return jsonify(stats_history[-100:])


@app.route('/api/debug/features')
def debug_features():
    dummy_raw = {
        'dur':1.0, 'sbytes':1000, 'dbytes':500, 'sttl':64, 'dttl':64,
        'sloss':0, 'dloss':0, 'Sload':8000.0, 'Dload':4000.0,
        'Spkts':5, 'Dpkts':3, 'swin':65535, 'dwin':65535,
        'stcpb':0, 'dtcpb':0, 'smeansz':200.0, 'dmeansz':166.0,
        'trans_depth':0, 'res_bdy_len':0,
        'Sjit':10.0, 'Djit':8.0, 'Sintpkt':200.0, 'Dintpkt':250.0,
        'tcprtt':0.0, 'synack':0.0, 'ackdat':0.0,
        'ct_state_ttl':0, 'ct_flw_http_mthd':0,
        'ct_srv_src':1, 'ct_srv_dst':1, 'ct_dst_ltm':1, 'ct_src_ltm':1,
        'ct_src_dport_ltm':1, 'ct_dst_sport_ltm':1, 'ct_dst_src_ltm':1,
        'is_sm_ips_ports':0, 'is_ftp_login':0, 'ct_ftp_cmd':0,
        'proto':'tcp', 'state':'CON', 'service':'http',
        'sport':54321, 'dsport':80,
        '_src_ip':'1.2.3.4', '_dst_ip':'5.6.7.8',
        '_sport':54321, '_dport':80, '_proto':'tcp', '_service':'http',
    }
    df_raw = pd.DataFrame([dummy_raw])
    df_pre = _preprocess_live(df_raw)
    produced = set(df_pre.columns)
    expected = set(FEATURE_COLS)
    return jsonify({
        "model_expects":         FEATURE_COLS,
        "pipeline_produces":     sorted(list(produced)),
        "matched":               sorted(produced & expected),
        "missing_from_pipeline": sorted(expected - produced),
        "extra_in_pipeline":     sorted(produced - expected),
        "match_pct":             round(len(produced & expected) / max(len(FEATURE_COLS), 1) * 100, 1),
    })


@app.route('/api/debug/capture')
def debug_capture():
    active_flows = []
    try:
        for key, f in list(extractor.flows.items())[:20]:
            active_flows.append({
                "key":   str(key),
                "spkts": f['spkts'], "dpkts": f['dpkts'],
                "sbytes":f['sbytes'],"dbytes":f['dbytes'],
                "dur":   round(time.time() - f['start_time'], 2),
                "idle":  round(time.time() - f['last_time'], 2),
            })
    except Exception:
        pass
    return jsonify({
        "capture_state":     capture_state,
        "sse_clients":       len(event_clients),
        "recent_alerts_count": len(recent_alerts),
        "active_flows_count":  len(extractor.flows) if extractor else 0,
        "active_flows_sample": active_flows,
        "models_ready":      MODELS_READY,
        "kitnet":            kitnet_engine.status,
    })


@app.route('/')
def index():
    html_path = Path(STATIC_DIR) / 'index.html'
    if html_path.exists():
        return send_from_directory(str(STATIC_DIR), 'index.html')
    return ("<h2>NetGuard backend running ✅</h2>"
            "<p>Place <code>index.html</code> next to <code>app.py</code> and restart.</p>"), 200


@app.route('/health')
def health():
    return jsonify({
        "ok":           True,
        "models_ready": MODELS_READY,
        "kitnet_ready": KITNET_READY,
        "feature_count": len(FEATURE_COLS),
        "kitnet":       kitnet_engine.status,
    })


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    print(f"🛡  NetGuard backend  →  http://0.0.0.0:{port}")
    print(f"   XGBoost ready : {MODELS_READY}")
    print(f"   KitNET  ready : {KITNET_READY}  (module={KITNET_AVAILABLE}, stub={not KITNET_AVAILABLE and KITNET_READY})")
    print(f"   Feature cols  : {len(FEATURE_COLS)}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)