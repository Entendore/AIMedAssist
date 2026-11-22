#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════╗
║          AI MEDICAL ASSISTANT v3.1 (Professional Desktop Edition)     ║
║        Powered by PySide6, llama.cpp & Medical/Multimodal LLMs       ║
║        Features: RAG, PDF, Image Analysis, Stats, Persistent Data    ║
║        Works on GTX 1080 (8GB) to High-End Systems                   ║
╚═══════════════════════════════════════════════════════════════════════╝

Usage:  python medical_assistant_v3.py
"""

import os, sys, json, time, threading, platform, subprocess, signal, gc, re, math, base64, io
from pathlib import Path
from collections import Counter
import importlib.util

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

APP_NAME = "AI Medical Assistant Pro"
APP_VERSION = "3.1.0"
BASE_DIR = Path.home() / ".medical_assistant_pro"
CONVOS_DIR = BASE_DIR / "conversations"
DEFAULT_MODELS_DIR = BASE_DIR / "models"

MODEL_CONFIGS = {
    "OpenBioLLM-8B-Q4_K_M": {
        "repo_id": "aaditya/Llama3-OpenBioLLM-8B-GGUF",
        "filename": "Llama3-OpenBioLLM-8B.Q4_K_M.gguf",
        "size_gb": 4.9, "min_ram_gb": 8, "min_vram_gb": 6,
        "rec_vram_gb": 8, "context_length": 4096,
        "description": "OpenBioLLM 8B Q4 — Best for GTX 1080",
        "tier": "low", "focus": "Clinical reasoning, medical Q&A", "multimodal": False,
    },
    "OpenBioLLM-8B-Q8_0": {
        "repo_id": "aaditya/Llama3-OpenBioLLM-8B-GGUF",
        "filename": "Llama3-OpenBioLLM-8B.Q8_0.gguf",
        "size_gb": 8.5, "min_ram_gb": 16, "min_vram_gb": 10,
        "rec_vram_gb": 12, "context_length": 4096,
        "description": "OpenBioLLM 8B Q8 — Near-lossless, RTX 3060+",
        "tier": "high", "focus": "Clinical reasoning, medical Q&A", "multimodal": False,
    },
    "BioMistral-7B-Q4_K_M": {
        "repo_id": "BioMistral/BioMistral-7B-GGUF",
        "filename": "biomistral-7b.Q4_K_M.gguf",
        "size_gb": 4.37, "min_ram_gb": 8, "min_vram_gb": 6,
        "rec_vram_gb": 8, "context_length": 32768,
        "description": "BioMistral 7B Q4 — Large context, GTX 1080",
        "tier": "low", "focus": "Biomedical literature, medical Q&A", "multimodal": False,
    },
    "LLaVA-1.6-7B-Q4 (Vision)": {
        "repo_id": "afrideva/llava-v1.6-vicuna-7b-GGUF",
        "filename": "llava-v1.6-vicuna-7b.Q4_K_M.gguf",
        "mmproj_filename": "llava-v1.6-vicuna-7b.mmproj-f16.gguf",
        "size_gb": 4.5, "min_ram_gb": 12, "min_vram_gb": 8,
        "rec_vram_gb": 10, "context_length": 4096,
        "description": "LLaVA 7B Q4 — Image/Medical Scan Analysis",
        "tier": "mid", "focus": "Visual analysis, radiology, dermatology", "multimodal": True,
    },
}

DEFAULT_SYSTEM_PROMPT = """You are an expert medical AI assistant with extensive knowledge of medicine, anatomy, pharmacology, and clinical practice. Provide helpful, accurate, balanced health information.

CRITICAL RULES:
1. ALWAYS include a disclaimer that you are an AI, not a doctor.
2. NEVER prescribe medications or recommend exact dosages.
3. If symptoms suggest an emergency, URGE immediate emergency care.
4. State clearly when something is uncertain or beyond your scope.
5. Be empathetic, professional, and use clear language.
6. If context from documents is provided, base your answer primarily on that context and cite it.
"""

QUICK_PROMPTS = [
    "What are common symptoms of diabetes?",
    "Explain Type 1 vs Type 2 diabetes",
    "What is hypertension and how is it managed?",
    "Recommended screenings for adults over 50",
    "What are warning signs of a stroke?",
    "How does the immune system work?",
]

STATS_DATA = {
    "total_prompt_tokens": 0, "total_completion_tokens": 0,
    "total_requests": 0, "start_time": time.time(),
    "total_tps": 0.0, "tps_count": 0
}

# ═══════════════════════════════════════════════════════════════
# DEPENDENCY MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def pip_install(pkg):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception: return False

def detect_cuda():
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception: return False

def check_dependencies():
    print(f"\n{'═'*60}\n  {APP_NAME} v{APP_VERSION} — Dependency Check\n{'═'*60}\n")
    deps = [("psutil","psutil"), ("huggingface_hub","huggingface-hub"), 
            ("PyPDF2","PyPDF2"), ("markdown","markdown"), ("PySide6","PySide6")]
    for mod, pip in deps:
        if importlib.util.find_spec(mod) is None:
            print(f"  ⬇ Installing {pip}...")
            if not pip_install(pip): print(f"  ✗ Failed"); sys.exit(1)
            print(f"  ✓ {pip}")
        else: print(f"  ✓ {pip}")
    
    if importlib.util.find_spec("llama_cpp") is None:
        has_cuda = detect_cuda()
        if has_cuda:
            print(f"  ⬇ Installing llama-cpp-python with CUDA...")
            installed = False
            for cu in ["cu121","cu122","cu123"]:
                try:
                    url = f"https://abetlen.github.io/llama-cpp-python/whl/{cu}"
                    subprocess.check_call([sys.executable,"-m","pip","install","llama-cpp-python",
                           "--extra-index-url", url, "-q"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    installed = True; print(f"  ✓ llama-cpp-python ({cu})"); break
                except Exception: continue
            if not installed: pip_install("llama-cpp-python"); print(f"  ⚠ CPU fallback installed")
        else:
            print(f"  ⬇ Installing llama-cpp-python (CPU)...")
            pip_install("llama-cpp-python")
    else: print(f"  ✓ llama-cpp-python")
    print(f"\n  All dependencies ready!\n")

# ═══════════════════════════════════════════════════════════════
# SYSTEM & SETTINGS CORE
# ═══════════════════════════════════════════════════════════════

class SystemInfo:
    def __init__(self):
        self.os_name = platform.system(); self.cpu = platform.processor() or "Unknown"
        self.ram_gb = 0; self.gpu = "None"; self.vram_gb = 0; self.has_cuda = False
        self._detect()

    def _detect(self):
        try:
            import psutil; self.ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
        except Exception: pass
        try:
            r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=5)
            if r.returncode == 0 and r.stdout.strip():
                parts = r.stdout.strip().split(",")
                self.gpu = parts[0].strip(); self.vram_gb = round(float(parts[1].strip()) / 1024, 1)
                self.has_cuda = True
        except Exception: pass

    def tier(self):
        if self.vram_gb >= 12: return "high"
        if self.vram_gb >= 8: return "mid"
        if self.vram_gb >= 6: return "low"
        return "cpu"

    def gpu_layers(self, cfg):
        if not self.has_cuda: return 0
        usable = self.vram_gb * 1024**3 - 1.5 * 1024**3
        if usable <= 0: return 0
        ratio = usable / (cfg["size_gb"] * 1024**3)
        if ratio >= 0.95: return -1
        return max(1, int(32 * ratio))

    def to_dict(self):
        return {"os":self.os_name,"cpu":self.cpu,"ram_gb":self.ram_gb,
                "gpu":self.gpu,"vram_gb":self.vram_gb,"has_cuda":self.has_cuda,"tier":self.tier()}

si = SystemInfo()

class Settings:
    def __init__(self):
        self.path = BASE_DIR / "settings.json"
        self.data = {
            "models_dir": str(DEFAULT_MODELS_DIR),
            "temperature": 0.7,
            "max_tokens": 1024,
            "system_prompt": DEFAULT_SYSTEM_PROMPT
        }
        self.load()

    def load(self):
        if self.path.exists():
            try: self.data.update(json.loads(self.path.read_text()))
            except Exception: pass
        Path(self.data["models_dir"]).mkdir(parents=True, exist_ok=True)

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2))

    def get(self, key, default=None): return self.data.get(key, default)
    def set(self, key, value): self.data[key] = value; self.save()

settings = Settings()

class SimpleRAG:
    def __init__(self): self.docs = []; self.idf = {}

    def add_document(self, text, name):
        chunks = [text[i:i+600] for i in range(0, len(text), 500)]
        for c in chunks:
            tokens = set(re.findall(r'\b\w+\b', c.lower()))
            self.docs.append({"text": c, "name": name, "tokens": tokens})
        self._calc_idf()

    def _calc_idf(self):
        N = len(self.docs)
        df = Counter()
        for d in self.docs: df.update(d["tokens"])
        self.idf = {w: math.log((N + 1) / (c + 1)) + 1 for w, c in df.items()}

    def search(self, query, top_k=3):
        if not self.docs: return []
        qtoks = set(re.findall(r'\b\w+\b', query.lower()))
        scores = []
        for d in self.docs:
            inter = qtoks & d["tokens"]
            score = sum(self.idf.get(w, 0) for w in inter)
            scores.append((score, d))
        scores.sort(reverse=True)
        return [s[1] for s in scores[:top_k] if s[0] > 0]

    def clear(self): self.docs = []; self.idf = {}

rag = SimpleRAG()

class ModelManager:
    def __init__(self):
        self.models_dir = Path(settings.get("models_dir"))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = self._load_registry()
        self._sync_with_folder()

    def _reg_path(self): 
        return self.models_dir / "registry.json"
        
    def _load_registry(self):
        p = self._reg_path()
        if p.exists():
            try: return json.loads(p.read_text())
            except Exception: pass
        return {}
        
    def _save_registry(self): 
        self._reg_path().write_text(json.dumps(self.registry, indent=2))

    def _sync_with_folder(self):
        """Checks the folder for existing model files and updates registry."""
        changed = False
        for key, cfg in MODEL_CONFIGS.items():
            main_path = self.models_dir / cfg["filename"]
            if main_path.exists():
                if key not in self.registry or self.registry[key].get("main") != str(main_path):
                    self.registry[key] = {"main": str(main_path)}
                    if cfg.get("mmproj_filename"):
                        mmproj_path = self.models_dir / cfg["mmproj_filename"]
                        if mmproj_path.exists():
                            self.registry[key]["mmproj"] = str(mmproj_path)
                    changed = True
            elif key in self.registry:
                # If file was deleted manually, remove from registry
                if not Path(self.registry[key]["main"]).exists():
                    del self.registry[key]
                    changed = True
        
        if changed: self._save_registry()

    def is_downloaded(self, key):
        return key in self.registry and Path(self.registry[key]["main"]).exists()

    def get_path(self, key):
        if self.is_downloaded(key): return self.registry[key]
        return None

    def download(self, key):
        cfg = MODEL_CONFIGS.get(key)
        if not cfg: return False, "Invalid model key"
        
        main_path = self.models_dir / cfg["filename"]
        if main_path.exists():
            self._sync_with_folder() # Just sync and return
            return True, "Already exists"

        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["filename"], local_dir=str(self.models_dir))
            
            reg_entry = {"main": str(self.models_dir / cfg["filename"])}
            if cfg.get("mmproj_filename"):
                hf_hub_download(repo_id=cfg["repo_id"], filename=cfg["mmproj_filename"], local_dir=str(self.models_dir))
                reg_entry["mmproj"] = str(self.models_dir / cfg["mmproj_filename"])
            
            self.registry[key] = reg_entry
            self._save_registry()
            return True, "Download Complete"
        except Exception as e:
            return False, str(e)

model_mgr = ModelManager()

class LLMEngine:
    def __init__(self):
        self.llm = None; self.model_key = None; self.loaded = False; self.loading = False
        self.load_msg = ""; self.is_multimodal = False

    def load(self, key=None):
        if self.loading: return False, "Already loading"
        self.loading = True; self.load_msg = "Loading model..."
        try:
            from llama_cpp import Llama
            paths = model_mgr.get_path(key)
            cfg = MODEL_CONFIGS.get(key, {"context_length": 4096})
            if not paths: self.loading = False; return False, "Model files not found. Please download first."
            
            ngl = si.gpu_layers(cfg) if si.has_cuda else 0
            ctx = cfg.get("context_length", 4096)
            if si.vram_gb > 0 and si.vram_gb < 8: ctx = min(ctx, 4096)
            
            kwargs = {"model_path": paths["main"], "n_gpu_layers": ngl, "n_ctx": ctx, "verbose": False}
            if paths.get("mmproj"):
                kwargs["mmproj"] = paths["mmproj"]; self.is_multimodal = True
            else: self.is_multimodal = False
            
            self.load_msg = f"Loading (GPU layers: {ngl}, ctx: {ctx})..."
            self.llm = Llama(**kwargs)
            self.model_key = key; self.loaded = True; self.loading = False; self.load_msg = ""
            return True, f"Loaded {key} (GPU: {ngl}, ctx: {ctx})"
        except Exception as e:
            self.loading = False; self.load_msg = ""
            return False, str(e)

    def unload(self):
        if not self.llm: return "No model loaded"
        try:
            del self.llm; self.llm = None
            self.model_key = None; self.loaded = False; self.is_multimodal = False
            gc.collect()
            return "Model unloaded successfully. VRAM freed."
        except Exception as e:
            return str(e)

    def chat(self, messages, image_base64=None):
        if not self.loaded or not self.llm: return None, "No model loaded", {}
        temperature = settings.get("temperature", 0.7)
        max_tokens = settings.get("max_tokens", 1024)
        start_time = time.time()
        try:
            user_query = messages[-1]["content"] if messages else ""
            rag_results = rag.search(user_query)
            if rag_results:
                rag_context = "\n\n[Retrieved Medical Document Context]:\n" + "\n---\n".join([f"Source: {r['name']}\n{r['text']}" for r in rag_results])
                messages[0]["content"] += rag_context

            kwargs = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
            
            if image_base64 and self.is_multimodal:
                img_data = image_base64
                if img_data.startswith("data:"): img_data = img_data.split(",")[1]
                messages[-1]["content"] = [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_data}"}},
                    {"type": "text", "text": user_query}
                ]

            r = self.llm.create_chat_completion(**kwargs)
            text = r["choices"][0]["message"]["content"]
            
            p_tok = r.get("usage", {}).get("prompt_tokens", 0)
            c_tok = r.get("usage", {}).get("completion_tokens", 0)
            elapsed = time.time() - start_time
            tps = c_tok / elapsed if elapsed > 0 else 0
            
            STATS_DATA["total_prompt_tokens"] += p_tok
            STATS_DATA["total_completion_tokens"] += c_tok
            STATS_DATA["total_requests"] += 1
            STATS_DATA["total_tps"] += tps; STATS_DATA["tps_count"] += 1
            
            return text, None, {"prompt_tokens": p_tok, "completion_tokens": c_tok, "tps": round(tps, 2)}
        except Exception as e:
            return None, str(e), {}

    def status(self):
        return {"loaded":self.loaded,"loading":self.loading,"load_msg":self.load_msg,
                "model":self.model_key,"multimodal":self.is_multimodal}

llm = LLMEngine()

class ConvMgr:
    def __init__(self):
        self.convs = {}; self.cur = None; self._nid = 1
        CONVOS_DIR.mkdir(parents=True, exist_ok=True)
        self._load_all()

    def _load_all(self):
        for f in CONVOS_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text())
                self.convs[data["id"]] = data
                if int(data["id"]) >= self._nid: self._nid = int(data["id"]) + 1
            except Exception: pass

    def _save(self, cid):
        if cid in self.convs:
            (CONVOS_DIR / f"{cid}.json").write_text(json.dumps(self.convs[cid], indent=2))

    def create(self, title="New Chat"):
        cid = str(self._nid); self._nid += 1
        self.convs[cid] = {"id":cid,"title":title,"messages":[],"ts":time.time()}
        self._save(cid); self.cur = cid; return cid

    def add(self, cid, role, content, stats=None):
        if cid not in self.convs: return
        msg = {"role":role,"content":content,"ts":time.time()}
        if stats: msg["stats"] = stats
        self.convs[cid]["messages"].append(msg)
        if role == "user" and len(self.convs[cid]["messages"]) <= 2:
            self.convs[cid]["title"] = content[:50] + ("..." if len(content)>50 else "")
        self._save(cid)

    def llm_messages(self, cid):
        if cid not in self.convs: return []
        sys_prompt = settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        msgs = [{"role":"system","content":sys_prompt}]
        for m in self.convs[cid]["messages"]:
            msgs.append({"role":m["role"],"content":m["content"]})
        return msgs

    def get(self, cid): return self.convs.get(cid)
    def list_all(self):
        return sorted([{"id":c["id"],"title":c["title"],"count":len(c["messages"]),"ts":c["ts"]} for c in self.convs.values()], key=lambda x:-x["ts"])
    def delete(self, cid):
        self.convs.pop(cid, None)
        (CONVOS_DIR / f"{cid}.json").unlink(missing_ok=True)
        if self.cur == cid: self.cur = None
    def clear(self, cid):
        if cid in self.convs: self.convs[cid]["messages"] = []; self._save(cid)

convos = ConvMgr()

# ═══════════════════════════════════════════════════════════════
# PYSIDE6 UI & THREADS
# ═══════════════════════════════════════════════════════════════

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QTabWidget, QPushButton, QTextEdit, QLabel, QListWidget, 
                               QListWidgetItem, QFileDialog, QLineEdit, QProgressBar, 
                               QSplitter, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
                               QFormLayout, QMessageBox)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QPixmap, QIcon, QFont, QColor, QImage

STYLESHEET = """
QMainWindow { background-color: #060a13; }
QWidget { color: #e8edf5; font-family: 'Inter', 'Segoe UI', sans-serif; }
QTabWidget::pane { border: 1px solid #1a2540; background: #0c1221; border-radius: 8px; }
QTabBar::tab { background: #0c1221; color: #5a6f94; padding: 12px 20px; border: 1px solid #1a2540; border-bottom: none; border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: bold; }
QTabBar::tab:selected { background: #141d33; color: #38bdf8; border-bottom: 2px solid #0ea5e9; }
QListWidget { background-color: #0c1221; border: 1px solid #1a2540; border-radius: 8px; padding: 4px; outline: none; }
QListWidget::item { padding: 10px; border-radius: 6px; margin-bottom: 2px; }
QListWidget::item:selected { background-color: #1c2942; color: #38bdf8; }
QPushButton { background-color: #141d33; color: #38bdf8; border: 1px solid #0ea5e9; padding: 8px 16px; border-radius: 6px; font-weight: bold; }
QPushButton:hover { background-color: #0ea5e9; color: white; }
QPushButton:disabled { background-color: #1c2942; color: #5a6f94; border-color: #1a2540; }
QPushButton#UnloadBtn { background-color: #ef4444; color: white; border: none; }
QPushButton#UnloadBtn:hover { background-color: #dc2626; }
QTextEdit { background-color: #141d33; border: 1px solid #1a2540; border-radius: 8px; padding: 12px; color: #e8edf5; font-size: 14px; }
QTextEdit:focus { border-color: #0ea5e9; }
QLineEdit { background-color: #141d33; border: 1px solid #1a2540; border-radius: 6px; padding: 8px; color: #e8edf5; }
QLineEdit:focus { border-color: #0ea5e9; }
QSpinBox, QDoubleSpinBox { background-color: #141d33; border: 1px solid #1a2540; border-radius: 6px; padding: 4px; color: #e8edf5; }
QProgressBar { border: 1px solid #1a2540; border-radius: 4px; text-align: center; background-color: #141d33; color: white; }
QProgressBar::chunk { background-color: #0ea5e9; border-radius: 3px; }
QLabel { color: #8b9dc3; }
QGroupBox { border: 1px solid #1a2540; border-radius: 8px; margin-top: 1em; padding-top: 10px; font-weight: bold; color: #38bdf8; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px; }
"""

# --- Worker Threads ---

class DownloadWorker(QThread):
    progress = Signal(int)
    finished = Signal(bool, str)

    def __init__(self, key):
        super().__init__(); self.key = key

    def run(self):
        self.progress.emit(10)
        ok, msg = model_mgr.download(self.key)
        self.finished.emit(ok, msg)

class LoadWorker(QThread):
    finished = Signal(bool, str)

    def __init__(self, key):
        super().__init__(); self.key = key

    def run(self):
        # Unload existing first to free VRAM
        if llm.loaded:
            llm.unload()
        ok, msg = llm.load(self.key)
        self.finished.emit(ok, msg)

class UnloadWorker(QThread):
    finished = Signal(str)

    def run(self):
        msg = llm.unload()
        self.finished.emit(msg)

class ChatWorker(QThread):
    finished = Signal(str, dict, str)

    def __init__(self, messages, image_base64=None):
        super().__init__(); self.messages = messages; self.image = image_base64

    def run(self):
        text, err, stats = llm.chat(self.messages, self.image)
        self.finished.emit(text or "", stats or {}, err or "")


# --- Main Window ---

class MedicalAssistantUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1200, 800)
        self.current_convo = None
        self.attached_image_base64 = None
        self.active_dl_key = None # Track which model is currently downloading
        self.workers = [] 

        self.init_ui()
        self.refresh_convos()
        self.refresh_models()
        
        # Stats Timer
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self.refresh_stats)
        self.stats_timer.start(2000)

        # Download Progress Timer
        self.dl_progress_timer = QTimer(self)
        self.dl_progress_timer.timeout.connect(self.update_download_progress)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Sidebar ---
        sidebar = QWidget()
        sidebar.setFixedWidth(260)
        sidebar_layout = QVBoxLayout(sidebar)
        
        lbl_title = QLabel("🩺 MedAssist Pro")
        lbl_title.setFont(QFont("", 16, QFont.Bold))
        lbl_title.setStyleSheet("color: #38bdf8; margin-bottom: 10px;")
        sidebar_layout.addWidget(lbl_title)

        btn_new = QPushButton("+ New Chat")
        btn_new.clicked.connect(self.new_convo)
        sidebar_layout.addWidget(btn_new)

        self.list_convos = QListWidget()
        self.list_convos.currentItemChanged.connect(self.load_convo)
        sidebar_layout.addWidget(self.list_convos)

        lbl_sys = QLabel(f"System: {si.tier().upper()}\nGPU: {si.gpu} ({si.vram_gb}GB)")
        lbl_sys.setStyleSheet("color: #5a6f94; font-size: 11px; margin-top: 10px;")
        sidebar_layout.addWidget(lbl_sys)

        # --- Main Tabs ---
        self.tabs = QTabWidget()
        
        # TAB: Chat
        chat_widget = QWidget()
        chat_layout = QVBoxLayout(chat_widget)
        
        self.txt_chat = QTextEdit()
        self.txt_chat.setReadOnly(True)
        self.txt_chat.setStyleSheet("background-color: #0c1221; border: none; font-size: 15px;")
        chat_layout.addWidget(self.txt_chat)

        # Image preview
        self.lbl_img_preview = QLabel()
        self.lbl_img_preview.setFixedSize(100, 100)
        self.lbl_img_preview.setStyleSheet("border: 1px solid #1a2540; border-radius: 6px; background: #141d33;")
        self.lbl_img_preview.hide()
        
        input_layout = QHBoxLayout()
        btn_attach = QPushButton("📎")
        btn_attach.setFixedSize(40, 40)
        btn_attach.clicked.connect(self.attach_image)
        btn_clear_img = QPushButton("✖")
        btn_clear_img.setFixedSize(30, 30)
        btn_clear_img.setStyleSheet("background: #ef4444; color: white; border: none;")
        btn_clear_img.clicked.connect(self.clear_image)

        self.txt_input = QTextEdit()
        self.txt_input.setFixedHeight(80)
        self.txt_input.setPlaceholderText("Ask a medical question...")
        
        self.btn_send = QPushButton("Send ➤")
        self.btn_send.setFixedHeight(80)
        self.btn_send.clicked.connect(self.send_message)

        input_layout.addWidget(btn_attach)
        input_layout.addWidget(btn_clear_img)
        input_layout.addWidget(self.lbl_img_preview)
        input_layout.addWidget(self.txt_input)
        input_layout.addWidget(self.btn_send)
        
        chat_layout.addLayout(input_layout)
        self.tabs.addTab(chat_widget, "💬 Chat")

        # TAB: Documents
        docs_widget = QWidget()
        docs_layout = QVBoxLayout(docs_widget)
        
        docs_layout.addWidget(QLabel("Upload Medical PDFs for RAG Context"))
        btn_upload_pdf = QPushButton("Upload PDF")
        btn_upload_pdf.clicked.connect(self.upload_pdf)
        docs_layout.addWidget(btn_upload_pdf)
        
        self.list_docs = QListWidget()
        docs_layout.addWidget(self.list_docs)
        
        btn_clear_docs = QPushButton("Clear All Documents")
        btn_clear_docs.setStyleSheet("background-color: #ef4444; color: white;")
        btn_clear_docs.clicked.connect(self.clear_docs)
        docs_layout.addWidget(btn_clear_docs)
        self.tabs.addTab(docs_widget, "📄 Documents")

        # TAB: Models
        self.models_widget = QWidget()
        self.models_layout = QVBoxLayout(self.models_widget)
        self.tabs.addTab(self.models_widget, "🧠 Models")

        # TAB: Stats
        stats_widget = QWidget()
        stats_layout = QVBoxLayout(stats_widget)
        self.lbl_stats = QLabel("Loading stats...")
        self.lbl_stats.setStyleSheet("font-family: monospace; font-size: 14px;")
        stats_layout.addWidget(self.lbl_stats)
        self.tabs.addTab(stats_widget, "📊 Stats")

        # TAB: Settings
        settings_widget = QWidget()
        s_layout = QVBoxLayout(settings_widget)
        
        g_gen = QGroupBox("Generation Settings")
        f_gen = QFormLayout()
        self.spin_temp = QDoubleSpinBox(); self.spin_temp.setRange(0.0, 2.0); self.spin_temp.setSingleStep(0.1)
        self.spin_max_tok = QSpinBox(); self.spin_max_tok.setRange(128, 8192); self.spin_max_tok.setSingleStep(128)
        f_gen.addRow("Temperature:", self.spin_temp)
        f_gen.addRow("Max Tokens:", self.spin_max_tok)
        g_gen.setLayout(f_gen)
        s_layout.addWidget(g_gen)

        g_sys = QGroupBox("System Prompt")
        v_sys = QVBoxLayout()
        self.txt_sys_prompt = QTextEdit()
        v_sys.addWidget(self.txt_sys_prompt)
        g_sys.setLayout(v_sys)
        s_layout.addWidget(g_sys)

        g_path = QGroupBox("Storage Paths")
        f_path = QFormLayout()
        self.txt_models_dir = QLineEdit()
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self.browse_models_dir)
        dir_lay = QHBoxLayout(); dir_lay.addWidget(self.txt_models_dir); dir_lay.addWidget(btn_browse)
        f_path.addRow("Models Dir:", dir_lay)
        g_path.setLayout(f_path)
        s_layout.addWidget(g_path)

        btn_save_settings = QPushButton("💾 Save Settings")
        btn_save_settings.setStyleSheet("background-color: #22c55e; color: white; font-size: 16px; padding: 10px;")
        btn_save_settings.clicked.connect(self.save_settings)
        s_layout.addWidget(btn_save_settings)
        s_layout.addStretch()

        self.tabs.addTab(settings_widget, "⚙️ Settings")

        # Assemble Main
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

        self.load_settings_ui()

    # --- UI Logic ---

    def new_convo(self):
        cid = convos.create()
        self.refresh_convos()
        self.select_convo(cid)

    def select_convo(self, cid):
        if not cid: return
        items = self.list_convos.findItems(cid, Qt.MatchExactly)
        if items: self.list_convos.setCurrentItem(items[0])

    def refresh_convos(self):
        self.list_convos.clear()
        for c in convos.list_all():
            item = QListWidgetItem(f"{c['title']} ({c['count']} msgs)")
            item.setData(Qt.UserRole, c['id'])
            self.list_convos.addItem(item)

    def load_convo(self, current, previous):
        if not current: return
        cid = current.data(Qt.UserRole)
        convos.cur = cid
        self.current_convo = cid
        self.render_chat()

    def render_chat(self):
        self.txt_chat.clear()
        c = convos.get(self.current_convo)
        if not c: return
        
        import markdown
        for m in c['messages']:
            role = m['role'].upper()
            content = m['content']
            html_content = markdown.markdown(content) if role == "ASSISTANT" else f"<p>{content}</p>"
            
            color = "#0ea5e9" if role == "USER" else "#14b8a6"
            stats_html = f"<span style='font-size:10px; color:#5a6f94;'>Tokens: {m['stats']['prompt_tokens']}+{m['stats']['completion_tokens']} | Speed: {m['stats']['tps']} tok/s</span>" if m.get('stats') else ""
            
            block = f"""
            <div style="margin-bottom: 15px; padding: 10px; background: #141d33; border-radius: 8px; border-left: 4px solid {color};">
                <b style="color:{color};">{role}</b> {stats_html}<br>
                <div style="margin-top:5px;">{html_content}</div>
            </div>
            """
            self.txt_chat.append(block)

    def attach_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Attach Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            with open(path, "rb") as f: img_data = base64.b64encode(f.read()).decode("utf-8")
            self.attached_image_base64 = img_data
            
            pixmap = QPixmap(path)
            self.lbl_img_preview.setPixmap(pixmap.scaled(100, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_img_preview.show()

    def clear_image(self):
        self.attached_image_base64 = None
        self.lbl_img_preview.hide()

    def send_message(self):
        if not llm.loaded:
            QMessageBox.warning(self, "No Model", "Please load a model first in the Models tab.")
            return
        
        txt = self.txt_input.toPlainText().strip()
        if not txt: return

        if not self.current_convo:
            self.new_convo()

        convos.add(self.current_convo, "user", txt)
        self.txt_input.clear()
        self.render_chat()

        self.btn_send.setEnabled(False)
        self.txt_chat.append("<div style='color: #5a6f94; padding:10px;'><i>Thinking...</i></div>")

        messages = convos.llm_messages(self.current_convo)
        worker = ChatWorker(messages, self.attached_image_base64)
        worker.finished.connect(self.on_chat_finished)
        self.workers.append(worker)
        worker.start()
        self.clear_image()

    def on_chat_finished(self, text, stats, error):
        self.btn_send.setEnabled(True)
        if error:
            self.txt_chat.append(f"<div style='color:#ef4444; padding:10px;'>Error: {error}</div>")
        else:
            convos.add(self.current_convo, "assistant", text, stats=stats)
        self.render_chat()
        self.refresh_convos()

    # --- Documents ---
    def upload_pdf(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Upload Medical PDFs", "", "PDFs (*.pdf)")
        for p in paths:
            try:
                from PyPDF2 import PdfReader
                reader = PdfReader(p)
                text = ""
                for page in reader.pages:
                    t = page.extract_text()
                    if t: text += t + "\n"
                if text.strip():
                    rag.add_document(text, Path(p).name)
            except Exception as e:
                QMessageBox.warning(self, "PDF Error", f"Failed to read {Path(p).name}:\n{e}")
        self.refresh_docs()

    def refresh_docs(self):
        self.list_docs.clear()
        grouped = {}
        for d in rag.docs: grouped[d["name"]] = grouped.get(d["name"], 0) + 1
        for n, c in grouped.items():
            self.list_docs.addItem(f"📄 {n} ({c} chunks)")

    def clear_docs(self):
        rag.clear(); self.refresh_docs()

    # --- Models ---
    def refresh_models(self):
        # Clear existing model widgets
        while self.models_layout.count():
            item = self.models_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
            elif item.layout(): 
                while item.layout().count():
                    sub = item.layout().takeAt(0)
                    if sub.widget(): sub.widget().deleteLater()

        for k, cfg in MODEL_CONFIGS.items():
            frame = QGroupBox(f"{k} ({cfg['size_gb']}GB)")
            frame_layout = QVBoxLayout(frame)
            frame_layout.addWidget(QLabel(cfg['description']))
            
            is_loaded = llm.model_key == k
            is_downloaded = model_mgr.is_downloaded(k)
            is_loading = llm.loading and llm.load_msg and llm.model_key is None # simplistic check

            btn_layout = QHBoxLayout()
            
            if is_loaded:
                btn_active = QPushButton("✓ Active")
                btn_active.setEnabled(False)
                btn_active.setStyleSheet("background-color: #22c55e; color: white; border: none;")
                btn_unload = QPushButton("Unload Model")
                btn_unload.setObjectName("UnloadBtn")
                btn_unload.clicked.connect(lambda checked, key=k: self.unload_model(key))
                btn_layout.addWidget(btn_active)
                btn_layout.addWidget(btn_unload)
            elif is_loading:
                btn_load = QPushButton("Loading... Please Wait")
                btn_load.setEnabled(False)
                btn_layout.addWidget(btn_load)
            elif is_downloaded:
                btn_load = QPushButton("Load Model")
                btn_load.setStyleSheet("background-color: #0ea5e9; color: white;")
                btn_load.clicked.connect(lambda checked, key=k: self.load_model(key))
                btn_layout.addWidget(btn_load)
            elif self.active_dl_key == k:
                prog = QProgressBar()
                prog.setMinimum(0); prog.setMaximum(100)
                prog.setObjectName(f"prog_{k}")
                btn_layout.addWidget(prog)
                self.dl_progress_timer.start(1000) # Check file size every second
            else:
                btn_dl = QPushButton(f"Download ({cfg['size_gb']}GB)")
                btn_dl.clicked.connect(lambda checked, key=k: self.download_model(key))
                btn_layout.addWidget(btn_dl)

            frame_layout.addLayout(btn_layout)
            self.models_layout.addWidget(frame)
        
        self.models_layout.addStretch()

    def download_model(self, key):
        self.active_dl_key = key
        worker = DownloadWorker(key)
        worker.finished.connect(self.on_download_finished)
        self.workers.append(worker)
        worker.start()
        self.refresh_models() # Show progress bar

    def update_download_progress(self):
        if not self.active_dl_key: 
            self.dl_progress_timer.stop()
            return
        
        cfg = MODEL_CONFIGS.get(self.active_dl_key)
        if not cfg: return

        target_file = Path(settings.get("models_dir")) / cfg["filename"]
        if target_file.exists():
            current_size_bytes = target_file.stat().st_size
            target_size_bytes = cfg["size_gb"] * 1024 * 1024 * 1024
            progress = int((current_size_bytes / target_size_bytes) * 100)
            
            # Find progress bar and update
            prog_bar = self.findChild(QProgressBar, f"prog_{self.active_dl_key}")
            if prog_bar:
                prog_bar.setValue(min(progress, 99)) # Cap at 99 until fully complete

    def on_download_finished(self, ok, msg):
        dl_key = self.active_dl_key
        self.active_dl_key = None
        self.dl_progress_timer.stop()
        
        if ok:
            QMessageBox.information(self, "Success", f"Model {dl_key} downloaded successfully!")
        else:
            QMessageBox.critical(self, "Download Error", f"Failed to download {dl_key}:\n{msg}")
        
        model_mgr._sync_with_folder() # Ensure registry is up to date
        self.refresh_models()

    def load_model(self, key):
        if llm.loaded:
            reply = QMessageBox.question(self, 'Unload Current Model', 
                                         f'A model ({llm.model_key}) is already loaded. Unload it and load {key} instead?',
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: return
        
        worker = LoadWorker(key)
        worker.finished.connect(self.on_load_finished)
        self.workers.append(worker)
        worker.start()
        self.refresh_models() # Will show 'Loading...'

    def on_load_finished(self, ok, msg):
        if ok: print(f"Model loaded: {msg}")
        else: QMessageBox.critical(self, "Load Error", msg)
        self.refresh_models()

    def unload_model(self, key):
        worker = UnloadWorker()
        worker.finished.connect(self.on_unload_finished)
        self.workers.append(worker)
        worker.start()
        self.refresh_models()

    def on_unload_finished(self, msg):
        print(msg)
        self.refresh_models()

    # --- Stats ---
    def refresh_stats(self):
        avg_tps = round(STATS_DATA["total_tps"]/STATS_DATA["tps_count"], 2) if STATS_DATA["tps_count"]>0 else 0
        up_secs = int(time.time() - STATS_DATA["start_time"])
        up_mins = up_secs // 60
        
        self.lbl_stats.setText(f"""
            <h3>Usage Statistics</h3>
            <p><b>Total Requests:</b> {STATS_DATA['total_requests']}</p>
            <p><b>Total Tokens:</b> {STATS_DATA['total_prompt_tokens'] + STATS_DATA['total_completion_tokens']}</p>
            <p><b>Avg Tokens/Sec:</b> {avg_tps}</p>
            <p><b>Uptime:</b> {up_mins} minutes</p>
            <hr>
            <h3>System Info</h3>
            <p><b>OS:</b> {si.os_name}</p>
            <p><b>CPU:</b> {si.cpu}</p>
            <p><b>RAM:</b> {si.ram_gb} GB</p>
            <p><b>GPU:</b> {si.gpu} ({si.vram_gb} GB VRAM)</p>
            <p><b>CUDA:</b> {'Yes' if si.has_cuda else 'No'}</p>
            <p><b>Tier:</b> {si.tier().upper()}</p>
        """)

    # --- Settings ---
    def load_settings_ui(self):
        self.spin_temp.setValue(settings.get("temperature", 0.7))
        self.spin_max_tok.setValue(settings.get("max_tokens", 1024))
        self.txt_sys_prompt.setPlainText(settings.get("system_prompt", DEFAULT_SYSTEM_PROMPT))
        self.txt_models_dir.setText(settings.get("models_dir", str(DEFAULT_MODELS_DIR)))

    def save_settings(self):
        settings.set("temperature", self.spin_temp.value())
        settings.set("max_tokens", self.spin_max_tok.value())
        settings.set("system_prompt", self.txt_sys_prompt.toPlainText())
        new_dir = self.txt_models_dir.text().strip()
        if new_dir != settings.get("models_dir"):
            settings.set("models_dir", new_dir)
            Path(new_dir).mkdir(parents=True, exist_ok=True)
            model_mgr.models_dir = Path(new_dir)
            model_mgr._sync_with_folder()
            self.refresh_models()
            
        QMessageBox.information(self, "Saved", "Settings saved successfully!")

    def browse_models_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Models Directory")
        if d: self.txt_models_dir.setText(d)

# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    check_dependencies()
    CONVOS_DIR.mkdir(parents=True, exist_ok=True)
    Path(settings.get("models_dir")).mkdir(parents=True, exist_ok=True)
    
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    
    window = MedicalAssistantUI()
    window.show()
    
    sys.exit(app.exec())