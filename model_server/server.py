import engine_v2
"""
Local FastAPI AI Model Translation Microservice (No External APIs)
"""
import re
import os
import sys
import shutil
import subprocess
import tempfile
import time
import threading
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from model_config import (
    SUPPORTED_LANGUAGES,
    normalize_language,
    build_translation_prompt,
    DEFAULT_BASE_MODEL,
    GENERATION_CONFIG
)

# --- Singleton AI Model Cache (loaded once at startup) ---
_ai_model = None
_ai_tokenizer = None
_ai_model_loaded = False

def load_ai_model():
    """Loads local HuggingFace DeepSeek-Coder model in bfloat16 (memory efficient, no OS 1455 error)."""
    global _ai_model, _ai_tokenizer, _ai_model_loaded
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[Altern] Loading AI model '{DEFAULT_BASE_MODEL}' (bfloat16, low_cpu_mem_usage)...")
        _ai_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
        _ai_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True
        )
        if torch.cuda.is_available():
            _ai_model = _ai_model.to("cuda")
        _ai_model_loaded = True
        print("[Altern] AI model loaded successfully in bfloat16!")
    except Exception as e:
        print(f"[Altern] AI model not loaded (will use structural engine fallback): {e}")
        _ai_model_loaded = False

from contextlib import asynccontextmanager
import asyncio

@asynccontextmanager
async def lifespan(app):
    """Startup hook — loads AI model in background so server starts immediately."""
    threading.Thread(target=load_ai_model, daemon=True).start()
    yield

app = FastAPI(
    title="Altern Local Translation AI Server",
    description="Self-hosted Python microservice for local code conversion between programming languages.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
FRONTEND_INDEX = os.path.join(FRONTEND_DIR, "index.html")

@app.get("/")
def root():
    """Root endpoint - serves web frontend UI if present or server status JSON."""
    if os.path.exists(FRONTEND_INDEX):
        return FileResponse(FRONTEND_INDEX)
    return {
        "status": "online",
        "service": "Altern Local Translation AI Server",
        "version": "1.0.0",
        "endpoints": {
            "translate": "POST /translate",
            "execute": "POST /execute",
            "health": "GET /health",
            "docs": "GET /docs"
        }
    }

@app.get("/health")
def health():
    """Health check endpoint for frontend status checks and monitoring."""
    return {
        "status": "ok",
        "service": "Altern Local Translation AI Server",
        "model_loaded": _ai_model_loaded,
        "base_model": DEFAULT_BASE_MODEL,
        "supported_languages": list(SUPPORTED_LANGUAGES.keys())
    }


class TranslationRequest(BaseModel):
    source_code: str = Field(..., description="Original code string to convert")
    source_lang: str = Field(..., description="Source programming language (cpp, java, python, javascript, typescript)")
    target_lang: str = Field(..., description="Target programming language (cpp, java, python, javascript, typescript)")

class TranslationResponse(BaseModel):
    status: str
    source_lang: str
    target_lang: str
    converted_code: str
    model_used: str

class ExecuteRequest(BaseModel):
    code: str = Field(..., description="Source or Target code to execute")
    language: str = Field(..., description="Language (cpp, java, python, javascript, typescript)")
    stdin: str = Field(default="", description="Optional stdin to pipe into the program (newline-separated)")

class ExecuteResponse(BaseModel):
    status: str
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float

def _build_node_shim() -> str:
    """Returns a Node.js preload shim that adds prompt/readline/alert/print.
    Uses synchronous, line-by-line reads from fd 0 so it works for both
    batch (POST /execute) and interactive (WS /ws/execute) execution."""
    return (
        "const fs = require('fs');\n"
        "const _buf1 = Buffer.alloc(1);\n"
        "function __readLine() {\n"
        "  let line = '';\n"
        "  try {\n"
        "    let n;\n"
        "    while ((n = fs.readSync(0, _buf1, 0, 1)) > 0) {\n"
        "      const ch = _buf1.toString('utf-8');\n"
        "      if (ch === '\\n') break;\n"
        "      if (ch !== '\\r') line += ch;\n"
        "    }\n"
        "  } catch (e) {}\n"
        "  return line;\n"
        "}\n"
        "global.prompt   = function(msg) { return __readLine(); };\n"
        "global.readline = function()    { return __readLine(); };\n"
        "global.alert    = function(...a){ console.log(...a); };\n"
        "global.print    = function(...a){ console.log(...a); };\n"
    )

def clean_model_output(output_text: str) -> str:
    """Strips markdown codeblock wrappers and trailing commentary."""
    text = output_text.strip()
    # Case 1: if wrapped in ```lang ... ```
    m = re.search(r'```(?:[a-zA-Z0-9_+#-]+)?\s*\n(.*?)```', text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    # Case 2: generated code is continuation of prompt ```lang, so it ends at ```
    if '```' in text:
        candidate = text.split('```')[0].strip()
        if candidate:
            return candidate
    clean = re.sub(r"^```[a-zA-Z]*\n", "", text, flags=re.MULTILINE)
    clean = re.sub(r"\n```$", "", clean, flags=re.MULTILINE)
    return clean.strip('` \n\r')

# Timeout (seconds) for local LLM inference
def _get_llm_timeout() -> int:
    try:
        import torch
        if torch.cuda.is_available():
            return 60
    except Exception:
        pass
    return 10

def _run_hf_inference(prompt: str, result_box: list) -> None:
    """Runs HuggingFace model.generate() in a thread; stores result in result_box[0]."""
    global _ai_model, _ai_tokenizer
    try:
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        class StopOnCodeBlock(StoppingCriteria):
            def __init__(self, tokenizer, prompt_len):
                self.tokenizer = tokenizer
                self.prompt_len = prompt_len
                self.backtick_token = 63

            def __call__(self, input_ids, scores, **kwargs):
                last_tok = input_ids[0][-1].item()
                if last_tok == 32021 or (self.tokenizer.eos_token_id and last_tok == self.tokenizer.eos_token_id):
                    return True
                # DeepSeek-Coder token 63 is '```'
                if last_tok == self.backtick_token:
                    gen_len = len(input_ids[0]) - self.prompt_len
                    if gen_len > 3:  # Only stop if some code has been generated
                        return True
                return False

        inputs = _ai_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        input_len = inputs.input_ids.shape[1]

        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
        else:
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 4)))

        calc_max_tokens = min(max(96, input_len * 2), 384)

        kwargs = {
            "max_new_tokens": calc_max_tokens,
            "do_sample": False,
            "pad_token_id": _ai_tokenizer.eos_token_id,
            "stopping_criteria": StoppingCriteriaList([StopOnCodeBlock(_ai_tokenizer, input_len)])
        }

        with torch.no_grad():
            outputs = _ai_model.generate(**inputs, **kwargs)
        decoded = _ai_tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        result_box[0] = clean_model_output(decoded)
    except Exception as e:
        print(f"[Altern] Model inference error: {e}")

def try_ai_model_translation(source_code: str, src_lang: str, tgt_lang: str) -> str:
    """Attempts local LLM translation. Falls through to structural engine if unavailable or timed out."""
    prompt = build_translation_prompt(source_code, src_lang, tgt_lang)

    # 1. Try local Ollama instance if available
    try:
        import requests as _req
        resp = _req.post(
            "http://localhost:11434/api/generate",
            json={"model": "deepseek-coder", "prompt": prompt, "stream": False},
            timeout=2.0
        )
        if resp.status_code == 200:
            data = resp.json()
            if "response" in data and data["response"].strip():
                return clean_model_output(data["response"])
    except Exception:
        pass

    # 2. Use pre-loaded local HuggingFace DeepSeek-Coder model
    global _ai_model, _ai_tokenizer
    if _ai_model is None or _ai_tokenizer is None:
        return None  # Still loading or not loaded — fall through to structural engine

    result_box = [None]
    timeout_secs = _get_llm_timeout()
    t = threading.Thread(target=_run_hf_inference, args=(prompt, result_box), daemon=True)
    t.start()
    t.join(timeout=timeout_secs)

    if t.is_alive():
        print(f"[Altern] LLM inference timed out after {timeout_secs}s — using structural fallback")
        return None

    return result_box[0]


def chunk_code_by_structures(code: str, max_lines: int = 35) -> list:
    """Splits large code into logical chunks by function/class boundaries."""
    lines = code.splitlines()
    if len(lines) <= max_lines:
        return [code]

    chunks = []
    current_chunk = []
    brace_depth = 0

    for line in lines:
        stripped = line.strip()
        current_chunk.append(line)
        brace_depth += stripped.count('{') - stripped.count('}')

        # Break chunk at top level boundary if chunk size exceeded
        if brace_depth <= 0 and len(current_chunk) >= max_lines:
            chunks.append("\n".join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def _extract_balanced_paren(raw: str, prefix: str) -> str | None:
    idx = raw.find(prefix + "(")
    if idx == -1:
        return None
    start = idx + len(prefix) + 1
    depth = 1
    i = start
    while i < len(raw) and depth > 0:
        if raw[i] == '(':
            depth += 1
        elif raw[i] == ')':
            depth -= 1
        i += 1
    if depth == 0:
        return raw[start:i-1].strip()
    return None

def _strip_typescript_types(code: str) -> str:
    """Pre-process TypeScript to JavaScript by stripping type annotations."""
    code = re.sub(r'\binterface\s+\w+\s*\{[^}]*\}', '', code, flags=re.DOTALL)
    code = re.sub(r'\btype\s+\w+\s*=[^;]+;', '', code)
    code = re.sub(r'\)\s*:\s*[\w\[\]<>|,\s]+(?=\s*\{)', ')', code)
    code = re.sub(r'(\w+)\s*\?\s*:\s*[\w\[\]<>|,\s]+', r'\1', code)
    code = re.sub(r'(\w+)\s*:\s*[\w\[\]<>|,\s]+(?=[,)])', r'\1', code)
    code = re.sub(r'(const|let|var)\s+(\w+)\s*:\s*[\w\[\]<>|,\s]+\s*=', r'\1 \2 =', code)
    code = re.sub(r'<[\w\s,|]+>', '', code)
    return code

def _normalize_code(code: str) -> str:
    """Safely normalize code lines while protecting initializer lists like = {10, 20, 30};"""
    protected = []
    def protect_init(m):
        content = m.group(0)
        inner = m.group(2)
        if ";" not in inner and "if " not in inner and "for " not in inner:
            idx = len(protected)
            protected.append(content)
            return f"__INIT_LIST_{idx}__"
        return content

    subbed = re.sub(r'(=|\breturn\b)\s*\{([^{}]+)\}', protect_init, code)

    out = []
    in_str = None
    for ch in subbed:
        if in_str:
            out.append(ch)
            if ch == in_str:
                in_str = None
        elif ch in ('"', "'", '`'):
            in_str = ch
            out.append(ch)
        elif ch == '{':
            out.append('\n{\n')
        elif ch == '}':
            out.append('\n}\n')
        else:
            out.append(ch)

    raw_lines = "".join(out).splitlines()
    cleaned = []
    for line in raw_lines:
        s = line.strip()
        if not s:
            continue
        if ";" in s and not s.startswith("for") and not s.startswith("//"):
            parts = [p.strip() + ";" for p in s.split(";") if p.strip()]
            cleaned.extend(parts)
        else:
            cleaned.append(s)

    res = "\n".join(cleaned)
    for idx, orig in enumerate(protected):
        res = res.replace(f"__INIT_LIST_{idx}__", orig)
    return res

def _python_to_braced_blocks(code: str) -> str:
    """Converts Python indent-based blocks into explicit { and } blocks."""
    lines = code.splitlines()
    out = []
    indent_stack = [0]
    
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        
        while indent < indent_stack[-1]:
            indent_stack.pop()
            out.append("}")
            
        out.append(stripped)
        if stripped.endswith(":"):
            indent_stack.append(indent + 1)
            out.append("{")
            
    while len(indent_stack) > 1:
        indent_stack.pop()
        out.append("}")
        
    return "\n".join(out)

def _parse_array_decl(raw: str):
    m = re.search(
        r'(?:(?:vector<[^>]+>|std::vector<[^>]+>|List<[^>]+>|ArrayList<[^>]+>|\w+\[\]|const|let|var|auto|int|double|float|string|String)\s+)*'
        r'(\w+)(?:\[\d*\])?\s*=\s*(?:new\s+[\w\[\]<>]+)?\s*[\{\[]\s*([^\{\}\[\]]*)\s*[\}\]];?',
        raw
    )
    if m and m.group(1) not in ("for", "if", "while", "return"):
        return m.group(1), m.group(2).strip()
    return None

def _parse_foreach(raw: str):
    m = re.search(r'for\s*\(\s*(?:const\s+)?(?:auto|int|float|double|var|let|const|[\w:<>]+)\s*&?\s*(\w+)\s*:\s*(\w+)\s*\)', raw)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r'for\s+(\w+)\s+in\s+(\w+):', raw)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r'for\s*\(\s*(?:const|let|var)?\s*(\w+)\s+(?:of|in)\s*(\w+)\s*\)', raw)
    if m:
        return m.group(1), m.group(2)
    return None

def rule_assisted_fallback_translation(source_code: str, src_lang: str, tgt_lang: str) -> str:
    """High-speed structural translation engine -- fully converts and structures all 20 language pairs."""
    return engine_v2.full_translate(source_code, src_lang, tgt_lang)

@app.post("/translate", response_model=TranslationResponse)
def translate_code(req: TranslationRequest):
    try:
        src_clean = normalize_language(req.source_lang)
        tgt_clean = normalize_language(req.target_lang)
    except ValueError as val_err:
        raise HTTPException(status_code=400, detail=str(val_err))

    if src_clean == tgt_clean:
        return TranslationResponse(
            status="success",
            source_lang=src_clean,
            target_lang=tgt_clean,
            converted_code=req.source_code,
            model_used="passthrough"
        )

    source_line_count = len(req.source_code.strip().splitlines())

    # 1. Primary engine: Real local DeepSeek-Coder AI model
    ai_result = try_ai_model_translation(req.source_code, src_clean, tgt_clean)
    if ai_result and ai_result.strip():
        return TranslationResponse(
            status="success",
            source_lang=src_clean,
            target_lang=tgt_clean,
            converted_code=clean_model_output(ai_result),
            model_used="deepseek-ai/deepseek-coder-1.3b-instruct"
        )

    # 2. Fallback: Fast structural engine if model is still loading or unavailable
    converted = rule_assisted_fallback_translation(req.source_code, src_clean, tgt_clean)
    return TranslationResponse(
        status="success",
        source_lang=src_clean,
        target_lang=tgt_clean,
        converted_code=clean_model_output(converted),
        model_used="structural_fallback"
    )


def run_code_execution(code: str, language: str, stdin_input: str = "") -> dict:
    try:
        lang = normalize_language(language)
    except ValueError:
        lang = language.lower()

    start_time = time.time()
    # Always pipe stdin — empty string gives immediate EOF, preventing hangs
    stdin_bytes = stdin_input if stdin_input else ""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        if lang == "python":
            file_path = os.path.join(temp_dir, "script.py")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            try:
                res = subprocess.run(
                    [sys.executable, file_path],
                    input=stdin_bytes,
                    capture_output=True,
                    text=True,
                    timeout=10.0
                )
                exec_time = round((time.time() - start_time) * 1000, 2)
                return {
                    "status": "success" if res.returncode == 0 else "error",
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "exit_code": res.returncode,
                    "execution_time_ms": exec_time
                }
            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "stdout": "",
                    "stderr": "Execution timed out (10s limit)",
                    "exit_code": -1,
                    "execution_time_ms": 10000.0
                }

        elif lang in ("javascript", "typescript"):
            js_code = _strip_typescript_types(code) if lang == "typescript" else code

            shim_path = os.path.join(temp_dir, "shim.js")
            with open(shim_path, "w", encoding="utf-8") as f:
                f.write(_build_node_shim())

            file_path = os.path.join(temp_dir, "script.js")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(js_code)

            node_path = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
            try:
                res = subprocess.run(
                    [node_path, "-r", shim_path, file_path],
                    input=stdin_bytes,
                    capture_output=True,
                    text=True,
                    timeout=10.0
                )
                exec_time = round((time.time() - start_time) * 1000, 2)
                return {
                    "status": "success" if res.returncode == 0 else "error",
                    "stdout": res.stdout,
                    "stderr": res.stderr,
                    "exit_code": res.returncode,
                    "execution_time_ms": exec_time
                }
            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "stdout": "",
                    "stderr": "Execution timed out (10s limit)",
                    "exit_code": -1,
                    "execution_time_ms": 10000.0
                }
            except Exception as e:
                return {
                    "status": "error",
                    "stdout": "",
                    "stderr": f"Node execution error: {str(e)}",
                    "exit_code": 1,
                    "execution_time_ms": 0
                }

        elif lang == "cpp":
            cpp_path = os.path.join(temp_dir, "main.cpp")
            exe_path = os.path.join(temp_dir, "main.exe")
            with open(cpp_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            gpp_path = shutil.which("g++") or r"C:\MinGW\bin\g++.exe"
            try:
                comp_res = subprocess.run(
                    [gpp_path, "-std=c++14", cpp_path, "-o", exe_path],
                    capture_output=True,
                    text=True,
                    timeout=8.0
                )
                if comp_res.returncode != 0:
                    return {
                        "status": "compile_error",
                        "stdout": "",
                        "stderr": f"Compilation Error:\n{comp_res.stderr}",
                        "exit_code": comp_res.returncode,
                        "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                
                exec_res = subprocess.run(
                    [exe_path],
                    input=stdin_bytes,
                    capture_output=True,
                    text=True,
                    timeout=5.0
                )
                exec_time = round((time.time() - start_time) * 1000, 2)
                return {
                    "status": "success" if exec_res.returncode == 0 else "error",
                    "stdout": exec_res.stdout,
                    "stderr": exec_res.stderr,
                    "exit_code": exec_res.returncode,
                    "execution_time_ms": exec_time
                }
            except subprocess.TimeoutExpired:
                return {
                    "status": "timeout",
                    "stdout": "",
                    "stderr": "Execution timed out (5s limit)",
                    "exit_code": -1,
                    "execution_time_ms": 5000.0
                }
            except Exception as e:
                return {
                    "status": "error",
                    "stdout": "",
                    "stderr": f"C++ compilation error: {str(e)}",
                    "exit_code": 1,
                    "execution_time_ms": 0
                }

        elif lang == "java":
            javac_path = shutil.which("javac")
            java_path = shutil.which("java")
            if not (javac_path and java_path):
                import glob
                for p in glob.glob(r"C:\Program Files\Eclipse Adoptium\*\bin\javac.exe"):
                    javac_path = p
                    java_path = p.replace("javac.exe", "java.exe")
                    break
            
            if javac_path and java_path:
                java_file = os.path.join(temp_dir, "Main.java")
                with open(java_file, "w", encoding="utf-8") as f:
                    f.write(code)
                try:
                    comp_res = subprocess.run(
                        [javac_path, java_file],
                        capture_output=True,
                        text=True,
                        timeout=8.0
                    )
                    if comp_res.returncode != 0:
                        return {
                            "status": "compile_error",
                            "stdout": "",
                            "stderr": f"Java Compilation Error:\n{comp_res.stderr}",
                            "exit_code": comp_res.returncode,
                            "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                        }
                    
                    exec_res = subprocess.run(
                        [java_path, "-cp", temp_dir, "Main"],
                        input=stdin_bytes,
                        capture_output=True,
                        text=True,
                        timeout=5.0
                    )
                    return {
                        "status": "success" if exec_res.returncode == 0 else "error",
                        "stdout": exec_res.stdout,
                        "stderr": exec_res.stderr,
                        "exit_code": exec_res.returncode,
                        "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                    }
                except Exception as e:
                    return {
                        "status": "error",
                        "stdout": "",
                        "stderr": str(e),
                        "exit_code": 1,
                        "execution_time_ms": 0
                    }
            else:
                # JDK not installed — cannot compile or run Java. Return a clear error.
                return {
                    "status": "error",
                    "stdout": "",
                    "stderr": (
                        "Java (JDK) is not installed or not on PATH.\n"
                        "Install JDK from https://adoptium.net and ensure 'javac' is on PATH,\n"
                        "then restart the server."
                    ),
                    "exit_code": 1,
                    "execution_time_ms": round((time.time() - start_time) * 1000, 2)
                }

    return {"status": "error", "stdout": "", "stderr": f"Unsupported language: {language}", "exit_code": 1, "execution_time_ms": 0}

@app.post("/execute", response_model=ExecuteResponse)
def execute_code(req: ExecuteRequest):
    if not req.code or not req.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    
    result = run_code_execution(req.code, req.language, req.stdin or "")
    return ExecuteResponse(
        status=result["status"],
        stdout=result["stdout"],
        stderr=result["stderr"],
        exit_code=result["exit_code"],
        execution_time_ms=result["execution_time_ms"]
    )

@app.websocket("/ws/execute")
async def ws_execute(websocket: WebSocket):
    """Interactive WebSocket terminal: streams stdout in real-time and accepts
    stdin keystrokes from the browser as the program runs."""
    await websocket.accept()
    proc = None
    tmp_dir_obj = None
    start_time = time.time()
    loop = asyncio.get_event_loop()

    async def _send(msg: dict):
        try:
            await websocket.send_json(msg)
        except Exception:
            pass

    try:
        # ── 1. Receive start message ──────────────────────────────────────────
        init = await asyncio.wait_for(websocket.receive_json(), timeout=20.0)
        if init.get("type") != "start":
            await websocket.close(code=1002)
            return

        raw_code = init.get("code", "").strip()
        raw_lang = init.get("language", "")

        if not raw_code:
            await _send({"type": "error", "message": "Code is empty."})
            await websocket.close()
            return

        try:
            lang = normalize_language(raw_lang)
        except ValueError:
            lang = raw_lang.lower()

        # ── 2. Set up temp directory + command ───────────────────────────────
        tmp_dir_obj = tempfile.TemporaryDirectory()
        tmp = tmp_dir_obj.name
        cmd = None

        if lang == "python":
            fpath = os.path.join(tmp, "script.py")
            with open(fpath, "w", encoding="utf-8") as fh:
                fh.write(raw_code)
            cmd = [sys.executable, "-u", fpath]   # -u = unbuffered

        elif lang in ("javascript", "typescript"):
            js_code = _strip_typescript_types(raw_code) if lang == "typescript" else raw_code
            shim_p = os.path.join(tmp, "shim.js")
            with open(shim_p, "w", encoding="utf-8") as fh:
                fh.write(_build_node_shim())
            fpath = os.path.join(tmp, "script.js")
            with open(fpath, "w", encoding="utf-8") as fh:
                fh.write(js_code)
            node_p = shutil.which("node") or r"C:\Program Files\nodejs\node.exe"
            cmd = [node_p, "-r", shim_p, fpath]

        elif lang == "cpp":
            cpp_p = os.path.join(tmp, "main.cpp")
            exe_p = os.path.join(tmp, "main.exe")
            with open(cpp_p, "w", encoding="utf-8") as fh:
                fh.write(raw_code)
            gpp = shutil.which("g++") or r"C:\MinGW\bin\g++.exe"
            await _send({"type": "output", "data": "\x1b[33mCompiling C++…\x1b[0m\r\n"})
            comp = await loop.run_in_executor(None, lambda: subprocess.run(
                [gpp, "-std=c++14", cpp_p, "-o", exe_p],
                capture_output=True, text=True, timeout=15.0
            ))
            if comp.returncode != 0:
                err = comp.stderr or comp.stdout
                await _send({"type": "output", "data": f"\x1b[31mCompilation Error:\x1b[0m\n{err}"})
                await _send({"type": "exit", "code": 1,
                             "time_ms": round((time.time() - start_time) * 1000, 2)})
                return
            await _send({"type": "output", "data": "\x1b[32mCompiled OK\x1b[0m\r\n\r\n"})
            cmd = [exe_p]

        elif lang == "java":
            javac_p = shutil.which("javac")
            java_p  = shutil.which("java")
            if not (javac_p and java_p):
                import glob
                for p in glob.glob(r"C:\Program Files\Eclipse Adoptium\*\bin\javac.exe"):
                    javac_p = p
                    java_p  = p.replace("javac.exe", "java.exe")
                    break
            if not (javac_p and java_p):
                await _send({"type": "error", "message": "Java (JDK) not found on PATH."})
                return
            jfile = os.path.join(tmp, "Main.java")
            with open(jfile, "w", encoding="utf-8") as fh:
                fh.write(raw_code)
            await _send({"type": "output", "data": "\x1b[33mCompiling Java…\x1b[0m\r\n"})
            comp = await loop.run_in_executor(None, lambda: subprocess.run(
                [javac_p, jfile], capture_output=True, text=True, timeout=15.0
            ))
            if comp.returncode != 0:
                await _send({"type": "output", "data": f"\x1b[31mCompilation Error:\x1b[0m\n{comp.stderr}"})
                await _send({"type": "exit", "code": 1,
                             "time_ms": round((time.time() - start_time) * 1000, 2)})
                return
            await _send({"type": "output", "data": "\x1b[32mCompiled OK\x1b[0m\r\n\r\n"})
            cmd = [java_p, "-cp", tmp, "Main"]

        else:
            await _send({"type": "error", "message": f"Unsupported language: {raw_lang}"})
            return

        # ── 3. Spawn the process ──────────────────────────────────────────────
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
        )

        # ── 4. Concurrent tasks: stream stdout + relay stdin ─────────────────
        async def _stream_stdout():
            """Read process stdout in chunks and forward to browser."""
            while True:
                chunk = await loop.run_in_executor(None, proc.stdout.read, 64)
                if not chunk:
                    break
                await _send({"type": "output",
                             "data": chunk.decode("utf-8", errors="replace")})

        async def _relay_stdin():
            """Forward browser keystrokes to process stdin; handle kill."""
            while True:
                try:
                    msg = await websocket.receive_json()
                    mtype = msg.get("type", "")
                    if mtype == "input":
                        data_bytes = msg.get("data", "").encode("utf-8")
                        try:
                            proc.stdin.write(data_bytes)
                            proc.stdin.flush()
                        except OSError:
                            return   # process already closed stdin
                    elif mtype == "kill":
                        try:
                            proc.kill()
                        except Exception:
                            pass
                        return
                except WebSocketDisconnect:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                    return
                except Exception:
                    return

        reader = asyncio.create_task(_stream_stdout())
        relay  = asyncio.create_task(_relay_stdin())

        # Wait for whichever finishes first (process exits OR kill signal)
        done, pending = await asyncio.wait(
            [reader, relay], return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except asyncio.CancelledError:
                pass

        # Ensure the process has fully exited
        try:
            await loop.run_in_executor(None, lambda: proc.wait(timeout=3.0))
        except Exception:
            pass

        exit_code = proc.returncode if proc.returncode is not None else -1
        exec_ms   = round((time.time() - start_time) * 1000, 2)
        await _send({"type": "exit", "code": exit_code, "time_ms": exec_ms})

    except WebSocketDisconnect:
        pass
    except asyncio.TimeoutError:
        await _send({"type": "error", "message": "Timed out waiting for start message."})
    except Exception as exc:
        await _send({"type": "error", "message": str(exc)})
    finally:
        if proc and proc.returncode is None:
            try:
                proc.kill()
            except Exception:
                pass
        if tmp_dir_obj:
            try:
                tmp_dir_obj.cleanup()
            except Exception:
                pass

if __name__ == "__main__":
    print("Starting Altern Local Translation AI Server on port 8000...")
    print("[Altern] Mode: Fast Structural Engine (instant startup, no RAM spike)")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
