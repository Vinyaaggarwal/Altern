# Altern — AI Code Translator & Interactive Execution Playground

Altern is a high-performance, self-hosted developer tool for translating code seamlessly between **C++**, **Java**, **Python**, **JavaScript**, and **TypeScript**. It combines an instant structural rule/AST translation engine with local AI model support (`deepseek-ai/deepseek-coder-1.3b-instruct`) and includes an interactive web IDE with live real-time terminal execution via WebSockets.

---

## 📋 System Requirements & Prerequisites

### 1. Core Requirements (Required)
- **Python**: Version `3.10` or higher (Python 3.10, 3.11, 3.12, or 3.13)
- **pip**: Python package manager
- **RAM**:
  - **Fast Structural Engine**: ~150 MB RAM (Instant startup, zero heavy requirements)
  - **Local AI LLM Mode**: ~3 GB RAM (Loads `deepseek-coder-1.3b-instruct` in `bfloat16`)
- **GPU (Optional)**: NVIDIA GPU with CUDA for faster local AI model generation (falls back to CPU automatically).

### 2. Optional Language Runtimes (For Built-in Terminal Execution)
The built-in web IDE contains an interactive terminal to run source and translated code. To run each language locally, install its respective compiler/runtime and add it to your system `PATH`:
- **Python**: Already available with your Python installation
- **JavaScript & TypeScript**: [Node.js](https://nodejs.org/) (v16+)
- **C++**: GCC/G++ via [MinGW-w64](https://www.mingw-w64.org/) or [MSYS2](https://www.msys2.org/)
- **Java**: [Eclipse Adoptium Temurin OpenJDK](https://adoptium.net/) or Oracle JDK (Java 11+)

*(Note: Code translation works completely even if language compilers are not installed; compilers are only needed if you click "▶ Run" in the execution terminal).*

---

## 🚀 Quick Start Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/Vinyaaggarwal/Altern.git
cd Altern
```

### Step 2: Create and Activate a Virtual Environment
It is recommended to use a virtual environment:

**Windows (PowerShell / Command Prompt):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r model_server/requirements.txt
```

### Step 4: Run the Server
From the root of the repository:
```bash
python model_server/server.py
```
Or navigate into `model_server`:
```bash
cd model_server
python server.py
```

### Step 5: Open the Web Application
Open your browser and navigate to:
```
http://localhost:8000
```
- **Web Interface**: `http://localhost:8000/` (Live translation IDE & interactive terminals)
- **API Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)
- **Health Check**: `http://localhost:8000/health`

## 📁 Repository Structure

```
Altern/
├── README.md                      # Complete project documentation & setup guide
├── .gitignore                     # Git ignore rules (pycache, venvs, artifacts)
├── frontend/
│   └── index.html                 # Full modern web IDE UI & WebSocket terminal
└── model_server/
    ├── server.py                  # FastAPI server (REST endpoints + WebSocket runner)
    ├── engine_v2.py               # Deterministic structural code translation engine
    ├── model_config.py            # AI model configurations, prompt templates, language registry
    ├── dataset_loader.py          # Data ingestion utilities
    ├── train_lora.py              # LoRA fine-tuning script for DeepSeek-Coder
    ├── requirements.txt           # Model server dependencies
    ├── test_all_samples.py        # Comprehensive test suite across supported languages
    ├── test_vector_samples.py     # Vector and array translation test cases
    ├── test_engine_fix.py         # Structural engine unit tests
    └── test_api.py                # REST API test runner
```

---

## 🧪 Testing & Validation

Run the test suite to verify code translation accuracy and engine functionality:

```bash
# Test all language translation samples
python model_server/test_all_samples.py

# Test standard library and vector translations
python model_server/test_vector_samples.py

# Test API endpoints
python model_server/test_api.py
```

---

## 🔌 API Reference

### `POST /translate`
Translates source code from one language to another.

**Request Body:**
```json
{
  "source_code": "def add(a, b):\n    return a + b",
  "source_lang": "python",
  "target_lang": "javascript"
}
```

**Response:**
```json
{
  "status": "success",
  "source_lang": "python",
  "target_lang": "javascript",
  "converted_code": "function add(a, b) {\n    return a + b;\n}",
  "model_used": "structural_engine_v2"
}
```

### `POST /execute`
Executes code on the host machine and returns stdout, stderr, and execution time.

### `WS /ws/execute`
Interactive WebSocket endpoint providing bi-directional real-time communication for standard input and streamed output (interactive console applications).

---

## ⚙️ Troubleshooting

- **Server Offline error in browser**: Ensure the backend is running (`python model_server/server.py`) and listening on port `8000`.
- **Compiler Not Found (e.g. `g++` or `javac`)**: If you try to run C++ or Java in the browser terminal, ensure MinGW / JDK is installed and present in your system `PATH`. Restart your terminal after updating `PATH`.
- **PyTorch installation on Windows**: If you have an NVIDIA GPU and want GPU acceleration, install the CUDA-enabled PyTorch build from [pytorch.org](https://pytorch.org/get-started/locally/). If no GPU is available, the CPU version installed via `requirements.txt` works seamlessly.
