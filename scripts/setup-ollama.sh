#!/bin/bash
# setup-ollama.sh — Install Ollama and pull the default model for AI Clipper
# Usage: bash scripts/setup-ollama.sh [model_name]
# Default model: llama3.1:8b

set -e

MODEL="${1:-llama3.1:8b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

echo "🔧 AI Clipper — Ollama Setup"
echo "   Model: $MODEL"
echo ""

# ── 1. Check if Ollama is already installed ──
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed ($(ollama --version 2>/dev/null || echo 'version unknown'))"
else
    echo "📦 Installing Ollama..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v curl &> /dev/null; then
            curl -fsSL https://ollama.com/install.sh | sh
        else
            echo "❌ curl is required. Install it with: sudo apt install curl"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "❌ Homebrew is required. Install it from https://brew.sh"
            echo "   Or download Ollama manually from https://ollama.com/download"
            exit 1
        fi
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
        # Windows (Git Bash / WSL)
        echo "⚠️  Windows detected. Please install Ollama manually:"
        echo "   1. Download from https://ollama.com/download"
        echo "   2. Run the installer"
        echo "   3. Restart this script"
        exit 0
    else
        echo "⚠️  Unknown OS. Please install Ollama manually from https://ollama.com/download"
        exit 0
    fi
    
    echo "✅ Ollama installed"
fi

# ── 2. Start Ollama if not running ──
if ss -tlnp 2>/dev/null | grep -q ':11434'; then
    echo "✅ Ollama is already running on :11434"
else
    echo "🚀 Starting Ollama..."
    nohup ollama serve > /tmp/ollama.log 2>&1 &
    
    # Wait for it to be ready
    for i in $(seq 1 15); do
        sleep 2
        if curl -s "$OLLAMA_URL/api/tags" > /dev/null 2>&1; then
            echo "✅ Ollama is ready"
            break
        fi
        if [ $i -eq 15 ]; then
            echo "⚠️  Ollama didn't start within 30s. Check /tmp/ollama.log"
            exit 1
        fi
        echo "   Waiting... ($((i * 2))s)"
    done
fi

# ── 3. Check if model is already pulled ──
echo ""
echo "📥 Checking model: $MODEL"

if ollama list 2>/dev/null | grep -q "$(echo $MODEL | cut -d: -f1)"; then
    echo "✅ Model '$MODEL' is already available"
else
    echo "⬇️  Pulling model: $MODEL (this may take a while)..."
    ollama pull "$MODEL"
    echo "✅ Model pulled successfully"
fi

# ── 4. Verify ──
echo ""
echo "🧪 Testing Ollama..."
RESPONSE=$(curl -s -X POST "$OLLAMA_URL/api/generate" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"$MODEL"'","prompt":"Reply with OK.","stream":false,"options":{"num_predict":5}}' \
    --max-time 30)

if echo "$RESPONSE" | grep -q "OK"; then
    echo "✅ Ollama is working! Model '$MODEL' responded correctly."
else
    echo "⚠️  Ollama responded but the model may still be loading."
    echo "   Response: $RESPONSE"
fi

echo ""
echo "🎉 Setup complete! You can now run AI Clipper:"
echo "   python main.py"
echo ""
echo "   Open http://127.0.0.1:7878 in your browser."
