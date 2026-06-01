#!/bin/bash
# Ollama Health Check & Auto-Start for AI Clipper
# Checks if Ollama is running, starts it if not, verifies the model is loaded

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen3.5}"
MAX_RETRIES=3
RETRY_DELAY=5

echo "🔍 Checking Ollama health..."

check_ollama() {
    curl -sf "${OLLAMA_URL}/api/tags" > /dev/null 2>&1
}

get_running_models() {
    curl -sf "${OLLAMA_URL}/api/tags" 2>/dev/null | \
        python3 -c "import json,sys; data=json.load(sys.stdin); print(','.join(m.get('name','') for m in data.get('models',[])))" 2>/dev/null
}

for i in $(seq 1 $MAX_RETRIES); do
    if check_ollama; then
        echo "  ✓ Ollama is running on ${OLLAMA_URL}"
        
        # Check if the model is loaded
        RUNNING_MODELS=$(get_running_models)
        if echo "$RUNNING_MODELS" | grep -q "$OLLAMA_MODEL"; then
            echo "  ✓ Model '${OLLAMA_MODEL}' is loaded"
            echo "✅ Ollama is healthy"
            exit 0
        else
            echo "  ⚠ Ollama running but model '${OLLAMA_MODEL}' not loaded"
            echo "  → Loading model..."
            curl -sf "${OLLAMA_URL}/api/pull" -d "{\"name\":\"${OLLAMA_MODEL}\"}" > /dev/null 2>&1 || true
            sleep 3
            # Retry the check
            continue
        fi
    else
        echo "  ✗ Ollama not running (attempt $i/$MAX_RETRIES)"
        if [ "$i" -lt "$MAX_RETRIES" ]; then
            echo "  → Starting Ollama..."
            # Try starting Ollama
            if command -v ollama &> /dev/null; then
                nohup ollama serve > /tmp/ollama.log 2>&1 &
                sleep $RETRY_DELAY
            else
                echo "  ✗ 'ollama' command not found. Install Ollama first."
                exit 1
            fi
        fi
    fi
done

echo "❌ Ollama health check failed after $MAX_RETRIES attempts"
echo "   Check logs: /tmp/ollama.log"
exit 1
