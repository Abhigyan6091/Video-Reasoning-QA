#!/bin/bash
# Linux Startup Script for Video Question Generator

echo "🚀 Starting Video Question Generator..."

# 1. Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running. Please start Ollama first (ollama serve)."
    exit 1
fi

echo "✅ Ollama is running."

# 2. Check for port conflicts (Port 9005)
if lsof -Pi :9005 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️ Port 9005 is already in use."
    read -p "Do you want to kill the existing process? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -t -i :9005 | xargs kill -9
        echo "✅ Existing process killed."
    else
        echo "❌ Cannot start backend while port 9005 is occupied."
        exit 1
    fi
fi

# 3. Check for required models
echo "🔍 Checking models..."
MODELS=$(ollama list)
if [[ ! $MODELS == *"llava:7b"* ]]; then
    echo "📥 Pulling llava:7b..."
    ollama pull llava:7b
fi
if [[ ! $MODELS == *"gemma4:e4b"* ]]; then
    echo "📥 Pulling gemma4:e4b..."
    ollama pull gemma4:e4b
fi

# 3. Start Backend
echo "🌐 Starting backend on http://localhost:9005..."
cd "$(dirname "$0")"
uvicorn backend.main:app --host 0.0.0.0 --port 9005 --reload
