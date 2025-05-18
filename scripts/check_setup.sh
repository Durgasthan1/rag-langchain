#!/bin/bash

echo "🔍 Checking setup for RAG project..."

# List of tools to check
TOOLS=("python3" "pip3" "uvicorn" "fastapi" "openai" "dotenv" "curl" "docker" "git")

ALL_OK=true

for TOOL in "${TOOLS[@]}"; do
  if command -v $TOOL >/dev/null 2>&1; then
    echo "✅ $TOOL is installed and in PATH"
  else
    echo "❌ $TOOL is NOT in PATH or not installed"
    ALL_OK=false
  fi
done

# Check Python version
echo -n "🔧 Python version: "
python3 --version

# Check Docker version
if command -v docker >/dev/null 2>&1; then
  echo -n "🐳 Docker version: "
  docker --version
fi

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" == *":$HOME/.local/bin:"* ]]; then
  echo "✅ ~/.local/bin is in your PATH"
else
  echo "⚠️  ~/.local/bin is NOT in your PATH"
  echo "👉 Run this to fix it:"
  echo 'echo "export PATH=\$HOME/.local/bin:\$PATH" >> ~/.bashrc && source ~/.bashrc'
  ALL_OK=false
fi

# Final message
if [ "$ALL_OK" = true ]; then
  echo -e "\n🎉 All checks passed! You're ready to build and run the RAG project."
else
  echo -e "\n⚠️ Some checks failed. Please fix the issues above before proceeding."
fi

