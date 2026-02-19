#!/bin/bash
set -e

echo "🚀 Starting Perps Bot..."
echo "   Paper mode: ${PAPER_TRADE:-true}"
echo "   Venue: ${VENUE:-hyperliquid}"
echo "   Symbols: ${SYMBOLS:-ETH,BTC,SOL}"
echo ""

# Create required directories
mkdir -p logs data

# Use specific python version that has dependencies installed
exec /opt/homebrew/bin/python3.12 main.py
