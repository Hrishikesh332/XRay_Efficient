#!/bin/bash
# Push script for coldstart folder to GitHub
# Usage: ./push.sh [YOUR_GITHUB_TOKEN]

cd ~/coldstart

if [ -z "$1" ]; then
    echo "=========================================="
    echo "GitHub Push Helper"
    echo "=========================================="
    echo ""
    echo "To push, you need a GitHub Personal Access Token"
    echo ""
    echo "1. Get your token from: https://github.com/settings/tokens"
    echo "   - Click 'Generate new token (classic)'"
    echo "   - Select 'repo' scope"
    echo "   - Copy the token"
    echo ""
    echo "2. Run this script with your token:"
    echo "   ./push.sh YOUR_TOKEN_HERE"
    echo ""
    echo "OR set it as environment variable:"
    echo "   export GITHUB_TOKEN=your_token"
    echo "   ./push.sh"
    echo ""
    exit 1
fi

TOKEN=${1:-$GITHUB_TOKEN}

if [ -z "$TOKEN" ]; then
    echo "Error: No token provided"
    exit 1
fi

echo "Pushing to GitHub..."
git remote set-url origin https://${TOKEN}@github.com/Hrishikesh332/XRay_Efficient.git
git push -u origin main

# Reset remote URL to remove token (for security)
git remote set-url origin https://github.com/Hrishikesh332/XRay_Efficient.git

echo ""
echo "✓ Push complete!"

