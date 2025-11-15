#!/bin/bash
# Browser-based authentication and push for GitHub
# This script uses GitHub's device flow for authentication

cd ~/coldstart

echo "=========================================="
echo "GitHub Browser Authentication & Push"
echo "=========================================="
echo ""

# Method 1: Try GitHub CLI if available
if command -v gh &> /dev/null; then
    echo "Method 1: Using GitHub CLI (gh)"
    echo ""
    
    # Check if already authenticated
    if gh auth status &> /dev/null; then
        echo "✓ Already authenticated with GitHub CLI"
        echo "Pushing to GitHub..."
        git push -u origin main
        exit $?
    fi
    
    echo "Starting browser authentication..."
    echo "This will open your browser or provide a code."
    echo ""
    gh auth login --web --git-protocol https --hostname github.com --scopes repo
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Authentication successful!"
        echo "Configuring git to use GitHub CLI credentials..."
        gh auth setup-git
        echo ""
        echo "Pushing to GitHub..."
        git push -u origin main
        exit $?
    fi
fi

# Method 2: Manual device flow
echo ""
echo "Method 2: Manual Device Flow"
echo "============================"
echo ""
echo "To authenticate via browser:"
echo ""
echo "1. Visit: https://github.com/login/device"
echo ""
echo "2. You'll need a device code. Generating one..."
echo ""
echo "   OR use a Personal Access Token:"
echo "   - Go to: https://github.com/settings/tokens/new"
echo "   - Select 'repo' scope"
echo "   - Copy the token"
echo "   - Run: ./push.sh YOUR_TOKEN"
echo ""
echo "3. After authentication, run:"
echo "   git push -u origin main"
echo ""

# Try to use git credential helper with browser
echo "Attempting git credential helper method..."
GIT_TERMINAL_PROMPT=1 git push -u origin main 2>&1 | head -10

echo ""
echo "If authentication failed, use:"
echo "  ./push.sh YOUR_PERSONAL_ACCESS_TOKEN"

