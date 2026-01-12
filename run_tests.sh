#!/bin/bash

# Kiva SDK Test Runner
# This script runs all tests (unit and e2e) using uv
# It enforces the use of uv and checks for required environment variables.

set -e  # Exit immediately if a command exits with a non-zero status.

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Kiva SDK Test Suite"
echo "=========================================="
echo ""

# 1. Enforce uv usage
if ! command -v uv &> /dev/null; then
    echo -e "${RED}Error: uv is not installed. Please install uv to run tests.${NC}"
    echo "Visit https://github.com/astral-sh/uv for installation instructions."
    exit 1
fi

# 2. Check Environment Variables
MISSING_VARS=0

if [ -z "$KIVA_API_BASE" ]; then
    echo -e "${RED}Error: KIVA_API_BASE environment variable is not set.${NC}"
    MISSING_VARS=1
fi

if [ -z "$KIVA_API_KEY" ]; then
    echo -e "${RED}Error: KIVA_API_KEY environment variable is not set.${NC}"
    MISSING_VARS=1
fi

if [ -z "$KIVA_MODEL" ]; then
    echo -e "${RED}Error: KIVA_MODEL environment variable is not set.${NC}"
    MISSING_VARS=1
fi

if [ $MISSING_VARS -eq 1 ]; then
    echo ""
    echo -e "${YELLOW}Please set the required environment variables:${NC}"
    echo "  export KIVA_API_BASE='http://your-api-endpoint/v1'"
    echo "  export KIVA_API_KEY='your-api-key'"
    echo "  export KIVA_MODEL='your-model-name'"
    exit 1
fi

echo "Configuration:"
echo "  API Base: $KIVA_API_BASE"
echo "  Model: $KIVA_MODEL"
echo ""

# 3. Run All Tests
echo "=========================================="
echo "Running All Tests (Unit + E2E)..."
echo "=========================================="

# Run all tests in the tests/ directory
uv run --dev pytest tests/ -v

echo ""
echo -e "${GREEN}✓ All tests completed successfully${NC}"
echo ""
