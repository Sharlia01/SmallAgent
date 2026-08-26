#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"

suite="${1:-all}"

if [[ $# -gt 0 ]]; then
    shift
fi

case "$suite" in
    all)
        pytest_args=(-v)
        ;;
    unit|api|integration)
        pytest_args=(-m "$suite" -v)
        ;;
    *)
        echo "Usage: ./run-tests.sh [all|unit|api|integration] [pytest options]"
        exit 2
        ;;
esac

docker compose \
    -f docker-compose.test.yml \
    run \
    --build \
    --rm \
    test_runner \
    pytest "${pytest_args[@]}" "$@"