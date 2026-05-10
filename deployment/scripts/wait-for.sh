#!/bin/bash
# =============================================================================
# Wait for a service to be ready before proceeding
# Usage: ./wait-for.sh host:port [--timeout=seconds] [-- command args...]
# =============================================================================

TIMEOUT=60
HOST=""
PORT=""
shift_number=0

for arg in "$@"; do
    if [[ "$arg" == --timeout=* ]]; then
        TIMEOUT="${arg#*=}"
    elif [[ "$arg" == -- ]]; then
        shift_number=$((shift_number + 1))
        break
    elif [[ "$arg" != --* ]]; then
        HOST_PORT="$arg"
        HOST="${HOST_PORT%:*}"
        PORT="${HOST_PORT#*:}"
        shift_number=$((shift_number + 1))
    fi
done

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 host:port [--timeout=seconds] [-- command args...]" >&2
    exit 1
fi

# Convert port to integer for validation
if ! [[ "$PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: Port must be a number, got '$PORT'" >&2
    exit 1
fi

CMD=("${@:$((shift_number + 1))}")

echo "Waiting for $HOST:$PORT (timeout: ${TIMEOUT}s)..."

START_TIME=$(date +%s)
while true; do
    # Try nc (netcat) first, fall back to bash /dev/tcp
    if command -v nc &>/dev/null; then
        if nc -z -w 2 "$HOST" "$PORT" 2>/dev/null; then
            ELAPSED=$(($(date +%s) - START_TIME))
            echo "$HOST:$PORT is available after ${ELAPSED}s"
            if [ ${#CMD[@]} -gt 0 ]; then
                exec "${CMD[@]}"
            fi
            exit 0
        fi
    elif command -v bash &>/dev/null; then
        if (echo > /dev/tcp/"$HOST"/"$PORT") 2>/dev/null; then
            ELAPSED=$(($(date +%s) - START_TIME))
            echo "$HOST:$PORT is available after ${ELAPSED}s"
            if [ ${#CMD[@]} -gt 0 ]; then
                exec "${CMD[@]}"
            fi
            exit 0
        fi
    else
        echo "Error: Neither nc nor bash /dev/tcp available" >&2
        exit 1
    fi

    ELAPSED=$(($(date +%s) - START_TIME))
    if [ $ELAPSED -ge $TIMEOUT ]; then
        echo "ERROR: Timed out after ${TIMEOUT}s waiting for $HOST:$PORT" >&2
        exit 1
    fi

    sleep 2
done
