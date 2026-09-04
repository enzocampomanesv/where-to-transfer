#!/bin/bash
# ============================================================================
# run_trainIso_all_cities.sh
# Trains ISO forests (including score-feature) for all source cities.
# Usage: bash run_trainIso_all_cities.sh
# ============================================================================

set -e  # Exit immediately on any error

SCRIPT="where_transfer_trainIso_BT_v2.py"
VOTER="communities"
LOG_DIR="logs/trainIso"

mkdir -p "$LOG_DIR"

# All source cities (7 individual + 6 aggregates)
CITIES=(
    # ACC NBO BEI MUM TEM KIS CHI
    # GHA KEN MOZ
    PRI SEC AFR
    PRI_ACC PRI_NBO PRI_BEI
    SEC_TEM SEC_KIS SEC_CHI
    AFR_ACC AFR_NBO AFR_BEI
    AFR_TEM AFR_KIS AFR_CHI
)

echo "============================================================"
echo "Training ISO forests for ${#CITIES[@]} cities"
echo "Voter: $VOTER"
echo "Log dir: $LOG_DIR"
echo "============================================================"

FAILED=()

for city in "${CITIES[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "[$city] Starting..."
    echo "------------------------------------------------------------"

    log_file="$LOG_DIR/${city}_${VOTER}.log"

    if python "$SCRIPT" "$VOTER" "$city" > "$log_file" 2>&1; then
        echo "[$city] ✓ Done — log: $log_file"
    else
        echo "[$city] ✗ FAILED — check $log_file"
        FAILED+=("$city")
    fi
done

echo ""
echo "============================================================"
echo "All cities processed."

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED cities: ${FAILED[*]}"
    exit 1
else
    echo "All cities completed successfully."
fi
echo "============================================================"