#!/usr/bin/env bash

set -e

# Usage:
#   ./hts_vr_connect.sh
#   ./hts_vr_connect.sh 8000
#   ./hts_vr_connect.sh --sudo
#   ./hts_vr_connect.sh --sudo 8000

USE_SUDO=0

if [ "${1:-}" = "--sudo" ]; then
    USE_SUDO=1
    shift
fi

PORT="${1:-8000}"

if [ "$USE_SUDO" -eq 1 ]; then
    ADB_CMD="sudo adb"
else
    ADB_CMD="adb"
fi

echo "========================================"
echo "HTS VR Connection Helper"
echo "Port: ${PORT}"
echo "ADB:  ${ADB_CMD}"
echo "========================================"

echo ""
echo "[1/4] Checking adb command..."
if ! command -v adb >/dev/null 2>&1; then
    echo "ERROR: adb not found. Please install it first:"
    echo "sudo apt install android-tools-adb"
    exit 1
fi

echo ""
echo "[2/4] Starting adb server..."
$ADB_CMD start-server

echo ""
echo "[3/4] Checking connected Quest device..."
DEVICE_OUTPUT="$($ADB_CMD devices 2>&1 || true)"
echo "$DEVICE_OUTPUT"

if echo "$DEVICE_OUTPUT" | grep -qi "insufficient permissions\|no permissions"; then
    echo ""
    echo "ERROR: ADB has insufficient USB permission."
    echo ""
    echo "Temporary solution:"
    echo "  ./hts_vr_connect.sh --sudo ${PORT}"
    echo ""
    echo "Permanent solution: add udev rule for Quest/Oculus device."
    exit 1
fi

if echo "$DEVICE_OUTPUT" | grep -qi "unauthorized"; then
    echo ""
    echo "ERROR: Quest is connected but unauthorized."
    echo "Please wear the Quest headset and allow USB debugging."
    exit 1
fi

if ! echo "$DEVICE_OUTPUT" | awk 'NR>1 && $2=="device" {found=1} END {exit !found}'; then
    echo ""
    echo "ERROR: No authorized Quest device found."
    echo "Please check USB-C cable, developer mode, and USB debugging permission."
    exit 1
fi

echo ""
echo "[4/4] Setting adb reverse tcp:${PORT} -> tcp:${PORT}..."
$ADB_CMD reverse tcp:${PORT} tcp:${PORT}

echo ""
echo "Current adb reverse list:"
$ADB_CMD reverse --list

echo ""
echo "========================================"
echo "SUCCESS"
echo "Quest app should use:"
echo "  Protocol: TCP"
echo "  Host/IP: 127.0.0.1"
echo "  Port: ${PORT}"
echo "========================================"
