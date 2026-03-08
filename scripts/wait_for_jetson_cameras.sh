#!/usr/bin/env bash

set -euo pipefail

NVARGUS_SERVICE="${NVARGUS_SERVICE:-nvargus-daemon.service}"
ZED_X_SERVICE="${ZED_X_SERVICE:-zed_x_daemon.service}"
VIDEO_DEVICE_GLOB="${VIDEO_DEVICE_GLOB:-/dev/video*}"
VIDEO_DEVICE_COUNT="${VIDEO_DEVICE_COUNT:-4}"
SERVICE_TIMEOUT_S="${SERVICE_TIMEOUT_S:-60}"
VIDEO_TIMEOUT_S="${VIDEO_TIMEOUT_S:-60}"
SETTLE_SLEEP_S="${SETTLE_SLEEP_S:-2}"

log() {
  printf '[lap-trackr boot] %s\n' "$*"
}

wait_for_active() {
  local service="$1"
  local timeout_s="$2"
  local start_ts=$SECONDS

  until systemctl is-active --quiet "$service"; do
    if (( SECONDS - start_ts >= timeout_s )); then
      log "Timed out waiting for ${service} to become active"
      systemctl --no-pager --full status "$service" || true
      return 1
    fi
    sleep 1
  done
}

wait_for_video_devices() {
  local expected_count="$1"
  local timeout_s="$2"
  local start_ts=$SECONDS
  local devices=()

  while true; do
    mapfile -t devices < <(compgen -G "$VIDEO_DEVICE_GLOB" || true)
    if (( ${#devices[@]} >= expected_count )); then
      log "Detected video devices: ${devices[*]}"
      return 0
    fi
    if (( SECONDS - start_ts >= timeout_s )); then
      log "Timed out waiting for ${expected_count} video devices"
      ls -l /dev/video* 2>/dev/null || true
      return 1
    fi
    sleep 1
  done
}

log "Restarting ${NVARGUS_SERVICE}"
systemctl restart "$NVARGUS_SERVICE"
wait_for_active "$NVARGUS_SERVICE" "$SERVICE_TIMEOUT_S"
sleep "$SETTLE_SLEEP_S"

log "Restarting ${ZED_X_SERVICE}"
systemctl restart "$ZED_X_SERVICE"
wait_for_active "$ZED_X_SERVICE" "$SERVICE_TIMEOUT_S"

if command -v udevadm >/dev/null 2>&1; then
  udevadm settle --timeout=10 || true
fi
sleep "$SETTLE_SLEEP_S"

wait_for_video_devices "$VIDEO_DEVICE_COUNT" "$VIDEO_TIMEOUT_S"
log "Jetson camera stack is ready"
