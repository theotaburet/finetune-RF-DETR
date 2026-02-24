#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${SOURCE_DIR:-$HOME/Github/finetune-RF-DETR/}"
REMOTE_USER="${REMOTE_USER:-theo}"
REMOTE_HOST="${REMOTE_HOST:-192.168.50.103}"
REMOTE_DIR="${REMOTE_DIR:-~/Github/finetune-RF-DETR/}"
SSH_OPTS="${SSH_OPTS:--T -o Compression=no}"

DRY_RUN=false
EXTRA_RSYNC_ARGS=()

usage() {
  cat <<'EOF'
Usage: scripts/sync_project.sh [options] [-- <extra rsync args>]

Options:
  --dry-run                 Show what would be transferred
  --source <path>           Local source directory (default: ~/Github/finetune-RF-DETR/)
  --remote-user <user>      Remote SSH user (default: theo)
  --remote-host <host>      Remote SSH host/IP (default: 192.168.50.103)
  --remote-dir <path>       Remote destination directory (default: ~/Github/finetune-RF-DETR/)
  -h, --help                Show this help

Environment overrides:
  SOURCE_DIR, REMOTE_USER, REMOTE_HOST, REMOTE_DIR, SSH_OPTS

Examples:
  scripts/sync_project.sh
  scripts/sync_project.sh --dry-run
  scripts/sync_project.sh --remote-host 192.168.50.200 --remote-user theo
  scripts/sync_project.sh -- --delete-excluded
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --source)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --remote-user)
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_RSYNC_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -d "$SOURCE_DIR" ]]; then
  echo "Source directory not found: $SOURCE_DIR" >&2
  exit 1
fi

RSYNC_ARGS=(
  -a
  --delete-delay
  --partial
  --inplace
  --info=progress2,stats2
  --human-readable
  --exclude=.git/
  --exclude=.venv/
  --exclude=__pycache__/
  --exclude='*.pyc'
  --exclude=data/
  --exclude=output/
  --exclude='*.pth'
  --exclude='.eggs/'
  --exclude='*.egg-info/'
  --exclude='.ruff_cache/'
  --exclude='.pytest_cache/'
  --exclude='wandb/'
)

if [[ "$DRY_RUN" == "true" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

set -x
rsync "${RSYNC_ARGS[@]}" \
  -e "ssh ${SSH_OPTS}" \
  "${EXTRA_RSYNC_ARGS[@]}" \
  "$SOURCE_DIR" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}"
