# Copyright (c) IBM.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -e

log() {
    local type=$1
    local message=$2

    local NC='\033[0m'
    local GREEN='\033[0;32m'
    local RED='\033[0;31m'
    local YELLOW='\033[0;33m'

    case "$type" in
        "INFO")
            color=$GREEN
            ;;
        "ERROR")
            color=$RED
            ;;
        "WARNING")
            color=$YELLOW
            ;;
        *)
            color=$NC
            ;;
    esac

    echo -e "[${color}$type${NC}] $message"
}