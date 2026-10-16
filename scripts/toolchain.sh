#!/usr/bin/env bash
# Source this helper, then call activate_toolchain PROJECT_ROOT ARCHITECTURE.
_toolchain_compiler_works() {
    [[ -n "$1" ]] && "$1" --version >/dev/null 2>&1
}

_toolchain_discover() {
    local candidate
    for candidate in "$@"; do
        if _toolchain_compiler_works "$candidate"; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

activate_toolchain() {
    local settings="$1/.toolchain-$2.sh"
    # Opt in per architecture; preserve existing environments without settings.
    [[ -f "$settings" ]] || return 0
    local TOOLCHAIN_MODULE_INIT='' TOOLCHAIN_CC='' TOOLCHAIN_CXX=''
    local -a TOOLCHAIN_MODULE_PATHS=() TOOLCHAIN_MODULES=()
    local selected_cc="${CC:-}" selected_cxx="${CXX:-}" entry
    if ! source "$settings"; then
        echo "Could not read toolchain settings: $settings" >&2
        return 1
    fi
    selected_cc="${selected_cc:-$TOOLCHAIN_CC}"
    selected_cxx="${selected_cxx:-$TOOLCHAIN_CXX}"
    selected_cc="${selected_cc:-$(_toolchain_discover gcc clang || true)}"
    selected_cxx="${selected_cxx:-$(_toolchain_discover g++ clang++ || true)}"
    if ! _toolchain_compiler_works "$selected_cc" || ! _toolchain_compiler_works "$selected_cxx"; then
        if ! type module >/dev/null 2>&1 && [[ -n "$TOOLCHAIN_MODULE_INIT" ]]; then
            if ! source "$TOOLCHAIN_MODULE_INIT"; then
                echo "Could not initialize modules using $TOOLCHAIN_MODULE_INIT (see $settings)." >&2
                return 1
            fi
        fi
        if (( ${#TOOLCHAIN_MODULE_PATHS[@]} + ${#TOOLCHAIN_MODULES[@]} )); then
            if ! type module >/dev/null 2>&1; then
                echo "Module command unavailable; set TOOLCHAIN_MODULE_INIT in $settings." >&2
                return 1
            fi
            for entry in "${TOOLCHAIN_MODULE_PATHS[@]}"; do
                if ! module use "$entry"; then
                    echo "Could not add module search path '$entry'; check $settings." >&2
                    return 1
                fi
            done
            for entry in "${TOOLCHAIN_MODULES[@]}"; do
                if ! module load "$entry"; then
                    echo "Could not load compiler module '$entry'; check $settings." >&2
                    return 1
                fi
            done
        fi
        hash -r
        selected_cc="${selected_cc:-$(_toolchain_discover gcc clang || true)}"
        selected_cxx="${selected_cxx:-$(_toolchain_discover g++ clang++ || true)}"
    fi
    if ! _toolchain_compiler_works "$selected_cc" || ! _toolchain_compiler_works "$selected_cxx"; then
        echo "Host compilers unusable (CC='$selected_cc', CXX='$selected_cxx'). Check CC/CXX overrides and $settings; configure working executables or compiler modules. See docs/toolchain-setup.md." >&2
        return 1
    fi
    export CC="$selected_cc" CXX="$selected_cxx"
}
