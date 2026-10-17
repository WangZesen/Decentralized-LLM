# Source this file for completion without activating or synchronizing an environment.
# The adapter needs Bash 4.4's mapfile delimiter support.
if [[ -z ${BASH_VERSION-} ]] || (( BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 4) )); then
    return 0 2>/dev/null || exit 0
fi

_decentralized_completion_root="$(CDPATH= builtin cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && builtin pwd -P)" || return 1
_decentralized_completion_architecture="$(uname -m)" || return 1

_decentralized_completion_in_checkout() {
    local physical_directory
    physical_directory="$(builtin pwd -P)" || return 1
    [[ $physical_directory == "$_decentralized_completion_root" ||
        $physical_directory == "$_decentralized_completion_root/"* ]]
}

_decentralized_complete_provider() {
    local offset=$1 python='' candidate environment replies
    # These options apply only to this completion attempt. Retain the registered
    # options for the previous handler when delegating outside the checkout.
    compopt +o default +o bashdefault +o nospace +o filenames +o noquote \
        +o dirnames +o plusdirs 2>/dev/null || :
    COMPREPLY=()
    environment=${UV_PROJECT_ENVIRONMENT-}
    if [[ -n $environment && $environment != /* ]]; then
        environment="$_decentralized_completion_root/$environment"
    fi
    for candidate in "${DECENTRALIZED_PYTHON-}" "${environment:+$environment/bin/python}" \
        "$_decentralized_completion_root/.venv-$_decentralized_completion_architecture/bin/python" \
        "$_decentralized_completion_root/.venv/bin/python"; do
        if [[ -n $candidate && -x $candidate ]]; then
            python=$candidate
            break
        fi
    done
    [[ -n $python ]] || return 0

    # Isolate argcomplete's output protocol from any global argcomplete setup.
    # -B prevents writing bytecode, and this never invokes uv or installs packages.
    if ! replies="$(
        unset _ARGCOMPLETE_STDOUT_FILENAME _ARGCOMPLETE_DFS _ARC_DEBUG
        DECENTRALIZED_COMPLETION_ROOT="$_decentralized_completion_root" \
        PYTHONPATH="$_decentralized_completion_root/src${PYTHONPATH:+:$PYTHONPATH}" \
        COMP_LINE="${COMP_LINE-}" COMP_POINT="${COMP_POINT-0}" COMP_TYPE="${COMP_TYPE-9}" \
        _ARGCOMPLETE_COMP_WORDBREAKS="${COMP_WORDBREAKS-}" \
        _ARGCOMPLETE="$offset" _ARGCOMPLETE_SHELL=bash _ARGCOMPLETE_SUPPRESS_SPACE=1 \
        _ARGCOMPLETE_IFS=$'\013' \
        "$python" -B -m decentralized_llm.completion 8>&1 9>/dev/null 1>/dev/null 2>/dev/null </dev/null
    )"; then
        return 0
    fi
    # Do not use an unquoted array assignment: it expands glob characters in paths.
    mapfile -d $'\013' -t COMPREPLY < <(printf '%s' "$replies")
    if [[ ${COMPREPLY[0]-} == *[=/:] ]]; then
        compopt -o nospace 2>/dev/null || :
    fi
    return 0
}

_decentralized_complete_direct() {
    if _decentralized_completion_in_checkout; then
        _decentralized_complete_provider 1
    elif [[ -n ${_decentralized_previous_direct_function-} ]] &&
        declare -F "$_decentralized_previous_direct_function" >/dev/null; then
        "$_decentralized_previous_direct_function" "$@"
    else
        COMPREPLY=()
    fi
}

_decentralized_complete_uv() {
    local index=2
    if _decentralized_completion_in_checkout && [[ ${COMP_WORDS[1]-} == run ]]; then
        while [[ $index -lt ${#COMP_WORDS[@]} ]]; do
            case ${COMP_WORDS[index]} in
                --locked|--frozen|--offline|--no-sync) ((index += 1)) ;;
                --) ((index += 1)); break ;;
                *) break ;;
            esac
        done
        if [[ ${COMP_WORDS[index]-} == decentralized-llm && ${COMP_CWORD-0} -gt $index ]]; then
            _decentralized_complete_provider "$((index + 1))"
            return 0
        fi
    fi
    if [[ -n ${_decentralized_previous_uv_function-} ]] &&
        declare -F "$_decentralized_previous_uv_function" >/dev/null; then
        "$_decentralized_previous_uv_function" "$@"
    else
        COMPREPLY=()
    fi
}

_decentralized_register_completion() {
    local command_name=$1 wrapper=$2 saved_function_variable=$3 specification function_name=''
    local index argument IFS=$' \t\n'
    local -a arguments=() options=()
    specification="$(builtin complete -p -- "$command_name" 2>/dev/null)" || specification=''

    # Resolve bash-completion's lazy loader before replacing its command handler.
    # Loading can return 124 to tell Readline to retry; that is not an error here.
    if [[ -z $specification || $specification =~ ^complete(\ -o\ [a-z]+)*\ -F\ (_completion_loader|_comp_complete_load)\ $command_name$ ]]; then
        if declare -F _completion_loader >/dev/null; then
            _completion_loader "$command_name" >/dev/null 2>&1 || :
        elif declare -F _comp_load >/dev/null; then
            _comp_load "$command_name" >/dev/null 2>&1 || :
        fi
        specification="$(builtin complete -p -- "$command_name" 2>/dev/null)" || specification=''
    fi
    if [[ -z $specification ]]; then
        options=(-o bashdefault -o default)
    else
        # Only wrap simple function completions. Avoid evaluating shell text or
        # replacing custom -C/-W/-A/filter/prefix/suffix completion specifications.
        read -r -a arguments <<< "$specification"
        [[ ${arguments[0]-} == complete && ${arguments[-1]-} == "$command_name" ]] || return 0
        index=1
        while (( index < ${#arguments[@]} - 1 )); do
            argument=${arguments[index]}
            case $argument in
                -o)
                    (( index + 1 < ${#arguments[@]} - 1 )) || return 0
                    options+=(-o "${arguments[index+1]}")
                    ((index += 2))
                    ;;
                -F)
                    [[ -z $function_name ]] && (( index + 1 < ${#arguments[@]} - 1 )) || return 0
                    function_name=${arguments[index+1]}
                    [[ $function_name =~ ^[a-zA-Z_][a-zA-Z0-9_.:-]*$ ]] || return 0
                    ((index += 2))
                    ;;
                *) return 0 ;;
            esac
        done
        [[ -n $function_name ]] || return 0
        # Re-sourcing can select another checkout without capturing our own handler.
        [[ $function_name != "$wrapper" ]] || return 0
    fi
    printf -v "$saved_function_variable" '%s' "$function_name"
    builtin complete "${options[@]}" -F "$wrapper" -- "$command_name"
}

_decentralized_register_completion decentralized-llm _decentralized_complete_direct _decentralized_previous_direct_function
_decentralized_register_completion uv _decentralized_complete_uv _decentralized_previous_uv_function
unset -f _decentralized_register_completion
