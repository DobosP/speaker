# PreToolUse guard for the `speaker` repo.
#
# Hard-blocks an autonomous session from touching the WORK git/SSH identity or
# pushing to main. Personal key (id_ed25519_personal) and feature-branch pushes
# are unaffected. Reads the tool-call JSON on stdin; emits a PreToolUse "deny"
# decision when a rule matches, otherwise stays silent and defers to the normal
# permission flow. Fails OPEN (exit 0, allow) on any parse error so it can never
# wedge a normal run.
#
# NOTE: this guard fires only OUTSIDE bypassPermissions mode. Launch the session
# normally (no --dangerously-skip-permissions) or these rules are skipped.

$ErrorActionPreference = 'SilentlyContinue'
try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { exit 0 }
    $j = $raw | ConvertFrom-Json
} catch { exit 0 }

$cmd  = [string]$j.tool_input.command
$path = [string]$j.tool_input.file_path
$hay  = "$cmd `n $path"

$rules = @(
    @{ p = 'id_rsa';                            why = 'the work SSH key (id_rsa) is off-limits' },
    @{ p = '[\\/]\.ssh[\\/]config';             why = 'the SSH config (holds work hosts) is off-limits' },
    @{ p = '[\\/]\.gitconfig';                  why = 'the global git identity is off-limits' },
    @{ p = 'git\s+config\s+--(global|system)'; why = 'changing global/system git identity is blocked' }
)

foreach ($r in $rules) {
    if ($hay -imatch $r.p) {
        $out = @{
            hookSpecificOutput = @{
                hookEventName            = 'PreToolUse'
                permissionDecision       = 'deny'
                permissionDecisionReason = "Blocked by speaker repo guard: $($r.why)."
            }
        }
        $out | ConvertTo-Json -Compress -Depth 5
        exit 0
    }
}
exit 0
