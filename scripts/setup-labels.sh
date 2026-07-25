#!/usr/bin/env bash
#
# Create the project's issue labels on GitHub. Idempotent: a label that already
# exists is updated to this colour and description rather than erroring.
#
# Requires the `gh` CLI, authenticated with write access to the repo:
#   gh auth login
#
# Usage:
#   ./scripts/setup-labels.sh                       # current repo
#   ./scripts/setup-labels.sh Infinex-Labs/Intelligence-OS
#
set -euo pipefail

REPO="${1:-}"
if [[ -n "$REPO" ]]; then
    ARGS=(--repo "$REPO")
else
    ARGS=()
fi

label() {
    local name="$1" color="$2" desc="$3"
    if gh label create "$name" --color "$color" --description "$desc" "${ARGS[@]}" 2>/dev/null; then
        echo "  created  $name"
    else
        gh label edit "$name" --color "$color" --description "$desc" "${ARGS[@]}" >/dev/null
        echo "  updated  $name"
    fi
}

echo "Triage"
label "needs triage"      "ededed" "Not yet looked at by a maintainer"
label "bug"               "d73a4a" "Behaves differently from what the docs or code say"
label "enhancement"       "a2eeef" "New capability or improvement"
label "question"          "d876e3" "Setup, tuning, or how something is meant to work"
label "documentation"     "0075ca" "Docs, examples, README"
label "discussion"        "c5def5" "Needs a decision before anyone writes code"

echo "Invitation"
label "good first issue"  "7057ff" "Scoped so you don't have to read the whole cascade first"
label "help wanted"       "008672" "We would actively like a hand with this"

echo "Impact"
label "breaking change"   "b60205" "Requires action from existing users"
label "schema"            "5319e7" "Touches the SQLite schema — migrations must stay additive"
label "security"          "b60205" "Security-relevant (public issues only; see SECURITY.md)"
label "privacy"           "e99695" "Retention, consent, deletion, biometric defaults"
label "performance"       "fbca04" "Throughput, latency, or cost per frame"

echo "Area — mirrors the cascade in docs/architecture.md"
label "area: identity"    "1d76db" "Face embedding, match-or-mint, the gallery"
label "area: detection"   "1d76db" "YOLO, tracking, object persistence"
label "area: memory"      "1d76db" "store.py, the graph, distillation"
label "area: ask"         "1d76db" "Natural-language queries and the assistant"
label "area: rules"       "1d76db" "Rule compiler, alerting, delivery"
label "area: ui"          "1d76db" "Dashboard and web API"
label "area: infra"       "1d76db" "Packaging, CI, deployment"

echo "Resolution"
label "wontfix"           "ffffff" "Deliberately not doing this — see ROADMAP.md"
label "duplicate"         "cfd3d7" "Already tracked elsewhere"

echo
echo "Done."
