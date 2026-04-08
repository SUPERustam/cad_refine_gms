#!/usr/bin/env bash
# Leaderboard: per-user RUNNING CPUs, GPUs (from AllocTRES / GRES-like strings), job count.
# Requires: sort, awk, squeue (Slurm).

set -euo pipefail

# %b = AllocTRES; %G = GRES (schedmd: may differ by site — tune if your %G is not textual)
read -r -d '' AWK_AGG <<'AWK' || true
function gpus(s,   sum, t, n) {
  sum = 0
  t = s
  while (match(t, /gpu:([0-9]+)/)) {
    n = substr(t, RSTART+4, RLENGTH-4) + 0
    sum += n
    t = substr(t, RSTART+RLENGTH)
  }
  t = s
  while (match(t, /gpu:[^|:]+:([0-9]+)/)) {
    n = substr(t, RSTART, RLENGTH)
    sub(/^gpu:[^:]*:/, "", n)
    sum += n + 0
    t = substr(t, RSTART+RLENGTH)
  }
  return sum
}
{
  u = $1
  c = $2 + 0
  g = gpus($3 " " $4)
  cpu[u] += c
  gpu[u] += g
  jobs[u]++
}
END {
  for (u in cpu)
    if (u != "")
      print u "|" cpu[u] "|" gpu[u] "|" jobs[u]
}
AWK

{
  printf '%s\n' 'USER|CPUS|GPUS|JOBS'
  squeue -t RUNNING -h -o '%u|%C|%b|%G' 2>/dev/null | awk -F'|' "$AWK_AGG" \
    | sort -t'|' -k3,3nr -k2,2nr
} | awk -F'|' 'NR==1 {
  printf "%-22s %10s %10s %8s\n", $1, $2, $3, $4
  next
}
{
  printf "%-22s %10d %10d %8d\n", $1, $2+0, $3+0, $4+0
}'
