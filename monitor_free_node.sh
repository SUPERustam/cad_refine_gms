#!/bin/bash
set -euo pipefail

JOB_SCRIPT="/scratch/498rustam/cad_refine_m/slurm_runner.sh"

EXCLUDE_NODES_REGEX='^(cn13|cn27|cn28)$'
PARTITION="batch"
POLL_INTERVAL_SEC=1

if [[ ! -f "${JOB_SCRIPT}" ]]; then
  echo "Job script not found: ${JOB_SCRIPT}" >&2
  exit 1
fi

echo "Monitoring idle nodes in partition '${PARTITION}'..."
echo "Excluding nodes: cn13, cn28"

while true; do
  # Find first idle node (fast path).
  IDLE_NODE="$(
    sinfo -h -N -p "${PARTITION}" -o "%N %T" \
      | awk '$2 == "idle" {print $1}' \
      | grep -Ev "${EXCLUDE_NODES_REGEX}" \
      | head -n 1
  )"

  if [[ -n "${IDLE_NODE}" ]]; then
    echo "Found idle node: ${IDLE_NODE}"
    perl -0777 -pi -e "s/^#SBATCH --nodelist=.*/#SBATCH --nodelist=${IDLE_NODE}/m" "${JOB_SCRIPT}"
    # NODE_NUM="${IDLE_NODE#cn}"
    # perl -0777 -pi -e "s/^#SBATCH --output=.*/#SBATCH --output=test-slurm-${NODE_NUM}.out/m" "${JOB_SCRIPT}"
    echo "Submitting job to ${IDLE_NODE}..."
    if sbatch "${JOB_SCRIPT}"; then
      echo "Job submitted successfully."
      break
    fi
    echo "sbatch failed, retrying..."
  fi

  sleep "${POLL_INTERVAL_SEC}"
done

sleep 1
echo "--- Your jobs (squeue | grep ${USER}) ---"
squeue | grep "${USER:0:6}"