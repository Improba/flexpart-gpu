#!/usr/bin/env bash
set -euo pipefail

# ===========================================================================
# ETEX-1 Real Data Validation Pipeline
#
# ETEX workflow for Phase D scientific validation:
#   GPU-only default:
#     1. Download ERA5 meteorological data (requires CDS account)
#     2. Prepare FLEXPART-compatible input files
#     3. Parse ETEX-1 station measurements
#     4. Run flexpart-gpu
#     5. Compare against observations
#     6. Generate validation report
#   Optional:
#     - Run FLEXPART Fortran reference (Docker) for side-by-side comparison
#
# Usage:
#   scripts/run-etex.sh [step]
#
# Steps:
#   all              Run GPU-only pipeline (default)
#   all-with-fortran Run GPU pipeline + optional Fortran reference
#   download     Download ERA5 data from CDS
#   prepare      Prepare FLEXPART input from ERA5
#   parse        Parse ETEX measurements
#   fortran      Run Fortran FLEXPART
#   gpu          Run flexpart-gpu
#   compare      Compare outputs against observations
#   report       Print final report
#   status       Show current pipeline status
#
# Prerequisites:
#   - CDS account with ~/.cdsapirc configured
#   - Python 3 with eccodes, numpy, cdsapi
#   - Rust toolchain (for flexpart-gpu)
#   - Docker + docker-compose (optional, Fortran comparison only)
# ===========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLEXPART_DIR="${PROJECT_ROOT}/../flexpart"
FORTRAN_DOCKER_DIR="${PROJECT_ROOT}/../flexpart-fortran-docker"

ETEX_DIR="${PROJECT_ROOT}/target/etex"
ERA5_RAW="${ETEX_DIR}/era5_raw"
METEO_DIR="${ETEX_DIR}/meteo"
FORTRAN_RUN="${ETEX_DIR}/fortran_run"
GPU_OUTPUT="${ETEX_DIR}/gpu_output.json"
MEASUREMENTS="${ETEX_DIR}/measurements.json"
REPORT="${ETEX_DIR}/comparison_report.json"
DATA_DIR="${PROJECT_ROOT}/fixtures/etex/data"
CONFIG_DIR="${PROJECT_ROOT}/fixtures/etex/real/config"

C_FLEXPART="/workspace/flexpart"
C_GPU="/workspace/flexpart-gpu"
C_DATA="/workspace/etex"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BLUE}=== Step: $* ===${NC}"; }

fortran_run_succeeded() {
    [ -f "${ETEX_DIR}/fortran.log" ] && grep -q "CONGRATULATIONS" "${ETEX_DIR}/fortran.log" 2>/dev/null
}

# ---------------------------------------------------------------------------
check_prereqs() {
    local ok=true
    if ! command -v python3 &>/dev/null; then
        log_error "python3 not found"; ok=false
    fi
    if ! python3 -c "import eccodes" 2>/dev/null; then
        log_warn "python3-eccodes not found (needed for GRIB processing)"
    fi
    if ! python3 -c "import numpy" 2>/dev/null; then
        log_warn "numpy not found (needed for comparison)"
    fi
    if ! command -v docker &>/dev/null; then
        log_warn "docker not found (optional, only needed for Fortran comparison)"
    fi
    $ok
}

# ---------------------------------------------------------------------------
step_download() {
    log_step "Download ERA5 data"
    mkdir -p "${ERA5_RAW}"

    if [ -f "${ERA5_RAW}/era5_pressure_levels.grib" ] && [ -f "${ERA5_RAW}/era5_single_levels.grib" ]; then
        log_info "ERA5 data already downloaded. Skipping."
        return 0
    fi

    if ! python3 -c "import cdsapi" 2>/dev/null; then
        log_error "cdsapi not installed. Run: pip install cdsapi"
        log_error "Then configure ~/.cdsapirc with your CDS Personal Access Token."
        log_error "See scripts/etex/download_era5.py for details."
        return 1
    fi

    if [ ! -f ~/.cdsapirc ]; then
        log_error "~/.cdsapirc not found. Create it with your CDS credentials:"
        echo "  url: https://cds.climate.copernicus.eu/api"
        echo "  key: <YOUR_PERSONAL_ACCESS_TOKEN>"
        return 1
    fi

    python3 "${PROJECT_ROOT}/scripts/etex/download_era5.py" \
        --output-dir "${ERA5_RAW}"
}

# ---------------------------------------------------------------------------
step_prepare() {
    log_step "Prepare FLEXPART input"
    mkdir -p "${METEO_DIR}"

    if [ ! -f "${ERA5_RAW}/era5_pressure_levels.grib" ]; then
        log_error "ERA5 data not downloaded. Run: scripts/run-etex.sh download"
        return 1
    fi

    python3 "${PROJECT_ROOT}/scripts/etex/prepare_flexpart_input.py" \
        --era5-dir "${ERA5_RAW}" \
        --output-dir "${METEO_DIR}" \
        --gpu-json "${ETEX_DIR}/gpu_meteo.json"
}

# ---------------------------------------------------------------------------
step_parse() {
    log_step "Parse ETEX-1 measurements"

    if [ ! -f "${DATA_DIR}/meas-t1.txt" ]; then
        log_error "ETEX measurement data not found at ${DATA_DIR}/meas-t1.txt"
        return 1
    fi

    python3 "${PROJECT_ROOT}/scripts/etex/parse_measurements.py" \
        --data-dir "${DATA_DIR}" \
        --output "${MEASUREMENTS}"
}

# ---------------------------------------------------------------------------
step_fortran() {
    log_step "Run FLEXPART Fortran"

    if ! command -v docker &>/dev/null; then
        log_error "docker not found. Fortran step requires Docker."
        log_error "Fortran step is optional. For GPU-only quickstart, run: scripts/run-etex.sh all"
        return 1
    fi
    if ! docker compose version &>/dev/null; then
        log_error "docker compose is not available."
        log_error "Fortran step is optional. For GPU-only quickstart, run: scripts/run-etex.sh all"
        return 1
    fi

    if [ ! -d "${FLEXPART_DIR}" ] || [ ! -d "${FLEXPART_DIR}/src" ]; then
        log_error "Fortran checkout not found at ${FLEXPART_DIR}"
        log_error "Fortran step is optional. For GPU-only quickstart, run: scripts/run-etex.sh all"
        return 1
    fi
    if [ ! -f "${FORTRAN_DOCKER_DIR}/docker-compose.yml" ]; then
        log_error "Fortran Docker compose not found at ${FORTRAN_DOCKER_DIR}/docker-compose.yml"
        log_error "Fortran step is optional. For GPU-only quickstart, run: scripts/run-etex.sh all"
        return 1
    fi

    if [ ! -d "${METEO_DIR}" ] || [ ! -f "${METEO_DIR}/AVAILABLE" ]; then
        log_error "Meteorological data not prepared. Run: scripts/run-etex.sh prepare"
        return 1
    fi

    mkdir -p "${FORTRAN_RUN}/options/SPECIES"
    rm -rf "${FORTRAN_RUN}/output"
    mkdir -p "${FORTRAN_RUN}/output"
    rm -f "${ETEX_DIR}/fortran.log"

    cat > "${FORTRAN_RUN}/pathnames" <<'PATHEOF'
./options/
./output/
../meteo/
../meteo/AVAILABLE
============================================
PATHEOF

    cp "${CONFIG_DIR}/COMMAND"     "${FORTRAN_RUN}/options/"
    cp "${CONFIG_DIR}/RELEASES"    "${FORTRAN_RUN}/options/"
    cp "${CONFIG_DIR}/OUTGRID"     "${FORTRAN_RUN}/options/"
    cp "${CONFIG_DIR}/AGECLASSES"  "${FORTRAN_RUN}/options/"
    cp "${CONFIG_DIR}/RECEPTORS"   "${FORTRAN_RUN}/options/"

    if [ -d "${FLEXPART_DIR}/options/SPECIES" ]; then
        cp -r "${FLEXPART_DIR}/options/SPECIES/"* "${FORTRAN_RUN}/options/SPECIES/"
    fi
    for f in IGBP_int1.dat surfdata.t surfdepo.t; do
        if [ -f "${FLEXPART_DIR}/options/${f}" ]; then
            cp "${FLEXPART_DIR}/options/${f}" "${FORTRAN_RUN}/options/"
        fi
    done

    log_info "Building Docker images..."
    cd "${PROJECT_ROOT}"
    docker compose -f "${FORTRAN_DOCKER_DIR}/docker-compose.yml" build flexpart-fortran

    log_info "Compiling FLEXPART Fortran..."
    docker compose -f "${FORTRAN_DOCKER_DIR}/docker-compose.yml" run --rm \
        -v "${ETEX_DIR}:/workspace/etex" \
        flexpart-fortran bash -c "
            cd ${C_FLEXPART}/src && make clean 2>/dev/null; \
            make serial \
                INCPATH1=/usr/lib/x86_64-linux-gnu/fortran/x86_64-linux-gnu-gfortran-11 \
                INCPATH2=/usr/include \
                LIBPATH1=/usr/lib/x86_64-linux-gnu \
                LIBS='-leccodes_f90 -leccodes -lm' \
                2>&1 | tail -5
        "

    log_info "Running FLEXPART Fortran (ETEX-1)..."
    docker compose -f "${FORTRAN_DOCKER_DIR}/docker-compose.yml" run --rm \
        -v "${ETEX_DIR}:/workspace/etex" \
        flexpart-fortran bash -c "
            cd /workspace/etex/fortran_run && ${C_FLEXPART}/src/FLEXPART
        " 2>&1 | tee "${ETEX_DIR}/fortran.log"

    if grep -q "CONGRATULATIONS" "${ETEX_DIR}/fortran.log" 2>/dev/null; then
        log_info "Fortran run completed successfully"
    else
        log_error "Fortran run failed. Check ${ETEX_DIR}/fortran.log"
        return 1
    fi
}

# ---------------------------------------------------------------------------
step_gpu() {
    log_step "Run flexpart-gpu"

    log_info "Building GPU ETEX binary..."
    cd "${PROJECT_ROOT}"
    cargo build --release --bin fortran-validation 2>&1 | tail -3

    log_info "Running flexpart-gpu (ETEX-1)..."
    OUTPUT_PATH="${GPU_OUTPUT}" \
    ETEX_METEO="${ETEX_DIR}/gpu_meteo.json" \
    PARTICLES=100000 \
        cargo run --release --bin fortran-validation 2>&1 \
        | tee "${ETEX_DIR}/gpu.log"

    if [ -f "${GPU_OUTPUT}" ]; then
        log_info "GPU output: ${GPU_OUTPUT}"
    else
        log_warn "GPU output not generated (may need ETEX-specific binary)"
    fi
}

# ---------------------------------------------------------------------------
step_compare() {
    log_step "Compare with observations"

    if [ ! -f "${MEASUREMENTS}" ]; then
        log_error "Measurements not parsed. Run: scripts/run-etex.sh parse"
        return 1
    fi

    local fortran_arg=""
    if fortran_run_succeeded && [ -d "${FORTRAN_RUN}/output" ]; then
        fortran_arg="--fortran-output ${FORTRAN_RUN}/output"
    fi

    local gpu_arg=""
    if [ -f "${GPU_OUTPUT}" ]; then
        gpu_arg="--gpu-output ${GPU_OUTPUT}"
    fi

    python3 "${PROJECT_ROOT}/scripts/etex/compare_with_observations.py" \
        --measurements "${MEASUREMENTS}" \
        ${fortran_arg} \
        ${gpu_arg} \
        --output "${REPORT}" \
        --verbose
}

# ---------------------------------------------------------------------------
step_status() {
    echo ""
    echo "ETEX-1 Validation Pipeline Status"
    echo "=================================="
    echo ""

    check_file() {
        if [ -e "$1" ]; then
            echo -e "  ${GREEN}[OK]${NC}  $2"
        else
            echo -e "  ${RED}[--]${NC}  $2"
        fi
    }

    check_file "${DATA_DIR}/meas-t1.txt"                        "ETEX measurements (DATEM)"
    check_file "${DATA_DIR}/stations.txt"                        "ETEX stations (DATEM)"
    check_file "${ERA5_RAW}/era5_pressure_levels.grib"           "ERA5 pressure levels"
    check_file "${ERA5_RAW}/era5_single_levels.grib"             "ERA5 single levels"
    check_file "${METEO_DIR}/AVAILABLE"                          "FLEXPART input prepared"
    check_file "${MEASUREMENTS}"                                 "Measurements parsed"
    if fortran_run_succeeded; then
        echo -e "  ${GREEN}[OK]${NC}  Fortran FLEXPART run"
    else
        echo -e "  ${RED}[--]${NC}  Fortran FLEXPART run"
    fi
    check_file "${GPU_OUTPUT}"                                   "GPU run"
    check_file "${REPORT}"                                       "Comparison report"

    echo ""

    if [ -f ~/.cdsapirc ]; then
        echo -e "  ${GREEN}[OK]${NC}  CDS API credentials (~/.cdsapirc)"
    else
        echo -e "  ${YELLOW}[!!]${NC}  CDS API credentials missing (~/.cdsapirc)"
        echo "        Create account: https://cds.climate.copernicus.eu"
        echo "        Then: echo 'url: https://cds.climate.copernicus.eu/api' > ~/.cdsapirc"
        echo "              echo 'key: <YOUR_TOKEN>' >> ~/.cdsapirc"
    fi
    echo ""
}

# ---------------------------------------------------------------------------
step_report() {
    log_step "Validation Report"
    if [ -f "${REPORT}" ]; then
        python3 -c "
import json, sys
with open('${REPORT}') as f:
    r = json.load(f)
print(json.dumps(r, indent=2))
"
    else
        log_error "No report found. Run: scripts/run-etex.sh compare"
    fi
}

# ---------------------------------------------------------------------------
STEP="${1:-all}"

case "${STEP}" in
    status)   step_status ;;
    download) step_download ;;
    prepare)  step_prepare ;;
    parse)    step_parse ;;
    fortran)  step_fortran ;;
    gpu)      step_gpu ;;
    compare)  step_compare ;;
    report)   step_report ;;
    all)
        check_prereqs || true
        step_parse
        step_download || {
            log_warn "ERA5 download failed (CDS credentials needed)."
            log_warn "Pipeline will continue with available data."
            log_warn "Run 'scripts/run-etex.sh status' to check requirements."
        }
        if [ -f "${ERA5_RAW}/era5_pressure_levels.grib" ]; then
            step_prepare
            step_gpu
        fi
        step_compare
        step_report
        ;;
    all-with-fortran)
        check_prereqs || true
        step_parse
        step_download || {
            log_warn "ERA5 download failed (CDS credentials needed)."
            log_warn "Pipeline will continue with available data."
            log_warn "Run 'scripts/run-etex.sh status' to check requirements."
        }
        if [ -f "${ERA5_RAW}/era5_pressure_levels.grib" ]; then
            step_prepare
            step_fortran
            step_gpu
        fi
        step_compare
        step_report
        ;;
    *)
        echo "Usage: scripts/run-etex.sh [all|all-with-fortran|status|download|prepare|parse|fortran|gpu|compare|report]"
        exit 2
        ;;
esac
