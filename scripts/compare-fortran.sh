#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# FLEXPART Fortran vs flexpart-gpu comparison
#
# Modes:
#   compose  — Docker containers (default, recommended)
#   nvidia   — Docker with NVIDIA GPU passthrough
#   local    — run directly on host (needs gfortran, eccodes, python3)
#
# Usage:
#   scripts/compare-fortran.sh [compose|nvidia|local] [setup|run|clean|all|validate|perf]
#
# Quick start:
#   scripts/compare-fortran.sh compose all
#
# Scientific validation (Fortran vs GPU concentration comparison):
#   scripts/compare-fortran.sh compose validate
#
# GPU performance profile (strict vs production + profiled timeloop):
#   scripts/compare-fortran.sh compose perf
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FLEXPART_DIR="$(cd "${PROJECT_ROOT}/../flexpart" && pwd)"
FORTRAN_DOCKER_DIR="$(cd "${PROJECT_ROOT}/../flexpart-fortran-docker" && pwd)"

C_FLEXPART="/workspace/flexpart"
C_GPU="/workspace/flexpart-gpu"
C_DATA="/workspace/comparison"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ---------------------------------------------------------------------------
# Fortran Docker is in a sibling directory (../flexpart-fortran-docker/)
# GPU Docker is in this project
fortran_compose_cmd() { local m="$1"; shift
  docker compose -f "${FORTRAN_DOCKER_DIR}/docker-compose.yml" "$@"
}
gpu_compose_cmd() { local m="$1"; shift
  case "$m" in
    compose) docker compose "$@" ;;
    nvidia)  docker compose -f docker-compose.yml -f docker-compose.nvidia.yml "$@" ;;
  esac
}
fortran_exec() { local m="$1"; shift; fortran_compose_cmd "$m" run --rm flexpart-fortran "$@"; }
gpu_exec()     { local m="$1"; shift; gpu_compose_cmd "$m" run --rm flexpart-gpu "$@"; }

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
do_setup() {
  local mode="$1"

  log_info "Building Docker images..."
  fortran_compose_cmd "$mode" build
  gpu_compose_cmd "$mode" build

  log_info "Compiling FLEXPART Fortran..."
  fortran_exec "$mode" bash -c "
    cd ${C_FLEXPART}/src && make clean 2>/dev/null; \
    make serial \
      INCPATH1=/usr/lib/x86_64-linux-gnu/fortran/x86_64-linux-gnu-gfortran-11 \
      INCPATH2=/usr/include \
      LIBPATH1=/usr/lib/x86_64-linux-gnu \
      LIBS='-leccodes_f90 -leccodes -lm' \
      2>&1 | tail -3
  "

  log_info "Generating synthetic GRIB data..."
  fortran_exec "$mode" python3 ${C_GPU}/scripts/generate_synthetic_grib.py \
    --output-dir "${C_DATA}/meteo" \
    --nx 32 --ny 32 --nz 12 \
    --u-wind 0.5 --v-wind -0.3 --w-wind 0.0 \
    --start-date 20240101 --hours 6

  log_info "Setting up FLEXPART run directory..."
  fortran_exec "$mode" bash -c "
    mkdir -p ${C_DATA}/fortran_run/options/SPECIES
    mkdir -p ${C_DATA}/fortran_run/output

    # pathnames (4 lines: options, output, meteo_dir, available_path)
    cat > ${C_DATA}/fortran_run/pathnames <<'EOF'
./options/
./output/
../meteo/
../meteo/AVAILABLE
============================================
EOF

    # COMMAND
    cat > ${C_DATA}/fortran_run/options/COMMAND <<'EOF'
&COMMAND
 LDIRECT=               1,
 IBDATE=         20240101,
 IBTIME=           000000,
 IEDATE=         20240101,
 IETIME=           010000,
 LOUTSTEP=           3600,
 LOUTAVER=           3600,
 LOUTSAMPLE=          900,
 ITSPLIT=        99999999,
 LSYNCTIME=           900,
 CTL=          -5.0000000,
 IFINE=                 4,
 IOUT=                  1,
 IPOUT=                 0,
 LSUBGRID=              0,
 LCONVECTION=           0,
 LAGESPECTRA=           0,
 IPIN=                  0,
 IOUTPUTFOREACHRELEASE= 1,
 IFLUX=                 0,
 MDOMAINFILL=           0,
 IND_SOURCE=            1,
 IND_RECEPTOR=          1,
 MQUASILAG=             0,
 NESTED_OUTPUT=         0,
 LINIT_COND=            0,
 SURF_ONLY=             0,
 CBLFLAG=               0,
 OHFIELDS_PATH= \"../../flexin/\",
 /
EOF

    # RELEASES
    cat > ${C_DATA}/fortran_run/options/RELEASES <<'EOF'
&RELEASES_CTRL
 NSPEC      =           1,
 SPECNUM_REL=          24,
 /
&RELEASE
 IDATE1  =       20240101,
 ITIME1  =         000000,
 IDATE2  =       20240101,
 ITIME2  =         000000,
 LON1    =         10.000,
 LON2    =         10.000,
 LAT1    =         10.000,
 LAT2    =         10.000,
 Z1      =         50.000,
 Z2      =         50.000,
 ZKIND   =              1,
 MASS    =       1.0000E0,
 PARTS   =           1000,
 COMMENT =    \"COMPARISON\",
 /
EOF

    # OUTGRID
    cat > ${C_DATA}/fortran_run/options/OUTGRID <<'EOF'
&OUTGRID
 OUTLON0=      0.00,
 OUTLAT0=      0.00,
 NUMXGRID=       32,
 NUMYGRID=       32,
 DXOUT=        1.00,
 DYOUT=        1.00,
 OUTHEIGHTS=  100.0, 500.0, 1000.0,
 /
EOF

    # RECEPTORS (old format, 5 header lines + 0 receptors)
    cat > ${C_DATA}/fortran_run/options/RECEPTORS <<'EOF'
*******************************************************************************
*                                                                             *
*  Input file for the Lagrangian particle dispersion model FLEXPART           *
*                        Please specify receptor points                       *
*******************************************************************************
                0
EOF

    # AGECLASSES
    cat > ${C_DATA}/fortran_run/options/AGECLASSES <<'EOF'
&AGECLASS
 NAGECLASS= 1,
 LAGE= 3600,
 /
EOF

    # Copy static data (landuse, surface params)
    cp ${C_FLEXPART}/options/IGBP_int1.dat  ${C_DATA}/fortran_run/options/
    cp ${C_FLEXPART}/options/surfdata.t     ${C_DATA}/fortran_run/options/
    cp ${C_FLEXPART}/options/surfdepo.t     ${C_DATA}/fortran_run/options/

    # Copy SPECIES definitions
    if [ -d ${C_FLEXPART}/options/SPECIES ]; then
      cp -r ${C_FLEXPART}/options/SPECIES/* ${C_DATA}/fortran_run/options/SPECIES/
    fi
  "

  log_info "Setup complete."
}

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
do_run() {
  local mode="$1"
  mkdir -p "${PROJECT_ROOT}/target"

  log_info "Running flexpart-gpu CPU vs GPU tests..."
  gpu_exec "$mode" cargo test --test cpu_gpu_comparison -- --nocapture 2>&1 \
    | tee "${PROJECT_ROOT}/target/gpu-comparison.log" || true

  log_info "Running FLEXPART Fortran..."
  fortran_exec "$mode" bash -c "
    cd ${C_DATA}/fortran_run && ${C_FLEXPART}/src/FLEXPART
  " 2>&1 | tee "${PROJECT_ROOT}/target/fortran-comparison.log" || true

  echo ""
  echo "================================================================"
  echo "  FLEXPART Fortran vs flexpart-gpu — Results"
  echo "================================================================"

  # GPU tests
  if grep -q "test result: ok" "${PROJECT_ROOT}/target/gpu-comparison.log" 2>/dev/null; then
    echo -e "  GPU tests:    ${GREEN}PASSED${NC}"
    grep -E "^test " "${PROJECT_ROOT}/target/gpu-comparison.log" | sed 's/^/    /'
  else
    echo -e "  GPU tests:    ${RED}FAILED or NOT RUN${NC}"
  fi
  echo ""

  # Fortran
  if grep -q "CONGRATULATIONS" "${PROJECT_ROOT}/target/fortran-comparison.log" 2>/dev/null; then
    echo -e "  Fortran run:  ${GREEN}COMPLETED${NC}"
    grep -E "Simulated|Particles|CONGRATULATIONS" "${PROJECT_ROOT}/target/fortran-comparison.log" | sed 's/^/    /'
  else
    echo -e "  Fortran run:  ${RED}FAILED${NC}"
    tail -5 "${PROJECT_ROOT}/target/fortran-comparison.log" 2>/dev/null | sed 's/^/    /'
  fi

  echo ""
  echo "  Logs:"
  echo "    target/gpu-comparison.log"
  echo "    target/fortran-comparison.log"
  echo "================================================================"
}

# ---------------------------------------------------------------------------
# VALIDATE — scientific comparison of concentration fields
# ---------------------------------------------------------------------------

# Shared parameters (must match src/bin/fortran-validation.rs constants)
V_U_WIND=5.0
V_V_WIND=-3.0
V_W_WIND=0.0
V_PARTICLES=10000
V_OUTLON0=9.50
V_OUTLAT0=8.50
V_NX=32
V_NY=32
V_DX=0.10
V_DY=0.10

do_validate_setup() {
  local mode="$1"

  log_info "Building Docker images..."
  fortran_compose_cmd "$mode" build

  log_info "Compiling FLEXPART Fortran..."
  fortran_exec "$mode" bash -c "
    cd ${C_FLEXPART}/src && make clean 2>/dev/null; \
    make serial \
      INCPATH1=/usr/lib/x86_64-linux-gnu/fortran/x86_64-linux-gnu-gfortran-11 \
      INCPATH2=/usr/include \
      LIBPATH1=/usr/lib/x86_64-linux-gnu \
      LIBS='-leccodes_f90 -leccodes -lm' \
      2>&1 | tail -3
  "

  log_info "Generating synthetic GRIB data (u=${V_U_WIND}, v=${V_V_WIND})..."
  fortran_exec "$mode" python3 ${C_GPU}/scripts/generate_synthetic_grib.py \
    --output-dir "${C_DATA}/meteo" \
    --nx 32 --ny 32 --nz 12 \
    --u-wind "${V_U_WIND}" --v-wind "${V_V_WIND}" --w-wind "${V_W_WIND}" \
    --start-date 20240101 --hours 6

  log_info "Setting up Fortran validation run directory..."
  fortran_exec "$mode" bash -c "
    rm -rf ${C_DATA}/validate_run
    mkdir -p ${C_DATA}/validate_run/options/SPECIES
    mkdir -p ${C_DATA}/validate_run/output

    cat > ${C_DATA}/validate_run/pathnames <<'EOF'
./options/
./output/
../meteo/
../meteo/AVAILABLE
============================================
EOF

    cat > ${C_DATA}/validate_run/options/COMMAND <<'EOF'
&COMMAND
 LDIRECT=               1,
 IBDATE=         20240101,
 IBTIME=           000000,
 IEDATE=         20240101,
 IETIME=           060000,
 LOUTSTEP=          21600,
 LOUTAVER=           1800,
 LOUTSAMPLE=          900,
 ITSPLIT=        99999999,
 LSYNCTIME=           900,
 CTL=          -5.0000000,
 IFINE=                 4,
 IOUT=                  1,
 IPOUT=                 2,
 LSUBGRID=              0,
 LCONVECTION=           0,
 LAGESPECTRA=           0,
 IPIN=                  0,
 IOUTPUTFOREACHRELEASE= 1,
 IFLUX=                 0,
 MDOMAINFILL=           0,
 IND_SOURCE=            1,
 IND_RECEPTOR=          1,
 MQUASILAG=             0,
 NESTED_OUTPUT=         0,
 LINIT_COND=            0,
 SURF_ONLY=             0,
 CBLFLAG=               0,
 OHFIELDS_PATH= \"../../flexin/\",
 /
EOF

    cat > ${C_DATA}/validate_run/options/RELEASES <<'EOF'
&RELEASES_CTRL
 NSPEC      =           1,
 SPECNUM_REL=          24,
 /
&RELEASE
 IDATE1  =       20240101,
 ITIME1  =         000000,
 IDATE2  =       20240101,
 ITIME2  =         000000,
 LON1    =         10.000,
 LON2    =         10.000,
 LAT1    =         10.000,
 LAT2    =         10.000,
 Z1      =         50.000,
 Z2      =         50.000,
 ZKIND   =              1,
 MASS    =       1.0000E0,
 PARTS   =         ${V_PARTICLES},
 COMMENT =    \"VALIDATION\",
 /
EOF

    cat > ${C_DATA}/validate_run/options/OUTGRID <<EOF
&OUTGRID
 OUTLON0=      ${V_OUTLON0},
 OUTLAT0=      ${V_OUTLAT0},
 NUMXGRID=       ${V_NX},
 NUMYGRID=       ${V_NY},
 DXOUT=        ${V_DX},
 DYOUT=        ${V_DY},
 OUTHEIGHTS=  100.0, 250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 5000.0,
 /
EOF

    cat > ${C_DATA}/validate_run/options/RECEPTORS <<'EOF'
*******************************************************************************
*                                                                             *
*  Input file for the Lagrangian particle dispersion model FLEXPART           *
*                        Please specify receptor points                       *
*******************************************************************************
                0
EOF

    cat > ${C_DATA}/validate_run/options/AGECLASSES <<'EOF'
&AGECLASS
 NAGECLASS= 1,
 LAGE= 21600,
 /
EOF

    cp ${C_FLEXPART}/options/IGBP_int1.dat  ${C_DATA}/validate_run/options/
    cp ${C_FLEXPART}/options/surfdata.t     ${C_DATA}/validate_run/options/
    cp ${C_FLEXPART}/options/surfdepo.t     ${C_DATA}/validate_run/options/
    if [ -d ${C_FLEXPART}/options/SPECIES ]; then
      cp -r ${C_FLEXPART}/options/SPECIES/* ${C_DATA}/validate_run/options/SPECIES/
    fi
  "

  log_info "Validation setup complete."
}

do_validate() {
  local mode="$1"
  mkdir -p "${PROJECT_ROOT}/target/validation"

  # Step 1: Setup
  do_validate_setup "$mode"

  # Step 2: Run Fortran FLEXPART
  log_info "Running FLEXPART Fortran (validation)..."
  fortran_exec "$mode" bash -c "
    cd ${C_DATA}/validate_run && ${C_FLEXPART}/src/FLEXPART
  " 2>&1 | tee "${PROJECT_ROOT}/target/validation/fortran.log"

  if grep -q "CONGRATULATIONS" "${PROJECT_ROOT}/target/validation/fortran.log" 2>/dev/null; then
    log_info "Fortran run completed successfully"
  else
    log_error "Fortran run failed"
    tail -10 "${PROJECT_ROOT}/target/validation/fortran.log"
    exit 1
  fi

  # Step 3: Build and run GPU validation binary
  log_info "Building GPU validation binary..."
  cargo build --release --bin fortran-validation 2>&1 \
    | tee "${PROJECT_ROOT}/target/validation/gpu-build.log"

  log_info "Running GPU validation..."
  OUTPUT_PATH="${PROJECT_ROOT}/target/validation/gpu_concentration.json" \
    PARTICLES=${V_PARTICLES} \
    cargo run --release --bin fortran-validation 2>&1 \
    | tee "${PROJECT_ROOT}/target/validation/gpu.log"

  # Step 4: Fortran output is now at the bind-mount path
  local FORTRAN_OUTPUT="${PROJECT_ROOT}/target/comparison/validate_run/output"
  if [ ! -d "${FORTRAN_OUTPUT}" ]; then
    log_error "Fortran output not found at ${FORTRAN_OUTPUT}. Check volume mounts."
    exit 1
  fi
  log_info "Fortran output at ${FORTRAN_OUTPUT}"
  ls -la "${FORTRAN_OUTPUT}/" | head -20

  # Step 5: Run comparison
  log_info "Comparing concentration fields..."
  python3 "${SCRIPT_DIR}/compare_concentrations.py" \
    --fortran-output "${FORTRAN_OUTPUT}" \
    --gpu-output "${PROJECT_ROOT}/target/validation/gpu_concentration.json" \
    --output-json "${PROJECT_ROOT}/target/validation/comparison_report.json" \
    --verbose 2>&1 | tee "${PROJECT_ROOT}/target/validation/comparison.log"

  log_info "Validation complete. Results in target/validation/"
}

# ---------------------------------------------------------------------------
# PERF — GPU-only performance profile (strict vs production)
# ---------------------------------------------------------------------------
do_perf() {
  local mode="$1"
  mkdir -p "${PROJECT_ROOT}/target/validation"

  local particles="${PERF_PARTICLES:-1000000}"
  local warmup_steps="${PERF_WARMUP_STEPS:-5}"
  local measure_steps="${PERF_MEASURE_STEPS:-30}"

  log_info "GPU performance profile (mode=${mode}, particles=${particles})"
  log_info "Output directory: target/validation/"

  log_info "Run 1/3: strict GPU baseline (SYNC_READBACK=1)"
  OUTPUT_PATH="${PROJECT_ROOT}/target/validation/perf_gpu_${particles}_strict.json" \
    PARTICLES="${particles}" \
    SYNC_READBACK=1 \
    /usr/bin/time -f 'REAL_SECONDS=%e' \
    cargo run --release --bin fortran-validation 2>&1 \
    | tee "${PROJECT_ROOT}/target/validation/perf_gpu_${particles}_strict.log"

  log_info "Run 2/3: production GPU baseline (SYNC_READBACK=0)"
  OUTPUT_PATH=/dev/null \
    PARTICLES="${particles}" \
    SYNC_READBACK=0 \
    /usr/bin/time -f 'REAL_SECONDS=%e' \
    cargo run --release --bin fortran-validation 2>&1 \
    | tee "${PROJECT_ROOT}/target/validation/perf_gpu_${particles}_prod.log"

  log_info "Run 3/3: profiled timeloop benchmark"
  FLEXPART_GPU_PROFILE=1 \
    PARTICLES="${particles}" \
    WARMUP_STEPS="${warmup_steps}" \
    MEASURE_STEPS="${measure_steps}" \
    cargo run --release --bin bench-timeloop 2>&1 \
    | tee "${PROJECT_ROOT}/target/validation/perf_gpu_${particles}_timeloop_profile.log"

  log_info "Performance profile complete."
  echo "Generated logs:"
  echo "  target/validation/perf_gpu_${particles}_strict.log"
  echo "  target/validation/perf_gpu_${particles}_prod.log"
  echo "  target/validation/perf_gpu_${particles}_timeloop_profile.log"
}

# ---------------------------------------------------------------------------
# CLEAN
# ---------------------------------------------------------------------------
do_clean() {
  local mode="$1"
  log_info "Cleaning up..."
  fortran_compose_cmd "$mode" down -v --remove-orphans 2>/dev/null || true
  gpu_compose_cmd "$mode" down -v --remove-orphans 2>/dev/null || true
  rm -f "${PROJECT_ROOT}/target/gpu-comparison.log"
  rm -f "${PROJECT_ROOT}/target/fortran-comparison.log"
  log_info "Done."
}

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
MODE="${1:-compose}"
ACTION="${2:-all}"

case "$MODE" in
  compose|nvidia)
    cd "${PROJECT_ROOT}"
    case "$ACTION" in
      setup)    do_setup    "$MODE" ;;
      run)      do_run      "$MODE" ;;
      clean)    do_clean    "$MODE" ;;
      validate) do_validate "$MODE" ;;
      perf)     do_perf     "$MODE" ;;
      all)      do_setup "$MODE"; do_run "$MODE" ;;
      *)        echo "Usage: $0 $MODE [setup|run|clean|all|validate|perf]"; exit 2 ;;
    esac
    ;;
  local)
    echo "Local mode: ensure gfortran, eccodes, python3 are installed."
    echo "Then adapt the paths in this script. Docker mode is recommended."
    exit 2
    ;;
  *)
    echo "Usage: $0 [compose|nvidia|local] [setup|run|clean|all]"
    exit 2
    ;;
esac
