#!/bin/bash

EXECUTABLE="./lucavinciguerra-bathesis"
ATTEMPTS=5
PARTITIONS=(8 4 2 1) # Partitions to be tested
THREADS=(10 8 6 4 2 1) # Thread counts to be tested (this script will ignore higher counts than AVAILABLE_THREADS)
AVAILABLE_THREADS=10 # Amount of threads to be used
CSV_FILE="benchmark_results.csv"

# CLI COLORS
RED='\033[0;33m'
GREEN='\033[0;32m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Parse available threads from command line input
if [ $# -eq 1 ]
  then
    AVAILABLE_THREADS=$1
fi
echo ""
echo -e "Using a maximum of ${PURPLE}$AVAILABLE_THREADS${NC} threads for this benchmark"

# Make sure the executable exists in this path
if [ -e $EXECUTABLE ]; then
    echo -e "Executable ${PURPLE}$EXECUTABLE${NC} found, will be used for the benchmark"
    echo ""
else
    echo -e "Executable ${PURPLE}$EXECUTABLE${NC} not found"
fi

# Run the attempt with the given values
function run_attempt {
  local threadcount=$1
  local partitioncount=$2
  local attempt=$3
  echo -e -n "${NC}Benchmarking ${RED}$partitioncount${NC} partitions with ${RED}$threadcount${NC} threads #${RED}$attempt${NC} "
  local filename="benchmark_attempt${attempt}_${partitioncount}partitions_${threadcount}threads.txt"
  start=$(date +%s.%3N)
  { time $EXECUTABLE "$threadcount" "$partitioncount" "$attempt" > /dev/null; } 2>> "$filename"
  end=$(date +%s.%3N)
  echo "TOTAL TIME: $(echo "$end - $start" | bc) seconds" >> "$filename"
  echo -e "${GREEN}[$(echo "$end - $start" | bc)s]"
  echo "$threadcount,$partitioncount,$attempt,$(echo "$end - $start" | bc)" >> "$CSV_FILE"
}

# Run the benchmarks for all possible combinations
for partition in ${PARTITIONS[@]};do
  for threads in ${THREADS[@]};do
      if ((threads <= AVAILABLE_THREADS));then
        for ((i = 1; i <= ATTEMPTS; i++)); do
          run_attempt "$threads" "$partition" "$i"
        done
      fi
    done
done