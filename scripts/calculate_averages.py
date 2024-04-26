#!/usr/bin/python3

import csv
import sys

ATTEMPTS = 5

benchmarkId = sys.argv[1]
sourceFilepath = f'../results/benchmarks/{benchmarkId}/benchmark_results.csv'
destinationFile = open(f'../results/benchmarks/{benchmarkId}_averaged.csv', "w")

# Read in csv file
with open(sourceFilepath, newline="") as csvfile:
    reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
    avgRuntime = 0.0
    for line in reader:
        # Read in the csv rows
        threads = line[0]
        partitions = line[1]
        attempt = int(line[2])
        runtime = float(line[3])
        # Add to the average runtime add write it to the file when the 5th attempt is reached
        avgRuntime += runtime
        if attempt == ATTEMPTS:
            avgRuntime /= ATTEMPTS
            destinationFile.write(f'{threads},{partitions},{avgRuntime}\n')
            avgRuntime = 0

destinationFile.close()
