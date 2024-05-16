#!/usr/bin/python3

import csv
import os
import statistics
import sys

ATTEMPTS = 5
DECIMAL_PLACES = 1


# Read in csv file
def generate_averages(benchmark_id):
    source_filepath = f'../results/benchmarks/{benchmark_id}/benchmark_results.csv'
    averages_file = open(f'../results/benchmarks/{benchmark_id}_averaged.csv', "w")
    with open(source_filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        avg_runtime = 0.0
        results = [] # stores the results for the current tries
        for line in reader:
            # Read in the csv rows
            threads = line[0]
            partitions = line[1]
            attempt = int(line[2])
            runtime = float(line[3])
            results.append(runtime)
            # Add to the average runtime add write it to the file when the 5th attempt is reached
            avg_runtime += runtime
            if attempt == ATTEMPTS:
                # calculate average and standard deviation
                avg_runtime /= ATTEMPTS
                avg_runtime = round(avg_runtime, DECIMAL_PLACES)
                stddev = round(statistics.stdev(results), DECIMAL_PLACES)
                variance = round(statistics.variance(results), DECIMAL_PLACES)
                # write results to csv file and reset variables
                averages_file.write(f'{threads},{partitions},{avg_runtime},{stddev},{variance}\n')
                avg_runtime = 0
                results = []
    averages_file.close()


def main() -> int:
    for benchmark_id in os.listdir("../results/benchmarks"):
        if os.path.isdir(f'../results/benchmarks/{benchmark_id}'):
            generate_averages(benchmark_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
