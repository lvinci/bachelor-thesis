#!/usr/bin/python3

import csv
import os
import statistics
import sys

ATTEMPTS = 5


# Read in csv file
def generate_averages(benchmark_id):
    source_filepath = f'../results/benchmarks/{benchmark_id}/benchmark_results.csv'
    averages_file = open(f'../results/benchmarks/{benchmark_id}_averaged.csv', "w")
    with open(source_filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        avg_runtime = 0.0
        results = []
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
                avg_runtime /= ATTEMPTS
                avg_runtime = round(avg_runtime, 1)
                stddev = round(statistics.stdev(results), 1)
                variance = round(statistics.variance(results), 1)
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
