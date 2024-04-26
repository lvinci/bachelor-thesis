#!/usr/bin/python3

import csv
import os
import sys

ATTEMPTS = 5


# Read in csv file
def generate_averages(benchmark_id):
    source_filepath = f'../results/benchmarks/{benchmark_id}/benchmark_results.csv'
    destination_file = open(f'../results/benchmarks/{benchmark_id}_averaged.csv', "w")
    with open(source_filepath, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        avg_runtime = 0.0
        for line in reader:
            # Read in the csv rows
            threads = line[0]
            partitions = line[1]
            attempt = int(line[2])
            runtime = float(line[3])
            # Add to the average runtime add write it to the file when the 5th attempt is reached
            avg_runtime += runtime
            if attempt == ATTEMPTS:
                avg_runtime /= ATTEMPTS
                destination_file.write(f'{threads},{partitions},{avg_runtime}\n')
                avg_runtime = 0
    destination_file.close()


def main() -> int:
    for benchmark_id in os.listdir("../results/benchmarks"):
        if os.path.isdir(f'../results/benchmarks/{benchmark_id}'):
            generate_averages(benchmark_id)
    return 0


if __name__ == '__main__':
    sys.exit(main())
