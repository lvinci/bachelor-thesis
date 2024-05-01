#!/usr/bin/python3
import csv
import os
import string
import subprocess
import sys
from typing import Dict


class DatasetRow:
    def __init__(self, threads: int, partitions: int, runtime: float):
        self.threads = threads
        self.partitions = partitions
        self.runtime = runtime


def read_csv_dataset(filename: string) -> list[DatasetRow]:
    datasets = []
    with open(filename, newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        for line in reader:
            datasets.append(DatasetRow(int(line[0]), int(line[1]), float(line[2])))
    return datasets


def generate_comp_all_threads_plot(benchmark_id: string, dataset: list[DatasetRow]):
    # Group the dataset rows into separate lines
    lines: Dict[int, list[DatasetRow]] = {}
    for row in dataset:
        if row.threads not in lines:
            lines[row.threads] = []
        lines[row.threads].append(row)
    # Create the data file
    data_filepath = f'../results/plots/{benchmark_id}/comp_all_threads.dat'
    os.makedirs(os.path.dirname(data_filepath), exist_ok=True)
    data_file = open(data_filepath, "w")
    index: int = 0
    # Write each line to the file
    for line, rows in lines.items():
        data_file.write(f'# {rows[0].threads} Threads (index: {index}) \n')
        data_file.write("# X   Y\n")
        for row in rows:
            data_file.write(f'  {row.partitions}   {row.runtime}\n')
        data_file.write("\n\n")
        index += 1
    data_file.close()
    # Generate the plot
    if len(lines) < 6:
        subprocess.run(["cp", "plots/comp_all_threads_small.gp", f'../results/plots/{benchmark_id}/comp_all_threads.gp'])
    else:
        subprocess.run(["cp", "plots/comp_all_threads.gp", f'../results/plots/{benchmark_id}/comp_all_threads.gp'])
    os.chdir(f'../results/plots/{benchmark_id}/')
    subprocess.run(["gnuplot", "comp_all_threads.gp"])
    os.chdir("../../../scripts")


def generate_comp_all_partitions_plot(benchmark_id: string, dataset: list[DatasetRow]):
    # Group the dataset rows into separate lines
    lines: Dict[int, list[DatasetRow]] = {}
    for row in dataset:
        if row.partitions not in lines:
            lines[row.partitions] = []
        lines[row.partitions].append(row)
    # Create the data file
    data_filepath = f'../results/plots/{benchmark_id}/comp_all_partitions.dat'
    os.makedirs(os.path.dirname(data_filepath), exist_ok=True)
    data_file = open(data_filepath, "w")
    index: int = 0
    # Write each line to the file
    for line, rows in lines.items():
        data_file.write(f'# {rows[0].partitions} Partitions (index: {index}) \n')
        data_file.write("# X   Y\n")
        for row in rows:
            data_file.write(f'  {row.threads}   {row.runtime}\n')
        data_file.write("\n\n")
        index += 1
    data_file.close()
    # Generate the plot
    if len(lines) < 4:
        subprocess.run(["cp", "plots/comp_all_partitions_small.gp", f'../results/plots/{benchmark_id}/comp_all_partitions.gp'])
    else:
        subprocess.run(["cp", "plots/comp_all_partitions.gp", f'../results/plots/{benchmark_id}/comp_all_partitions.gp'])
    os.chdir(f'../results/plots/{benchmark_id}/')
    subprocess.run(["gnuplot", "comp_all_partitions.gp"])
    os.chdir("../../../scripts")


def main() -> int:
    for benchmark_id in os.listdir("../results/benchmarks"):
        if os.path.isdir(f'../results/benchmarks/{benchmark_id}'):
            dataset = read_csv_dataset(f'../results/benchmarks/{benchmark_id}_averaged.csv')
            generate_comp_all_threads_plot(benchmark_id, dataset)
            generate_comp_all_partitions_plot(benchmark_id, dataset)
    return 0


if __name__ == '__main__':
    sys.exit(main())
