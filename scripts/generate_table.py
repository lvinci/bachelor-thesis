#!/usr/bin/python3
import csv
import os

RESULTS_DIRECTORY = "../results/benchmarks"
TOTAL_DATASET_COUNT = 19020
OUTPUT_FILE = "../results/results_table.tex"


# Represents a set of 5 results with their average runtime and standard deviation
class Result:
    def __init__(self, device: str, threads: int, partitions: int, runtime: float, stddev: float):
        self.threads = threads
        self.partitions = partitions
        self.runtime = runtime
        self.stddev = stddev
        self.device = device.replace("_averaged.csv", "")  # sanitize name string


# Reads the results from all csv files
def read_results() -> list[Result]:
    results: list[Result] = []
    # Iterate through all files and directories in the results directory
    for result_file in os.listdir(RESULTS_DIRECTORY):
        # Make sure that the found file is a file and also is a csv file
        if os.path.isfile(f'{RESULTS_DIRECTORY}/{result_file}') and "csv" in result_file:
            # Open the file and read its csv content
            with open(f'{RESULTS_DIRECTORY}/{result_file}', newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
                # Read all lines from the csv file and append them to the results list
                for line in reader:
                    results.append(Result(result_file, int(line[0]), int(line[1]), float(line[2]), float(line[3])))
    return results


# Finds all the unique names of devices in the given result set
def find_devices(results: list[Result]) -> list[str]:
    devices: list[str] = []
    for result in results:
        if result.device not in devices:
            devices.append(result.device)
    return sorted(devices)


# Pretty prints the full device name from the given device id
def full_device_name(device: str) -> str:
    if device == "m1pro":
        return "Apple M1 Pro"
    elif device == "3600xt":
        return "Ryzen 3600XT"
    elif device == "raspberrypi3":
        return "Raspberry Pi 3"
    elif device == "vps":
        return "VPS Server"
    return device


# Find the result with the given filters.
# If the result could not be found, an empty result with negative runtime and stddev is returned.
def find_result(results: list[Result], thread: int, partitions: int, device: str) -> Result:
    for result in results:
        if result.threads == thread and result.partitions == partitions and result.device == device:
            return result
    return Result(device, 0, 0, -2, -2)


# Formats the given float as an element in latex
def fmt(v: float) -> str:
    if v >= 0:
        return str(v).replace(".", ",")
    return "\\text{-}"


# Generates the TeX Table for the given results in the output file
def generate_table(output_file: str, results: list[Result]):
    devices = find_devices(results)
    # Open the file and start specifying the table size
    table = open(output_file, "w")
    table.write("\\begin{tabular}{ |c|r|")
    for _ in devices:
        table.write("S S|")
    # Write the first line of the table header
    table.write(" } \n\t%Tabellenkopf\n\t\\hline & ")
    for device in devices:
        table.write(f'& \\multicolumn{{2}}{{c|}}{{{full_device_name(device)}}} ')
    # Write the second line of the table header
    table.write("\\\\\n\t\\text{Threads} & \\text{Datenmenge} ")
    for _ in devices:
        table.write("& $t$ & $\sigma$ ")
    # Start writing the table data
    table.write("\\\\\\hline\n\t%Daten\n")
    partitions = (1, 2, 4, 8)
    threads = (1, 2, 4, 6, 8, 10)
    for thread in threads:
        for partition in partitions:
            datacount = int(TOTAL_DATASET_COUNT / partition)
            table.write(f'\t${thread}$ & ${datacount}$ ')
            for device in devices:
                result = find_result(results, thread, partition, device)
                table.write(f'& {fmt(result.runtime)} & {fmt(result.stddev)} ')
            table.write(" \\\\ \n")
    table.write("\\hline \\end{tabular} \n")


if __name__ == '__main__':
    generate_table(OUTPUT_FILE, read_results())
