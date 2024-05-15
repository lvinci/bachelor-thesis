#!/usr/bin/python3
import csv
import os
import sys


class Result:
    def __init__(self, device: str, threads: int, partitions: int, runtime: float, stddev: float):
        self.threads = threads
        self.partitions = partitions
        self.runtime = runtime
        self.stddev = stddev
        self.device = device

def read_results() -> list[Result]:
    results = []
    for result_file in os.listdir("../results/benchmarks"):
        if os.path.isfile(f'../results/benchmarks/{result_file}') and "csv" in result_file:
            with open(f'../results/benchmarks/{result_file}', newline="") as csvfile:
                reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
                for line in reader:
                    results.append(Result(result_file.replace("_averaged.csv", ""), int(line[0]), int(line[1]), float(line[2]), float(line[3])))
    return results


def find_result(results: list[Result], thread: int, partitions: int, device: str) -> Result:
    for result in results:
        if result.threads == thread and result.partitions == partitions and result.device == device:
            return result
    return Result(device, 0, 0, -2, -2)


def fmt(v: float) -> str:
    if v >= 0:
        return str(v).replace(".", ",")
    return "\\text{-}"


def generate_table(output_file: str, results: list[Result]):
    table = open(output_file, "w")
    table.write("\\begin{tabular}{ |c|r|S S|S S|S S|S S| } \n")
    table.write("\\hline & & \\multicolumn{2}{c|}{M1 Pro} & \\multicolumn{2}{c|}{Raspberry Pi} & \\multicolumn{2}{c|}{\\text{Ryzen 3600XT}} & \\multicolumn{2}{c|}{VPS Server}\\\\\n")
    table.write("\\text{Threads} & \\text{Daten} & \\text{Zeit} & \\text{SD} & \\text{Zeit} & \\text{SD} & \\text{Zeit}& \\text{SD} & \\text{Zeit} & \\text{SD}  \\\\\\hline\n")
    partitions = (1, 2, 4, 8)
    threads = (1, 2, 4, 6, 8, 10)
    TOTAL_DATA = 19020
    for thread in threads:
        for partition in partitions:
            datacount = int(TOTAL_DATA / partition)
            m1pro = find_result(results, thread, partition, "m1pro")
            ryzen = find_result(results, thread, partition, "3600xt")
            raspi = find_result(results, thread, partition, "raspberrypi3")
            vps = find_result(results, thread, partition, "vps")
            table.write(f'${thread}$ & ${datacount}$ & {fmt(m1pro.runtime)} & {fmt(m1pro.stddev)} & {fmt(raspi.runtime)} & {fmt(raspi.stddev)} & {fmt(ryzen.runtime)} & {fmt(ryzen.stddev)} & {fmt(vps.runtime)} & {fmt(vps.stddev)} \\\\ \n')
    table.write("\\hline \\end{tabular} \n")


def main() -> int:
    results = read_results()
    generate_table("../results/results_table.tex", results)
    return 0


if __name__ == '__main__':
    sys.exit(main())
