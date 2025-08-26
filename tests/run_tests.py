
# 
# Copyright (C) 2025 Vastai-tech Company.
# All rights reserved.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# 
import os
import subprocess
from multiprocessing import Process, Queue
import argparse
import ast
import crayons


def find_test_files(folder_path, file_start="test_"):
    test_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.startswith(file_start) and file.endswith(".py"):
                test_files.append(os.path.join(root, file))
    return test_files

def run_test(test_file, device_id, report_path, mark):
    print(f"Running {test_file} on device_id {device_id}...")
    subprocess.run([
        "pytest",
        test_file,
        f"-m {mark}",
        f"--html={report_path}",
        "--self-contained-html",
        "--dist=loadfile",
        f"--device_id={device_id}" 
    ])
    print(f"Finished {test_file} on device_id:{device_id}. Report saved to {report_path}")

def worker(test_queue, device, report_dir,case_type):
    while not test_queue.empty():
        test_file = test_queue.get()
        report_path = os.path.join(report_dir, f"report_{device}_{os.path.basename(test_file)}.html")
        run_test(test_file, device, report_path,case_type)
    print(crayons.green(f"The worker work on device_id:{device} exit."))


def main(folder_path, devices, report_dir, merged_html, case_type = "fast"):
    file_start = "test_"
    test_files = find_test_files(folder_path,file_start)
    if not test_files:
        print("No test files found!")
        return

    print(f"Found {len(test_files)} test files:")
    for file in test_files:
        print(f"  - {file}")

    os.makedirs(report_dir, exist_ok=True)

    test_queue = Queue()
    for test_file in test_files:
        test_queue.put(test_file)

    processes = []
    for device in devices:
        p = Process(target=worker, args=(test_queue, device, report_dir,case_type))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    merge_cmd = [
        "pytest_html_merger",
        "-i", report_dir,
        "-o", merged_html
    ]
    subprocess.run(merge_cmd, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests in parallel on multiple devices.")
    parser.add_argument("--folder", type=str, help="Path to the folder containing test files.")
    parser.add_argument("--device_ids", type=str, help="List of devices to run tests on.")
    parser.add_argument("--merged_html", type=str, default="vsx_samples_merge_report.html", help="Test result file.")
    parser.add_argument("--report_dir", type=str, default="test_reports", help="Test result file.")
    parser.add_argument(
        "--case_type",
        type=str,
        choices=["slow", "ai_integration", "fast", "codec", "codec_integration"],  
        required=True, 
        help="Choose test case type, support [slow,ai_integration,fast,codec,codec_integration]"
    )
    args = parser.parse_args()

    device_ids = ast.literal_eval(args.device_ids)

    main(args.folder, device_ids, args.report_dir, args.merged_html, args.case_type)