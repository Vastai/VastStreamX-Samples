import subprocess


def run_cmd(command, check_stderr=True):
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # 获取命令的输出和错误
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        raise RuntimeError(
            f"returncode({process.returncode}) != 0, Failed to run: {command},stderr:{stderr.decode()}"
        )

    if check_stderr and stderr:
        raise RuntimeError(f"Failed to run {command}, stderr: {stderr.decode()}")

    return stdout.decode()
