
import sys
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.init as init


def get_mean_and_std(dataset):
    """Compute mean and std of a 3-channel dataset."""
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    mean = torch.zeros(3)
    std = torch.zeros(3)

    print("==> Computing mean and std..")

    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()

    mean.div_(len(dataset))
    std.div_(len(dataset))

    return mean, std


def init_params(net):
    """Initialize neural network parameters."""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            if m.weight is not None:
                init.constant_(m.weight, 1)
            if m.bias is not None:
                init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


TOTAL_BAR_LENGTH = 65
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time

    if total <= 0:
        raise ValueError("total must be greater than 0")

    if current == 0:
        begin_time = time.time()
        last_time = begin_time

    term_width = shutil.get_terminal_size(
        fallback=(100, 20)
    ).columns

    cur_len = int(
        TOTAL_BAR_LENGTH * (current + 1) / total
    )
    cur_len = min(cur_len, TOTAL_BAR_LENGTH)

    rest_len = TOTAL_BAR_LENGTH - cur_len

    bar = "=" * cur_len + "." * rest_len

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    message = (
        f" Step: {format_time(step_time)}"
        f" | Tot: {format_time(tot_time)}"
    )

    if msg:
        message += f" | {msg}"

    output = (
        f"[{bar}] {current + 1}/{total}{message}"
    )

    if current < total - 1:
        sys.stdout.write(
            "\r" + output[:term_width - 1]
        )
    else:
        sys.stdout.write(
            "\r" + output + "\n"
        )

    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds // 86400)
    seconds %= 86400

    hours = int(seconds // 3600)
    seconds %= 3600

    minutes = int(seconds // 60)
    seconds %= 60

    secondsf = int(seconds)
    millis = int((seconds - secondsf) * 1000)

    parts = []

    if days > 0:
        parts.append(f"{days}D")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secondsf > 0:
        parts.append(f"{secondsf}s")
    if millis > 0:
        parts.append(f"{millis}ms")

    return "".join(parts[:2]) if parts else "0ms"
