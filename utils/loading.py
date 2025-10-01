
import sys
import time

def progress_bar(iteration, total, length=50, start_time=None):
    percent = int(100 * (iteration / float(total)))
    filled = int(length * iteration // total)
    bar = "#" * filled + "." * (length - filled)

    # ETA
    if iteration > 0 and start_time is not None:
        elapsed = time.time() - start_time
        eta = elapsed * (total / iteration - 1)
        eta_str = f" | ETA: {eta:.1f}s"
    else:
        eta_str = ""

    sys.stdout.write(f"\r[{bar}] {percent}%{eta_str}")
    sys.stdout.flush()
    if iteration == total:
        print()