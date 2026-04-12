# import os
# import subprocess
# import sys
# import time

# BASE_DIR = r"D:\COS30018-Intelligent Systtem\COS30018-Intelligent-Systems-master\COS30018-Intelligent-Systems-master"

# TASKS = {
#     "Task C.1 (v01)": os.path.join(BASE_DIR, "W3", "stock_prediction_v01.py"),
#     "Task C.2 (v02)": os.path.join(BASE_DIR, "W4", "stock_prediction_v02.py"),
#     "Task C.3 (v03)": os.path.join(BASE_DIR, "W5", "stock_prediction_v03.py"),
#     "Task C.4 (v04)": os.path.join(BASE_DIR, "W6", "stock_prediction_v04.py"),
#     "Task C.5 (v05)": os.path.join(BASE_DIR, "W7", "stock_prediction_v07.py"),
#     "Task C.6 (v06)": os.path.join(BASE_DIR, "W8", "stock_prediction_v06.py"),
#     "Task C.7 (v07)": os.path.join(BASE_DIR, "W9", "stock_prediction_v07.py"),
# }

# # This code is injected into every subprocess before the target script runs.
# # It replaces plt.show() with a 5-second pause then auto-close.
# PATCH = """
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as _plt
# import time as _time

# _orig_show = _plt.show

# def _patched_show(*args, **kwargs):
#     try:
#         _orig_show(block=False)
#         for num in _plt.get_fignums():
#             fig = _plt.figure(num)
#             fig.canvas.draw()
#             fig.canvas.flush_events()
#         for _ in range(50):        # 5 seconds, 10 ticks/sec
#             for num in _plt.get_fignums():
#                 _plt.figure(num).canvas.flush_events()
#             _time.sleep(0.1)
#     except Exception:
#         _time.sleep(5)
#     finally:
#         _plt.close('all')

# _plt.show = _patched_show

# import runpy, sys
# runpy.run_path(sys.argv[1], run_name='__main__')
# """


# def print_banner(text):
#     print("\n" + "=" * 60)
#     print(f" STARTING: {text}".center(60))
#     print("=" * 60 + "\n")


# def run_tasks():
#     if not os.path.exists(BASE_DIR):
#         print(f"FATAL ERROR: Base directory not found:\n  {BASE_DIR}")
#         return

#     for task_name, script_path in TASKS.items():
#         print_banner(task_name)

#         if not os.path.exists(script_path):
#             print(f"ERROR: Script not found at:\n  {script_path}")
#             print("Stopping. Check your file naming.")
#             break

#         try:
#             script_dir = os.path.dirname(script_path)

#             result = subprocess.run(
#                 [sys.executable, "-c", PATCH, script_path],
#                 cwd=script_dir,
#                 check=True,
#             )

#             print(f"\n  {task_name} completed successfully.")
#             time.sleep(1)

#         except subprocess.CalledProcessError as e:
#             print(f"\nERROR: {task_name} failed (exit code {e.returncode}).")
#             print("Stopping execution.")
#             break

#     print("\n" + "=" * 60)
#     print(" ALL TASKS COMPLETED.".center(60))
#     print("=" * 60)


# if __name__ == "__main__":
#     print("Initializing Full Stock Prediction Pipeline Demo...\n")
#     run_tasks()




import os
import subprocess
import sys
import time

BASE_DIR = r"D:\COS30018-Intelligent Systtem\COS30018-Intelligent-Systems-master\COS30018-Intelligent-Systems-master"

TASKS = {
    # "Task C.5 (v05)": os.path.join(BASE_DIR, "W7", "stock_prediction_v05.py"),
    # "Task C.6 (v06)": os.path.join(BASE_DIR, "W8", "stock_prediction_v06.py"),
    "Task C.7 (v07)": os.path.join(BASE_DIR, "W9", "stock_prediction_v07.py"),
}

PATCH = """
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as _plt
import time as _time

_orig_show = _plt.show

def _patched_show(*args, **kwargs):
    try:
        _orig_show(block=False)
        for num in _plt.get_fignums():
            fig = _plt.figure(num)
            fig.canvas.draw()
            fig.canvas.flush_events()
        for _ in range(50):
            for num in _plt.get_fignums():
                _plt.figure(num).canvas.flush_events()
            _time.sleep(0.1)
    except Exception:
        _time.sleep(5)
    finally:
        _plt.close('all')

_plt.show = _patched_show

import runpy, sys
runpy.run_path(sys.argv[1], run_name='__main__')
"""


def print_banner(text):
    print("\n" + "=" * 60)
    print(f" STARTING: {text}".center(60))
    print("=" * 60 + "\n")


def run_tasks():
    if not os.path.exists(BASE_DIR):
        print(f"FATAL ERROR: Base directory not found:\n  {BASE_DIR}")
        return

    for task_name, script_path in TASKS.items():
        print_banner(task_name)

        if not os.path.exists(script_path):
            print(f"ERROR: Script not found at:\n  {script_path}")
            print("Stopping. Check your file naming.")
            break

        try:
            script_dir = os.path.dirname(script_path)
            subprocess.run(
                [sys.executable, "-c", PATCH, script_path],
                cwd=script_dir,
                check=True,
            )
            print(f"\n  {task_name} completed successfully.")
            time.sleep(1)

        except subprocess.CalledProcessError as e:
            print(f"\nERROR: {task_name} failed (exit code {e.returncode}).")
            print("Stopping execution.")
            break

    print("\n" + "=" * 60)
    print(" ALL TASKS COMPLETED.".center(60))
    print("=" * 60)


if __name__ == "__main__":
    print("Running C.5 → C.7 Pipeline...\n")
    run_tasks()