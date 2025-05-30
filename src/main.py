import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import segment.val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import SegmentationModel
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import GenericLogger
from utils.plots import plot_evolve, plot_labels
from utils.segment.dataloaders import create_dataloader
from utils.segment.loss import ComputeLoss
from utils.segment.metrics import KEYS, fitness
from utils.segment.plots import plot_images_and_masks, plot_results_with_masks
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()

import cv2
import mss
import numpy as np
import torch
import win32api
from colorama import Fore, Style
from line_profiler import profile

try:
    from controller_setup import initialize_pygame_and_controller, get_left_trigger, get_right_trigger
    from core.send_targets import send_targets
    from gui.main_window import MainWindow
    from spawn_utils.config_manager import ConfigManager
    from spawn_utils.yolo_handler import YOLOHandler
except ImportError:
    # Try relative imports if the above fails
    from .controller_setup import initialize_pygame_and_controller, get_left_trigger, get_right_trigger
    from .core.send_targets import send_targets
    from .gui.main_window import MainWindow
    from .spawn_utils.config_manager import ConfigManager
    from .spawn_utils.yolo_handler import YOLOHandler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use CuPy for GPU acceleration if available
if torch.cuda.is_available():
    pass

# --- Global Variables ---
MODELS_PATH = os.path.join(SCRIPT_DIR, "models")

screen = None
random_x, random_y, arduino = 0, 0, None

@profile
def calculate_targets_numba(boxes, width, height, headshot_percent):
    width_half = width / 2
    height_half = height / 2
    
    x = ((boxes[:, 0] + boxes[:, 2]) / 2) - width_half
    y = ((boxes[:, 1] + boxes[:, 3]) / 2) + headshot_percent * (boxes[:, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2)) - height_half
    
    targets = np.column_stack((x, y))
    distances = np.sqrt(np.sum(targets**2, axis=1))
    
    return targets, distances

def preprocess_frame(np_frame):
    if np_frame.shape[2] == 4:
        return np_frame[:, :, :3]
    return np_frame

# --- Utility Functions ---
def pr_red(text):
    print(Fore.RED + text, Style.RESET_ALL)

def pr_green(text):
    print(Fore.GREEN + text + Style.RESET_ALL)

def pr_yellow(text):
    print(Fore.YELLOW + text + Style.RESET_ALL)

def pr_blue(text):
    print(Fore.BLUE + text + Style.RESET_ALL)

def pr_purple(text):
    print(Fore.MAGENTA + text + Style.RESET_ALL)

def pr_cyan(text):
    print(Fore.CYAN + text + Style.RESET_ALL)

def get_keycode(key):
    """Gets the virtual key code for a given key."""
    return config_manager.get_key_code(key) or win32api.VkKeyScan(key)

def update_aim_shake():
    """Updates random aim shake offsets."""
    global random_x, random_y
    if config_manager.get_setting("aim_shake"):
        aim_shake_strength = int(config_manager.get_setting("aim_shake_strength"))
        random_x = random.randint(-aim_shake_strength, aim_shake_strength)
        random_y = random.randint(-aim_shake_strength, aim_shake_strength)
    else:
        random_x = 0
        random_y = 0

def mask_frame(frame):
    """Masks out specified regions of the frame."""
    if config_manager.get_setting("mask_left"):
        frame[
            int(config_manager.get_setting("height") - config_manager.get_setting("mask_height")) : config_manager.get_setting("height"),
            0 : int(config_manager.get_setting("mask_width")),
            :,
        ] = 0
    if config_manager.get_setting("mask_right"):
        frame[
            int(config_manager.get_setting("height") - config_manager.get_setting("mask_height")) : config_manager.get_setting("height"),
            int(config_manager.get_setting("width") - config_manager.get_setting("mask_width")) : config_manager.get_setting("width"),
            :,
        ] = 0
    return frame

def initialize_game_window():
    global screen
    left = int(win32api.GetSystemMetrics(0) / 2 - config_manager.get_setting("width") / 2)
    top = int(win32api.GetSystemMetrics(1) / 2 - config_manager.get_setting("height") / 2)
    right = left + config_manager.get_setting("width")
    bottom = top + config_manager.get_setting("height")

    try:
        screen = mss.mss()
    except Exception as e:
        print(f"Error initializing mss: {e}")
        # Fallback to OpenCV for screen capture
        screen = None
        
    return {"top": top, "left": left, "width": config_manager.get_setting("width"), "height": config_manager.get_setting("height")}

@profile
def process_detections(detections):
    targets = np.empty((0, 2), dtype=np.float32)
    distances = np.empty(0, dtype=np.float32)
    coordinates = np.empty((0, 4), dtype=np.float32)

    if detections.shape[0] > 0:
        # Process detections
        valid_detections = detections[:, 4] > config_manager.get_setting("confidence") / 100
        boxes = detections[valid_detections, :4]
        confs = detections[valid_detections, 4]
        classes = detections[valid_detections, 5]

        if boxes.shape[0] > 0:
            targets, distances = calculate_targets_numba(
                boxes,
                config_manager.get_setting("width"),
                config_manager.get_setting("height"),
                config_manager.get_setting("headshot") / 100,
            )
            coordinates = boxes

    if config_manager.get_setting("fov_enabled"):
        fov_size = config_manager.get_setting("fov_size")
        fov_mask = np.sum(targets**2, axis=1) <= fov_size**2
        targets = targets[fov_mask]
        distances = distances[fov_mask]
        coordinates = coordinates[fov_mask]

    return targets, distances, coordinates

def capture_screen(monitor):
    """Capture screen using either mss or OpenCV as fallback"""
    global screen
    
    if screen is not None:
        # Use mss if available
        try:
            return np.array(screen.grab(monitor))
        except Exception as e:
            print(f"Error with mss screen capture: {e}")
            screen = None  # Fall back to OpenCV
    
    # Fallback to OpenCV
    try:
        x, y, w, h = monitor["left"], monitor["top"], monitor["width"], monitor["height"]
        # Use OpenCV to capture screen
        import cv2
        hwnd = None  # 0 for the entire screen
        screen_img = np.array(cv2.captureWindow(hwnd))
        return screen_img[y:y+h, x:x+w]
    except Exception as e:
        print(f"Error with OpenCV screen capture: {e}")
        # Return a blank frame as last resort
        return np.zeros((monitor["height"], monitor["width"], 3), dtype=np.uint8)

@profile
def main_loop(controller, main_window, yolo_handler, monitor):
    start_time = time.time()
    frame_count = 0
    pressing = False
    loop_times = []

    def process_frame():
        nonlocal frame_count, start_time, pressing
        loop_start = time.time()

        frame_count += 1
        
        # Capture screen
        np_frame = capture_screen(monitor)
        np_frame = preprocess_frame(np_frame)
        
        pygame.event.pump()

        if np_frame.shape[2] == 4:
            np_frame = np_frame[:, :, :3]

        with torch.no_grad():
            frame = mask_frame(np_frame)
            detections = yolo_handler.detect(frame)
            targets, distances, coordinates = process_detections(detections)

        send_targets(
            controller,
            config_manager.settings,
            targets,
            distances,
            random_x,
            random_y,
            get_left_trigger,
            get_right_trigger,
        )

        main_window.update_preview(np_frame, coordinates, targets, distances)

        if config_manager.get_setting("overlay"):
           main_window.update_overlay(coordinates)

        elapsed_time = time.time() - start_time
        if elapsed_time >= 0.2:
           main_window.update_fps_label(round(frame_count / elapsed_time))
           frame_count = 0
           start_time = time.time()
           update_aim_shake()

        if config_manager.get_setting("toggle"):
           trigger_value = get_left_trigger(controller)
           if trigger_value > 0.5 and not pressing:
               pressing = True
               main_window.toggle_auto_aim()
           elif trigger_value <= 0.5 and pressing:
               pressing = False

        if win32api.GetKeyState(config_manager.get_setting("quit_key")) in (-127, -128):
            main_window.on_closing()
            return

        loop_end = time.time()
        loop_times.append(loop_end - loop_start)

        if len(loop_times) >= 100:
            avg_loop_time = sum(loop_times) / len(loop_times)
            print(f"Average loop time: {avg_loop_time:.6f} seconds")
            loop_times.clear()

        if main_window.running:
            main_window.root.after(1, process_frame)

    main_window.root.after(1, process_frame)
    main_window.run()

@profile
def main(**argv):
    global config_manager

    # Initialize ConfigManager
    config_manager = ConfigManager(argv['settingsProfile'])

    # Print welcome message and instructions
    pr_purple(
        """
  ██████  ██▓███   ▄▄▄      █     █░ ███▄    █      ▄▄▄       ██▓ ███▄ ▄███▓ ▄▄▄▄    ▒█████  ▄▄▄█████▓
▒██    ▒ ▓██░  ██ ▒████▄   ▓█░ █ ░█░ ██ ▀█   █     ▒████▄   ▒▓██▒▓██▒▀█▀ ██▒▓█████▄ ▒██▒  ██▒▓  ██▒ ▓▒
░ ▓██▄   ▓██░ ██▓▒▒██  ▀█▄ ▒█░ █ ░█ ▓██  ▀█ ██▒    ▒██  ▀█▄ ▒▒██▒▓██    ▓██░▒██▒ ▄██▒██░  ██▒▒ ▓██░ ▒░
  ▒   ██▒▒██▄█▓▒ ▒░██▄▄▄▄██░█░ █ ░█ ▓██▒  ▐▌██▒    ░██▄▄▄▄██░░██░▒██    ▒██ ▒██░█▀  ▒██   ██░░ ▓██▓ ░ 
▒██████▒▒▒██▒ ░  ░▒▓█   ▓██░░██▒██▓ ▒██░   ▓██░     ▓█   ▓██░░██░▒██▒   ░██▒░▓█  ▀█▓░ ████▓▒░  ▒██▒ ░ 
▒ ▒▓▒ ▒ ░▒▓▒░ ░  ░░▒▒   ▓▒█░ ▓░▒ ▒  ░ ▒░   ▒ ▒      ░   ▒▒ ░ ▒ ░░ ▒░   ░  ░░▒▓███▀▒░ ▒░▒░▒░   ▒ ░░   
░ ░▒  ░  ░▒ ░     ░ ░   ▒▒   ▒ ░ ░  ░ ░░   ░ ▒░      ░    ░    ░         ░    ░    ░ ░ ░ ░ ▒    ░      
░  ░  ░  ░░         ░   ▒    ░   ░     ░   ░ ░       ░    ░    ░         ░    ░          ░ ░"""
    )
    print("https://github.com/spawn9859/Spawn-Aim")
    pr_yellow("\nMake sure your game is in the center of your screen!")

    # Load launcher settings from JSON file
    with open(
        os.path.join(
            SCRIPT_DIR,
            "configuration",
            f"{argv['settingsProfile'].lower()}.json",
        ),
        "r",
    ) as f:
        launcher_settings = json.load(f)

    # Detect available serial ports
    ports = []
    try:
        import serial.tools.list_ports
        ports = [port[0] for port in serial.tools.list_ports.comports()]
    except:
        ports = ["COM1"]  # Default if serial not available
        
    default_port = "COM1"  # Default if no Arduino port found
    for port in ports:
        if "Arduino" in port:
            default_port = port
            break

    # Initialize Pygame and controller
    controller = initialize_pygame_and_controller()

    # Get activation and quit key codes
    activation_key = get_keycode(launcher_settings["activationKey"])
    quit_key = get_keycode(launcher_settings["quitKey"])

    # Initialize settings from launcher settings
    config_manager.update_setting("auto_aim", True)
    config_manager.update_setting("trigger_bot", True if launcher_settings["autoFire"] else False)
    config_manager.update_setting("toggle", True if launcher_settings["toggleable"] else False)
    config_manager.update_setting("recoil", False)
    config_manager.update_setting("aim_shake", True if launcher_settings["aimShakey"] else False)
    config_manager.update_setting("overlay", False)
    config_manager.update_setting("preview", True if launcher_settings["visuals"] else False)
    config_manager.update_setting("mask_left", True if launcher_settings["maskLeft"] and launcher_settings["useMask"] else False)
    config_manager.update_setting("mask_right", True if not launcher_settings["maskLeft"] and launcher_settings["useMask"] else False)
    config_manager.update_setting("sensitivity", launcher_settings["movementAmp"] * 100)
    config_manager.update_setting("headshot", launcher_settings["headshotDistanceModifier"] * 100 if launcher_settings["headshotMode"] else 40)
    config_manager.update_setting("trigger_bot_distance", launcher_settings["autoFireActivationDistance"])
    config_manager.update_setting("confidence", launcher_settings["confidence"])
    config_manager.update_setting("recoil_strength", 0)
    config_manager.update_setting("aim_shake_strength", launcher_settings["aimShakeyStrength"])
    config_manager.update_setting("max_move", 100)
    config_manager.update_setting("height", launcher_settings["screenShotHeight"])
    config_manager.update_setting("width", launcher_settings["screenShotHeight"])
    config_manager.update_setting("mask_width", launcher_settings["maskWidth"])
    config_manager.update_setting("mask_height", launcher_settings["maskHeight"])
    config_manager.update_setting("yolo_version", f"v{argv['yoloVersion']}")
    config_manager.update_setting("yolo_model", "v5_Fortnite_taipeiuser")
    config_manager.update_setting("yolo_mode", "tensorrt")
    config_manager.update_setting("yolo_device", {1: "cpu", 2: "amd", 3: "nvidia"}.get(launcher_settings["onnxChoice"]))
    config_manager.update_setting("activation_key", activation_key)
    config_manager.update_setting("quit_key", quit_key)
    config_manager.update_setting("activation_key_string", launcher_settings["activationKey"])
    config_manager.update_setting("quit_key_string", launcher_settings["quitKey"])
    config_manager.update_setting("mouse_input", "default")
    config_manager.update_setting("arduino", default_port)
    config_manager.update_setting("fov_enabled", True if launcher_settings["fovToggle"] else False)
    config_manager.update_setting("fov_size", launcher_settings["fovSize"])

    # Create main window
    main_window = MainWindow(config_manager)

    # Initialize YOLOHandler
    yolo_handler = YOLOHandler(config_manager, MODELS_PATH)

    if config_manager.get_setting("overlay"):
        main_window.toggle_overlay()

    try:
        # Initialize screen capture
        monitor = initialize_game_window()

        main_loop(controller, main_window, yolo_handler, monitor)

    except Exception as e:
        pr_red(f"An error occurred: {str(e)}")
    finally:
        pr_green("Goodbye!")

if __name__ == "__main__":
   main(settingsProfile="config", yoloVersion=5, version=0)