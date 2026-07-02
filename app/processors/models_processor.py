import threading
import os
import subprocess as sp
import gc
import traceback
import multiprocessing
import re
import time
from typing import Dict, TYPE_CHECKING, Optional, Any
from PIL import Image
from packaging import version
import numpy as np

try:
    import onnxruntime
except ImportError as _ort_err:
    print("\n" + "=" * 70)
    print("[FATAL] onnxruntime failed to import.")
    print(f"  Error: {_ort_err}")
    print()
    print("  COMMON FIXES:")
    print("  1. Install Visual C++ Redistributable 2019 (x64) from Microsoft.")
    print("     Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("  2. Ensure CUDA 12.x runtime DLLs are present (cudart64_12.dll etc.).")
    print(
        "     Install CUDA Toolkit 12.x from https://developer.nvidia.com/cuda-downloads"
    )
    print("  3. On Windows 10 older than 1903: update Windows or install KB4571756.")
    print("  4. Portable install: run 'Check / Update Dependencies' in the Launcher.")
    print("=" * 70 + "\n")
    raise
import torch
import onnx
from torchvision.transforms import v2

from app.processors.utils import faceutil

# --- Optional Imports & Fallbacks ---

# KORNIA IMPORT
try:
    import kornia.color as K
except ImportError:
    K = None  # Fallback if Kornia is not installed
    print(
        "[WARN] Kornia library not found. Color space conversions will use power-law approximation."
    )

from PySide6 import QtCore

# TENSORRT IMPORT
try:
    import tensorrt as trt

    TENSORRT_AVAILABLE = True
except ModuleNotFoundError:
    print("[WARN] No TensorRT Found")
    TENSORRT_AVAILABLE = False
    trt = None

# --- Internal Project Imports ---

from app.processors.face_detectors import FaceDetectors
from app.processors.face_landmark_detectors import FaceLandmarkDetectors
from app.processors.face_masks import FaceMasks
from app.processors.face_restorers import FaceRestorers
from app.processors.face_swappers import FaceSwappers
from app.processors.frame_enhancers import FrameEnhancers
from app.processors.face_editors import FaceEditors
from app.processors.face_reaging import FaceReaging
from app.processors.perform_recast import PerformRecast
from app.processors.utils.dfm_model import DFMModel
from app.processors.models_data import (
    models_list,
    arcface_mapping_model_dict,
    fp16_safe_models_list,
    tensorrt_shape_infer_models,
)
from app.helpers.miscellaneous import is_file_exists
from app.helpers.downloader import download_file
from app.processors.utils.ref_ldm_kv_embedding import KVExtractor

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

# --- Global Configuration ---

onnxruntime.set_default_logger_severity(4)
onnxruntime.log_verbosity_level = -1

SRGB_GAMMA = (
    2.2  # More precise sRGB gamma handling is complex, this is an approximation
)


# --- Isolated Process Workers ---
# These functions run in a separate process to prevent fatal C++/CUDA
# crashes (like segmentation faults) from killing the main application.
def _probe_onnx_model_worker(
    model_path, providers_list, trt_options, session_options_dict
):
    """
    Worker function to be run in an isolated process to "warm up"
    an ONNX model, especially for the TensorRT provider.
    This triggers the engine cache build without freezing the main thread.
    """
    # Move all imports to top of function so sys.exit(1) is always available
    import os
    import sys
    import traceback
    import onnxruntime
    import torch

    try:
        # Create the SessionOptions object *inside* the worker process.
        session_options = onnxruntime.SessionOptions()
        if session_options_dict:
            for key, value in session_options_dict.items():
                # Use setattr to configure the SessionOptions object
                setattr(session_options, key, value)

        # Set the CUDA device to match the TRT provider's device_id
        gpu_id = trt_options.get("device_id", 0)
        if gpu_id != 0 and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        # Reconstruct the providers tuple
        providers = []
        for p in providers_list:
            name = p[0] if isinstance(p, tuple) else p
            if name == "TensorrtExecutionProvider":
                providers.append((name, trt_options))
            elif isinstance(p, tuple) and len(p) > 1:
                providers.append(p)
            else:
                providers.append(name)

        print(f"[ONNX Prober]: Attempting to load {os.path.basename(model_path)}...")
        # This line is the one that triggers the build/cache generation
        session = onnxruntime.InferenceSession(
            model_path, sess_options=session_options, providers=providers
        )

        # Force this prober process to wait until all CUDA operations
        # (i.e., the engine build and serialization to disk)
        # are *fully* complete before this process exits.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # If we get here, the load and the synchronization worked.
        del session
        print("[ONNX Prober]: Load successful. TRT engine cache built and flushed.")
        sys.exit(0)  # Success
    except Exception:
        print("[ONNX Prober]: ERROR during model load probe.")
        traceback.print_exc()
        sys.exit(1)  # Failure


def gamma_encode_linear_rgb_to_srgb(linear_rgb: torch.Tensor, gamma=SRGB_GAMMA):
    """Converts linear RGB to sRGB. Uses Kornia if available for better accuracy."""
    if K is not None:
        # Kornia expects input in range [0, 1] and handles tensor dimensions correctly.
        return K.linear_rgb_to_srgb(linear_rgb.clamp(0.0, 1.0))
    else:
        # Fallback to the original power-law approximation
        return torch.pow(linear_rgb.clamp(0.0, 1.0), 1.0 / gamma)


def gamma_decode_srgb_to_linear_rgb(srgb: torch.Tensor, gamma=SRGB_GAMMA):
    """Converts sRGB to linear RGB. Uses Kornia if available for better accuracy."""
    if K is not None:
        # Kornia expects input in range [0, 1]
        return K.srgb_to_linear_rgb(srgb.clamp(0.0, 1.0))
    else:
        # Fallback to the original power-law approximation
        return torch.pow(srgb.clamp(0.0, 1.0), gamma)


class ModelsProcessor(QtCore.QObject):
    """
    Central hub for managing AI models (ONNX, TensorRT, PyTorch).
    Handles:
    - Model Loading/Unloading (Thread-safe)
    - TensorRT Engine compilation and caching
    - Inference wrapper methods for various tasks (detection, swapping, restoration)
    - GPU memory management
    """

    processing_complete = QtCore.Signal()
    model_loaded = QtCore.Signal()  # Signal emitted with Onnx InferenceSession

    # Signal to request the GUI thread to show the build dialog
    # Arguments: (str: window_title, str: label_text)
    show_build_dialog = QtCore.Signal(str, str)
    # Signal to request the GUI thread to hide the build dialog
    hide_build_dialog = QtCore.Signal()

    def __init__(self, main_window: "MainWindow", device="cuda"):
        """
        Initialises the ModelsProcessor.

        Sets up all model dictionaries, TensorRT options, provider lists, sub-processors
        (face detectors, masks, restorers, etc.), and helper state (locks, sync vectors).

        Args:
            main_window: The application's MainWindow, used to access UI controls and signals.
            device: Torch/ONNX device string — ``"cuda"`` or ``"cpu"``.
        """
        super().__init__()
        self.main_window = main_window
        self.gpu_id = getattr(main_window, "gpu_id", 0)
        self.K = K  # Assign the module-level K to an instance attribute
        self.provider_name = "TensorRT"
        # NOTE: internal_deep_copied_kv_map / internal_kv_map_source_filename were
        # placeholder attributes for a planned per-session KV-map cache.  They are
        # currently unused (never written after __init__).  If a future feature
        # populates them, ensure a matching cleanup path is added to the force-unload
        # path (delete_models_dfm / force_unload path) so the tensors are freed.
        self.internal_deep_copied_kv_map: Dict[str, Dict[str, torch.Tensor]] | None = (
            None
        )
        self.internal_kv_map_source_filename: str | None = None
        self.kv_extractor: Optional[KVExtractor] = None
        self.kv_extraction_lock = threading.Lock()
        self.device = f"{device}:{self.gpu_id}" if device != "cpu" else device
        self.device_type = device
        if self.gpu_id != 0 and device != "cpu":
            torch.cuda.set_device(self.gpu_id)
        self.model_lock = threading.RLock()  # Reentrant lock for model access

        self.cuda_graph_capture_lock = threading.Lock()

        # --- TENSORRT WORKSPACE ---
        MIN_WORKSPACE_SIZE = 1073741824  # 1 GB
        FALLBACK_WORKSPACE_SIZE = 4294967296  # 4 GB

        try:
            # Get total GPU memory in bytes
            total_vram = torch.cuda.get_device_properties(self.gpu_id).total_memory

            # Safely allocate 40% of total VRAM for TensorRT workspace
            calculated_workspace = int(total_vram * 0.40)

            # Enforce a minimum of 1 GB to avoid compilation failures on very low-end GPUs
            workspace_size = max(calculated_workspace, MIN_WORKSPACE_SIZE)
        except Exception:
            # Fallback to 4GB if PyTorch fails to detect the GPU
            workspace_size = FALLBACK_WORKSPACE_SIZE

        # Default TensorRT options
        self.trt_ep_options = {
            "device_id": self.gpu_id,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "tensorrt-engines",
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": "tensorrt-engines",
            "trt_dump_ep_context_model": True,
            "trt_ep_context_file_path": "tensorrt-engines",
            "trt_layer_norm_fp32_fallback": True,
            "trt_max_workspace_size": workspace_size,
            "trt_builder_optimization_level": 5,
        }
        # A set to keep track of models that have been loaded but
        # have not had their engine built (lazy build).
        self.models_pending_build: set = set()
        self.providers = [
            ("TensorrtExecutionProvider", self.trt_ep_options),
            ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
            ("CPUExecutionProvider"),
        ]
        self.syncvec = torch.empty((1, 1), dtype=torch.float32, device=self.device)
        self.nThreads = 1

        # Initialize models and models_path dictionaries
        self.models: Dict[str, onnxruntime.InferenceSession] = {}
        self.models_path = {}
        self.models_data = {}

        for model_data in models_list:
            model_name, model_path = model_data["model_name"], model_data["local_path"]
            self.models[model_name] = None  # Model Instance placeholder
            self.models_path[model_name] = model_path
            self.models_data[model_name] = {
                "local_path": model_data["local_path"],
                "hash": model_data["hash"],
                "url": model_data.get("url"),
            }

        self.dfm_models: Dict[str, DFMModel] = {}
        self.dfm_inference_lock = threading.Lock()
        self.force_unload_in_progress = False

        # --- SMART UNLOAD STATE ---
        self.deferred_unloads: Dict[str, Dict[str, Any]] = {}

        # Initialize Sub-Processors
        self.face_detectors = FaceDetectors(self)
        self.face_landmark_detectors = FaceLandmarkDetectors(self)
        self.face_masks = FaceMasks(self)
        self.face_restorers = FaceRestorers(self)
        self.face_swappers = FaceSwappers(self)
        self.frame_enhancers = FrameEnhancers(self)
        self.face_editors = FaceEditors(self)
        self.face_reaging = FaceReaging(self)
        self.perform_recast = PerformRecast(self)

        # Initialize Mask Latent
        self.lp_mask_crop_latent = faceutil.create_faded_inner_mask(
            size=(64, 64),
            border_thickness=3,
            fade_thickness=8,
            blur_radius=3,
            device=self.device,
        )
        self.lp_mask_crop_latent = torch.unsqueeze(
            self.lp_mask_crop_latent, 0
        )  # Shape: [1, 64, 64]

        # Denoiser specific initializations (VR180 feature compatible)
        num_ddpm_timesteps = 1000
        linear_start_val = 0.0015
        linear_end_val = 0.0155
        self.betas_np = ModelsProcessor.make_beta_schedule(
            schedule="linear",
            n_timestep=num_ddpm_timesteps,
            linear_start=linear_start_val,
            linear_end=linear_end_val,
        )
        self.alphas_np = 1.0 - self.betas_np
        self.alphas_cumprod_np = np.cumprod(self.alphas_np, axis=0)
        self.alphas_cumprod_torch = (
            torch.from_numpy(self.alphas_cumprod_np).float().to(self.device)
        )
        # NOTE: vae_scale_factor=1.0 is intentional for this model's specific VAE configuration
        self.vae_scale_factor = 1.0

        # Cache for DDIM schedule tensors, keyed by (ddim_steps, ddim_eta).
        # Bounded LRU: each entry holds ~4 GPU tensors; at most 20 unique step/eta
        # combos are expected in practice (steps 1–100, eta 0.0 or 1.0).
        from collections import OrderedDict as _OD

        self._ddim_schedule_cache: _OD = _OD()
        self._DDIM_CACHE_MAX = 20

        self.clip_session: list = []

        # --- Face Analysis Constants (ArcFace/Landmarks) ---
        self.arcface_dst = np.array(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            dtype=np.float32,
        )
        self.FFHQ_kps = np.array(
            [
                [192.98138, 239.94708],
                [318.90277, 240.1936],
                [256.63416, 314.01935],
                [201.26117, 371.41043],
                [313.08905, 371.15118],
            ]
        )
        self.mean_lmk: list = []
        self.anchors: list = []
        self.emap: list = []
        self.LandmarksSubsetIdxs = [
            0,
            1,
            4,
            5,
            6,
            7,
            8,
            10,
            13,
            14,
            17,
            21,
            33,
            37,
            39,
            40,
            46,
            52,
            53,
            54,
            55,
            58,
            61,
            63,
            65,
            66,
            67,
            70,
            78,
            80,
            81,
            82,
            84,
            87,
            88,
            91,
            93,
            95,
            103,
            105,
            107,
            109,
            127,
            132,
            133,
            136,
            144,
            145,
            146,
            148,
            149,
            150,
            152,
            153,
            154,
            155,
            157,
            158,
            159,
            160,
            161,
            162,
            163,
            168,
            172,
            173,
            176,
            178,
            181,
            185,
            191,
            195,
            197,
            234,
            246,
            249,
            251,
            263,
            267,
            269,
            270,
            276,
            282,
            283,
            284,
            285,
            288,
            291,
            293,
            295,
            296,
            297,
            300,
            308,
            310,
            311,
            312,
            314,
            317,
            318,
            321,
            323,
            324,
            332,
            334,
            336,
            338,
            356,
            361,
            362,
            365,
            373,
            374,
            375,
            377,
            378,
            379,
            380,
            381,
            382,
            384,
            385,
            386,
            387,
            388,
            389,
            390,
            397,
            398,
            400,
            402,
            405,
            409,
            415,
            454,
            466,
            468,
            469,
            470,
            471,
            472,
            473,
            474,
            475,
            476,
            477,
        ]

        self.normalize = v2.Normalize(
            mean=[0.0, 0.0, 0.0], std=[1 / 1.0, 1 / 1.0, 1 / 1.0]
        )

        self.lp_mask_crop = self.face_editors.lp_mask_crop
        self.lp_lip_array = self.face_editors.lp_lip_array
        self.rgb_to_linear_rgb_converter = None
        self.linear_rgb_to_rgb_converter = None

    @property
    def binding_device_id(self) -> int:
        return self.gpu_id if self.device_type != "cpu" else 0

    def _ensure_trt_ready_onnx(self, model_name: str, onnx_path: str) -> str:
        """Return an ONNX path that the TensorRT EP can build an engine from.

        Some models (see ``tensorrt_shape_infer_models``) contain ops — notably
        5-D ``GridSample`` in the PerformRecast warping module — whose output
        tensors carry no static shape. The TensorRT EP refuses such graphs with
        "has no shape specified. Please run shape inference on the onnx model
        first." We fix this once by pinning the batch dimension to 1 (the app
        always feeds a single face) and running ONNX Runtime's symbolic shape
        inference, then caching the result next to the original as
        ``*.trtshape.onnx``. The cached file is reused unless the source ONNX is
        newer. For models not in the list, the original path is returned as-is.
        """
        if model_name not in tensorrt_shape_infer_models:
            return onnx_path
        if not onnx_path.lower().endswith(".onnx"):
            return onnx_path

        sidecar_path = onnx_path[: -len(".onnx")] + ".trtshape.onnx"
        try:
            if os.path.exists(sidecar_path) and (
                os.path.getmtime(sidecar_path) >= os.path.getmtime(onnx_path)
            ):
                return sidecar_path

            print(
                f"[INFO] Preparing TensorRT-ready (shape-inferred) ONNX for {model_name}..."
            )
            # This shape-inference pass can take a noticeable amount of time for
            # large graphs (the PerformRecast warping module is ~200 MB) and runs
            # *before* the engine-build probe, so without a dialog the UI looks
            # frozen with no indication of what is happening. Surface a dialog for
            # this preprocessing step too. It is only paid once (result cached).
            self.show_build_dialog.emit(
                "Preparing TensorRT Model",
                f"Running shape inference for:\n{model_name}\n\n"
                f"This one-time step prepares the model for the TensorRT engine "
                f"build and may take a moment.",
            )
            try:
                from onnxruntime.tools.onnx_model_utils import make_dim_param_fixed
                from onnxruntime.tools.symbolic_shape_infer import (
                    SymbolicShapeInference,
                )

                model = onnx.load(onnx_path)
                # Pin the dynamic 'batch' axis to 1 so symbolic dims (e.g.
                # "50*batch") resolve to concrete values the TensorRT builder
                # accepts.
                try:
                    make_dim_param_fixed(model.graph, "batch", 1)
                except Exception as dim_err:
                    # Not fatal — symbolic shape inference may still add shapes.
                    print(f"[WARN] Could not pin batch dim for {model_name}: {dim_err}")
                model = SymbolicShapeInference.infer_shapes(
                    model, auto_merge=True, guess_output_rank=True
                )
                onnx.save(model, sidecar_path)
                del model
                gc.collect()
            finally:
                self.hide_build_dialog.emit()
            print(f"[INFO] Wrote shape-inferred ONNX: {os.path.basename(sidecar_path)}")
            return sidecar_path
        except Exception as e:
            print(
                f"[WARN] Shape-inference preprocessing failed for {model_name} ({e}). "
                f"Falling back to the original ONNX."
            )
            traceback.print_exc()
            return onnx_path

    def _check_tensorrt_cache(self, model_name: str, onnx_path: str) -> bool:
        """
        Checks if a valid TensorRT cache (ctx and engine file) exists for the given model.
        Returns True if a valid cache is found, False otherwise.
        """
        try:
            cache_dir = "tensorrt-engines"
            base_onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]
            ctx_file_name = f"{base_onnx_name}_ctx.onnx"
            ctx_file_path = os.path.join(cache_dir, ctx_file_name)

            if os.path.exists(ctx_file_path):
                with open(ctx_file_path, "rb") as f:
                    content = f.read()

                # Look for the engine name embedded in the context file
                match = re.search(b"TensorrtExecutionProvider_.*?\\.engine", content)
                if not match:
                    return False

                engine_name = match.group(0).decode("utf-8")
                engine_subdirectory_name = os.path.basename(cache_dir)
                engine_file_path = os.path.join(
                    cache_dir, engine_subdirectory_name, engine_name
                )

                if os.path.exists(engine_file_path):
                    return True
                else:
                    return False
            else:
                return False

        except Exception as e:
            print(f"[ERROR] Failed TensorRT cache check: {e}")
            return False

    def _clean_tensorrt_cache(self, onnx_path: str, trt_options: dict) -> None:
        """
        Cleans up potentially corrupted TensorRT cache files for a specific model.
        Safely ignores missing files or locked files to prevent crashes during the cleanup process.

        Args:
            onnx_path (str): The local path to the ONNX model.
            trt_options (dict): The TensorRT options containing the dynamic cache path.
        """
        import os
        import re

        cache_dir = trt_options.get("trt_engine_cache_path", "tensorrt-engines")
        base_onnx_name = os.path.splitext(os.path.basename(onnx_path))[0]

        # 1. Try to read the context file to find the specific engine file before deleting it
        ctx_file_name = f"{base_onnx_name}_ctx.onnx"
        ctx_file_path = os.path.join(cache_dir, ctx_file_name)

        engine_file_paths_to_check = []
        if os.path.exists(ctx_file_path) and os.path.isfile(ctx_file_path):
            try:
                with open(ctx_file_path, "rb") as f:
                    content = f.read()

                # Extract the engine name generated by ONNX Runtime
                match = re.search(b"TensorrtExecutionProvider_.*?\\.engine", content)
                if match:
                    engine_name = match.group(0).decode("utf-8")

                    # Failsafe: ORT pathing behavior varies.
                    engine_subdirectory_name = os.path.basename(cache_dir)
                    engine_file_paths_to_check.extend(
                        [
                            os.path.join(cache_dir, engine_name),
                            os.path.join(
                                cache_dir, engine_subdirectory_name, engine_name
                            ),
                        ]
                    )
            except Exception as e:
                print(
                    f"[WARN] Could not read corrupted context file {ctx_file_path} to find engine name: {e}"
                )

        # 2. Delete the context file
        if os.path.exists(ctx_file_path) and os.path.isfile(ctx_file_path):
            try:
                os.remove(ctx_file_path)
                print(
                    f"[INFO] Deleted corrupted TensorRT context file: {ctx_file_path}"
                )
            except Exception as e:
                print(
                    f"[WARN] Failed to delete {ctx_file_path} (it might be locked or missing): {e}"
                )

        # 3. Delete the engine file(s) if we found them
        for engine_path in engine_file_paths_to_check:
            if (
                engine_path
                and os.path.exists(engine_path)
                and os.path.isfile(engine_path)
            ):
                try:
                    os.remove(engine_path)
                    print(
                        f"[INFO] Deleted corrupted TensorRT engine file: {engine_path}"
                    )
                except Exception as e:
                    print(f"[WARN] Failed to delete engine file {engine_path}: {e}")

        # 4. Delete any associated timing cache, profile files, or general cache files
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            try:
                for file_name in os.listdir(cache_dir):
                    # Catch model-specific files (e.g., SomeModel.profile)
                    is_model_specific = file_name.startswith(base_onnx_name) and (
                        file_name.endswith(".profile")
                        or file_name.endswith(".cache")
                        or file_name.endswith(".timing")
                    )

                    # Catch exact generic names (like DFM's "timing.cache")
                    is_generic_timing = file_name == "timing.cache"

                    # Catch ORT's global architecture-based timing caches
                    # Example: TensorrtExecutionProvider_cache_sm120.timing
                    is_ort_global_timing = file_name.startswith(
                        "TensorrtExecutionProvider_"
                    ) and (
                        file_name.endswith(".timing") or file_name.endswith(".profile")
                    )

                    if is_model_specific or is_generic_timing or is_ort_global_timing:
                        target_path = os.path.join(cache_dir, file_name)
                        if os.path.isfile(target_path):
                            try:
                                os.remove(target_path)
                                print(
                                    f"[INFO] Deleted TensorRT auxiliary/timing file: {target_path}"
                                )
                            except Exception as e:
                                print(
                                    f"[WARN] Failed to delete auxiliary file {target_path}: {e}"
                                )
            except Exception as e:
                print(
                    f"[WARN] Failed to clean profile/timing/cache files in {cache_dir}: {e}"
                )

    def load_model(self, model_name, session_options=None):
        """
        Loads an AI model (ONNX) with thread safety.
        Handles checking for existing TensorRT caches and launching the build probe if needed.
        """
        with self.model_lock:
            if self.models.get(model_name):
                return self.models[model_name]

            model_instance = None
            onnx_path = self.models_path.get(model_name)
            if not onnx_path:
                print(
                    f"[ERROR] Model path for '{model_name}' not found in models_data."
                )
                return None

            # Some models need a shape-inferred graph before the TensorRT EP can
            # build an engine. This transparently swaps in a cached sidecar; the
            # original path stays untouched for download/integrity checks.
            onnx_path = self._ensure_trt_ready_onnx(model_name, onnx_path)

            build_was_triggered = (
                False  # MP-05: flag to track if build dialog was shown
            )

            # --- DYNAMIC PRECISION CONFIGURATION (WHITELIST FP16) ---
            model_trt_options = dict(self.trt_ep_options)

            # Check if the model is explicitly marked as safe for FP16 in models_data.py
            if model_name in fp16_safe_models_list:
                model_trt_options["trt_fp16_enable"] = True
                print(f"[INFO] FP16 Acceleration ENABLED for {model_name}")
            else:
                model_trt_options["trt_fp16_enable"] = False

            # Reconstruct the providers with model-specific options
            model_providers = []
            for p in self.providers:
                if isinstance(p, tuple) and p[0] == "TensorrtExecutionProvider":
                    model_providers.append(
                        ("TensorrtExecutionProvider", model_trt_options)
                    )
                elif p == "TensorrtExecutionProvider":
                    model_providers.append(
                        ("TensorrtExecutionProvider", model_trt_options)
                    )
                else:
                    model_providers.append(p)

            is_tensorrt_load = any(
                (p[0] if isinstance(p, tuple) else p) == "TensorrtExecutionProvider"
                for p in model_providers
            )

            if onnx_path.lower().endswith(".onnx"):
                # Only run the isolated probe if TensorRT is the target provider
                if is_tensorrt_load:
                    # Check if engine config file exists...
                    cache_is_valid = self._check_tensorrt_cache(model_name, onnx_path)

                    # If no engine config file or cache file exists run the probe
                    if not cache_is_valid:
                        print(
                            f"[INFO] TensorRT load detected for {model_name}. Running isolated probe..."
                        )

                        try:
                            # We emit signals to ask the main GUI thread to show the dialog.
                            dialog_title = "Building TensorRT Cache"
                            dialog_text = (
                                f"Building TensorRT engine cache for:\n"
                                f"{os.path.basename(onnx_path)}\n\n"
                                f"This may take several minutes.\n"
                                f"The application will continue once finished."
                            )

                            # The trt engine build worker process use this SessionOptions
                            # to use only 1 thread for building engines
                            sess_options_dict = {"intra_op_num_threads": 1}

                            # Ask the main thread to show the dialog
                            self.show_build_dialog.emit(dialog_title, dialog_text)

                            probe_successful = False
                            last_exit_code = None
                            max_retries = 3

                            for attempt in range(max_retries):
                                print(
                                    f"[INFO] Probe attempt {attempt + 1} of {max_retries} for {model_name}..."
                                )

                                # Use 'spawn' context for CUDA/TRT safety
                                ctx = multiprocessing.get_context("spawn")
                                # Pass full providers list (with tuples) so the worker
                                # can reconstruct them with device_id options.
                                current_providers_list = list(model_providers)
                                probe_process = ctx.Process(
                                    target=_probe_onnx_model_worker,
                                    args=(
                                        onnx_path,
                                        current_providers_list,
                                        model_trt_options,  # On passe les options dynamiques
                                        sess_options_dict,
                                    ),
                                )

                                try:
                                    probe_process.start()
                                    build_was_triggered = True

                                    # Force timeout at 15Min if the compilator silently crashed, enabling app to stay alive
                                    probe_process.join(timeout=900)

                                    if probe_process.is_alive():
                                        print(
                                            f"[ERROR] Probe process for {model_name} timed out! Terminating."
                                        )
                                        probe_process.terminate()
                                        probe_process.join()

                                        # Clean up corrupted caches caused by the timeout before raising
                                        print(
                                            f"[INFO] Cleaning up corrupted TensorRT cache for {model_name} due to timeout..."
                                        )
                                        self._clean_tensorrt_cache(
                                            onnx_path, model_trt_options
                                        )

                                        raise RuntimeError(
                                            "TensorRT Engine build timed out."
                                        )
                                except Exception as e:
                                    print(f"[ERROR] Process execution failed: {e}")

                                # Process finished, get exit code
                                exitcode = probe_process.exitcode
                                last_exit_code = exitcode

                                if exitcode == 0:
                                    print(
                                        f"[INFO] Probe successful for {model_name}. Cache should be built."
                                    )
                                    probe_successful = True
                                    break  # Exit the retry loop on success
                                else:
                                    print(
                                        f"[WARN] Probe attempt {attempt + 1} failed with exit code {exitcode}."
                                    )

                                    # Wipe corrupted artifacts before attempting the next retry
                                    print(
                                        f"[INFO] Cleaning up potentially corrupted TensorRT cache for {model_name}..."
                                    )
                                    self._clean_tensorrt_cache(
                                        onnx_path, model_trt_options
                                    )

                                    if attempt < max_retries - 1:
                                        print("[INFO] Retrying in 2 seconds...")
                                        time.sleep(2.0)

                            if not probe_successful:
                                raise RuntimeError(
                                    f"[ERROR] ONNX/TensorRT probe process failed after {max_retries} attempts. Last exit code: {last_exit_code}"
                                )

                        except Exception:
                            # MP-05: only emit hide_build_dialog when build was triggered
                            if build_was_triggered:
                                self.hide_build_dialog.emit()

                            print(f"[ERROR] Isolated probe failed for {model_name}.")
                            print(
                                "[ERROR] The model will not be loaded. This is likely a fatal TensorRT/CUDA error."
                            )
                            traceback.print_exc()
                            self.models[model_name] = (
                                None  # Ensure it's marked as not loaded
                            )
                            return None  # Abort the load

            # Now, proceed with the *actual* load in the main thread.
            try:
                # MP-01: Double-checked load after re-acquiring the lock.
                # Another thread may have loaded this model while we were in the probe.
                if self.models.get(model_name):
                    print(
                        f"[INFO] Skipped loading: {model_name} is already loaded in memory (post-probe check)."
                    )
                    return self.models.get(model_name)

                if session_options is None:
                    session_options = onnxruntime.SessionOptions()

                # Force log_severity_level to 3 (ERROR) for the actual load as well, to suppress non-critical warnings from ONNX Runtime that can clutter the console.
                session_options.log_severity_level = 3

                model_instance = onnxruntime.InferenceSession(
                    onnx_path,
                    sess_options=session_options,
                    providers=model_providers,
                )

                # This ensures the CUDA context is synchronized after a new TRT
                # engine build, before we try to load it.
                if build_was_triggered:
                    if torch.cuda.is_available():
                        # Only synchronise current stream
                        torch.cuda.current_stream().synchronize()

                    # Check cache AGAIN.
                    # If the probe succeeded BUT the cache STILL doesn't exist,
                    # it's a "Lazy Build" model.
                    if not self._check_tensorrt_cache(model_name, onnx_path):
                        print(
                            f"[INFO] Model {model_name} requires a lazy build (engine not found after probe)."
                        )
                        self.models_pending_build.add(model_name)

                self.models[model_name] = model_instance
                print(
                    f"[INFO] Loading model: {model_name} with provider: {self.provider_name}"
                )
                if model_name == "Inswapper128":
                    graph = onnx.load(self.models_path[model_name]).graph
                    emap_initializer = None
                    for initializer in graph.initializer:
                        if initializer.name == "emap":
                            emap_initializer = initializer
                            break

                    if emap_initializer:
                        self.emap = onnx.numpy_helper.to_array(emap_initializer)
                    else:
                        self.emap = onnx.numpy_helper.to_array(graph.initializer[-1])
                    # MP-17: release large ONNX graph object after emap extraction
                    del graph
                    gc.collect()
                return model_instance

            except Exception:
                # This catch is still valuable for non-fatal errors
                print(f"[ERROR] Failed to load model {model_name} (even after probe).")
                traceback.print_exc()
                if model_instance is not None:
                    del model_instance
                    gc.collect()
                self.models[model_name] = None
                return None

            finally:
                # MP-05: Only emit hide_build_dialog when a build was triggered.
                if build_was_triggered:
                    self.hide_build_dialog.emit()

    def check_and_clear_pending_build(self, model_name: str) -> bool:
        """
        Checks if a model is pending its first-run lazy build.
        If it is, it clears the flag and returns True.
        """
        with self.model_lock:
            if model_name in self.models_pending_build:
                print(
                    f"[INFO] Model '{model_name}' is triggering its first-run lazy build."
                )
                # MP-08: use discard for atomic, safe removal (no KeyError)
                self.models_pending_build.discard(model_name)
                return True
        return False

    def load_dfm_model(self, dfm_model):
        """Loads a DeepFaceLab model instance."""
        with self.model_lock:
            if self.dfm_models.get(dfm_model):
                return self.dfm_models[dfm_model]

            self.main_window.model_loading_signal.emit()
            try:
                max_models_to_keep = self.main_window.control["MaxDFMModelsSlider"]
                total_loaded_models = len(self.dfm_models)
                # Ensure max_models_to_keep > 0 to avoid evicting when set to 0 (unlimited)
                if total_loaded_models >= max_models_to_keep and max_models_to_keep > 0:
                    print("[INFO] Clearing DFM Model (max capacity reached)")
                    model_name, model_instance = list(self.dfm_models.items())[0]
                    del model_instance
                    self.dfm_models.pop(model_name)
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # --- Isolate TensorRT cache and bypass DFM internal garbage names ---
                import copy
                import re
                import os

                dfm_providers = copy.deepcopy(self.providers)

                if (
                    dfm_providers
                    and isinstance(dfm_providers[0], tuple)
                    and dfm_providers[0][0] == "TensorrtExecutionProvider"
                ):
                    trt_options = dict(dfm_providers[0][1])

                    # 1. Clean the filename to create a safe string
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "", dfm_model)
                    if safe_name.lower().endswith(".dfm"):
                        safe_name = safe_name[:-4]

                    # 2. Use absolute paths for the dedicated cache directory to avoid OS pathing bugs
                    cache_base = os.path.abspath("tensorrt-engines")
                    dedicated_cache_dir = os.path.join(
                        cache_base, "dfm_caches", safe_name
                    )
                    os.makedirs(dedicated_cache_dir, exist_ok=True)

                    # 3. Route cache paths
                    trt_options["trt_engine_cache_path"] = dedicated_cache_dir
                    trt_options["trt_timing_cache_path"] = os.path.join(
                        dedicated_cache_dir, "timing.cache"
                    )

                    # 4. Override the internal ONNX model name.
                    trt_options["trt_engine_cache_prefix"] = safe_name

                    # 5. Disable Context dumping for DFM.
                    trt_options["trt_dump_ep_context_model"] = False
                    if "trt_ep_context_file_path" in trt_options:
                        del trt_options["trt_ep_context_file_path"]

                    dfm_providers[0] = ("TensorrtExecutionProvider", trt_options)

                self.dfm_models[dfm_model] = DFMModel(
                    self.main_window.dfm_model_manager.get_models_data()[dfm_model],
                    dfm_providers,
                    self.device,
                    self.gpu_id,
                )
            except Exception:
                print(f"[ERROR] Failed to load DFM model {dfm_model}.")
                traceback.print_exc()
                self.dfm_models[dfm_model] = None
            finally:
                self.main_window.model_loaded_signal.emit()

            return self.dfm_models.get(dfm_model)

    def delete_models(self):
        """Unloads all ONNX models."""
        model_names_to_unload = list(self.models.keys())
        for model_name in model_names_to_unload:
            self.unload_model(model_name)
        self.clip_session = []

    def delete_models_dfm(self):
        """Unloads all DFM models."""
        model_names_to_unload = list(self.dfm_models.keys())
        for model_name in model_names_to_unload:
            self.unload_dfm_model(model_name)

    def unload_dfm_model(self, model_name_to_unload, force_immediate=False):
        """
        Unloads a single DFM model instance from memory.

        Respects the KeepModelsAliveToggle control unless a force-unload is in progress.
        Frees the Python object, runs gc.collect(), and clears the CUDA cache.
        """
        # Check if unloading should be skipped
        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return  # Skip unloading

        # --- SMART UNLOAD: Intercept if video is playing ---
        if not force_immediate and not self.force_unload_in_progress:
            vp = getattr(self.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                # Video is playing, get the feeder's current frame and defer
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.model_lock:
                    self.deferred_unloads[model_name_to_unload] = {
                        "type": "dfm",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring DFM '{model_name_to_unload}' unload after frame {target_frame}"
                )
                return

        with self.model_lock:
            if (
                model_name_to_unload
                and model_name_to_unload in self.dfm_models
                and self.dfm_models.get(model_name_to_unload) is not None
            ):
                print(f"[INFO] Unloading DFM model: {model_name_to_unload}")
                model_instance = self.dfm_models.pop(model_name_to_unload, None)
                if model_instance:
                    del model_instance
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def unload_model(self, model_name_to_unload, force_immediate=False):
        """
        Unloads a single ONNX model from memory.

        Handles the ``self.models`` (ONNX) dictionary. Respects the KeepModelsAliveToggle
        control unless a force-unload is in progress. Frees the Python object, runs
        gc.collect(), and clears the CUDA cache when something was actually unloaded.
        """
        # Check if unloading should be skipped
        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return  # Skip unloading

        # --- SMART UNLOAD: Intercept if video is playing ---
        if not force_immediate and not self.force_unload_in_progress:
            vp = getattr(self.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                # Video is playing, get the feeder's current frame and defer
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.model_lock:
                    self.deferred_unloads[model_name_to_unload] = {
                        "type": "onnx",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring ONNX '{model_name_to_unload}' unload after frame {target_frame}"
                )
                return

        with self.model_lock:
            unloaded = False

            # Handle ONNX models (for CUDA, CPU, and TensorRT providers)
            if model_name_to_unload and model_name_to_unload in self.models:
                model_instance = self.models[model_name_to_unload]

                if model_instance is not None:
                    print(f"[INFO] Unloading ONNX model: {model_name_to_unload}")
                    # MP-06: set dict entry to None first, then del the instance
                    self.models[model_name_to_unload] = None
                    # Explicitly delete the object to trigger its __del__ method
                    del model_instance
                    unloaded = True
                else:
                    self.models[model_name_to_unload] = None

            if unloaded:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def is_model_active_in_ui(self, model_name: str) -> bool:
        """
        Live Verification JIT: Checks the UI state dynamically to see if a model is still needed.
        Used exclusively before a deferred unload to prevent accidental purging.
        """
        from app.ui.widgets.models_toggle_data import MODELS_TOGGLE_MAP

        # Thread Safety: Use dict() to safely create shallow copies and avoid 'dictionary changed size' errors
        try:
            params = dict(getattr(self.main_window, "parameters", {}))
            ctrl = dict(getattr(self.main_window, "control", {}))
        except Exception:
            params = getattr(self.main_window, "parameters", {})
            ctrl = getattr(self.main_window, "control", {})

        # Create a unified flat dictionary for global lookups
        live_global = (
            {**params, **ctrl}
            if isinstance(params, dict) and isinstance(ctrl, dict)
            else params
        )

        # 1. SPECIAL CASE: FACE RESTORERS
        if hasattr(self, "face_restorers") and hasattr(
            self.face_restorers, "model_map"
        ):
            expected_combo = None
            for combo_str, ort_name in self.face_restorers.model_map.items():
                if ort_name == model_name:
                    expected_combo = combo_str
                    break

            if expected_combo is not None:

                def is_restorer_requested(p) -> bool:
                    if not hasattr(p, "get"):
                        return False

                    if (
                        p.get("FaceRestorerEnableToggle", False)
                        and p.get("FaceRestorerTypeSelection") == expected_combo
                    ):
                        return True
                    if p.get("FaceRestorerEnable2Toggle", False) and not p.get(
                        "FaceRestorerEnable2EndToggle", False
                    ):
                        if p.get("FaceRestorerType2Selection") == expected_combo:
                            return True
                    if (
                        p.get("FaceRestorerEnable2EndToggle", False)
                        and p.get("FaceRestorerType2Selection") == expected_combo
                    ):
                        return True
                    return False

                # 1. Global check
                if is_restorer_requested(live_global):
                    return True

                # 2. Exhaustive verification for each active face in memory
                if hasattr(params, "items"):
                    for face_id, face_params in params.items():
                        if is_restorer_requested(face_params):
                            return True

                return False  # No face requested this restorer, we can safely unload it

        # 2. GENERAL CASE: MODELS TOGGLE MAP
        toggles = MODELS_TOGGLE_MAP.get(model_name)
        if not toggles:
            return True  # Core models without UI toggles are always assumed needed

        for toggle in toggles:
            target_key = toggle.key

            if hasattr(ctrl, "get") and ctrl.get(target_key, False):
                return True

            if hasattr(params, "get") and params.get(target_key, False):
                return True

            if hasattr(params, "items"):
                for face_id, face_params in params.items():
                    if hasattr(face_params, "get") and face_params.get(
                        target_key, False
                    ):
                        return True

        return False

    def check_deferred_unloads(self, current_displayed_frame: int):
        """
        Checks if any pending model unloads have reached their failsafe trigger.
        Includes a Just-In-Time (JIT) UI state verification.
        """
        if not self.deferred_unloads:
            return

        with self.model_lock:
            to_unload = []
            for model_name, data in self.deferred_unloads.items():
                if current_displayed_frame >= data["target_frame"]:
                    to_unload.append((model_name, data["type"]))

            for model_name, m_type in to_unload:
                # Remove from pending list regardless
                del self.deferred_unloads[model_name]

                # --- JIT LIVE CHECK ---
                if self.is_model_active_in_ui(model_name):
                    print(
                        f"[INFO] Smart Unload JIT: Option re-enabled for '{model_name}'. Cancelling unload."
                    )
                    continue  # Safe! We skip the unload.

                print(
                    f"[INFO] Smart Unload: Trigger reached. Actually releasing '{model_name}'."
                )
                if m_type == "onnx":
                    self.unload_model(model_name, force_immediate=True)
                elif m_type == "dfm":
                    self.unload_dfm_model(model_name, force_immediate=True)
                elif m_type == "kv":
                    self.unload_kv_extractor(force_immediate=True)

    def execute_all_deferred_unloads(self):
        """
        Forces the immediate unload of all deferred models.
        Ideal for the 'Stop' function (Smart Stop).
        Includes JIT verification to keep re-enabled models.
        """
        with self.model_lock:
            if not self.deferred_unloads:
                return

            print("[INFO] Smart Stop: Executing pending UI unloads...")
            for model_name, data in list(self.deferred_unloads.items()):
                del self.deferred_unloads[model_name]

                # --- LIVE UI VERIFICATION (JIT) ---
                if self.is_model_active_in_ui(model_name):
                    print(
                        f"[INFO] Smart Stop JIT: Option is active for '{model_name}'. Keeping model in VRAM."
                    )
                    continue  # We skip the unload, leaving the model ready for the next run!

                print(f"[INFO] Smart Stop: Releasing '{model_name}'.")
                if data["type"] == "onnx":
                    self.unload_model(model_name, force_immediate=True)
                elif data["type"] == "dfm":
                    self.unload_dfm_model(model_name, force_immediate=True)
                elif data["type"] == "kv":
                    self.unload_kv_extractor(force_immediate=True)

    def showModelLoadingProgressBar(self):
        """Shows the model-loading progress dialog in the UI."""
        self.main_window.model_load_dialog.show()

    def hideModelLoadProgressBar(self):
        """Closes the model-loading progress dialog if it is open."""
        if self.main_window.model_load_dialog:
            self.main_window.model_load_dialog.close()

    def switch_providers_priority(self, provider_name):
        """
        Reconfigures the ONNX Runtime provider list and the active device.

        Supported values for *provider_name*: ``"TensorRT"``, ``"TensorRT-Engine"``,
        ``"CUDA"``, ``"CPU"``.  Raises ``RuntimeError`` if TensorRT is requested but
        not installed, and ``ValueError`` for any unknown provider name.

        Returns:
            str: The resolved provider name (may differ from the input when TensorRT
                 is downgraded due to a version constraint).
        """
        # Release existing ORT sessions whenever the provider is changed so that
        # GPU memory from the old provider is freed before new sessions are allocated.
        self.face_detectors.unload_models()
        self.face_masks.unload_models()
        self.face_swappers.unload_models()
        self.face_landmark_detectors.unload_models()
        self.face_restorers.unload_models()

        match provider_name:
            case "TensorRT" | "TensorRT-Engine":
                # MP-04: guard against TensorRT not being installed
                if not TENSORRT_AVAILABLE or trt is None:
                    raise RuntimeError("TensorRT is not installed.")
                providers = [
                    ("TensorrtExecutionProvider", self.trt_ep_options),
                    ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
                    ("CPUExecutionProvider"),
                ]
                self.device = f"cuda:{self.gpu_id}"
                self.device_type = "cuda"
                if (
                    version.parse(trt.__version__) < version.parse("10.2.0")
                    and provider_name == "TensorRT-Engine"
                ):
                    print(
                        "[WARN] TensorRT-Engine provider cannot be used when TensorRT version is lower than 10.2.0."
                    )
                    provider_name = "TensorRT"

            case "CPU":
                providers = [("CPUExecutionProvider")]
                self.device = "cpu"
                self.device_type = "cpu"
            case "CUDA":
                providers = [
                    ("CUDAExecutionProvider", {"device_id": self.gpu_id}),
                    ("CPUExecutionProvider"),
                ]
                self.device = f"cuda:{self.gpu_id}"
                self.device_type = "cuda"
            case _:
                # MP-22: raise on unknown provider name
                raise ValueError(f"Unknown provider: {provider_name}")

        self.providers = providers
        self.provider_name = provider_name
        self.lp_mask_crop = self.lp_mask_crop.to(self.device)
        # Also move auxiliary tensors that are used alongside lp_mask_crop so
        # they remain on the same device and do not cause device-mismatch errors.
        self.lp_mask_crop_latent = self.lp_mask_crop_latent.to(self.device)
        self.alphas_cumprod_torch = self.alphas_cumprod_torch.to(self.device)

        return self.provider_name

    def set_number_of_threads(self, value):
        """Sets the ONNX thread count. TRT engine reloading is no longer needed here."""
        self.nThreads = value

    def get_gpu_memory(self):
        """
        Returns GPU memory usage as ``(used_MB, total_MB)``.

        Queries nvidia-smi for accuracy; falls back to ``torch.cuda`` device properties
        if nvidia-smi is unavailable.  Returns ``(0, 0)`` when no GPU is detected.
        """
        # MP-13: use a single nvidia-smi call for both total and free memory
        try:
            command = f"nvidia-smi --id={self.gpu_id} --query-gpu=memory.total,memory.free --format=csv,noheader,nounits"
            output = sp.check_output(command.split()).decode("ascii").strip()
            # Output format: "total, free" (one line per GPU)
            first_line = output.split("\n")[0]
            parts = first_line.split(",")
            memory_total_val = int(parts[0].strip())
            memory_free_val = int(parts[1].strip())
            memory_used = memory_total_val - memory_free_val
            return memory_used, memory_total_val
        except Exception:
            # Fallback to torch.cuda if nvidia-smi is unavailable
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(self.gpu_id)
                memory_total_val = props.total_memory // (1024 * 1024)
                memory_free_val = (
                    props.total_memory - torch.cuda.memory_reserved(self.gpu_id)
                ) // (1024 * 1024)
                memory_used = memory_total_val - memory_free_val
                return memory_used, memory_total_val
            return 0, 0

    def clear_gpu_memory(self):
        """
        Force-unloads every loaded model (ONNX, DFM, KV Extractor, CLIP) and
        releases all GPU memory.

        Bypasses the KeepModelsAliveToggle by temporarily setting
        ``force_unload_in_progress = True``.  Stops any active video processing first
        to ensure no worker threads are using the models during unload.
        """
        print("[INFO] Clearing GPU Memory: Unloading all models...")
        self.main_window.video_processor.stop_processing()  # Ensure no workers are active

        # Set the force_unload flag to bypass the 'KeepModelsAlive' check
        self.force_unload_in_progress = True
        try:
            # Explicitly call unloaders for each category
            self.face_detectors.unload_models()
            self.face_landmark_detectors.unload_models()
            self.face_masks.unload_models()
            self.face_restorers.unload_models()
            self.face_swappers.unload_models()
            self.frame_enhancers.unload_models()
            self.face_editors.unload_models()

            # Unload any remaining models in the main dictionaries
            self.delete_models()
            self.delete_models_dfm()

            # Unload the Clip and KV Extractor models specifically
            self.unload_kv_extractor()
            if self.clip_session:
                del self.clip_session
                self.clip_session = []
        finally:
            self.force_unload_in_progress = False

        # Finally, clear caches
        print("[INFO] Running garbage collection and clearing CUDA cache.")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[INFO] GPU Memory Cleared.")

    # --- KV Extractor (Thread-Safe Loading) ---

    def get_kv_map_for_face(
        self, input_face_image_pil: "Image.Image"
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Loads the KV Extractor, extracts K/V maps, and unloads.
        Callers are responsible for holding kv_extraction_lock around this call.
        """
        kv_map = {}
        try:
            # 1. Load the extractor
            self.ensure_kv_extractor_loaded()

            if self.kv_extractor is None:
                raise RuntimeError("KV Extractor model failed to load.")

            # 2. Perform the extraction
            print("[INFO] Extracting K/V from reference image...")
            kv_map = self.kv_extractor.extract_kv(input_face_image_pil)
            print(
                f"[INFO] Successfully extracted K/V for {len(kv_map)} attention layers."
            )

        except Exception as e:
            print(f"[ERROR] Failed the K/V extraction: {e}")
            traceback.print_exc()
            kv_map = {}  # Return empty map if failed

        finally:
            # 3. Unload the extractor
            self.unload_kv_extractor()

        return kv_map

    def ensure_kv_extractor_loaded(self):
        """
        Guarantees that the KVExtractor (Ref-LDM) model is loaded and ready.

        Downloads the required config and checkpoint files on first use, then
        instantiates ``KVExtractor`` inside the model lock.  Safe to call multiple
        times; no-ops when the extractor is already loaded.
        """
        # MP-25: Check file existence and download outside lock to avoid blocking
        # other threads during potentially slow network I/O.
        base_path = "model_assets/ref-ldm_embedding"
        configs_path = os.path.join(base_path, "configs")
        ckpts_path = os.path.join(base_path, "ckpts")
        os.makedirs(configs_path, exist_ok=True)
        os.makedirs(ckpts_path, exist_ok=True)

        ref_ldm_files = {
            "configs/ldm.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/ldm.yaml",
            "configs/refldm.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/refldm.yaml",
            "configs/vqgan.yaml": "https://raw.githubusercontent.com/Glat0s/ref-ldm-onnx/slim-fast/configs/vqgan.yaml",
            "ckpts/refldm.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/refldm.ckpt",
            "ckpts/vqgan.ckpt": "https://github.com/ChiWeiHsiao/ref-ldm/releases/download/1.0.0/vqgan.ckpt",
        }

        for rel_path, url in ref_ldm_files.items():
            full_path = os.path.join(base_path, rel_path)
            if not is_file_exists(full_path):
                print(
                    f"[INFO] Downloading ReF-LDM file: {os.path.basename(full_path)}..."
                )
                download_file(os.path.basename(full_path), full_path, None, url)

        config_path = os.path.join(configs_path, "refldm.yaml")
        model_path = os.path.join(ckpts_path, "refldm.ckpt")
        vae_path = os.path.join(ckpts_path, "vqgan.ckpt")

        if not all(os.path.exists(p) for p in [config_path, model_path, vae_path]):
            print(
                "[ERROR] ReF-LDM model files not found even after download attempt. Cannot load KV Extractor."
            )
            return

        # MP-25: Only lock during the final KVExtractor instantiation.
        with self.model_lock:
            if self.kv_extractor is not None:
                return  # Already loaded (another thread may have loaded it)

            try:
                print("[INFO] Loading KV Extractor...")
                self.kv_extractor = KVExtractor(
                    model_config_path=config_path,
                    model_ckpt_path=model_path,
                    vae_ckpt_path=vae_path,
                    device=self.device,
                )
                print("[INFO] KV Extractor loaded.")
            except Exception as e:
                print(f"[ERROR] Failed to load KV Extractor: {e}")
                traceback.print_exc()
                self.kv_extractor = None

    def ensure_denoiser_models_loaded(self):
        """Loads the UNet and VAE models if they are not already loaded."""
        with self.model_lock:
            unet_model_name = self.main_window.fixed_unet_model_name
            vae_encoder_name = "RefLDMVAEEncoder"
            vae_decoder_name = "RefLDMVAEDecoder"

            if not self.models.get(unet_model_name):
                self.models[unet_model_name] = self.load_model(unet_model_name)

            if not self.models.get(vae_encoder_name):
                self.models[vae_encoder_name] = self.load_model(vae_encoder_name)

            if not self.models.get(vae_decoder_name):
                self.models[vae_decoder_name] = self.load_model(vae_decoder_name)

    def unload_denoiser_models(self):
        """Unloads the UNet and VAE models."""
        with self.model_lock:
            print("[INFO] Unloading denoiser models (UNet, VAEs)...")
            self.unload_model(self.main_window.fixed_unet_model_name)
            self.unload_model("RefLDMVAEEncoder")
            self.unload_model("RefLDMVAEDecoder")

    def unload_kv_extractor(self, force_immediate=False):
        """Unloads the KVExtractor model and clears associated memory."""

        # Check if unloading should be skipped
        if not self.force_unload_in_progress:
            if self.main_window.control.get("KeepModelsAliveToggle", False):
                return  # Skip unloading

        # --- SMART UNLOAD: Intercept if video is playing ---
        if not force_immediate and not self.force_unload_in_progress:
            vp = getattr(self.main_window, "video_processor", None)
            if vp and getattr(vp, "processing", False):
                # Video is playing, get the feeder's current frame and defer
                target_frame = getattr(vp, "current_frame_number", 0) + 1
                with self.model_lock:
                    self.deferred_unloads["KVExtractor"] = {
                        "type": "kv",
                        "target_frame": target_frame,
                    }
                print(
                    f"[INFO] Smart Unload: Deferring KV Extractor unload after frame {target_frame}"
                )
                return

        with self.model_lock:
            if self.kv_extractor is not None:
                print("[INFO] Unloading KV Extractor...")
                del self.kv_extractor
                self.kv_extractor = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # --- Wrapper Unloaders ---

    def unload_face_detector_models(self):
        """Unloads the active face detector model under the model lock."""
        with self.model_lock:
            self.face_detectors.unload_models()

    def unload_face_landmark_detector_models(self):
        """Unloads the active face landmark detector model under the model lock."""
        with self.model_lock:
            self.face_landmark_detectors.unload_models()

    def unload_face_editor_models(self):
        """Unloads all loaded face editor models under the model lock."""
        with self.model_lock:
            self.face_editors.unload_models()

    def unload_perform_recast_models(self):
        """Unloads all loaded PerformRecast (Recast mode) models under the lock."""
        with self.model_lock:
            self.perform_recast.unload_models()

    def unload_face_mask_models(self):
        """Unloads all loaded face mask models under the model lock."""
        with self.model_lock:
            self.face_masks.unload_models()

    def unload_frame_enhancer_models(self):
        """Unloads all loaded frame enhancer models under the model lock."""
        with self.model_lock:
            self.frame_enhancers.unload_models()

    def unload_face_restorer_models(self):
        """Unloads all loaded face restorer models under the model lock."""
        with self.model_lock:
            self.face_restorers.unload_models()

    # --- Static Math Helpers ---

    @staticmethod
    def print_tensor_stats(tensor: torch.Tensor, name: str, enabled: bool = True):
        if not enabled:
            return
        if isinstance(tensor, torch.Tensor):
            if tensor.dtype == torch.uint8:
                tensor_float = tensor.float() / 255.0
                print(
                    f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor_float.mean().item():.4f}, std={tensor_float.std().item():.4f} (stats on [0,1] float)"
                )
            elif tensor.dtype == torch.float16 or tensor.dtype == torch.float32:
                print(
                    f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}, mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}"
                )
            else:
                print(
                    f"DEBUG DENOISER STATS for {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device} (stats not computed for this dtype)"
                )
        else:
            print(
                f"DEBUG DENOISER STATS for {name}: Not a tensor, type is {type(tensor)}"
            )

    @staticmethod
    def make_beta_schedule(
        schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
    ) -> np.ndarray:
        if schedule == "linear":
            betas = (
                torch.linspace(
                    linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
                )
                ** 2
            )
        elif schedule == "cosine":
            timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep
                + cosine_s
            )
            alphas = timesteps / (1 + cosine_s) * np.pi / 2  # type: ignore
            alphas = torch.cos(alphas).pow(2)
            alphas = alphas / alphas[0]
            betas = 1 - alphas[1:] / alphas[:-1]
            betas = np.clip(betas.numpy(), a_min=0, a_max=0.999)  # type: ignore
        elif schedule == "sqrt_linear":
            betas = torch.linspace(
                linear_start, linear_end, n_timestep, dtype=torch.float64
            )
        elif schedule == "sqrt":
            betas = (
                torch.linspace(
                    linear_start, linear_end, n_timestep, dtype=torch.float64
                )
                ** 0.5
            )
        else:
            raise ValueError(f"schedule '{schedule}' unknown.")
        return betas.numpy() if isinstance(betas, torch.Tensor) else betas

    @staticmethod
    def make_ddim_timesteps(
        ddim_discr_method: str,
        num_ddim_timesteps: int,
        num_ddpm_timesteps: int,
        verbose: bool = True,
    ) -> np.ndarray:
        if ddim_discr_method == "uniform":
            c = num_ddpm_timesteps // num_ddim_timesteps
            if c == 0:
                c = 1
            ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
        elif ddim_discr_method == "uniform_trailing":
            c = num_ddpm_timesteps // num_ddim_timesteps
            if c == 0:
                c = 1
            ddim_timesteps = np.arange(num_ddpm_timesteps, 0, -c).astype(int)[::-1] - 2
            ddim_timesteps = np.clip(ddim_timesteps, 0, num_ddpm_timesteps - 1)
        elif ddim_discr_method == "quad":
            ddim_timesteps = (
                (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps))
                ** 2
            ).astype(int)
        else:
            raise NotImplementedError(
                f'There is no ddim discretization method called "{ddim_discr_method}"'
            )

        steps_out = np.unique(ddim_timesteps)
        steps_out.sort()

        if verbose:
            print(f"Selected DDPM timesteps for DDIM sampler (0-indexed): {steps_out}")
        return steps_out

    @staticmethod
    def make_ddim_sampling_parameters(
        alphacums: np.ndarray,
        ddim_timesteps: np.ndarray,
        eta: float,
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _prev_t = np.concatenate(
            ([-1], ddim_timesteps[:-1])
        )  # Use -1 to signify "before first step"
        _alphas_prev = np.array([alphacums[pt] if pt != -1 else 1.0 for pt in _prev_t])
        _alphas = alphacums[ddim_timesteps]
        sigmas = eta * np.sqrt(
            (1 - _alphas_prev) / (1 - _alphas) * (1 - _alphas / _alphas_prev)
        )
        sigmas = np.nan_to_num(sigmas, nan=0.0)
        return sigmas, _alphas, _alphas_prev

    # --- Processing Wrappers ---

    def apply_vgg_mask_simple(
        self,
        swapped_face: torch.Tensor,  # [3,512,512] uint8
        original_face: torch.Tensor,  # [3,512,512] uint8
        swap_mask_128: torch.Tensor,  # [1,128,128] float mask (0..1)
        center_pct: float,  # 0..100, e.g. parameters['VGGMaskThresholdSlider']
        softness_pct: float,  # 0..100, e.g. parameters['VGGMaskSoftnessSlider']
        feature_layer: str = "combo_relu3_3_relu3_1",
        mode: str = "smooth",  # 'smooth' (smoothstep) or 'linear'
    ):
        """
        Returns:
          mask_vgg: [1,512,512] float 0..1 (soft difference mask)
          diff_norm_texture: [1,128,128] float 0..1 (normalized raw difference in 128 resolution)
        """
        # 1) Get raw difference via existing ONNX pipeline in 128x128 (without mapping).
        #    We use apply_perceptual_diff_onnx in pass-through mode (ExcludeVGGMaskEnableToggle=False),
        #    and ignore the complex threshold parameters (they are replaced below).
        dummy_lower = 0.0
        dummy_upper = 1.0
        dummy_upper_v = 1.0
        dummy_middle_v = 0.5

        diff_mapped_128, diff_norm_128 = self.apply_perceptual_diff_onnx(
            swapped_face,
            original_face,
            swap_mask_128,
            dummy_lower,
            0.0,
            dummy_upper,
            dummy_upper_v,
            dummy_middle_v,
            feature_layer,
            ExcludeVGGMaskEnableToggle=False,
        )
        # diff_norm_128: [1,128,128] in [0..1]
        d = diff_mapped_128.squeeze(0)  # [128,128]

        # 2) Two-slider-Mapping -> lower/upper threshold derived from (center, softness)
        center = float(center_pct) / 100.0  # 0..1
        softness = float(softness_pct) / 100.0  # 0..1

        # Width of the transition band (practical values):
        #  - Min width 0.04, Max width 0.40
        band = 0.04 + 0.36 * softness
        lo = max(0.0, center - band * 0.5)
        hi = min(1.0, center + band * 0.5)

        # 3) Curve Shape
        x = (d - lo) / max(1e-6, (hi - lo))
        x = x.clamp(0.0, 1.0)
        if mode == "smooth":
            # Smoothstep
            x = x * x * (3.0 - 2.0 * x)
        # else: 'linear' -> x remains linear

        # 4) Upscale to 512x512 (bilinear)
        x_512 = torch.nn.functional.interpolate(
            x.unsqueeze(0).unsqueeze(0),
            size=(512, 512),
            mode="bilinear",
            align_corners=True,
        ).squeeze(0)

        return x_512.clamp(0, 1), diff_norm_128

    def run_detect(
        self,
        img,
        detect_mode="RetinaFace",
        max_num=1,
        score=0.5,
        input_size=(512, 512),
        use_landmark_detection=False,
        landmark_detect_mode="203",
        landmark_score=0.5,
        from_points=False,
        rotation_angles=None,
        **kwargs,
    ):
        rotation_angles = rotation_angles or [0]
        return self.face_detectors.run_detect(
            img,
            detect_mode,
            max_num,
            score,
            input_size,
            use_landmark_detection,
            landmark_detect_mode,
            landmark_score,
            from_points,
            rotation_angles,
            **kwargs,
        )

    def run_detect_landmark(
        self,
        img,
        bbox,
        det_kpss,
        detect_mode="203",
        score=0.5,
        from_points=False,
        **kwargs,
    ):
        return self.face_landmark_detectors.run_detect_landmark(
            img, bbox, det_kpss, detect_mode, score, from_points, **kwargs
        )

    def get_arcface_model(self, face_swapper_model):
        if face_swapper_model in arcface_mapping_model_dict:
            return arcface_mapping_model_dict[face_swapper_model]
        else:
            raise ValueError(f"Face swapper model {face_swapper_model} not found.")

    def run_recognize_direct(
        self, img, kps, similarity_type="Auto", arcface_model="Inswapper128ArcFace"
    ):
        return self.face_swappers.run_recognize_direct(
            img, kps, similarity_type, arcface_model
        )

    def calc_inswapper_latent(self, source_embedding):
        return self.face_swappers.calc_inswapper_latent(source_embedding)

    def run_inswapper(self, image, embedding, output):
        self.face_swappers.run_inswapper(image, embedding, output)

    def run_inswapper_batched(self, images, embedding, output):
        self.face_swappers.run_inswapper_batched(images, embedding, output)

    def calc_swapper_latent_iss(self, source_embedding, version="A"):
        return self.face_swappers.calc_swapper_latent_iss(source_embedding, version)

    def run_iss_swapper(self, image, embedding, output, version="A"):
        self.face_swappers.run_iss_swapper(image, embedding, output, version)

    def calc_swapper_latent_simswap512(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_simswap512(source_embedding)

    def run_swapper_simswap512(self, image, embedding, output):
        self.face_swappers.run_swapper_simswap512(image, embedding, output)

    def calc_swapper_latent_ghost(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_ghost(source_embedding)

    def run_swapper_ghostface(
        self, image, embedding, output, swapper_model="GhostFace-v2"
    ):
        self.face_swappers.run_swapper_ghostface(
            image, embedding, output, swapper_model
        )

    def calc_swapper_latent_cscs(self, source_embedding):
        return self.face_swappers.calc_swapper_latent_cscs(source_embedding)

    def run_swapper_cscs(self, image, embedding, output):
        self.face_swappers.run_swapper_cscs(image, embedding, output)

    def run_occluder(self, image, output):
        self.face_masks.run_occluder(image, output)

    def run_dfl_xseg(self, image, output):
        self.face_masks.run_dfl_xseg(image, output)

    def run_faceparser(self, image, output):
        self.face_masks.run_faceparser(image, output)

    def run_CLIPs(self, img, CLIPText, CLIPAmount):
        return self.face_masks.run_CLIPs(img, CLIPText, CLIPAmount)

    def lp_motion_extractor(self, img, face_editor_type="Human-Face", **kwargs) -> dict:
        return self.face_editors.lp_motion_extractor(img, face_editor_type, **kwargs)

    def lp_appearance_feature_extractor(self, img, face_editor_type="Human-Face"):
        return self.face_editors.lp_appearance_feature_extractor(img, face_editor_type)

    def lp_retarget_eye(
        self,
        kp_source: torch.Tensor,
        eye_close_ratio: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_retarget_eye(
            kp_source, eye_close_ratio, face_editor_type
        )

    def lp_retarget_lip(
        self,
        kp_source: torch.Tensor,
        lip_close_ratio: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_retarget_lip(
            kp_source, lip_close_ratio, face_editor_type
        )

    def lp_stitch(
        self,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_stitch(kp_source, kp_driving, face_editor_type)

    def lp_stitching(
        self,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_stitching(kp_source, kp_driving, face_editor_type)

    def lp_warp_decode(
        self,
        feature_3d: torch.Tensor,
        kp_source: torch.Tensor,
        kp_driving: torch.Tensor,
        face_editor_type="Human-Face",
    ) -> torch.Tensor:
        return self.face_editors.lp_warp_decode(
            feature_3d, kp_source, kp_driving, face_editor_type
        )

    def findCosineDistance(self, vector1, vector2):
        vector1 = vector1.ravel()
        vector2 = vector2.ravel()
        cos_dist = 1 - np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )  # 2..0
        return 100 - cos_dist * 50

    def apply_facerestorer(
        self,
        swapped_face_upscaled,
        restorer_det_type,
        restorer_type,
        restorer_blend,
        fidelity_weight,
        detect_score,
        target_kps,
        slot_id: int = 1,
    ):
        return self.face_restorers.apply_facerestorer(
            swapped_face_upscaled,
            restorer_det_type,
            restorer_type,
            restorer_blend,
            fidelity_weight,
            detect_score,
            target_kps,
            slot_id=slot_id,
        )

    def apply_occlusion(self, img, amount):
        return self.face_masks.apply_occlusion(img, amount)

    def apply_dfl_xseg(self, img, amount, mouth, parameters, inner_mouth_mask):
        return self.face_masks.apply_dfl_xseg(
            img, amount, mouth, parameters, inner_mouth_mask
        )

    def process_masks_and_masks(
        self, swap_restorecalc, original_face_512, parameters, control
    ):
        return self.face_masks.process_masks_and_masks(
            swap_restorecalc, original_face_512, parameters, control
        )

    def apply_face_makeup(self, img, parameters):
        return self.face_editors.apply_face_makeup(img, parameters)

    def restore_mouth(
        self,
        img_orig,
        img_swap,
        kpss_orig,
        blend_alpha=0.5,
        feather_radius=10,
        size_factor=0.5,
        radius_factor_x=1.0,
        radius_factor_y=1.0,
        x_offset=0,
        y_offset=0,
    ):
        return self.face_masks.restore_mouth(
            img_orig,
            img_swap,
            kpss_orig,
            blend_alpha,
            feather_radius,
            size_factor,
            radius_factor_x,
            radius_factor_y,
            x_offset,
            y_offset,
        )

    def restore_eyes(
        self,
        img_orig,
        img_swap,
        kpss_orig,
        blend_alpha=0.5,
        feather_radius=10,
        size_factor=3.5,
        radius_factor_x=1.0,
        radius_factor_y=1.0,
        x_offset=0,
        y_offset=0,
        eye_spacing_offset=0,
    ):
        return self.face_masks.restore_eyes(
            img_orig,
            img_swap,
            kpss_orig,
            blend_alpha,
            feather_radius,
            size_factor,
            radius_factor_x,
            radius_factor_y,
            x_offset,
            y_offset,
            eye_spacing_offset,
        )

    def apply_fake_diff(
        self,
        swapped_face,
        original_face,
        lower_limit_thresh,
        lower_value,
        upper_thresh,
        upper_value,
        middle_value,
        parameters,
    ):
        return self.face_masks.apply_fake_diff(
            swapped_face,
            original_face,
            lower_limit_thresh,
            lower_value,
            upper_thresh,
            upper_value,
            middle_value,
            parameters,
        )

    def run_onnx(self, image, output, model_key):
        return self.face_masks.run_onnx(image, output, model_key)

    def apply_perceptual_diff_onnx(
        self,
        swapped_face,
        original_face,
        swap_mask,
        lower_limit_thresh,
        lower_value,
        upper_thresh,
        upper_value,
        middle_value,
        feature_layer,
        ExcludeVGGMaskEnableToggle,
    ):
        return self.face_masks.apply_perceptual_diff_onnx(
            swapped_face,
            original_face,
            swap_mask,
            lower_limit_thresh,
            lower_value,
            upper_thresh,
            upper_value,
            middle_value,
            feature_layer,
            ExcludeVGGMaskEnableToggle,
        )

    @staticmethod
    def extract_into_tensor_torch(
        a: torch.Tensor, t: torch.Tensor, x_shape: tuple
    ) -> torch.Tensor:
        if t.ndim == 0:
            t = t.unsqueeze(0)
        b = t.shape[0]
        out = torch.gather(a, 0, t.long())
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def apply_denoiser_unet(
        self,
        image_cxhxw_uint8: torch.Tensor,
        reference_kv_map: Dict | None,
        use_reference_exclusive_path: bool,
        denoiser_mode: str = "Single Step (Fast)",
        denoiser_single_step_t: int = 1,
        denoiser_ddim_steps: int = 20,
        denoiser_cfg_scale: float = 1.0,
        denoiser_ddim_eta: float = 0.0,
        base_seed: int = 220,
        latent_sharpening_strength: float = 0.0,
        color_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Runs the Diffusion-based Denoiser/Restorer (ReF-LDM).
        Supports 'Single Step' (Fast) and 'Full Restore' (DDIM) modes.
        Also handles pixel sharpening and histogram matching for color consistency.
        """
        # --- CONFIGURATION ---
        ENABLE_PIXEL_SHARPENING = latent_sharpening_strength > 0.0
        PIXEL_SHARPEN_STRENGTH = latent_sharpening_strength

        ENABLE_COLOR_MATCH = True

        # P2-04: enable debug output via env var: set VISOMASTER_DEBUG_DENOISER=1
        DEBUG_DENOISER = os.environ.get("VISOMASTER_DEBUG_DENOISER", "0") == "1"
        unet_model_name = self.main_window.fixed_unet_model_name
        vae_encoder_name = "RefLDMVAEEncoder"
        vae_decoder_name = "RefLDMVAEDecoder"

        if DEBUG_DENOISER:
            print(
                f"\n--- Denoiser Pass Start: Mode='{denoiser_mode}', CFG Scale={denoiser_cfg_scale}, VAE Scale Factor={self.vae_scale_factor} ---"
            )
            ModelsProcessor.print_tensor_stats(
                image_cxhxw_uint8, "Initial input image_cxhxw_uint8", DEBUG_DENOISER
            )

        with self.model_lock:
            self.ensure_denoiser_models_loaded()
            unet_session = self.models.get(unet_model_name)
            vae_enc_session = self.models.get(vae_encoder_name)
            vae_dec_session = self.models.get(vae_decoder_name)

            if not (unet_session and vae_enc_session and vae_dec_session):
                return image_cxhxw_uint8

        kv_tensor_map_for_this_run: Dict[str, Dict[str, torch.Tensor]] | None = None
        if reference_kv_map:
            try:
                # The cache from ref_ldm_kv_embedding.py is already on GPU and
                # immutable from this code path's perspective (ORT reads the
                # bound buffers, never writes). The previous .clone().to(device)
                # round was meaningful only when the cache lived on CPU — it
                # forced a PCIe upload. With the cache on GPU, .to() was a no-op
                # but .clone() still allocated a fresh GPU buffer per layer per
                # UNet step (32 layers × 20-50 DDIM steps × 8 workers — many GB
                # of GPU memory churn per frame). Bind the cached tensors
                # directly; if a future code path needs to mutate them, that
                # path should clone locally instead of paying the cost here.
                kv_tensor_map_for_this_run = {
                    layer: {"k": tens_dict["k"], "v": tens_dict["v"]}
                    for layer, tens_dict in reference_kv_map.items()
                    if tens_dict
                    and isinstance(tens_dict.get("k"), torch.Tensor)
                    and isinstance(tens_dict.get("v"), torch.Tensor)
                }
            except Exception as e:
                print(f"[ERROR] Denoiser: Error deep copying K/V map: {e}. Skipping.")
                return image_cxhxw_uint8

        if (
            denoiser_mode == "Full Restore (DDIM)"
            and use_reference_exclusive_path
            and not kv_tensor_map_for_this_run
        ):
            print(
                "[ERROR] Denoiser (Full Restore): Reference K/V tensor file selected for use, but K/V map is empty. Skipping."
            )
            return image_cxhxw_uint8
        if (
            denoiser_mode == "Single Step (Fast)"
            and use_reference_exclusive_path
            and not kv_tensor_map_for_this_run
        ):
            print(
                "[ERROR] Denoiser (Single Step): Reference K/V tensor file selected for use, but K/V map is empty. Skipping."
            )
            return image_cxhxw_uint8

        target_proc_dim = 512
        _, h_input, w_input = image_cxhxw_uint8.shape
        if h_input != target_proc_dim or w_input != target_proc_dim:
            # OPTIMIZED: Functional resize avoids slow class instantiation
            image_to_process_cxhxw_uint8 = v2.functional.resize(
                image_cxhxw_uint8,
                [target_proc_dim, target_proc_dim],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            image_to_process_cxhxw_uint8 = image_cxhxw_uint8

        h_proc, w_proc = (
            image_to_process_cxhxw_uint8.shape[1],
            image_to_process_cxhxw_uint8.shape[2],
        )

        image_srgb_float_minus1_1 = (image_to_process_cxhxw_uint8.float() / 127.5) - 1.0
        image_srgb_float_minus1_1_batched = image_srgb_float_minus1_1.unsqueeze(
            0
        ).contiguous()

        latent_h, latent_w = h_proc // 8, w_proc // 8
        encoded_latent_direct_vae_out_bchw = torch.empty(
            (1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device
        ).contiguous()

        self.face_restorers.run_vae_encoder(
            image_srgb_float_minus1_1_batched, encoded_latent_direct_vae_out_bchw
        )

        lq_latent_x0_scaled_for_unet = (
            encoded_latent_direct_vae_out_bchw * self.vae_scale_factor
        )
        # MP-16: del encoded latent buffer and input image float as soon as done
        del encoded_latent_direct_vae_out_bchw
        del image_srgb_float_minus1_1_batched
        final_denoised_latent_x0_scaled = None

        # OPTIMIZATION VRAM/PCIe: Create boolean tensors DIRECTLY on the GPU.
        # Passing a Python list to torch.tensor() allocates CPU RAM first and forces a PCIe transfer.
        # torch.ones/torch.zeros directly on the device avoids CPU-GPU sync overhead.
        if use_reference_exclusive_path:
            is_ref_flag_tensor_for_unet = torch.ones(
                1, dtype=torch.bool, device=self.device
            )
        else:
            is_ref_flag_tensor_for_unet = torch.zeros(
                1, dtype=torch.bool, device=self.device
            )

        actual_use_exclusive_path_tensor_for_unet = is_ref_flag_tensor_for_unet
        false_tensor_for_unet = torch.zeros(1, dtype=torch.bool, device=self.device)

        rng = torch.Generator(device=self.device)
        rng.manual_seed(base_seed)

        # --- PROCESS: Single Step ---
        if denoiser_mode == "Single Step (Fast)":
            rng.manual_seed(base_seed + denoiser_single_step_t)
            noise_sample = torch.randn(
                lq_latent_x0_scaled_for_unet.shape,
                device=self.device,
                dtype=lq_latent_x0_scaled_for_unet.dtype,
                generator=rng,
            )

            current_t_idx = min(
                max(0, denoiser_single_step_t), len(self.alphas_cumprod_np) - 1
            )
            alpha_t_bar_val = self.alphas_cumprod_np[current_t_idx]

            # OPTIMIZATION VRAM: Direct GPU creation
            sqrt_alpha_bar_t_torch = torch.sqrt(
                torch.full(
                    (1,), alpha_t_bar_val, dtype=torch.float32, device=self.device
                )
            )
            sqrt_one_minus_alpha_bar_t_torch = torch.sqrt(
                1.0
                - torch.full(
                    (1,), alpha_t_bar_val, dtype=torch.float32, device=self.device
                )
            )

            xt_noisy_scaled_8_channel = (
                lq_latent_x0_scaled_for_unet * sqrt_alpha_bar_t_torch
                + noise_sample * sqrt_one_minus_alpha_bar_t_torch
            )
            unet_input_16_channel = torch.cat(
                (xt_noisy_scaled_8_channel, lq_latent_x0_scaled_for_unet), dim=1
            )

            # OPTIMIZATION VRAM: Direct GPU creation
            timesteps_tensor_unet = torch.full(
                (1,), current_t_idx, dtype=torch.int64, device=self.device
            )

            predicted_noise_from_unet = torch.empty(
                (1, 8, latent_h, latent_w), dtype=torch.float32, device=self.device
            ).contiguous()

            # CUDA Stream Sync: Ensure all non-blocking PCIe transfers are complete
            if torch.cuda.is_available():
                torch.cuda.current_stream().synchronize()

            self.face_restorers.run_ref_ldm_unet(
                x_noisy_plus_lq_latent=unet_input_16_channel,
                timesteps_tensor=timesteps_tensor_unet,
                is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                kv_tensor_map=kv_tensor_map_for_this_run,
                output_unet_tensor=predicted_noise_from_unet,
            )
            final_denoised_latent_x0_scaled = (
                xt_noisy_scaled_8_channel
                - sqrt_one_minus_alpha_bar_t_torch * predicted_noise_from_unet
            ) / sqrt_alpha_bar_t_torch

        # --- PROCESS: Full Restore (DDIM) ---
        elif denoiser_mode == "Full Restore (DDIM)":
            num_ddpm_timesteps = self.alphas_cumprod_np.shape[0]

            _ddim_raw_ddpm_timesteps_np = ModelsProcessor.make_ddim_timesteps(
                ddim_discr_method="uniform",
                num_ddim_timesteps=denoiser_ddim_steps,
                num_ddpm_timesteps=num_ddpm_timesteps,
                verbose=DEBUG_DENOISER,
            )
            _ddim_sigmas_np, _ddim_alphas_np, _ddim_alphas_prev_np = (
                ModelsProcessor.make_ddim_sampling_parameters(
                    alphacums=self.alphas_cumprod_np,
                    ddim_timesteps=_ddim_raw_ddpm_timesteps_np,
                    eta=denoiser_ddim_eta,
                    verbose=DEBUG_DENOISER,
                )
            )

            # Numpy -> Tensor directly on device
            ddim_sigmas = (
                torch.from_numpy(_ddim_sigmas_np)
                .float()
                .to(self.device, non_blocking=True)
            )
            ddim_alphas = (
                torch.from_numpy(_ddim_alphas_np)
                .float()
                .to(self.device, non_blocking=True)
            )
            ddim_alphas_prev = (
                torch.from_numpy(_ddim_alphas_prev_np)
                .float()
                .to(self.device, non_blocking=True)
            )

            ddim_sqrt_one_minus_alphas = torch.sqrt(
                torch.clamp(1.0 - ddim_alphas, min=0.0)
            )

            current_latent_xt_scaled = torch.randn(
                lq_latent_x0_scaled_for_unet.shape,
                device=self.device,
                dtype=lq_latent_x0_scaled_for_unet.dtype,
                generator=rng,
            )
            time_range_ddpm_indices = np.flip(_ddim_raw_ddpm_timesteps_np).copy()
            total_steps = len(time_range_ddpm_indices)

            pred_x0_scaled_current_step = torch.empty_like(lq_latent_x0_scaled_for_unet)

            # --- OPTIMIZATION VRAM: Pre-allocate DDIM Loop Buffers ---
            # Allocating these inside the loop causes massive VRAM fragmentation
            # and stalls the CUDA allocator. We allocate once per function call.
            ts_unet = torch.empty((1,), dtype=torch.int64, device=self.device)
            schedule_idx_tensor = torch.empty(
                (1,), dtype=torch.long, device=self.device
            )
            e_t_cond = torch.empty_like(lq_latent_x0_scaled_for_unet)
            e_t_uncond = (
                torch.empty_like(lq_latent_x0_scaled_for_unet)
                if denoiser_cfg_scale != 1.0
                else None
            )
            noise_ddim_buffer = torch.empty_like(lq_latent_x0_scaled_for_unet)

            for i, step_ddpm_idx in enumerate(time_range_ddpm_indices):
                index_for_schedules = total_steps - 1 - i

                # OPTIMIZATION VRAM: In-place update of pre-allocated tensors
                ts_unet.fill_(step_ddpm_idx)
                schedule_idx_tensor.fill_(index_for_schedules)

                unet_input_cond = torch.cat(
                    [current_latent_xt_scaled, lq_latent_x0_scaled_for_unet], dim=1
                )

                if torch.cuda.is_available():
                    torch.cuda.current_stream().synchronize()

                # The ONNX runtime writes directly into our pre-allocated e_t_cond buffer
                self.face_restorers.run_ref_ldm_unet(
                    x_noisy_plus_lq_latent=unet_input_cond,
                    timesteps_tensor=ts_unet,
                    is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                    use_reference_exclusive_path_globally_tensor=actual_use_exclusive_path_tensor_for_unet,
                    kv_tensor_map=kv_tensor_map_for_this_run,
                    output_unet_tensor=e_t_cond,
                )
                e_t = e_t_cond

                if denoiser_cfg_scale != 1.0:
                    unet_input_uncond = torch.cat(
                        [current_latent_xt_scaled, lq_latent_x0_scaled_for_unet], dim=1
                    )

                    if torch.cuda.is_available():
                        torch.cuda.current_stream().synchronize()

                    self.face_restorers.run_ref_ldm_unet(
                        x_noisy_plus_lq_latent=unet_input_uncond,
                        timesteps_tensor=ts_unet,
                        is_ref_flag_tensor=is_ref_flag_tensor_for_unet,
                        use_reference_exclusive_path_globally_tensor=false_tensor_for_unet,
                        kv_tensor_map=None,
                        output_unet_tensor=e_t_uncond,
                    )
                    e_t = e_t_uncond + denoiser_cfg_scale * (e_t_cond - e_t_uncond)

                a_t = ModelsProcessor.extract_into_tensor_torch(
                    ddim_alphas, schedule_idx_tensor, current_latent_xt_scaled.shape
                )
                a_prev = ModelsProcessor.extract_into_tensor_torch(
                    ddim_alphas_prev,
                    schedule_idx_tensor,
                    current_latent_xt_scaled.shape,
                )
                sigma_t = ModelsProcessor.extract_into_tensor_torch(
                    ddim_sigmas, schedule_idx_tensor, current_latent_xt_scaled.shape
                )
                sqrt_one_minus_a_t = ModelsProcessor.extract_into_tensor_torch(
                    ddim_sqrt_one_minus_alphas,
                    schedule_idx_tensor,
                    current_latent_xt_scaled.shape,
                )

                pred_x0_scaled_current_step = (
                    current_latent_xt_scaled - sqrt_one_minus_a_t * e_t
                ) / torch.sqrt(a_t).clamp(min=1e-8)

                dir_xt = (
                    torch.sqrt(torch.clamp(1.0 - a_prev - sigma_t**2, min=1e-8)) * e_t
                )

                # OPTIMIZATION VRAM: Reuse noise buffer instead of generating a new tensor every step
                noise_ddim_buffer.normal_(generator=rng)
                noise_ddim = sigma_t * noise_ddim_buffer

                current_latent_xt_scaled = (
                    torch.sqrt(a_prev) * pred_x0_scaled_current_step
                    + dir_xt
                    + noise_ddim
                )

            final_denoised_latent_x0_scaled = pred_x0_scaled_current_step
        else:
            print(
                f"[ERROR] Denoiser: Unknown mode '{denoiser_mode}'. Skipping denoiser pass."
            )
            return image_cxhxw_uint8

        if final_denoised_latent_x0_scaled is None:
            return image_cxhxw_uint8

        latent_for_vae_decoder = final_denoised_latent_x0_scaled / self.vae_scale_factor
        # MP-16: del denoised latent once VAE decoder input is computed
        del final_denoised_latent_x0_scaled
        decoded_image_normalized_bchw = torch.empty(
            (1, 3, h_proc, w_proc), dtype=torch.float32, device=self.device
        ).contiguous()

        self.face_restorers.run_vae_decoder(
            latent_for_vae_decoder, decoded_image_normalized_bchw
        )
        # MP-16: del VAE decoder input latent after use
        del latent_for_vae_decoder

        decoded_image_soft_clamped_bchw = torch.tanh(decoded_image_normalized_bchw)
        # MP-16: del raw decoder output after soft-clamping
        del decoded_image_normalized_bchw
        image_after_postproc_float_0_1 = (
            decoded_image_soft_clamped_bchw.squeeze(0) + 1.0
        ) / 2.0
        image_after_postproc_float_0_1 = torch.clamp(
            image_after_postproc_float_0_1, 0.0, 1.0
        )

        # --- IMPROVEMENT B: Color Matching (DFL Orig - LAB Reinhard) ---
        # Applied before sharpening so that histogram normalisation does not
        # undo the local-contrast enhancement added by the unsharp mask.
        if ENABLE_COLOR_MATCH:
            ref_tensor = image_to_process_cxhxw_uint8
            res_tensor = image_after_postproc_float_0_1 * 255.0

            if color_mask is not None:
                mask = color_mask.clone()
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0)
                if mask.shape[-1] != ref_tensor.shape[-1]:
                    mask = v2.functional.resize(
                        mask,
                        [ref_tensor.shape[-2], ref_tensor.shape[-1]],
                        antialias=True,
                    ).squeeze(0)
                else:
                    mask = mask.squeeze(0)
            else:
                mask = (ref_tensor.sum(dim=0) > 0).float()

            try:
                # Apply DFL_Orig Transfer
                matched_result = faceutil.histogram_matching_DFL_Orig(
                    ref_tensor,
                    res_tensor,
                    mask,
                    100,  # Blend strength 100%
                )
                image_after_postproc_float_0_1 = matched_result / 255.0
            except Exception as e:
                print(f"[WARN] Color matching failed: {e}")
        # --- END IMPROVEMENT B ---

        # --- IMPROVEMENT A: Pixel Sharpening (Unsharp Mask) ---
        # Applied after colour matching so the histogram normalisation does not
        # cancel out the local edge-contrast boost from the unsharp mask.
        if ENABLE_PIXEL_SHARPENING:
            # OPTIMIZED: Functional gaussian blur avoids class instantiation
            blurred = v2.functional.gaussian_blur(
                image_after_postproc_float_0_1.unsqueeze(0), [5, 5], [1.0, 1.0]
            ).squeeze(0)
            detail = image_after_postproc_float_0_1 - blurred
            image_after_postproc_float_0_1 = (
                image_after_postproc_float_0_1 + detail * PIXEL_SHARPEN_STRENGTH
            )
            image_after_postproc_float_0_1 = image_after_postproc_float_0_1.clamp(
                0.0, 1.0
            )
        # --- END IMPROVEMENT A ---

        final_image_uint8 = (image_after_postproc_float_0_1 * 255.0).byte()

        if h_proc != h_input or w_proc != w_input:
            # OPTIMIZED: Functional resize avoids slow class instantiation
            output_image_cxhxw_uint8 = v2.functional.resize(
                final_image_uint8,
                [h_input, w_input],
                interpolation=v2.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            output_image_cxhxw_uint8 = final_image_uint8

        # VRAM Leak Prevention: Explicitly delete the massive cloned K/V map
        # and other large tensors to free GPU memory before the GC cycle.
        if kv_tensor_map_for_this_run is not None:
            del kv_tensor_map_for_this_run

        del image_after_postproc_float_0_1
        del final_image_uint8

        return output_image_cxhxw_uint8
