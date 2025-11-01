import copy
import io
from pathlib import Path
import folder_paths
import zipfile
import json
import time
import os

from .utils import *

TOOLTIP_LIMIT_OPUS_FREE = "Limit image size and steps for free generation by Opus."


class PromptToNAID:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "forceInput": True,
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "weight_per_brace": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.05, "max": 0.10, "step": 0.05},
                ),
                "syntax_mode": (["brace", "numeric"], {"default": "brace"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"

    def convert(self, text, weight_per_brace, syntax_mode):
        nai_prompt = prompt_to_nai(text, weight_per_brace, syntax_mode)
        return (nai_prompt,)


class ImageToNAIMask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/utils"

    def convert(self, image):
        s = resize_to_naimask(image)
        return (s,)


class ModelOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    [
                        "nai-diffusion-2",
                        "nai-diffusion-furry-3",
                        "nai-diffusion-3",
                        "nai-diffusion-4-curated-preview",
                        "nai-diffusion-4-full",
                        "nai-diffusion-4-5-curated",
                        "nai-diffusion-4-5-full",
                    ],
                    {"default": "nai-diffusion-4-5-full"},
                ),
            },
            "optional": {"option": ("NAID_OPTION",)},
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"

    def set_option(self, model, option=None):
        option = copy.deepcopy(option) if option else {}
        option["model"] = model
        return (option,)


class Img2ImgOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.70,
                        "min": 0.01,
                        "max": 0.99,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "noise": (
                    "FLOAT",
                    {
                        "default": 0.00,
                        "min": 0.00,
                        "max": 0.99,
                        "step": 0.02,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"

    def set_option(self, image, strength, noise):
        option = {}
        option["img2img"] = (image, strength, noise)
        return (option,)


class InpaintingOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("IMAGE",),
                "add_original_image": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"

    def set_option(self, image, mask, add_original_image):
        option = {}
        option["infill"] = (image, mask, add_original_image)
        return (option,)


class VibeTransferOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "information_extracted": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {
                        "default": 0.6,
                        "min": 0.01,
                        "max": 1.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
            },
            "optional": {"option": ("NAID_OPTION",)},
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"

    def set_option(self, image, information_extracted, strength, option=None):
        option = copy.deepcopy(option) if option else {}
        if "vibe" not in option:
            option["vibe"] = []

        option["vibe"].append((image, information_extracted, strength))
        return (option,)


class NetworkOption:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ignore_errors": ("BOOLEAN", {"default": True}),
                "timeout_sec": (
                    "INT",
                    {
                        "default": 120,
                        "min": 30,
                        "max": 3000,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "retry": (
                    "INT",
                    {
                        "default": 3,
                        "min": 1,
                        "max": 100,
                        "step": 1,
                        "display": "number",
                    },
                ),
            },
            "optional": {"option": ("NAID_OPTION",)},
        }

    RETURN_TYPES = ("NAID_OPTION",)
    FUNCTION = "set_option"
    CATEGORY = "NovelAI"

    def set_option(self, ignore_errors, timeout_sec, retry, option=None):
        option = copy.deepcopy(option) if option else {}
        option["ignore_errors"] = ignore_errors
        option["timeout"] = timeout_sec
        option["retry"] = retry
        return (option,)


class GenerateNAID:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 64,
                        "max": 1600,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1216,
                        "min": 64,
                        "max": 1600,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "positive": (
                    "STRING",
                    {
                        "default": ", best quality, amazing quality, very aesthetic, absurdres",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "negative": (
                    "STRING",
                    {"default": "lowres", "multiline": True, "dynamicPrompts": False},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 28,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                "variety": ("BOOLEAN", {"default": False}),
                "decrisper": ("BOOLEAN", {"default": False}),
                "smea": (["none", "SMEA", "SMEA+DYN"], {"default": "none"}),
                "sampler": (
                    [
                        "k_euler",
                        "k_euler_ancestral",
                        "k_dpmpp_2s_ancestral",
                        "k_dpmpp_2m_sde",
                        "k_dpmpp_2m",
                        "k_dpmpp_sde",
                        "ddim",
                    ],
                    {"default": "k_euler"},
                ),
                "scheduler": (
                    ["native", "karras", "exponential", "polyexponential"],
                    {"default": "native"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999999999,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "uncond_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.5,
                        "step": 0.05,
                        "display": "number",
                    },
                ),
                "cfg_rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.02,
                        "display": "number",
                    },
                ),
                "keep_alpha": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable to further process output images locally",
                    },
                ),
            },
            "optional": {"option": ("NAID_OPTION",)},
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    def generate(
        self,
        limit_opus_free,
        width,
        height,
        positive,
        negative,
        steps,
        cfg,
        decrisper,
        variety,
        smea,
        sampler,
        scheduler,
        seed,
        uncond_scale,
        cfg_rescale,
        keep_alpha,
        option=None,
    ):
        width, height = calculate_resolution(width * height, (width, height))
        params = {
            "params_version": 1,
            "width": width,
            "height": height,
            "scale": cfg,
            "sampler": sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": 1,
            "ucPreset": 3,
            "qualityToggle": False,
            "sm": (smea == "SMEA" or smea == "SMEA+DYN") and sampler != "ddim",
            "sm_dyn": smea == "SMEA+DYN" and sampler != "ddim",
            "dynamic_thresholding": decrisper,
            "skip_cfg_above_sigma": None,
            "controlnet_strength": 1.0,
            "legacy": False,
            "add_original_image": False,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": scheduler,
            "legacy_v3_extend": False,
            "uncond_scale": uncond_scale,
            "negative_prompt": negative,
            "prompt": positive,
            "reference_image_multiple": [],
            "reference_information_extracted_multiple": [],
            "reference_strength_multiple": [],
            "extra_noise_seed": seed,
            "v4_prompt": {
                "use_coords": False,
                "use_order": False,
                "caption": {"base_caption": positive, "char_captions": []},
            },
            "v4_negative_prompt": {
                "use_coords": False,
                "use_order": False,
                "caption": {"base_caption": negative, "char_captions": []},
            },
        }
        model = "nai-diffusion-4-5-full"
        action = "generate"

        if sampler == "k_euler_ancestral" and scheduler != "native":
            params["deliberate_euler_ancestral_bug"] = False
            params["prefer_brownian"] = True

        if option:
            if "img2img" in option:
                action = "img2img"
                image, strength, noise = option["img2img"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["strength"] = strength
                params["noise"] = noise
            elif "infill" in option:
                action = "infill"
                image, mask, add_original_image = option["infill"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["mask"] = naimask_to_base64(
                    resize_to_naimask(mask, (width, height), "4" in model)
                )
                params["add_original_image"] = add_original_image

            if "vibe" in option:
                for vibe in option["vibe"]:
                    image, information_extracted, strength = vibe
                    params["reference_image_multiple"].append(
                        image_to_base64(resize_image(image, (width, height)))
                    )
                    params["reference_information_extracted_multiple"].append(
                        information_extracted
                    )
                    params["reference_strength_multiple"].append(strength)

            if "model" in option:
                model = option["model"]
            if "v4_prompt" in option:
                params["v4_prompt"].update(option["v4_prompt"])

        timeout = option["timeout"] if option and "timeout" in option else None
        retry = option["retry"] if option and "retry" in option else None

        if limit_opus_free:
            pixel_limit = 1024 * 1024
            if width * height > pixel_limit:
                max_width, max_height = calculate_resolution(
                    pixel_limit, (width, height)
                )
                params["width"] = max_width
                params["height"] = max_height
            if steps > 28:
                params["steps"] = 28

        if variety:
            params["skip_cfg_above_sigma"] = calculate_skip_cfg_above_sigma(
                params["width"], params["height"]
            )

        if sampler == "ddim" and model not in ("nai-diffusion-2"):
            params["sampler"] = "ddim_v3"

        if action == "infill" and model not in ("nai-diffusion-2"):
            model = f"{model}-inpainting"

        image = blank_image()
        try:
            zipped_bytes = generate_image(
                self.access_token, positive, model, action, params, timeout, retry
            )
            zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
            image_bytes = zipped.read(zipped.infolist()[0])
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path("NAI_autosave", self.output_dir)
            )
            file = f"{filename}_{counter:05}_.png"
            d = Path(full_output_folder)
            d.mkdir(exist_ok=True)
            (d / file).write_bytes(image_bytes)
            image = bytes_to_image(image_bytes, keep_alpha)
        except Exception as e:
            if "ignore_errors" in option and option["ignore_errors"]:
                print("ignore error:", e)
            else:
                raise e
        return (image,)


# ## NODE: GenerateNAID_V4Advanced ##
class GenerateNAID_V4Advanced:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()
        self.output_dir = os.path.join(self.output_dir, "NAI_autosaves")
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        char_inputs = {}
        for i in range(1, 7):
            char_inputs[f"char_prompt_{i}"] = (
                "STRING",
                {"multiline": True, "default": "", "rows": 6},
            )
            char_inputs[f"char_uc_{i}"] = (
                "STRING",
                {
                    "multiline": True,
                    "default": "",
                    "rows": 6,
                    "tooltip": "Character-specific negative prompt",
                },
            )
            char_inputs[f"char_x_{i}"] = (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            )
            char_inputs[f"char_y_{i}"] = (
                "FLOAT",
                {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
            )

        return {
            "required": {
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "width": (
                    "INT",
                    {
                        "default": 832,
                        "min": 64,
                        "max": 1600,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 1216,
                        "min": 64,
                        "max": 1600,
                        "step": 64,
                        "display": "number",
                    },
                ),
                "positive": (
                    "STRING",
                    {
                        "default": ", best quality, amazing quality, very aesthetic, absurdres",
                        "multiline": True,
                        "rows": 12,
                    },
                ),
                "negative": (
                    "STRING",
                    {"default": "lowres", "multiline": True, "rows": 12},
                ),
                "steps": (
                    "INT",
                    {
                        "default": 28,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 5.0,
                        "min": 0.0,
                        "max": 10.0,
                        "step": 0.1,
                        "display": "number",
                    },
                ),
                # ### MODIFIED SECTION START ###
                "ucPreset": (
                    ["Heavy", "Light", "Furry Focus", "Human Focus", "None"],
                    {"default": "Light"},
                ),
                # ### MODIFIED SECTION END ###
                "sampler": (
                    [
                        "k_euler",
                        "k_euler_ancestral",
                        "k_dpmpp_2s_ancestral",
                        "k_dpmpp_2m_sde",
                        "k_dpmpp_2m",
                        "k_dpmpp_sde",
                        "ddim",
                    ],
                    {"default": "k_euler_ancestral"},
                ),
                "scheduler": (
                    ["native", "karras", "exponential", "polyexponential"],
                    {"default": "karras"},
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 9999999999,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "cfg_rescale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.02,
                        "display": "number",
                    },
                ),
                "smea": (["none", "SMEA", "SMEA+DYN"], {"default": "none"}),
                "decrisper": ("BOOLEAN", {"default": False}),
                "variety": ("BOOLEAN", {"default": False}),
                "wait_time": (  # ### NEW PARAMETER START ###
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 300,
                        "step": 1,
                        "display": "number",
                        "tooltip": "在调用API开始生图前等待指定的秒数。",
                    },
                ),  # ### NEW PARAMETER END ###
                "keep_alpha": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Disable to further process output images locally",
                    },
                ),
            },
            "optional": {
                "option": ("NAID_OPTION",),
                # REMOVED: The confusing v4_config toggle has been removed. The section is now always visible.
                "use_coords": ("BOOLEAN", {"default": False}),
                "use_order": ("BOOLEAN", {"default": True}),
                **char_inputs,
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "api_payload")
    FUNCTION = "generate"
    CATEGORY = "NovelAI"

    def generate(
        self,
        limit_opus_free,
        width,
        height,
        positive,
        negative,
        steps,
        cfg,
        ucPreset,
        sampler,
        scheduler,
        seed,
        cfg_rescale,
        smea,
        decrisper,
        variety,
        keep_alpha,
        wait_time,  # <-- New parameter added
        option=None,
        **kwargs,
    ):
        width, height = calculate_resolution(width * height, (width, height))

        uc_preset_map = {"Heavy": 0, "Light": 1, "Furry Focus": 2, "Human Focus": 3, "None": 4}
        uc_preset_value = uc_preset_map.get(ucPreset, 1)

        char_captions_positive = []
        char_captions_negative = []
        character_prompts_list = []

        use_coords = kwargs.get("use_coords", True)
        use_order = kwargs.get("use_order", True)

        for i in range(1, 7):
            prompt = kwargs.get(f"char_prompt_{i}", "")
            uc = kwargs.get(f"char_uc_{i}", "")
            x = kwargs.get(f"char_x_{i}", 0.5)
            y = kwargs.get(f"char_y_{i}", 0.5)

            if prompt and prompt.strip():
                center_obj = {"x": x, "y": y}
                centers_list_obj = {"centers": [center_obj]}
                char_captions_positive.append(
                    {"char_caption": prompt, **centers_list_obj}
                )
                char_captions_negative.append({"char_caption": uc, **centers_list_obj})
                character_prompts_list.append(
                    {"prompt": prompt, "uc": uc, "center": center_obj, "enabled": True}
                )

        params = {
            "params_version": 3,
            "width": width,
            "height": height,
            "scale": cfg,
            "sampler": sampler,
            "steps": steps,
            "seed": seed,
            "n_samples": 1,
            "ucPreset": uc_preset_value,
            "qualityToggle": False,
            "dynamic_thresholding": decrisper,
            "controlnet_strength": 1.0,
            "legacy": False,
            "add_original_image": True,
            "cfg_rescale": cfg_rescale,
            "noise_schedule": scheduler,
            "legacy_v3_extend": False,
            "negative_prompt": negative,
            "prompt": positive,
            "characterPrompts": character_prompts_list,
            "v4_prompt": {
                "use_coords": use_coords,
                "use_order": use_order,
                "caption": {
                    "base_caption": positive,
                    "char_captions": char_captions_positive,
                },
            },
            "v4_negative_prompt": {
                "use_coords": use_coords,
                "use_order": use_order,
                "caption": {
                    "base_caption": negative,
                    "char_captions": char_captions_negative,
                },
            },
        }

        model = "nai-diffusion-4-5-full"
        action = "generate"

        if sampler == "k_euler_ancestral" and scheduler != "native":
            params["deliberate_euler_ancestral_bug"] = False
            params["prefer_brownian"] = True

        if option:
            if "img2img" in option:
                action = "img2img"
                image, strength, noise = option["img2img"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["strength"] = strength
                params["noise"] = noise
            elif "infill" in option:
                action = "infill"
                image, mask, add_original_image = option["infill"]
                params["image"] = image_to_base64(resize_image(image, (width, height)))
                params["mask"] = naimask_to_base64(
                    resize_to_naimask(mask, (width, height), "4" in model)
                )
                params["add_original_image"] = add_original_image

            if "vibe" in option:
                print(
                    "Warning: Vibe Transfer is not officially supported with V4 models and may not work as expected."
                )
                params["reference_image_multiple"] = []
                params["reference_information_extracted_multiple"] = []
                params["reference_strength_multiple"] = []
                for vibe in option["vibe"]:
                    image, information_extracted, strength = vibe
                    params["reference_image_multiple"].append(
                        image_to_base64(resize_image(image, (width, height)))
                    )
                    params["reference_information_extracted_multiple"].append(
                        information_extracted
                    )
                    params["reference_strength_multiple"].append(strength)

            if "model" in option:
                model = option["model"]

        timeout = option["timeout"] if option and "timeout" in option else 120
        retry = option["retry"] if option and "retry" in option else 3

        if limit_opus_free:
            pixel_limit = 1024 * 1024
            if width * height > pixel_limit:
                max_width, max_height = calculate_resolution(
                    pixel_limit, (width, height)
                )
                params["width"] = max_width
                params["height"] = max_height
            if steps > 28:
                params["steps"] = 28

        if variety:
            params["skip_cfg_above_sigma"] = calculate_skip_cfg_above_sigma(
                params["width"], params["height"]
            )

        if sampler == "ddim" and model not in ("nai-diffusion-2"):
            params["sampler"] = "ddim_v3"

        payload_string = json.dumps(params, indent=4)
        image = blank_image()
        try:
            if wait_time > 0:
                print(f"ComfyUI_NAIDGenerator: Waiting for {wait_time} seconds before generation...")
                time.sleep(wait_time)

            zipped_bytes = generate_image(
                self.access_token, positive, model, action, params, timeout, retry
            )
            zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
            image_bytes = zipped.read(zipped.infolist()[0])
            full_output_folder, filename, counter, subfolder, filename_prefix = (
                folder_paths.get_save_image_path("NAI_autosave", self.output_dir)
            )
            file = f"{filename}_{counter:05}_.png"
            d = Path(full_output_folder)
            d.mkdir(exist_ok=True)
            (d / file).write_bytes(image_bytes)
            image = bytes_to_image(image_bytes, keep_alpha)
        except Exception as e:
            if "ignore_errors" in option and option["ignore_errors"]:
                print("ignore error:", e)
            else:
                raise e
        return (image, payload_string)


def base_augment(
    access_token,
    output_dir,
    limit_opus_free,
    ignore_errors,
    req_type,
    image,
    options=None,
):
    image = image.movedim(-1, 1)
    w, h = (image.shape[3], image.shape[2])
    image = image.movedim(1, -1)

    if limit_opus_free:
        pixel_limit = 1024 * 1024
        if w * h > pixel_limit:
            w, h = calculate_resolution(pixel_limit, (w, h))
    base64_image = image_to_base64(resize_image(image, (w, h)))
    result_image = blank_image()
    try:
        request = {"image": base64_image, "req_type": req_type, "width": w, "height": h}
        if options:
            if "defry" in options:
                request["defry"] = options["defry"]
            if "prompt" in options:
                request["prompt"] = options["prompt"]

        zipped_bytes = augment_image(
            access_token, req_type, w, h, base64_image, options=options
        )
        zipped = zipfile.ZipFile(io.BytesIO(zipped_bytes))
        image_bytes = zipped.read(zipped.infolist()[0])
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path("NAI_autosave", output_dir)
        )
        file = f"{filename}_{counter:05}_.png"
        d = Path(full_output_folder)
        d.mkdir(exist_ok=True)
        (d / file).write_bytes(image_bytes)
        result_image = bytes_to_image(image_bytes)
    except Exception as e:
        if ignore_errors:
            print("ignore error:", e)
        else:
            raise e
    return (result_image,)


class RemoveBGAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "ignore_errors": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"

    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(
            self.access_token,
            self.output_dir,
            limit_opus_free,
            ignore_errors,
            "bg-removal",
            image,
        )


class LineArtAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "ignore_errors": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"

    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(
            self.access_token,
            self.output_dir,
            limit_opus_free,
            ignore_errors,
            "lineart",
            image,
        )


class SketchAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "ignore_errors": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"

    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(
            self.access_token,
            self.output_dir,
            limit_opus_free,
            ignore_errors,
            "sketch",
            image,
        )


class ColorizeAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "ignore_errors": ("BOOLEAN", {"default": False}),
                "defry": (
                    "INT",
                    {"default": 0, "min": 0, "max": 5, "step": 1, "display": "number"},
                ),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"

    def augment(self, image, limit_opus_free, ignore_errors, defry, prompt):
        return base_augment(
            self.access_token,
            self.output_dir,
            limit_opus_free,
            ignore_errors,
            "colorize",
            image,
            options={"defry": defry, "prompt": prompt},
        )


class EmotionAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    strength_list = [
        "normal",
        "slightly_weak",
        "weak",
        "even_weaker",
        "very_weak",
        "weakest",
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "ignore_errors": ("BOOLEAN", {"default": False}),
                "mood": (
                    [
                        "neutral",
                        "happy",
                        "sad",
                        "angry",
                        "scared",
                        "surprised",
                        "tired",
                        "excited",
                        "nervous",
                        "thinking",
                        "confused",
                        "shy",
                        "disgusted",
                        "smug",
                        "bored",
                        "laughing",
                        "irritated",
                        "aroused",
                        "embarrassed",
                        "worried",
                        "love",
                        "determined",
                        "hurt",
                        "playful",
                    ],
                    {"default": "neutral"},
                ),
                "strength": (s.strength_list, {"default": "normal"}),
                "prompt": (
                    "STRING",
                    {"default": "", "multiline": True, "dynamicPrompts": False},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"

    def augment(self, image, limit_opus_free, ignore_errors, mood, strength, prompt):
        prompt = f"{mood};;{prompt}"
        defry = EmotionAugment.strength_list.index(strength)
        return base_augment(
            self.access_token,
            self.output_dir,
            limit_opus_free,
            ignore_errors,
            "emotion",
            image,
            options={"defry": defry, "prompt": prompt},
        )


class DeclutterAugment:
    def __init__(self):
        self.access_token = get_access_token()
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "limit_opus_free": (
                    "BOOLEAN",
                    {"default": True, "tooltip": TOOLTIP_LIMIT_OPUS_FREE},
                ),
                "ignore_errors": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "augment"
    CATEGORY = "NovelAI/director_tools"

    def augment(self, image, limit_opus_free, ignore_errors):
        return base_augment(
            self.access_token,
            self.output_dir,
            limit_opus_free,
            ignore_errors,
            "declutter",
            image,
        )


class V4BasePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_caption": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/v4"

    def convert(self, base_caption):
        return (base_caption,)


class V4NegativePrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "negative_caption": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "NovelAI/v4"

    def convert(self, negative_caption):
        return (negative_caption,)


NODE_CLASS_MAPPINGS = {
    "GenerateNAID": GenerateNAID,
    "GenerateNAID_V4Advanced": GenerateNAID_V4Advanced,
    "ModelOptionNAID": ModelOption,
    "Img2ImgOptionNAID": Img2ImgOption,
    "InpaintingOptionNAID": InpaintingOption,
    "VibeTransferOptionNAID": VibeTransferOption,
    "NetworkOptionNAID": NetworkOption,
    "MaskImageToNAIMask": ImageToNAIMask,
    "PromptToNAID": PromptToNAID,
    "RemoveBGNAID": RemoveBGAugment,
    "LineArtNAID": LineArtAugment,
    "SketchNAID": SketchAugment,
    "ColorizeNAID": ColorizeAugment,
    "EmotionNAID": EmotionAugment,
    "DeclutterNAID": DeclutterAugment,
    "V4BasePrompt": V4BasePrompt,
    "V4NegativePrompt": V4NegativePrompt,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenerateNAID": "Generate ✒️🅝🅐🅘",
    "GenerateNAID_V4Advanced": "Generate (V4.5 Advanced) ✒️🅝🅐🅘",
    "ModelOptionNAID": "ModelOption ✒️🅝🅐🅘",
    "Img2ImgOptionNAID": "Img2ImgOption ✒️🅝🅐🅘",
    "InpaintingOptionNAID": "InpaintingOption ✒️🅝🅐🅘",
    "VibeTransferOptionNAID": "VibeTransferOption ✒️🅝🅐🅘",
    "NetworkOptionNAID": "NetworkOption ✒️🅝🅐🅘",
    "MaskImageToNAID": "Convert Mask Image ✒️🅝🅐🅘",
    "PromptToNAID": "Convert Prompt ✒️🅝🅐🅘",
    "RemoveBGNAID": "Remove BG ✒️🅝🅐🅘",
    "LineArtNAID": "LineArt ✒️🅝🅐🅘",
    "SketchNAID": "Sketch ✒️🅝🅐🅘",
    "ColorizeNAID": "Colorize ✒️🅝🅐🅘",
    "EmotionNAID": "Emotion ✒️🅝🅐🅘",
    "DeclutterNAID": "Declutter ✒️🅝🅐🅘",
    "V4BasePrompt": "V4 Base Prompt ✒️🅝🅐🅘",
    "V4NegativePrompt": "V4 Negative Prompt ✒️🅝🅐🅘",
}