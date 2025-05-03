import json
import math
import os
import glob
import shutil
import sys
from datetime import datetime, timezone

import torch
from pytorch_lightning import seed_everything

class JoePennaDreamboothConfigSchemaV1:
    def __init__(self, schema=1):
        super().__init__()
        self.schema = schema

    def saturate(
        self,
        project_name: str,
        max_training_steps: int,
        save_every_x_steps: int,
        training_images_folder_path: str,
        regularization_images_folder_path: str,
        token: str,
        class_word: str,
        mirror_prob: float,
        learning_rate: float,
        model_path: str,
        batch_size: int,
        num_workers: int,
        repeats: int,
        val_repeats: int,
        resolution: int,
        resampler: str,
        seed: int,
        debug: bool,
        gpu: int,
        token_only: bool,
        center_crop: bool,
        test: str,
        accum_num_grads: int,
        model_repo_id: str=None,
        run_seed_everything: bool=True
    ):
        self.project_name = project_name
        self.date_time_config = datetime.now(timezone.utc).strftime("%m-%d--%H-%M")
        self.project_config_filename = f"{self.date_time_config}-{self.project_name}-config.json"
        self.debug = debug
        self.gpu = gpu
        self.seed = seed
        self.save_every_x_steps = save_every_x_steps
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.repeats = repeats
        self.val_repeats = val_repeats
        self.resolution = resolution
        self.resampler = resampler
        self.center_crop = center_crop
        self.test = test
        self.learning_rate = learning_rate
        self.token = token
        self.token_only = token_only
        self.accum_num_grads = accum_num_grads


        if os.path.exists(training_images_folder_path):
            self.training_images_folder_path = os.path.relpath(training_images_folder_path)
        else:
            raise Exception(f"Training Images Path Not Found: '{os.path.relpath(self.training_images_folder_path)}'.")

        seed_everything(self.seed)

        _training_images_paths = [i for i in
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.jpeg'), recursive=True) +
            glob.glob(os.path.join(self.training_images_folder_path, '**', '*.png'), recursive=True)
        ]
        _training_images_paths = [os.path.relpath(i, self.training_images_folder_path) for i in _training_images_paths]

        if len(_training_images_paths) <= 0:
            raise Exception(f"No Training Images (*.png, *.jpg, *.jpeg) found in '{self.training_images_folder_path}'.")

        if max_training_steps is not None:
            self.max_training_steps = max_training_steps
        else:
             self.max_training_steps = len(_training_images_paths) * self.repeats

        if self.token_only is False:
            self.class_word = class_word
            if regularization_images_folder_path and os.path.exists(regularization_images_folder_path):
                self.regularization_images_folder_path = os.path.relpath(regularization_images_folder_path)
            else:
                raise Exception(f"Regularization Images Path Not Found: '{os.path.relpath(self.regularization_images_folder_path)}'.")

        if model_path.endswith('.ckpt') and os.path.exists(model_path):
            self.model_path = os.path.relpath(model_path)
        else:
            from huggingface_hub import hf_hub_download
            import joblib

            model_path = model_path.replace('/', '.')
            REPO_ID, FILENAME = os.path.splitext(model_path)
            self.model_path = joblib.load(hf_hub_download(repo_id=REPO_ID, filename=FILENAME))

        if mirror_prob > 0 and mirror_prob < 1:
            self.mirror_prob = mirror_prob
        else:
            raise Exception("--mirror_prob: must be between 0 and 1")

        self.validate_gpu_vram()
        self._create_log_folders()

    def validate_gpu_vram(self):
        def convert_size(size_bytes):
            if size_bytes == 0:
                return "0B"
            size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
            i = int(math.floor(math.log(size_bytes, 1024)))
            p = math.pow(1024, i)
            s = round(size_bytes / p, 2)
            return "%s %s" % (s, size_name[i])

            # Check total available GPU memory

        gpu_vram = int(torch.cuda.get_device_properties(self.gpu).total_memory)
        print(f"gpu_vram: {convert_size(gpu_vram)}")
        twenty_one_gigabytes = 22548578304
        if gpu_vram < twenty_one_gigabytes:
            raise Exception(f"VRAM: Currently unable to run on less than {convert_size(twenty_one_gigabytes)} of VRAM.")

    def saturate_from_file(
            self,
            config_file_path: str,
    ):
        if not os.path.exists(config_file_path):
            print(f"{config_file_path} not found.", file=sys.stderr)
            return None
        else:
            config_file = open(config_file_path)
            config_parsed = json.load(config_file)

            if config_parsed['schema'] == 1:
                self.saturate(
                    project_name=config_parsed['project_name'],
                    max_training_steps=config_parsed['max_training_steps'],
                    save_every_x_steps=config_parsed['save_every_x_steps'],
                    training_images_folder_path=config_parsed['training_images_folder_path'],
                    regularization_images_folder_path=config_parsed['regularization_images_folder_path'],
                    token=config_parsed['token'],
                    class_word=config_parsed['class_word'],
                    flip_percent=config_parsed['flip_percent'],
                    learning_rate=config_parsed['learning_rate'],
                    model_path=config_parsed['model_path'],
                    config_date_time=config_parsed['config_date_time'],
                    seed=config_parsed['seed'],
                    debug=config_parsed['debug'],
                    gpu=config_parsed['gpu'],
                    model_repo_id=config_parsed['model_repo_id'],
                    token_only=config_parsed['token_only'],
                )
            else:
                print(f"Unrecognized schema: {config_parsed['schema']}", file=sys.stderr)

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def create_checkpoint_file_name(self, steps: str):
        date_string = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
        return f"{date_string}_{self.project_name}_" \
               f"{int(steps):05d}_steps_" \
               f"{self.training_images_count}_training_images_" \
               f"{self.token}_token_" \
               f"{self.class_word}_class_word.ckpt".replace(" ", "_")

    def save_config_to_file(
            self,
            save_path: str,
            create_active_config: bool = False,
    ):
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        project_config_json = self.toJSON()
        config_save_path = os.path.join(save_path, self.project_config_filename)
        with open(config_save_path, "w") as config_file:
            config_file.write(project_config_json)

        if create_active_config:
            shutil.copy(config_save_path, os.path.join(save_path, "active-config.json"))
            print(project_config_json)
            print(f"✅ {self.project_config_filename} successfully generated.  Proceed to training.")

    def get_training_folder_name(self) -> str:
        return f"{self.config_date_time}_{self.project_name}"

    def log_directory(self) -> str:
        return os.path.join("logs", self.get_training_folder_name())

    def log_checkpoint_directory(self) -> str:
        return os.path.join(self.log_directory(), "ckpts")

    def log_intermediate_checkpoints_directory(self) -> str:
        return os.path.join(self.log_checkpoint_directory(), "trainstep_ckpts")

    def log_config_directory(self) -> str:
        return os.path.join(self.log_directory(), "configs")

    def trained_models_directory(self) -> str:
        return "trained_models"

    def _create_log_folders(self):
        os.makedirs(self.log_directory(), exist_ok=True)
        os.makedirs(self.log_checkpoint_directory(), exist_ok=True)
        os.makedirs(self.log_config_directory(), exist_ok=True)
        os.makedirs(self.trained_models_directory(), exist_ok=True)
