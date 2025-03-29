import torch
import inspect
import torch.nn.functional as F
import random
import logging
import inspect
import os
from torch.distributions import Beta
from helpers.training.wrappers import unwrap_model
from helpers.training.multi_process import _get_rank
from helpers.training.custom_schedule import apply_flow_schedule_shift, generate_timestep_weights, segmented_timestep_selection
from helpers.training.min_snr_gamma import compute_snr
from transformers.utils import ContextManagers
from helpers.training.deepspeed import (
    deepspeed_zero_init_disabled_context_manager,
    prepare_model_for_deepspeed,
)
from abc import ABC, abstractmethod
from enum import Enum

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)

flow_matching_model_families = ["flux", "sana", "ltxvideo", "wan", "sd3"]
upstream_config_sources = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "kolors": "stabilityai/stable-diffusion-xl-base-1.0",
    "sd3": "stabilityai/stable-diffusion-3-large",
    "sana": "terminusresearch/sana-1.6b-1024px",
    "flux": "black-forest-labs/flux.1-dev",
    "legacy": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "ltxvideo": "Lightricks/LTX-Video",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}

def get_model_config_path(model_family: str, model_path: str):
    if model_path.endswith(".safetensors"):
        if model_family in upstream_config_sources:
            return upstream_config_sources[model_family]
        else:
            raise ValueError(
                "Cannot find noise schedule config for .safetensors file in architecture {}".format(
                    model_family
                )
            )

    return model_path

class PipelineTypes(Enum):
    IMG2IMG = "img2img"
    TEXT2IMG = "text2img"

class PredictionTypes(Enum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"
    FLOW_MATCHING = "flow_matching"

class ModelFoundation(ABC):
    """
    Base class that contains all the universal logic:
      - Noise schedule, prediction target (epsilon, sample, v_prediction, flow-matching)
      - Batch preparation (moving to device, sampling noise, etc.)
      - Loss calculation (including optional SNR weighting)
    """
    def __init__(self, config: dict, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.noise_schedule = None
        self.setup_noise_schedule()

    @abstractmethod
    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        """
        Run a forward pass on the model.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("model_predict must be implemented in the child class.")

    @abstractmethod
    def _encode_prompts(self, text_batch: list):
        """
        Encodes a batch of text using the text encoder.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("_encode_prompts must be implemented in the child class.")


    @abstractmethod
    def load_lora_weights(self, models, input_dir):
        """
        Loads the LoRA weights.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("load_lora_weights must be implemented in the child class.")

    def save_lora_weights(self, *args, **kwargs):
        self.PIPELINE_CLASS.save_lora_weights(*args, **kwargs)

    def _model_config_path(self):
        return get_model_config_path(
            model_family=self.config.model_family,
            model_path=self.config.pretrained_model_name_or_path,
        )

    def unwrap_model(self, model = None):
        return unwrap_model(self.accelerator, model or self.model)

    def get_vae(self):
        """
        Returns the VAE model.
        """
        if not getattr(self, 'AUTOENCODER_CLASS', None):
            return
        if not hasattr(self, "vae") or self.vae is None:
            self.load_vae()
        return self.vae

    def load_vae(self, move_to_device: bool = True):
        if not getattr(self, 'AUTOENCODER_CLASS', None):
            return

        logger.info(f"Loading VAE from {self.config.vae_path}")
        self.vae = None
        self.config.vae_kwargs = {
            "pretrained_model_name_or_path": get_model_config_path(self.config.model_family, self.config.vae_path),
            "subfolder": "vae",
            "revision": self.config.revision,
            "force_upcast": False,
            "variant": self.config.variant,
        }
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            try:
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(**self.config.vae_kwargs)
            except Exception as e:
                logger.warning(
                    "Couldn't load VAE with default path. Trying without a subfolder.."
                )
                logger.error(e)
                self.config.vae_kwargs["subfolder"] = None
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(**self.config.vae_kwargs)
        if self.vae is None:
            raise ValueError(
                "Could not load VAE. Please check the model path and ensure the VAE is compatible."
            )
        if self.config.vae_enable_tiling:
            if hasattr(self.vae, "enable_tiling"):
                logger.info("Enabling VAE tiling.")
                self.vae.enable_tiling()
            else:
                logger.warning(
                    f"VAE tiling is enabled, but not yet supported by {self.config.model_family}."
                )
        if self.config.vae_enable_slicing:
            if hasattr(self.vae, "enable_slicing"):
                logger.info("Enabling VAE slicing.")
                self.vae.enable_slicing()
            else:
                logger.warning(
                    f"VAE slicing is enabled, but not yet supported by {self.config.model_family}."
                )
        if move_to_device and self.vae.device != self.accelerator.device:
            # The VAE is in bfloat16 to avoid NaN losses.
            _vae_dtype = torch.bfloat16
            if hasattr(self.config, "vae_dtype"):
                # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
                if self.config.vae_dtype == "bf16":
                    _vae_dtype = torch.bfloat16
                elif self.config.vae_dtype == "fp16":
                    raise ValueError(
                        "fp16 is not supported for SDXL's VAE. Please use bf16 or fp32."
                    )
                elif self.config.vae_dtype == "fp32":
                    _vae_dtype = torch.float32
                elif (
                    self.config.vae_dtype == "none"
                    or self.config.vae_dtype == "default"
                ):
                    _vae_dtype = torch.bfloat16
            logger.info(
                f"Loading VAE onto accelerator, converting from {self.vae.dtype} to {_vae_dtype}"
            )
            self.vae.to(self.accelerator.device, dtype=_vae_dtype)

    def unload_vae(self):
        if self.vae is not None:
            if hasattr(self.vae, 'to'):
                self.vae.to('meta')
            self.vae = None

    def load_text_tokenizer(self):
        if self.TEXT_ENCODER_CONFIGURATION is None or len(self.TEXT_ENCODER_CONFIGURATION) == 0:
            return
        self.tokenizers = []
        tokenizer_kwargs = {
            "pretrained_model_name_or_path": get_model_config_path(
                self.config.model_family, self.config.pretrained_model_name_or_path
            ),
            "subfolder": "tokenizer",
            "revision": self.config.revision,
            "use_fast": False,
        }
        tokenizer_idx = 0
        for attr_name, text_encoder_config in self.TEXT_ENCODER_CONFIGURATION.items():
            tokenizer_idx += 1
            tokenizer_cls = text_encoder_config.get("tokenizer")
            tokenizer_kwargs["subfolder"] = text_encoder_config.get("tokenizer_subfolder", "tokenizer")
            tokenizer_kwargs["use_fast"] = text_encoder_config.get("use_fast", False)
            logger.info("Loading tokenizer %i: %s", tokenizer_idx, tokenizer_cls.__name__)
            tokenizer = tokenizer_cls.from_pretrained(**tokenizer_kwargs)          
            self.tokenizers.append(tokenizer)
            setattr(self, f"tokenizer_{tokenizer_idx}", tokenizer)

    def load_text_encoder(self, move_to_device: bool = True):
        self.text_encoders = []
        if self.TEXT_ENCODER_CONFIGURATION is None or len(self.TEXT_ENCODER_CONFIGURATION) == 0:
            return
        self.load_text_tokenizer()

        text_encoder_idx = 0
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            for attr_name, text_encoder_config in self.TEXT_ENCODER_CONFIGURATION.items():
                text_encoder_idx += 1
                # load_tes returns a variant and three text encoders
                signature = inspect.signature(text_encoder_config["model"])
                extra_kwargs = {}
                if 'torch_dtype' in signature.parameters:
                    extra_kwargs["torch_dtype"] = self.config.weight_dtype
                logger.info(f"Loading {text_encoder_config.get('name')} text encoder")
                text_encoder = text_encoder_config["model"].from_pretrained(
                    get_model_config_path(
                        self.config.model_family, self.config.pretrained_model_name_or_path
                    ),
                    variant=self.config.variant,
                    revision=self.config.revision,
                    subfolder=text_encoder_config.get("subfolder", "text_encoder"),
                    **extra_kwargs,
                )
                if move_to_device:
                    logger.info(f"Moving {text_encoder_config.get('name')} to GPU")
                    text_encoder.to(
                        self.accelerator.device,
                        dtype=self.config.weight_dtype,
                    )
                setattr(self, f"text_encoder_{text_encoder_idx}", text_encoder)
                self.text_encoders.append(text_encoder)

    def get_text_encoder(self, index: int):
        return self.text_encoders[index] if self.text_encoders is not None else None

    def unload_text_encoder(self):
        if self.text_encoders is not None:
            for text_encoder in self.text_encoders:
                if hasattr(text_encoder, 'to'):
                    text_encoder.to('meta')
            self.text_encoders = None
        if self.tokenizers is not None:
            self.tokenizers = None

    def load_model(self, move_to_device: bool = True):
        logger.info(f"Loading diffusion model from {self.config.pretrained_model_name_or_path}")
        # Stub: load your UNet (or transformer) model using your diffusion model loader.
        pretrained_load_args = {
            "revision": self.config.revision,
            "variant": self.config.variant,
            "torch_dtype": self.config.weight_dtype,
            "use_safetensors": True,
        }
        if "nf4-bnb" == self.config.base_model_precision:
            from diffusers import BitsAndBytesConfig
            pretrained_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.config.weight_dtype,
            )
        loader_fn = self.MODEL_CLASS.from_pretrained
        model_path = (
            self.config.pretrained_transformer_model_name_or_path
            or self.config.pretrained_model_name_or_path
        )
        if model_path.endswith('.safetensors'):
            loader_fn = self.MODEL_CLASS.from_single_file
        self.model = loader_fn(
            model_path,
            subfolder=self.MODEL_SUBFOLDER,
            **pretrained_load_args,
        )
        if move_to_device and self.model is not None:
            self.model.to(self.accelerator.device, dtype=self.config.weight_dtype)

    def set_prepared_model(self, model):
        # after accelerate prepare, we'll set the model again.
        self.model = model

    def freeze_components(self):
        # Freeze vae and text_encoders
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoders is not None and len(self.text_encoders) > 0:
            for text_encoder in self.text_encoders:
                text_encoder.requires_grad_(False)
        if "lora" in self.config.model_type:
            if self.model is not None:
                self.model.requires_grad_(False)

    def get_trained_component(self):
        return self.model

    def _load_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG):
        """
        Loads the pipeline class for the model.
        This is a stub and should be implemented in subclasses.
        """
        pipeline_kwargs = {
            "pretrained_model_name_or_path": self._model_config_path(),
        }
        if not hasattr(self, "PIPELINE_CLASSES"):
            raise NotImplementedError("Pipeline class not defined.")
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(
                f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}."
            )
        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]
        if not hasattr(pipeline_class, "from_pretrained"):
            raise NotImplementedError(f"Pipeline class {pipeline_class} does not have from_pretrained method.")
        signature = inspect.signature(pipeline_class.from_pretrained)
        if "transformer" in signature.parameters:
            pipeline_kwargs["transformer"] = unwrap_model(self.model)
        elif "unet" in signature.parameters:
            pipeline_kwargs["unet"] = unwrap_model(self.model)
        if "vae" in signature.parameters:
            pipeline_kwargs["vae"] = unwrap_model(self.vae)
        if "text_encoder" in signature.parameters:
            pipeline_kwargs["text_encoder"] = unwrap_model(self.text_encoders[0])
        if "text_encoder_2" in signature.parameters:
            pipeline_kwargs["text_encoder_2"] = unwrap_model(self.text_encoders[1])
        if "text_encoder_3" in signature.parameters:
            pipeline_kwargs["text_encoder_3"] = unwrap_model(self.text_encoders[2])
        if "controlnet" in signature.parameters and self.config.controlnet:
            pipeline_kwargs["controlnet"] = self.controlnet

        logger.info(f"Initialising pipeline with components: {pipeline_kwargs.keys()}")
        self.pipeline = pipeline_class.from_pretrained(
            **pipeline_kwargs,
        )

        return self.pipeline
    
    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG):
        if not hasattr(self, "pipeline") or self.pipeline is None:
            return self._load_pipeline(pipeline_type)
        return self.pipeline

    def setup_noise_schedule(self):
        """Loads the noise schedule from the config."""
        flow_matching = False
        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            from diffusers import FlowMatchEulerDiscreteScheduler

            self.noise_schedule = FlowMatchEulerDiscreteScheduler.from_pretrained(
                get_model_config_path(
                    self.config.model_family, self.config.pretrained_model_name_or_path
                ),
                subfolder="scheduler",
                shift=self.config.flow_schedule_shift,
            )
            flow_matching = True
        elif self.PREDICTION_TYPE in [PredictionTypes.EPSILON, PredictionTypes.V_PREDICTION, PredictionTypes.SAMPLE]:
            if self.config.model_family == "legacy":
                raise NotImplemented("Legacy models need ZSNR config moved out.")
                self.config.rescale_betas_zero_snr = True
                self.config.training_scheduler_timestep_spacing = "trailing"

            from diffusers import DDPMScheduler

            self.noise_schedule = DDPMScheduler.from_pretrained(
                get_model_config_path(
                    self.config.model_family, self.config.pretrained_model_name_or_path
                ),
                subfolder="scheduler",
                rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
                timestep_spacing=self.config.training_scheduler_timestep_spacing,
            )
            self.config.prediction_type = self.noise_schedule.config.prediction_type
        else:
            raise NotImplementedError(f"Unknown prediction type {self.PREDICTION_TYPE}.")

        return self.config, self.noise_schedule

    def get_prediction_target(self, prepared_batch: dict):
        """
        Returns the target used in the loss function.
        Depending on the noise schedule prediction type or flow-matching settings,
        the target is computed differently.
        """
        if self.config.flow_matching:
            if self.config.flow_matching_loss == "diffusers":
                target = prepared_batch["latents"]
            elif self.config.flow_matching_loss == "compatible":
                target = prepared_batch["noise"] - prepared_batch["latents"]
            elif self.config.flow_matching_loss == "sd35":
                sigma_reshaped = prepared_batch["sigmas"].view(-1, 1, 1, 1)
                target = (prepared_batch["noisy_latents"] - prepared_batch["latents"]) / sigma_reshaped
            else:
                target = prepared_batch["latents"]
        elif self.noise_scheduler.config.prediction_type == self.PREDICTION_TYPE_EPSILON:
            target = prepared_batch["noise"]
        elif self.noise_scheduler.config.prediction_type == self.PREDICTION_TYPE_V_PREDICTION:
            target = self.noise_scheduler.get_velocity(
                prepared_batch["latents"],
                prepared_batch["noise"],
                prepared_batch["timesteps"]
            )
        elif self.noise_schedule.config.prediction_type == self.PREDICTION_TYPE_SAMPLE:
            target = prepared_batch["latents"]
        else:
            raise ValueError(f"Unknown prediction type {self.noise_schedule.config.prediction_type}.")
        return target

    def prepare_batch(self, batch: dict) -> dict:
        """
        Moves the batch to the proper device/dtype,
        samples noise, timesteps and, if applicable, flow-matching sigmas.
        This code is mostly common across models.
        """
        if not batch:
            return batch

        target_device_kwargs = {
            "device": self.accelerator.device,
            "dtype": self.config.weight_dtype,
        }

        # Ensure the encoder hidden states are on device
        batch["encoder_hidden_states"] = batch["prompt_embeds"].to(**target_device_kwargs)

        # Process additional conditioning if provided
        pooled_embeds = batch.get("add_text_embeds")
        time_ids = batch.get("batch_time_ids")
        batch["added_cond_kwargs"] = {}
        if pooled_embeds is not None and hasattr(pooled_embeds, "to"):
            batch["added_cond_kwargs"]["text_embeds"] = pooled_embeds.to(**target_device_kwargs)
        if time_ids is not None and hasattr(time_ids, "to"):
            batch["added_cond_kwargs"]["time_ids"] = time_ids.to(**target_device_kwargs)

        # Process latents (assumed to be in 'latent_batch')
        latents = batch.get("latent_batch")
        if not hasattr(latents, "to"):
            raise ValueError("Received invalid value for latents.")
        batch["latents"] = latents.to(**target_device_kwargs)

        encoder_attention_mask = batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None and hasattr(encoder_attention_mask, "to"):
            batch["encoder_attention_mask"] = encoder_attention_mask.to(**target_device_kwargs)

        # Sample noise and add potential input perturbation.
        noise = torch.randn_like(batch["latents"])
        bsz = batch["latents"].shape[0]
        if self.config.input_perturbation != 0 and (
            not getattr(self.config, "input_perturbation_steps", None) or global_step < self.config.input_perturbation_steps
        ):
            input_perturbation = self.config.input_perturbation
            if getattr(self.config, "input_perturbation_steps", None):
                input_perturbation *= 1.0 - (global_step / self.config.input_perturbation_steps)
            batch["noise"] = noise + input_perturbation * torch.randn_like(batch["latents"])
        else:
            batch["noise"] = noise

        # Flow matching branch: set sigmas and timesteps.
        if self.config.flow_matching:
            if not self.config.flux_fast_schedule and not any(
                [self.config.flow_use_beta_schedule, self.config.flow_use_uniform_schedule]
            ):
                batch["sigmas"] = torch.sigmoid(
                    self.config.flow_sigmoid_scale * torch.randn((bsz,), device=self.accelerator.device)
                )
                batch["sigmas"] = apply_flow_schedule_shift(self.config, self.noise_scheduler, batch["sigmas"], batch["noise"])
            elif self.config.flow_use_uniform_schedule:
                batch["sigmas"] = torch.rand((bsz,), device=self.accelerator.device)
                batch["sigmas"] = apply_flow_schedule_shift(self.config, self.noise_scheduler, batch["sigmas"], batch["noise"])
            elif self.config.flow_use_beta_schedule:
                alpha = self.config.flow_beta_schedule_alpha
                beta = self.config.flow_beta_schedule_beta
                beta_dist = Beta(alpha, beta)
                batch["sigmas"] = beta_dist.sample((bsz,)).to(device=self.accelerator.device)
                batch["sigmas"] = apply_flow_schedule_shift(self.config, self.noise_scheduler, batch["sigmas"], noise)
            else:
                available_sigmas = [1.0] * 7 + [0.75, 0.5, 0.25]
                batch["sigmas"] = torch.tensor(random.choices(available_sigmas, k=bsz),
                                               device=self.accelerator.device)
            batch["timesteps"] = batch["sigmas"] * 1000.0
            # Ensure sigmas is reshaped appropriately (default is 4D, may be overriden in video subclass)
            batch["sigmas"] = batch["sigmas"].view(-1, 1, 1, 1)
        else:
            # If not flow matching, possibly apply an offset to noise
            if self.config.offset_noise:
                if self.config.noise_offset_probability == 1.0 or random.random() < self.config.noise_offset_probability:
                    noise = noise + self.config.noise_offset * torch.randn(
                        batch["latents"].shape[0], batch["latents"].shape[1], 1, 1, device=batch["latents"].device
                    )
            weights = generate_timestep_weights(self.config, self.noise_scheduler.config.num_train_timesteps).to(self.accelerator.device)
            if bsz > 1 and not self.config.disable_segmented_timestep_sampling:
                batch["timesteps"] = segmented_timestep_selection(
                    actual_num_timesteps=self.noise_scheduler.config.num_train_timesteps,
                    bsz=bsz,
                    weights=weights,
                    use_refiner_range=False  # You can override in subclass if needed.
                ).to(self.accelerator.device)
            else:
                batch["timesteps"] = torch.multinomial(weights, bsz, replacement=True).long()
            batch["noisy_latents"] = self.noise_scheduler.add_noise(
                batch["latents"].float(), batch["noise"].float(), batch["timesteps"]
            ).to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        return batch

    def encode_text_batch(self, text_batch: list):
        """
        Encodes a batch of text using the text encoder.
        """
        if not self.TEXT_ENCODER_CONFIGURATION:
            raise ValueError("No text encoder configuration found.")
        encoded_text = self._encode_prompts(text_batch)
        return self._format_text_embedding(encoded_text)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        Args:
            text_embedding (torch.Tensor): The embed to adjust.
        
        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        return text_embedding

    def loss(self, prepared_batch: dict, model_pred, target, apply_conditioning_mask: bool = True):
        """
        Computes the loss between the model prediction and the target.
        Optionally applies SNR weighting and a conditioning mask.
        """
        if self.PREDICTION_TYPE == PredictionTypes.FLOW_MATCHING:
            loss = (model_pred.float() - target.float()) ** 2
        elif self.PREDICTION_TYPE in [PredictionTypes.EPSILON, PredictionTypes.V_PREDICTION]:
            if self.config.snr_gamma is None or self.config.snr_gamma == 0:
                loss = self.config.snr_weight * F.mse_loss(model_pred.float(), target.float(), reduction="none")
            else:
                snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                snr_divisor = snr
                if self.noise_schedule.config.prediction_type == self.PREDICTION_TYPE_V_PREDICTION or (
                    self.config.flow_matching and self.config.flow_matching_loss == "diffusion"
                ):
                    snr_divisor = snr + 1
                mse_loss_weights = torch.stack([snr, self.config.snr_gamma * torch.ones_like(prepared_batch["timesteps"])], dim=1).min(dim=1)[0] / snr_divisor
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                loss = loss * mse_loss_weights
        else:
            raise NotImplementedError(f"Loss calculation not implemented for prediction type {self.PREDICTION_TYPE}.")

        conditioning_type = prepared_batch.get("conditioning_type")
        if conditioning_type == "mask" and apply_conditioning_mask:
            mask_image = prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)[:, 0].unsqueeze(1)
            mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
            mask_image = mask_image / 2 + 0.5
            loss = loss * mask_image

        # Average over channels and spatial dims, then over batch.
        loss = loss.mean(dim=list(range(1, len(loss.shape)))).mean()
        return loss


class ImageModelFoundation(ModelFoundation):
    """
    Implements logic common to image-based diffusion models.
    Handles typical VAE, text encoder loading and a UNet forward pass.
    """

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.has_vae = True
        self.has_text_encoder = True
        # These will be set by your loading methods
        self.vae = None
        self.model = None
        self.text_encoders = None
        self.tokenizers = None

class VideoModelFoundation(ImageModelFoundation):
    """
    Base class for video models. Provides default 5D handling and optional
    text encoder instantiation. The actual text encoder classes and their
    attributes can be stored in a hardcoded dict if needed. This base class
    does not do it by default.
    """

    def __init__(self, config, accelerator):
        """
        :param config: The training configuration object/dict.
        """
        super().__init__(config, accelerator)
        self.config = config
        # Optionally, store or initialize text encoders here.
        # But do NOT automatically do it unless your code requires it.

        # For example, if you have a dictionary of text-encoder descriptors:
        # self.text_encoders = {
        #     "encoder1": {"class": MyTextEncoderClass, "attr_name": "text_encoder_1"},
        #     "encoder2": {"class": AnotherTextEncoder, "attr_name": "text_encoder_2"},
        # }
        # The trainer or child class might call self._init_text_encoders() at the right time.

    def prepare_5d_inputs(self, tensor):
        """
        Example method to handle default 5D shape. The typical shape might be:
        (batch_size, frames, channels, height, width).

        You can reshape or permute as needed for the underlying model. 
        """
        # Pseudocode for typical flattening:
        # B, F, C, H, W = tensor.shape
        # return tensor.view(B * F, C, H, W)
        return tensor

    def get_pipeline_class(self):
        """
        By default, return None or a placeholder. 
        Child classes can override this if they have a specific pipeline to load.
        """
        return None

    def get_model_class(self):
        """
        By default, return None or a placeholder.
        Child classes can override this if they have a specific model class to instantiate.
        """
        return None

    # Example text encoder initialization if you do want to handle that automatically
    def _init_text_encoders(self):
        # for enc_name, enc_info in self.text_encoders.items():
        #     klass = enc_info["class"]
        #     attr_name = enc_info["attr_name"]
        #     setattr(self, attr_name, klass(self.config))
        pass

class WanVideo(VideoModelFoundation):
    """
    Wan-specific video foundation class for training. Inherits from the general
    VideoModelFoundation and overrides pipeline/model retrieval as needed.
    """
    TRANSFORMER_PATH = "Wan-AI/Wan2.1-T2V-1.3B"
    VAE_PATH = "Wan-AI/Wan2.1-T2V-1.3B"
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": "Wan-AI/Wan2.1-T2V-1.3B"
    }

    def get_pipeline_class(self):
        """
        Return the WAN pipeline class. If it needs to vary by config,
        do the logic here before returning.
        """
        # return WanPipeline
        return None  # placeholder if pipeline is truly optional

    def get_model_class(self):
        """
        Return the WAN model class. Again, if it depends on config details,
        handle that here.
        """
        # return WanTransformer3DModel
        return None  # placeholder

    def __call__(self, *args, **kwargs):
        """
        The forward pass (training-time) logic that was previously in model_predict.
        We want to keep inference-specific logic out of here.
        """
        # example:
        # 1. prepare inputs (handle 5D shape if needed)
        # 2. forward pass through model
        # 3. return training outputs (loss, logits, etc.)
        #
        # inputs_5d = self.prepare_5d_inputs(kwargs.get('video_tensor'))
        # outputs = self.model.forward(inputs_5d, ...)
        # return outputs
        
        pass




# pipeline_cls = self._pipeline_cls()
# extra_pipeline_kwargs = {
#     "text_encoder": self.text_encoder_1,
#     "tokenizer": self.tokenizer_1,
#     "vae": self.vae,
# }
# if self.args.model_family in ["legacy"]:
#     extra_pipeline_kwargs["safety_checker"] = None
# if self.args.model_family in ["sd3", "sdxl", "flux"]:
#     extra_pipeline_kwargs["text_encoder_2"] = None
# if self.args.model_family in ["sd3"]:
#     extra_pipeline_kwargs["text_encoder_3"] = None
# if type(pipeline_cls) is StableDiffusionXLPipeline:
#     del extra_pipeline_kwargs["text_encoder"]
#     del extra_pipeline_kwargs["tokenizer"]
#     if validation_type == "final":
#         if self.text_encoder_1 is not None:
#             extra_pipeline_kwargs["text_encoder_1"] = unwrap_model(
#                 self.accelerator, self.text_encoder_1
#             )
#             extra_pipeline_kwargs["tokenizer_1"] = self.tokenizer_1
#             if self.text_encoder_2 is not None:
#                 extra_pipeline_kwargs["text_encoder_2"] = unwrap_model(
#                     self.accelerator, self.text_encoder_2
#                 )
#                 extra_pipeline_kwargs["tokenizer_2"] = self.tokenizer_2
#     else:
#         extra_pipeline_kwargs["text_encoder_1"] = None
#         extra_pipeline_kwargs["tokenizer_1"] = None
#         extra_pipeline_kwargs["text_encoder_2"] = None
#         extra_pipeline_kwargs["tokenizer_2"] = None

# if self.args.model_family == "smoldit":
#     extra_pipeline_kwargs["transformer"] = unwrap_model(
#         self.accelerator, self.transformer
#     )
#     extra_pipeline_kwargs["tokenizer"] = self.tokenizer_1
#     extra_pipeline_kwargs["text_encoder"] = self.text_encoder_1
#     extra_pipeline_kwargs["scheduler"] = self.setup_scheduler()

# if self.args.controlnet:
#     # ControlNet training has an additional adapter thingy.
#     extra_pipeline_kwargs["controlnet"] = unwrap_model(
#         self.accelerator, self.controlnet
#     )
# if self.unet is not None:
#     extra_pipeline_kwargs["unet"] = unwrap_model(
#         self.accelerator, self.unet
#     )

# if self.transformer is not None:
#     extra_pipeline_kwargs["transformer"] = unwrap_model(
#         self.accelerator, self.transformer
#     )

# if self.args.model_family == "sd3" and self.args.train_text_encoder:
#     if self.text_encoder_1 is not None:
#         extra_pipeline_kwargs["text_encoder"] = unwrap_model(
#             self.accelerator, self.text_encoder_1
#         )
#         extra_pipeline_kwargs["tokenizer"] = self.tokenizer_1
#     if self.text_encoder_2 is not None:
#         extra_pipeline_kwargs["text_encoder_2"] = unwrap_model(
#             self.accelerator, self.text_encoder_2
#         )
#         extra_pipeline_kwargs["tokenizer_2"] = self.tokenizer_2
#     if self.text_encoder_3 is not None:
#         extra_pipeline_kwargs["text_encoder_3"] = unwrap_model(
#             self.accelerator, self.text_encoder_3
#         )
#         extra_pipeline_kwargs["tokenizer_3"] = self.tokenizer_3

# if self.vae is None or not hasattr(self.vae, "device"):
#     extra_pipeline_kwargs["vae"] = self.init_vae()
# else:
#     logger.info(f"Found VAE: {self.vae.config}")
# if (
#     "vae" in extra_pipeline_kwargs
#     and extra_pipeline_kwargs.get("vae") is not None
#     and extra_pipeline_kwargs["vae"].device != self.inference_device
# ):
#     extra_pipeline_kwargs["vae"] = extra_pipeline_kwargs["vae"].to(
#         self.inference_device
#     )

# pipeline_kwargs = {
#     "pretrained_model_name_or_path": self.args.pretrained_model_name_or_path,
#     "revision": self.args.revision,
#     "variant": self.args.variant,
#     "torch_dtype": self.weight_dtype,
#     **extra_pipeline_kwargs,
# }
# logger.debug(f"Initialising pipeline with kwargs: {pipeline_kwargs}")
# attempt = 0
# while attempt < 3:
#     attempt += 1
#     try:
#         if self.args.model_family == "smoldit":
#             self.pipeline = pipeline_cls(
#                 vae=self.vae,
#                 transformer=unwrap_model(
#                     self.accelerator, self.transformer
#                 ),
#                 tokenizer=self.tokenizer_1,
#                 text_encoder=self.text_encoder_1,
#                 scheduler=self.setup_scheduler(),
#             )
#         else:
#             self.pipeline = pipeline_cls.from_pretrained(**pipeline_kwargs)
#     except Exception as e:
#         import traceback

#         logger.error(e)
#         logger.error(traceback.format_exc())
#         continue
#     break
# if self.args.validation_torch_compile:
#     if self.deepspeed:
#         logger.warning(
#             "DeepSpeed does not support torch compile. Disabling. Set --validation_torch_compile=False to suppress this warning."
#         )
#     elif self.args.lora_type.lower() == "lycoris":
#         logger.warning(
#             "LyCORIS does not support torch compile for validation due to graph compile breaks. Disabling. Set --validation_torch_compile=False to suppress this warning."
#         )
#     else:
#         if self.unet is not None and not is_compiled_module(self.unet):
#             logger.warning(
#                 f"Compiling the UNet for validation ({self.args.validation_torch_compile})"
#             )
#             self.pipeline.unet = torch.compile(
#                 self.pipeline.unet,
#                 mode=self.args.validation_torch_compile_mode,
#                 fullgraph=False,
#             )
#         if self.transformer is not None and not is_compiled_module(
#             self.transformer
#         ):
#             logger.warning(
#                 f"Compiling the transformer for validation ({self.args.validation_torch_compile})"
#             )
#             self.pipeline.transformer = torch.compile(
#                 self.pipeline.transformer,
#                 mode=self.args.validation_torch_compile_mode,
#                 fullgraph=False,
#             )






        # unet_ = None
        # transformer_ = None
        # denoiser = None
        # text_encoder_one_ = None
        # text_encoder_two_ = None

        # while len(models) > 0:
        #     model = models.pop()

        #     if isinstance(
        #         unwrap_model(self.accelerator, model),
        #         type(unwrap_model(self.accelerator, self.get_trained_component())),
        #     ):
        #         unet_ = model
        #         denoiser = unet_
        #     elif isinstance(
        #         unwrap_model(self.accelerator, model),
        #         type(unwrap_model(self.accelerator, self.transformer)),
        #     ):
        #         transformer_ = model
        #         denoiser = transformer_
        #     elif isinstance(
        #         unwrap_model(self.accelerator, model),
        #         type(unwrap_model(self.accelerator, self.text_encoder_1)),
        #     ):
        #         text_encoder_one_ = model
        #     elif isinstance(
        #         unwrap_model(self.accelerator, model),
        #         type(unwrap_model(self.accelerator, self.text_encoder_2)),
        #     ):
        #         text_encoder_two_ = model
        #     else:
        #         raise ValueError(
        #             f"unexpected save model: {model.__class__}"
        #             f"\nunwrapped: {unwrap_model(self.accelerator, model).__class__}"
        #             f"\nunet: {unwrap_model(self.accelerator, self.get_trained_component()).__class__}"
        #         )

        # key_to_replace = self.model.MODEL_SUBFOLDER
        # lora_state_dict = self.pipeline_class.lora_state_dict(input_dir)

        # denoiser_state_dict = {
        #     f'{k.replace(f"{key_to_replace}.", "")}': v
        #     for k, v in lora_state_dict.items()
        #     if k.startswith(f"{key_to_replace}.")
        # }
        # denoiser_state_dict = convert_unet_state_dict_to_peft(denoiser_state_dict)
        # incompatible_keys = set_peft_model_state_dict(
        #     denoiser, denoiser_state_dict, adapter_name="default"
        # )

        # if incompatible_keys is not None:
        #     # check only for unexpected keys
        #     unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
        #     if unexpected_keys:
        #         logger.warning(
        #             f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
        #             f" {unexpected_keys}. "
        #         )

        # if self.args.train_text_encoder:
        #     # Do we need to call `scale_lora_layers()` here?
        #     from diffusers.training_utils import _set_state_dict_into_text_encoder

        #     _set_state_dict_into_text_encoder(
        #         lora_state_dict,
        #         prefix="text_encoder.",
        #         text_encoder=text_encoder_one_,
        #     )

        #     _set_state_dict_into_text_encoder(
        #         lora_state_dict,
        #         prefix="text_encoder_2.",
        #         text_encoder=text_encoder_two_,
        #     )

