import functools
import importlib
import json
import typing
from pathlib import Path

from hydra.conf import HydraConf
from omegaconf import DictConfig
from pydantic import BaseModel, ConfigDict, model_validator

from hydra.utils import instantiate
from hydra.conf import HydraConf

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel

ConfigT = typing.TypeVar('ConfigT')
FunParamSpec = typing.ParamSpec('FunParamSpec')
FunResult = typing.TypeVar('FunResult')

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from modeling.unet import UnetModel
from modeling.diffusion import DiffusionModel


HYDRA_TARGET_KEY = '_target_'
HYDRA_PARTIAL_KEY = '_partial_'
NOT_PRIVATE_EXTRA_FIELDS = {HYDRA_TARGET_KEY, HYDRA_PARTIAL_KEY}


def from_omegaconf(cls: typing.Type[ConfigT], cfg: DictConfig, resolve: bool = False) -> ConfigT:
    return cls(**typing.cast(dict, OmegaConf.to_container(cfg, resolve=resolve)))


def to_omegaconf(config: typing.Any) -> DictConfig:
    return typing.cast(DictConfig, OmegaConf.create(config))


def with_hydra_conf(
    conf: HydraConf, fn: typing.Callable[FunParamSpec, FunResult]
) -> typing.Callable[FunParamSpec, FunResult]:
    @functools.wraps(fn)
    def _wrapper(*args: FunParamSpec.args, **kwargs: FunParamSpec.kwargs):
        config = HydraConfig()
        if config.cfg is not None:
            raise ValueError("HydraConfig was already set")

        try:
            config.set_config(DictConfig({'hydra': conf}))
            return fn(*args, **kwargs)
        finally:
            config.cfg = None

    return _wrapper


class BaseConfig(BaseModel):
    @classmethod
    def from_omegaconf(cls, cfg: DictConfig, hydra_conf: HydraConf | None = None, resolve: bool = False) -> typing.Self:
        fn = from_omegaconf if not hydra_conf else with_hydra_conf(hydra_conf, from_omegaconf)
        return fn(cls, cfg, resolve=resolve)

    def to_omegaconf(self, exclude_unset: bool = True) -> DictConfig:
        return to_omegaconf(self.model_dump(exclude_unset=exclude_unset))

    def hash(self, exclude_unset: bool = True, exclude: set[str] | dict[str, typing.Any] = set()) -> int:
        return hash(json.dumps(self.model_dump(exclude=exclude, exclude_unset=exclude_unset), indent=4, sort_keys=True))


class StrongConfig(BaseConfig, extra='forbid'):
    __ignore__: tuple[str, ...] = ()

    @model_validator(mode='before')
    @classmethod
    def clean_ignored(cls, data: dict[str, typing.Any]):
        for key in cls.__ignore__:
            data.pop(key, None)
        return data


class HydraConfig(BaseConfig, extra='allow'):
    _target_: str

    def instantiate(self, *args: typing.Any, **kwargs: typing.Any):
        return instantiate(dict(self), *args, **kwargs)

    def target(self) -> typing.Type:
        assert hasattr(self, '_target_'), f'{self.__class__.__name__} must have a target'
        package, target = self._target_.rsplit('.', 1)
        return getattr(importlib.import_module(package), target)

    @model_validator(mode='before')
    @classmethod
    def check(cls, data: dict[str, typing.Any]):
        assert isinstance(data, dict) and HYDRA_TARGET_KEY in data, f"Missing '{HYDRA_TARGET_KEY}' key in the config"
        return data

    def hydra_model_dump(
        self,
        target_alias: str | None,
        target_formatter: typing.Callable[[str], str] | None = None,
        **kwargs: typing.Any,
    ) -> dict:
        cfg = self.model_dump(**kwargs)
        if target_alias and HYDRA_TARGET_KEY in cfg:
            cfg[target_alias] = cfg.pop(HYDRA_TARGET_KEY)
            if target_formatter:
                cfg[target_alias] = target_formatter(cfg[target_alias])
        cfg.pop(HYDRA_TARGET_KEY, None)
        cfg.pop(HYDRA_PARTIAL_KEY, None)
        return cfg

    # a small hack to allow pydantic to store fields starting with '_'
    def __getattr__(self, name: str) -> typing.Any:
        if name not in NOT_PRIVATE_EXTRA_FIELDS:
            if not typing.TYPE_CHECKING:
                return super().__getattr__(name)
        else:
            assert self.model_extra is not None, "Extra fields are not enabled, please contact the developer"
            return self.model_extra.get(name)

    def __setattr__(self, name: str, value: typing.Any):
        if name not in NOT_PRIVATE_EXTRA_FIELDS:
            super().__setattr__(name, value)
        else:
            assert self.model_extra is not None, "Extra fields are not enabled, please contact the developer"
            self.model_extra[name] = value

    def __delattr__(self, item: str):
        if item not in NOT_PRIVATE_EXTRA_FIELDS:
            return super().__delattr__(item)
        else:
            assert self.model_extra is not None, "Extra fields are not enabled, please contact the developer"
            del self.model_extra[item]


class UNetConfig(HydraConfig):
    _target_: str = "modeling.unet.UnetModel"
    in_channels: int = 3
    out_channels: int = 3
    hidden_size: int = 128

    def instantiate(self, **kwargs: typing.Any) -> UnetModel:
        return super().instantiate(**kwargs)


class DiffusionConfig(HydraConfig):
    _target_: str = "modeling.diffusion.DiffusionModel"
    eps_model: UNetConfig
    betas: tuple[float, float] = (0.0001, 0.02)
    num_timesteps: int = 1000

    def instantiate(self, **kwargs: typing.Any) -> DiffusionModel:
        return super().instantiate(**kwargs)


class OptimizerConfig(HydraConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _target_: str = "torch.optim.Adam"
    lr: float = 0.00001
    params: list[torch.nn.Parameter]

    def instantiate(self, **kwargs: typing.Any) -> torch.optim.Adam:
        return super().instantiate(**kwargs)


class AugmentationConfig(HydraConfig):
    _target_: str = "torchvision.transforms.RandomHorizontalFlip"
    p: float = 0.5

    def instantiate(self, **kwargs: typing.Any) -> transforms.RandomHorizontalFlip:
        return super().instantiate(**kwargs)


class SaveConfig(StrongConfig):
    model_directory: Path = Path("saved_models/")
    samples_directory: Path = Path("train_samples/")
    samples_frequency_epochs: int = 1


class WandbConfig(StrongConfig):
    project: str = "diffusion_hw_1"
    run_name: str = "dvc"


class LoopConfig(StrongConfig):
    batch_size: int = 8
    num_workers: int = 2
    num_epochs: int = 2


class TrainConfig(StrongConfig):
    optimizer: OptimizerConfig
    loop: LoopConfig
    augmentations: list[AugmentationConfig]
    save: SaveConfig
    wandb: WandbConfig


class CIFAR10Config(HydraConfig):
    _target_: str = "torchvision.datasets.CIFAR10"
    download: bool = True

    def instantiate(self, **kwargs: typing.Any) -> CIFAR10:
        return super().instantiate(**kwargs)


class Config(StrongConfig):
    defaults: list[dict[str, str]] = [{"dataset": "cifar10"}, {"train": "default_loop"}, {"model": "default_diffusion"}]
    dataset: CIFAR10Config
    train: TrainConfig
    model: DiffusionConfig
