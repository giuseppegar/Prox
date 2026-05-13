import os
from dataclasses import dataclass, field
from typing import Optional

import litellm
import yaml

PROX_CONFIG_DIR = os.path.expanduser("~/.prox")
DEFAULT_CONFIG_PATH = os.path.join(PROX_CONFIG_DIR, "config.yaml")


@dataclass
class AgentModelConfig:
    model: str = "deepseek/deepseek-chat"
    temperature: float = 0.1
    max_tokens: int = 4096
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class ModelRegistry:
    main_orchestrator: AgentModelConfig = field(default_factory=AgentModelConfig)
    mini_orchestrator: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(model="deepseek/deepseek-chat"))
    director: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(model="deepseek/deepseek-chat"))
    worker: AgentModelConfig = field(default_factory=lambda: AgentModelConfig(model="gpt-4o-mini"))

    @classmethod
    def from_config(cls, path: str = DEFAULT_CONFIG_PATH) -> "ModelRegistry":
        if not os.path.exists(path):
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        llm_data = data.get("llm", {})
        return cls(
            main_orchestrator=AgentModelConfig(**llm_data.get("main_orchestrator", {})),
            mini_orchestrator=AgentModelConfig(**llm_data.get("mini_orchestrator", {})),
            director=AgentModelConfig(**llm_data.get("director", {})),
            worker=AgentModelConfig(**llm_data.get("worker", {})),
        )

    def to_config_dict(self) -> dict:
        return {
            "llm": {
                "main_orchestrator": self.main_orchestrator.__dict__,
                "mini_orchestrator": self.mini_orchestrator.__dict__,
                "director": self.director.__dict__,
                "worker": self.worker.__dict__,
            }
        }

    def get_model(self, tier: str) -> str:
        config = getattr(self, tier, self.worker)
        return config.model


def load_registry(config_path: str = DEFAULT_CONFIG_PATH) -> ModelRegistry:
    return ModelRegistry.from_config(config_path)
