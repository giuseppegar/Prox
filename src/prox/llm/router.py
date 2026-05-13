import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import yaml
import litellm

CONFIG_PATH = os.path.expanduser("~/.prox/config.yaml")
MODELS_FRESHNESS_PATH = os.path.expanduser("~/.prox/synapse/models.yaml")


class ProviderType(Enum):
    NATIVE = "native"
    PROXY = "proxy"


class Role(Enum):
    MAIN_ORCHESTRATOR = "main_orchestrator"
    MINI_ORCHESTRATOR = "mini_orchestrator"
    DIRECTOR = "director"
    CODER = "coder"
    REVIEWER = "reviewer"
    RESEARCHER = "researcher"
    TESTER = "tester"
    MEMORY = "memory"


@dataclass
class Provider:
    name: str
    type: ProviderType
    key_env: str
    key: Optional[str] = None
    models: list[str] = field(default_factory=list)

    def is_configured(self) -> bool:
        return bool(self.key or os.environ.get(self.key_env))

    def get_key(self) -> Optional[str]:
        return self.key or os.environ.get(self.key_env)


@dataclass
class ModelInfo:
    id: str
    provider: str
    provider_type: ProviderType
    cost_per_1m_input: float = 99.0
    cost_per_1m_output: float = 99.0
    context_window: int = 4096
    speed: str = "unknown"
    code_gen_score: float = 0.0
    reasoning_score: float = 0.0
    supports_tools: bool = True
    supports_vision: bool = False
    last_checked: float = 0.0


ROLE_CRITERIA = {
    Role.MAIN_ORCHESTRATOR: {
        "criteria": ["speed", "cost"],
        "min_context": None,
        "prefer_native": True,
        "max_cost_input": 5.0,
    },
    Role.MINI_ORCHESTRATOR: {
        "criteria": ["reasoning", "context_window"],
        "min_context": 128000,
        "prefer_native": True,
        "max_cost_input": 15.0,
    },
    Role.DIRECTOR: {
        "criteria": ["balanced", "consistency"],
        "min_context": 32000,
        "prefer_native": False,
        "max_cost_input": 10.0,
    },
    Role.CODER: {
        "criteria": ["code_generation", "context_window"],
        "min_context": 32000,
        "prefer_native": False,
        "max_cost_input": 15.0,
    },
    Role.REVIEWER: {
        "criteria": ["reasoning", "analysis"],
        "min_context": 32000,
        "prefer_native": False,
        "max_cost_input": 10.0,
    },
    Role.RESEARCHER: {
        "criteria": ["cost", "speed"],
        "min_context": None,
        "prefer_native": True,
        "max_cost_input": 1.0,
    },
    Role.TESTER: {
        "criteria": ["cost", "consistency"],
        "min_context": None,
        "prefer_native": True,
        "max_cost_input": 1.0,
    },
    Role.MEMORY: {
        "criteria": ["cost"],
        "min_context": None,
        "prefer_native": True,
        "max_cost_input": 1.0,
    },
}


SPEED_SCORES = {"fast": 100, "medium": 50, "slow": 20, "unknown": 30}


KNOWN_MODEL_CAPABILITIES = {
    "deepseek/deepseek-v4-pro": ModelInfo(
        id="deepseek/deepseek-v4-pro", provider="deepseek", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=0.14, cost_per_1m_output=0.28, context_window=128000,
        speed="fast", code_gen_score=90, reasoning_score=92,
    ),
    "deepseek/deepseek-v4-flash": ModelInfo(
        id="deepseek/deepseek-v4-flash", provider="deepseek", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=0.10, cost_per_1m_output=0.20, context_window=128000,
        speed="fast", code_gen_score=82, reasoning_score=78,
    ),
    "openai/gpt-4o": ModelInfo(
        id="openai/gpt-4o", provider="openai", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=2.50, cost_per_1m_output=10.0, context_window=128000,
        speed="fast", code_gen_score=92, reasoning_score=90, supports_vision=True,
    ),
    "openai/gpt-4o-mini": ModelInfo(
        id="openai/gpt-4o-mini", provider="openai", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=0.15, cost_per_1m_output=0.60, context_window=128000,
        speed="fast", code_gen_score=78, reasoning_score=72,
    ),
    "openai/o3-mini": ModelInfo(
        id="openai/o3-mini", provider="openai", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=1.10, cost_per_1m_output=4.40, context_window=200000,
        speed="fast", code_gen_score=88, reasoning_score=93,
    ),
    "anthropic/claude-sonnet-4-20250514": ModelInfo(
        id="anthropic/claude-sonnet-4-20250514", provider="anthropic", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=3.0, cost_per_1m_output=15.0, context_window=200000,
        speed="medium", code_gen_score=95, reasoning_score=92, supports_vision=True,
    ),
    "anthropic/claude-opus-4-20250514": ModelInfo(
        id="anthropic/claude-opus-4-20250514", provider="anthropic", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=15.0, cost_per_1m_output=75.0, context_window=200000,
        speed="slow", code_gen_score=94, reasoning_score=98, supports_vision=True,
    ),
    "google/gemini-2.5-pro": ModelInfo(
        id="google/gemini-2.5-pro", provider="google", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=1.25, cost_per_1m_output=10.0, context_window=1048576,
        speed="fast", code_gen_score=90, reasoning_score=91, supports_vision=True,
    ),
    "google/gemini-2.5-flash": ModelInfo(
        id="google/gemini-2.5-flash", provider="google", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=0.15, cost_per_1m_output=0.60, context_window=1048576,
        speed="fast", code_gen_score=82, reasoning_score=80,
    ),
    "meta-llama/llama-4-maverick": ModelInfo(
        id="meta-llama/llama-4-maverick", provider="meta", provider_type=ProviderType.NATIVE,
        cost_per_1m_input=0.20, cost_per_1m_output=0.90, context_window=128000,
        speed="medium", code_gen_score=80, reasoning_score=78,
    ),
    "openrouter/auto": ModelInfo(
        id="openrouter/auto", provider="openrouter", provider_type=ProviderType.PROXY,
        cost_per_1m_input=0.0, cost_per_1m_output=0.0, context_window=128000,
        speed="fast", code_gen_score=90, reasoning_score=90,
    ),
}


class ModelRouter:
    def __init__(self, config_path: str = CONFIG_PATH):
        self._config_path = config_path
        self._providers: dict[str, Provider] = {}
        self._models: list[ModelInfo] = []
        self._assignments: dict[Role, str] = {}
        self._load_config()

    @property
    def assignments(self) -> dict[Role, str]:
        return self._assignments

    def get_model(self, role: Role) -> str:
        return self._assignments.get(role, self._fallback(role))

    def get_all_models(self) -> list[ModelInfo]:
        return list(self._models)

    def get_models_by_provider(self, provider_name: str) -> list[ModelInfo]:
        return [m for m in self._models if m.provider == provider_name]

    def get_native_models(self) -> list[ModelInfo]:
        return [m for m in self._models if m.provider_type == ProviderType.NATIVE]

    def get_proxy_models(self) -> list[ModelInfo]:
        return [m for m in self._models if m.provider_type == ProviderType.PROXY]

    def discover_and_assign(self) -> dict[Role, str]:
        self._discover_models()
        self._deduplicate()
        self._assign_all()
        self._save_assignments()
        return self._assignments

    def add_provider(self, name: str, provider_type: str, key: str = None, key_env: str = None) -> None:
        self._providers[name] = Provider(
            name=name,
            type=ProviderType(provider_type),
            key_env=key_env or f"{name.upper()}_API_KEY",
            key=key,
        )
        self._save_config()

    def remove_provider(self, name: str) -> None:
        self._providers.pop(name, None)
        self._save_config()

    def list_providers(self) -> list[Provider]:
        return list(self._providers.values())

    def summary(self) -> str:
        lines = []
        native = [p for p in self._providers.values() if p.type == ProviderType.NATIVE]
        proxy = [p for p in self._providers.values() if p.type == ProviderType.PROXY]
        lines.append(f"Provider: {len(native)} native, {len(proxy)} proxy")
        lines.append(f"Modelli disponibili: {len(self._models)}")
        lines.append("")
        for role in Role:
            model = self._assignments.get(role, "non assegnato")
            provider = model.split("/")[0] if "/" in model else "?"
            lines.append(f"  {role.value:25} → {model} ({provider})")
        return "\n".join(lines)

    def _discover_models(self) -> None:
        self._models = []

        for prov_name, provider in self._providers.items():
            if provider.type == ProviderType.NATIVE:
                native_models = [
                    m for m in KNOWN_MODEL_CAPABILITIES.values()
                    if m.provider == prov_name
                ]
                if native_models:
                    for m in native_models:
                        self._models.append(ModelInfo(
                            id=m.id, provider=prov_name,
                            provider_type=ProviderType.NATIVE,
                            cost_per_1m_input=m.cost_per_1m_input,
                            cost_per_1m_output=m.cost_per_1m_output,
                            context_window=m.context_window,
                            speed=m.speed, code_gen_score=m.code_gen_score,
                            reasoning_score=m.reasoning_score,
                            supports_tools=m.supports_tools,
                            supports_vision=m.supports_vision,
                        ))
                else:
                    self._models.append(ModelInfo(
                        id=f"{prov_name}/default", provider=prov_name,
                        provider_type=ProviderType.NATIVE,
                        speed="unknown",
                    ))

            elif provider.type == ProviderType.PROXY:
                for mi in KNOWN_MODEL_CAPABILITIES.values():
                    self._models.append(ModelInfo(
                        id=mi.id, provider=prov_name,
                        provider_type=ProviderType.PROXY,
                        cost_per_1m_input=mi.cost_per_1m_input,
                        cost_per_1m_output=mi.cost_per_1m_output,
                        context_window=mi.context_window,
                        speed=mi.speed, code_gen_score=mi.code_gen_score,
                        reasoning_score=mi.reasoning_score,
                        supports_tools=mi.supports_tools,
                        supports_vision=mi.supports_vision,
                    ))

                    if mi.id not in KNOWN_MODEL_CAPABILITIES:
                        pass

        self._load_freshness_scores()

    def _load_freshness_scores(self) -> None:
        if not os.path.exists(MODELS_FRESHNESS_PATH):
            return
        with open(MODELS_FRESHNESS_PATH) as f:
            data = yaml.safe_load(f) or {}
        for model_id, scores in data.items():
            for m in self._models:
                if m.id == model_id:
                    if "code_gen_score" in scores:
                        m.code_gen_score = scores["code_gen_score"]
                    if "reasoning_score" in scores:
                        m.reasoning_score = scores["reasoning_score"]
                    if "last_checked" in scores:
                        m.last_checked = scores["last_checked"]

    def _deduplicate(self) -> None:
        seen: dict[str, ModelInfo] = {}
        native_models = [m for m in self._models if m.provider_type == ProviderType.NATIVE]
        proxy_models = [m for m in self._models if m.provider_type == ProviderType.PROXY]

        for m in native_models:
            seen[m.id] = m

        for m in proxy_models:
            if m.id not in seen:
                seen[m.id] = m

        self._models = list(seen.values())

    def _assign_all(self) -> None:
        for role in Role:
            self._assignments[role] = self._assign(role)

    def _assign(self, role: Role) -> str:
        criteria = ROLE_CRITERIA[role]
        candidates = list(self._models)

        if criteria.get("prefer_native"):
            native = [m for m in candidates if m.provider_type == ProviderType.NATIVE]
            if native:
                candidates = native

        if criteria.get("min_context"):
            min_ctx = criteria["min_context"]
            candidates = [m for m in candidates if m.context_window >= min_ctx]

        max_cost = criteria.get("max_cost_input", 999)
        candidates = [m for m in candidates if m.cost_per_1m_input <= max_cost]

        if not candidates:
            candidates = list(self._models)
        if not candidates:
            return "deepseek/deepseek-v4-pro"

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0].id if scored else "deepseek/deepseek-v4-pro"

    def _score_model(self, model: ModelInfo, role: Role) -> float:
        criteria = ROLE_CRITERIA[role]
        primary = criteria["criteria"]
        score = 0.0

        for c in primary:
            if c == "speed":
                w = SPEED_SCORES.get(model.speed, 30)
                score += float(w)
            elif c == "cost":
                w = max(0, 100 - model.cost_per_1m_input * 20)
                score += float(w)
            elif c == "reasoning":
                score += float(model.reasoning_score)
            elif c == "code_generation":
                score += float(model.code_gen_score)
            elif c == "context_window":
                score += min(100, model.context_window / 10000.0)
            elif c == "balanced":
                score += model.code_gen_score * 0.4 + model.reasoning_score * 0.4 + max(0, 100 - model.cost_per_1m_input * 10) * 0.2
            elif c == "analysis":
                score += model.reasoning_score * 0.6 + model.code_gen_score * 0.4
            elif c == "consistency":
                score += 80.0

        if model.provider_type == ProviderType.NATIVE:
            score *= 1.05

        return score

    def _fallback(self, role: Role) -> str:
        criteria = ROLE_CRITERIA[role]
        for m in KNOWN_MODEL_CAPABILITIES.values():
            if m.cost_per_1m_input <= 5.0:
                return m.id
        return "deepseek/deepseek-v4-pro"

    def _load_config(self) -> None:
        if not os.path.exists(self._config_path):
            return
        with open(self._config_path) as f:
            data = yaml.safe_load(f) or {}

        for name, info in data.get("providers", {}).items():
            self._providers[name] = Provider(
                name=name,
                type=ProviderType(info.get("type", "native")),
                key_env=info.get("key_env", f"{name.upper()}_API_KEY"),
                key=info.get("key"),
            )

        assignments = data.get("model_assignments", {})
        for role_str, model_id in assignments.items():
            try:
                role = Role(role_str)
                self._assignments[role] = model_id
            except ValueError:
                pass

    def _save_config(self) -> None:
        data = {}
        if os.path.exists(self._config_path):
            with open(self._config_path) as f:
                data = yaml.safe_load(f) or {}

        data["providers"] = {}
        for name, prov in self._providers.items():
            data["providers"][name] = {
                "type": prov.type.value,
                "key_env": prov.key_env,
            }
            if prov.key:
                data["providers"][name]["key"] = prov.key

        data["model_assignments"] = {
            role.value: model_id
            for role, model_id in self._assignments.items()
        }

        os.makedirs(os.path.dirname(self._config_path), exist_ok=True)
        with open(self._config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)

    def _save_assignments(self) -> None:
        self._save_config()


router = ModelRouter()


def get_router() -> ModelRouter:
    return router


def discover_and_assign() -> dict[Role, str]:
    return router.discover_and_assign()


def get_model_for_role(role: Role) -> str:
    return router.get_model(role)
