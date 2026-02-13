# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import List, Literal, Optional

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedObjective

logger = logging.getLogger(__name__)

# Maps the user-facing model name to the corresponding HuggingFace dataset ID.
_HCOT_DATASET_IDS: dict[str, str] = {
    "o1": "DukeCEICenter/Malicious_Educator_hcot_o1",
    "o3-mini": "DukeCEICenter/Malicious_Educator_hcot_o3-mini",
    "deepseek-r1": "DukeCEICenter/Malicious_Educator_hcot_DeepSeek-R1",
    "gemini": "DukeCEICenter/Malicious_Educator_hcot_Gemini-2.0-Flash-Thinking",
}

# The 10 harm categories present in the Malicious-Educator benchmark.
# Source: https://github.com/dukeceicenter/jailbreak-reasoning-openai-o1o3-deepseek-r1
HCOT_HARM_CATEGORIES = [
    "Child Harm",
    "Cybercrime",
    "Drug",
    "Economic Crime",
    "Firearms",
    "Human Trafficking",
    "Privacy",
    "Self Harm",
    "Sexual",
    "Violence",
]


class _MaliciousEducatorHCoTDataset(_RemoteDatasetLoader):
    """
    Loader for the Malicious-Educator H-CoT dataset.

    The Malicious-Educator benchmark contains 50 harmful goal/request pairs
    paired with Hijacking Chain-of-Thought (H-CoT) prefixes that were shown
    to bypass the safety reasoning mechanisms of large reasoning models (LRMs)
    including OpenAI o1/o3, DeepSeek-R1, and Gemini 2.0 Flash Thinking.

    Each example exposes the full attack payload (H-CoT prefix + harmful
    request) as a SeedObjective, with the raw ``Goal``, ``Request``, and
    ``H_CoT`` fields stored in metadata.

    Four model-specific variants are available via the ``model`` parameter:

    - ``"o1"``          → DukeCEICenter/Malicious_Educator_hcot_o1
    - ``"o3-mini"``     → DukeCEICenter/Malicious_Educator_hcot_o3-mini
    - ``"deepseek-r1"`` → DukeCEICenter/Malicious_Educator_hcot_DeepSeek-R1
    - ``"gemini"``      → DukeCEICenter/Malicious_Educator_hcot_Gemini-2.0-Flash-Thinking

    License: CC BY-NC-SA 4.0

    Reference:
        Kuo et al. (2025). "H-CoT: Hijacking the Chain-of-Thought Safety
        Reasoning Mechanism to Jailbreak Large Reasoning Models."
        https://arxiv.org/abs/2502.12893
        https://github.com/dukeceicenter/jailbreak-reasoning-openai-o1o3-deepseek-r1
    """

    def __init__(
        self,
        *,
        model: Literal["o1", "o3-mini", "deepseek-r1", "gemini"] = "o1",
        harm_categories: Optional[
            List[
                Literal[
                    "Child Harm",
                    "Cybercrime",
                    "Drug",
                    "Economic Crime",
                    "Firearms",
                    "Human Trafficking",
                    "Privacy",
                    "Self Harm",
                    "Sexual",
                    "Violence",
                ]
            ]
        ] = None,
        token: Optional[str] = None,
    ):
        """
        Initialize the H-CoT dataset loader.

        Args:
            model: Which model-specific dataset variant to load.
                One of ``"o1"``, ``"o3-mini"``, ``"deepseek-r1"``, or ``"gemini"``.
                Defaults to ``"o1"``.
            harm_categories: Optional list of harm categories to filter by.
                Defaults to None (all categories included). Only examples whose
                ``Category`` field matches at least one entry are returned.
            token: Optional HuggingFace authentication token. The datasets are
                public so this is not required in most cases.

        Raises:
            ValueError: If ``model`` is not one of the supported model names.
            ValueError: If any entry in ``harm_categories`` is not a valid category.
        """
        if model not in _HCOT_DATASET_IDS:
            raise ValueError(
                f"Unsupported model '{model}'. "
                f"Choose one of: {sorted(_HCOT_DATASET_IDS.keys())}"
            )

        if harm_categories:
            invalid = {
                cat for cat in harm_categories if cat not in HCOT_HARM_CATEGORIES
            }
            if invalid:
                raise ValueError(
                    f"Invalid harm categories: {invalid}. "
                    f"Valid categories are: {HCOT_HARM_CATEGORIES}"
                )

        self.model = model
        self.harm_categories_filter = harm_categories
        self.token = token
        self._hf_dataset_name = _HCOT_DATASET_IDS[model]

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return f"malicious-educator-hcot-{self.model}"

    async def fetch_dataset(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch the H-CoT dataset and return as a SeedDataset.

        Args:
            cache: Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: A SeedDataset containing H-CoT attack payloads,
                optionally filtered by harm category.
        """
        logger.info(
            f"Loading Malicious-Educator H-CoT dataset for model '{self.model}'"
        )

        hf_dataset = await self._fetch_from_huggingface(
            dataset_name=self._hf_dataset_name,
            split="train",
            cache=cache,
            token=self.token,
        )

        seeds = []
        for example in hf_dataset:
            category = example["Category"]

            if self.harm_categories_filter is not None:
                if category not in self.harm_categories_filter:
                    continue

            seed = SeedObjective(
                value=example["Full_Input (H-CoT + Request)"],
                name=example["Goal"],
                dataset_name=self.dataset_name,
                harm_categories=[category],
                description=(
                    "A Malicious-Educator H-CoT example. Each entry contains a harmful "
                    "goal/request prefixed with a fake chain-of-thought (H-CoT) designed "
                    "to hijack the safety reasoning of large reasoning models. "
                    "Intended for security research and AI red-teaming only. "
                    "Paper: https://arxiv.org/abs/2502.12893"
                ),
                source=f"https://huggingface.co/datasets/{self._hf_dataset_name}",
                authors=[
                    "Martin Kuo",
                    "Jianyi Zhang",
                    "Aolin Ding",
                    "Qinsi Wang",
                    "Louis DiValentin",
                    "Yujia Bao",
                    "Wei Wei",
                    "Hai Li",
                    "Yiran Chen",
                ],
                metadata={
                    "goal": example["Goal"],
                    "request": example["Request"],
                    "h_cot": example["H_CoT"],
                    "model": self.model,
                },
            )
            seeds.append(seed)

        logger.info(
            f"Successfully loaded {len(seeds)} H-CoT examples "
            f"from Malicious-Educator ({self.model})"
        )

        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)
