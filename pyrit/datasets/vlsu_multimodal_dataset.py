# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
import uuid
from pathlib import Path
from typing import List, Optional

from pyrit.common.net_utility import make_request_and_raise_if_error_async
from pyrit.datasets.dataset_helper import fetch_examples
from pyrit.models import SeedDataset, SeedPrompt, data_serializer_factory

logger = logging.getLogger(__name__)

ML_VLSU_CATEGORIES = [
    "C1: Slurs, Hate Speech, Hate Symbols",
    "C2: Discrimination and Unequal Treatment",
    "C3: Drug Abuse",
    "C4: Self-Harm and Suicide",
    "C5: Animal Violence and Gore",
    "C6: Adult Explicit Sexual Material",
    "C7: Adult Racy Material",
    "C8: Warfare and Armed Conflicts",
    "C9: Interpersonal Violence",
    "C10: Weapons and Dangerous Objects",
    "C11: Gore and Graphic Content",
    "C12: Terrorism and Violent Extremism",
    "C13: Jailbreaks",
    "C14: Inauthentic Practices/Fraud",
    "C15: Human Exploitation",
]


async def fetch_ml_vlsu_dataset_async(
    *,
    source: str = "https://raw.githubusercontent.com/apple/ml-vlsu/main/data/VLSU.csv",
    source_type: str = "public_url",
    cache: bool = True,
    data_home: Optional[Path] = None,
    categories: Optional[List[str]] = None,
    follow_redirects=True,
    limit: Optional[int] = None,  # limit option for testing subsample
    random_sample: bool = False,  # random sampling option for testing
    random_seed: Optional[int] = None,
) -> SeedDataset:
    """
    Fetch ML-VLSU examples and create a SeedDataset.

    Args:
        source (str): The source from which to fetch examples.
        source_type (str): The type of source. Defaults to 'public_url'.
        cache (bool): Whether to cache the fetched examples.
        data_home (Optional[Path]): Directory to store cached data.
        categories (Optional[List[str]]): List of categories to filter examples. Must be in ML_VLSU_CATEGORIES.
        limit (Optional[int]): Limit the number of examples to fetch. Useful for testing subsamples.
        random_sample (bool): Whether to randomly sample examples when limit is set.
        random_seed (Optional[int]): Seed for random sampling to ensure reproducibility.

    Returns:
        SeedDataset: A SeedDataset containing the ML-VLSU examples.

    Raises:
        ValueError: If any of the specified categories are invalid.
    """

    if categories:
        invalid = set(categories) - set(ML_VLSU_CATEGORIES)
        if invalid:
            raise ValueError(f"Invalid categories: {invalid}. Must be in ML_VLSU_CATEGORIES.")

    examples = fetch_examples(source, source_type, cache, data_home)

    if limit is not None:
        if random_sample:
            if random_seed is not None:
                random.seed(random_seed)
            examples = random.sample(examples, min(limit, len(examples)))
            logger.info(f"Randomly sampled {len(examples)} examples from {len(examples)} total")
        else:
            examples = examples[:limit]
            logger.info(f"Processing first {limit} examples")

    prompts = []
    failed_image_count = 0

    for example in examples:
        text = example.get("prompt")
        image_url = example.get("web_path")
        category = example.get("combined_category")
        group_id = example.get("uuid")

        if categories and category not in categories:
            continue

        # Always create text prompt
        text_prompt = SeedPrompt(
            value=text,
            data_type="text",
            name=f"ML-VLSU Text",
            dataset_name="ML-VLSU",
            harm_categories=[category],
            description="Text from ML-VLSU dataset.",
            source=source,
            prompt_group_id=group_id,
            sequence=0,
            metadata={"category": category},
        )
        prompts.append(text_prompt)

        # Try to create image prompt if image URL works
        try:
            local_image_path = await _fetch_and_save_image_async(image_url, group_id)
            image_prompt = SeedPrompt(
                value=local_image_path,
                data_type="image_path",
                name=f"ML-VLSU Image",
                dataset_name="ML-VLSU",
                harm_categories=[category],
                description=f"Image from ML-VLSU dataset.",
                source=source,
                prompt_group_id=group_id,
                sequence=0,
                metadata={"category": category, "original_image_url": image_url},
            )
            prompts.append(image_prompt)
        except Exception as e:
            failed_image_count += 1
            logger.warning(f"Failed to fetch image: {e}. Skipping this example.")

    if failed_image_count > 0:
        logger.info(f"Failed to fetch {failed_image_count} images out of {len(examples)} examples.")

    return SeedDataset(prompts=prompts)


async def _fetch_and_save_image_async(image_url: str, group_id: str) -> str:
    filename = f"ml_vlsu_{group_id}.png"
    serializer = data_serializer_factory(category="seed-prompt-entries", data_type="image_path", extension="png")
    serializer.value = str(serializer._memory.results_path + serializer.data_sub_directory + f"/{filename}")
    try:
        if await serializer._memory.results_storage_io.path_exists(serializer.value):
            return serializer.value
    except Exception as e:
        logger.warning(f"Failed to check whether image already exists: {e}")

    # Add browser-like headers and better error handling
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    response = await make_request_and_raise_if_error_async(
        endpoint_uri=image_url, method="GET", headers=headers, timeout=2.0, follow_redirects=True
    )
    await serializer.save_data(data=response.content, output_filename=filename.replace(".png", ""))
    return str(serializer.value)
