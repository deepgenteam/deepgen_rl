# Copyright 2025 Ruihang Li and DeepGen Team @ Shanghai Innovation Institute

"""
UniGenBench Evaluation Module for DeepGen-RL.

This module provides UniGenBench scoring functionality for evaluating
text-to-image generation models using a VLM judge model.

Based on: https://github.com/CodeGoat24/UnifiedReward/UniGenBench

CSV Format (unified columns):
    index,prompt,sub_dims

    - index: Integer index for the prompt
    - prompt: The text prompt
    - sub_dims: JSON string with testpoints info, e.g.:
      {"Testpoints": ["Style", "World Knowledge"], "Testpoint Description": ["ink painting", "pyramids"]}

Usage:
    1. Deploy UniGenBench-EvalModel via vLLM:
       vllm serve CodeGoat24/UniGenBench-EvalModel-qwen-72b-v1 \\
           --host localhost --port 8080 ...

    2. Set environment variable:
       export UNIGENBENCH_API_URL=http://localhost:8080

    3. Configure in eval.yaml:
       datasets:
         - name: unigenbench_en
           path: unigenbench/test_prompts_en.csv
           duplicates: 4
           scoring: unigenbench
"""

import os
import re
import ast
import json
import base64
import pandas as pd
from io import BytesIO
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
import random

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class UniGenBenchScoringConfig:
    """Configuration for UniGenBench scoring."""
    type: str = "unigenbench"
    # CSV path is set from the dataset's path field
    csv_path: Optional[str] = None
    # Language for judge prompt: "en" (English) or "zh" (Chinese)
    language: str = "en"

    @classmethod
    def from_string(cls, scoring_type: str) -> "UniGenBenchScoringConfig":
        """
        Create config from simple string format.

        Supports formats:
            - "unigenbench" -> defaults to English
            - "unigenbench/en" -> English
            - "unigenbench/zh" -> Chinese

        Args:
            scoring_type: Scoring type string (e.g., "unigenbench", "unigenbench/en", "unigenbench/zh")

        Returns:
            UniGenBenchScoringConfig instance
        """
        # Parse language from scoring type (e.g., "unigenbench/en" -> "en")
        if "/" in scoring_type:
            parts = scoring_type.split("/")
            base_type = parts[0]
            language = parts[1] if len(parts) > 1 else "en"
        else:
            base_type = scoring_type
            language = "en"  # Default to English

        return cls(type=base_type, language=language)


# ============================================================================
# VLM Client for API Calls
# ============================================================================

# Default model name for UniGenBench evaluation
# Can be overridden via UNIGENBENCH_MODEL_NAME environment variable
DEFAULT_UNIGENBENCH_MODEL_NAME = "UniGenBench-EvalModel-qwen3vl-32b-v1"


class VLMJudgeClient:
    """
    Client for calling VLM judge model via vLLM API.

    Handles image encoding, request batching, retries, and response parsing.
    """

    def __init__(
        self,
        api_url: str,
        model_name: Optional[str] = None,
        timeout_base: int = 120,
        max_retries: int = 10,
        backoff_base: float = 2.0,
        backoff_cap: float = 30.0,
        pool_maxsize: int = 16,
    ):
        self.api_url = api_url
        # Get model name from parameter, env var, or default
        self.model_name = model_name or os.environ.get(
            "UNIGENBENCH_MODEL_NAME", DEFAULT_UNIGENBENCH_MODEL_NAME
        )
        self.timeout_base = timeout_base
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.pool_maxsize = pool_maxsize
        self._local = threading.local()

    def _get_session(self) -> requests.Session:
        """Get thread-local session with connection pooling."""
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=self.pool_maxsize,
                pool_maxsize=self.pool_maxsize,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _encode_image(self, image_path: str) -> str:
        """Encode image file to base64 string."""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_messages(self, image_path: str, system_prompt: str) -> List[Dict]:
        """Build chat messages for VLM API."""
        base64_image = self._encode_image(image_path)
        image_url = f"data:image/jpeg;base64,{base64_image}"

        return [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]

    # English explanation dictionary for testpoints
    EXPLANATION_DICT_EN = {
        "Relationship - Comparison": "Comparison of attributes between two entities",
        "Relationship - Composition": "An entity is composed of one or more other entities",
        "Relationship - Inclusion": "A container contains an entity; the container can also be a plane, e.g., a snake in a painting on a wall",
        "Relationship - Similarity": "Existence of similarities between different entities",
        "Compound - Imagination": "Things that are impossible in real life",
        "Compound - Feature Matching": "Different entities possess different types of attribute features",
        "Attribute - Size": "Assessment of the subject's size, height, length, thickness, width, or tallness/shortness",
        "Attribute - Expression": "Distinguishing expressions from facial actions; expressions must convey a clear emotion",
        "Attribute - Quantity": "Focuses on the challenge of depicting three or more items accurately",
        "Attribute - Material": "Evaluation of different material types and textures",
        "Attribute - Color": "Assessment of different colors",
        "Attribute - Shape": "Assessment of different shapes",
        "Entity Layout - Two-Dimensional Space": "Arrangement and positioning of entities in two-dimensional space",
        "Entity Layout - Three-Dimensional Space": "Arrangement and positioning of entities in three-dimensional space",
        "Action - Full-body (Character/Anthropomorphic)": "Full-body actions by characters or anthropomorphized entities, such as running, diving, breakdancing, swinging, or hanging upside down",
        "Action - Hand (Character/Anthropomorphic)": "Focuses on hand structure—checking if fingers are missing, broken, or distorted",
        "Action - Animal": "Actions performed by animals",
        "Action - Contact Interaction": "Physical interactions between entities",
        "Action - Non-contact Interaction": "For example, two people making eye contact—testing if the model can accurately depict such interactions",
        "Action - State": "A sustained state of an entity, typically expressed with a verb",
        "Grammar - Negation": "Tests the model's understanding of negation grammar",
        "Grammar - Pronoun Reference": "Tests if the model can resolve ambiguous pronoun references correctly",
        "Grammar - Consistency": "Evaluation of shared attributes among entities",
        "World Knowledge": "Covers knowledge of celebrities, architecture, basic domain knowledge, and internet slang. Celebrities with modern copyright risk should be avoided",
        "Style": "Art, painting, photography, design styles, and corresponding artist names",
        "Text Generation": "The text content model needed to accurately generate without any omissions or extra words",
        "Text Generation (Case-insensitive)": "The text content model needed to accurately generate without any omissions or extra words, but ignore the case of the text - which means even the generated text is not in the same case as the prompt while the spelling is correct, it should still be considered as a correct answer",
        "Logical Reasoning": "Requires the model to deeply understand the intent and perform reasoning",
    }

    # Chinese explanation dictionary for testpoints
    EXPLANATION_DICT_ZH = {
        '关系-比较关系': '两者的属性对比',
        '关系-构成关系': '一个实体由另一种或几种实体构成',
        '关系-包含关系': '容器对实体的包含关系，容器也可以是平面的，比如：墙上的画里有一只蛇',
        '关系-相似关系': '不同实体中存在的相似关系',
        '复合考点-想象力': '现实生活中不可能发生的事情',
        '复合考点-不同实体特征匹配': '不同实体拥有不同类的属性特征',
        '实体布局-三维空间': '对于三维空间实体的摆放布局',
        '实体布局-二维空间': '对于二维空间实体的摆放布局',
        '属性-大小': '对主体 大小/高低/长短/粗细/宽窄/高矮',
        '属性-表情': '区分表情和脸部动作，脸部动作组成表情，但表情是一定要体现出某种情绪的。',
        '属性-数量': '重点考察三个或三个以上的数字难点',
        '属性-材质': '考察不同材质',
        '动作-人物/拟人全身动作': '人物或拟人全身性的动作，比如奔跑、跳水、跳街舞、荡秋千、倒挂金钩等',
        '动作-人物/拟人手部动作': '针对手部结构的考点，考核手指是否有缺失、崩坏等问题',
        '动作-动物动作': '动物的动作',
        '动作-实体间有接触互动': '各种实体间的有接触互动',
        '动作-实体间无接触互动': '比如两个人对视，考核模型能否把对视关系画对',
        '动作-状态': '实体持续的状态，一般是一个动词。',
        '语法-否定': '考察模型对于否定语法的掌握程度',
        '语法-代词指代': '这里的代词通常是有一些迷惑性的，考察模型能否正确对应',
        '语法-统一性': '实体共同属性的考察',
        '世界知识': '名人、建筑、基础的领域知识、网络流行语。其中名人不要使用当代有版权风险的名人',
        '风格': '艺术、绘画、摄影、设计风格，及对应艺术家名称',
        '逻辑推理': '需要模型深入理解意图并进行一定的推理',
        '文本生成': '考察模型能否准确生成不同语言，字体和长、短文字',
    }

    def evaluate_single(
        self,
        image_path: str,
        prompt: str,
        testpoints: List[str],
        testpoint_descriptions: List[str],
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Evaluate a single image using the VLM judge.

        Args:
            image_path: Path to the image file
            prompt: The generation prompt
            testpoints: List of testpoint names
            testpoint_descriptions: List of testpoint descriptions
            language: Language for judge prompt ("en" or "zh")

        Returns:
            Dict with evaluation results
        """
        # Select explanation dictionary based on language
        explanation_dict = self.EXPLANATION_DICT_ZH if language == "zh" else self.EXPLANATION_DICT_EN

        # Build explanation and system prompt based on language
        if language == "zh":
            # Chinese version
            explanation = "考点说明：「"
            for point in testpoints:
                if point in explanation_dict:
                    explanation += f"\n{point}: {explanation_dict[point]}"
                else:
                    explanation += f"\n{point}: (无定义)"
            explanation += "\n」"

            test_explanation = "考点描述说明：「"
            for idx, point in enumerate(testpoints):
                desc = testpoint_descriptions[idx] if idx < len(testpoint_descriptions) else ""
                test_explanation += f"\n{point}: {desc}"
            test_explanation += "\n」"

            system_prompt = f'''你是一个精确且客观的中文图像描述系统。我会给你一段生成图像的提示词，以及对应的生成图像，同时对于生成图像与提示词之间相关性的考点及对应说明，你需要逐个考点来判断生成的图像是否遵从了提示词中所包含的对应考点要求。

针对每张图像，你需要按照顺序完成如下的任务：
1. 这张生成图像对应的提示词为「{prompt}」，你需要根据{testpoints}中的这些角度逐个对图像内容进行更进一步的详细分析，考点的详细说明如下：{explanation}，各个考点在这条prompt中对应的描述说明如下：{test_explanation}, 你需要根据考点逐一判断生成图像是否满足了考点对应的要求
2. 综合上述回答，你需要逐个考点判断生成的图像在考点关注维度上是否符合输入的prompt，如果满足要求则该考点得分为1，否则为0

约束条件：
- 仅描述直接可见的内容；不要进行解读、推测或暗示背景故事。
- 专注于能够确定性陈述的视觉细节。
- 省略不确定或不清晰的细节。
- 即使输入中存在，也不要描述抽象实体、情感或推测。

请严格遵循以下输出格式：

<description>
    <prompt>{prompt}</prompt>
    <checkpoint>{testpoints}</checkpoint>
    <analysis>按照步骤1对于给定考点进行逐项详细分析，格式为一个方括号列表，**确保列表的长度与考点的数量相等**，每个元素为一个字符串，表示对于对应考点的分析</analysis>
    <score>按照步骤2逐个对考点进行打分，格式为一个方括号列表，**确保列表的长度与考点的数量相等**，每个元素为0或者1，表示对应考点是否完成</score>
</description>
'''
        else:
            # English version (default)
            explanation = "Checkpoints Definition:「"
            for point in testpoints:
                if point in explanation_dict:
                    explanation += f"\n{point}: {explanation_dict[point]}"
                else:
                    explanation += f"\n{point}: (No definition available)"
            explanation += "\n」"

            test_explanation = "Checkpoints Description:「"
            for idx, point in enumerate(testpoints):
                desc = testpoint_descriptions[idx] if idx < len(testpoint_descriptions) else ""
                test_explanation += f"\n{point}: {desc}"
            test_explanation += "\n」"

            system_prompt = f'''You are a precise and objective English-language image description system. I will provide you with a prompt for image generation, as well as the corresponding generated image. You will be given a set of evaluation criteria (checkpoints) and their explanations that define the relevance between the prompt and the image. You must evaluate whether the generated image fulfills the requirements implied by each checkpoint in the prompt.

For each image, follow the steps below in order:

1. The prompt for the generated image is: 「{prompt}」. You are to analyze the image content in detail from the angles specified in {testpoints}. Detailed definitions of these checkpoints are provided here: {explanation}. The specific description of each checkpoint in the context of the prompt is: {test_explanation}. You must analyze whether the image meets the requirements for each checkpoint individually.

2. Based on the above analysis, determine whether the generated image satisfies each checkpoint in terms of its visual alignment with the prompt. If the image meets the requirements of a checkpoint, assign a score of 1 to that checkpoint; otherwise, assign a score of 0.

Constraints:
- Only describe content that is directly visible; do not interpret, speculate, or infer any background story.
- Focus solely on visually verifiable details.
- Omit any uncertain or ambiguous elements.
- Even if mentioned in the input, do not describe abstract entities, emotions, or speculative ideas.

Please strictly follow the output format below:

<description>
    <prompt>{prompt}</prompt>
    <checkpoint>{testpoints}</checkpoint>
    <analysis>A list using square brackets `[]`, where each element is a string of detailed analysis corresponding to one checkpoint, as required in Step 1. **Ensure the list length matches the number of checkpoints**. Each element should be a string representing the analysis for that specific checkpoint.</analysis>
    <score>A list using square brackets `[]`, where each element is a binary score (0 or 1) corresponding to a checkpoint, as required in Step 2. **Ensure the list length matches the number of checkpoints**. Each element should be either 0 or 1, indicating whether the checkpoint was satisfied.</score>
</description>
'''

        # Call VLM API with retries
        attempt = 0
        last_error = None

        while attempt < self.max_retries:
            try:
                attempt += 1
                session = self._get_session()
                messages = self._build_messages(image_path, system_prompt)

                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "do_sample": False,
                    "max_tokens": 4096,
                }

                response = session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout_base + attempt * 10,
                )

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(f"Retryable HTTP {response.status_code}")

                response.raise_for_status()
                output = response.json()["choices"][0]["message"]["content"]

                # Parse response
                return self._parse_response(output, testpoints, prompt, image_path)

            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    sleep_time = min(
                        self.backoff_base ** attempt + random.uniform(0, 1),
                        self.backoff_cap,
                    )
                    time.sleep(sleep_time)

        # All retries failed
        return {
            "success": False,
            "error": last_error,
            "prompt": prompt,
            "image_path": image_path,
            "testpoints": testpoints,
            "scores": [0] * len(testpoints),  # Default to 0 on failure
        }

    def _parse_response(
        self,
        text: str,
        testpoints: List[str],
        prompt: str,
        image_path: str,
    ) -> Dict[str, Any]:
        """Parse VLM response to extract scores."""
        try:
            analysis_match = re.search(r'<analysis>(.*?)</analysis>', text, re.DOTALL)
            score_match = re.search(r'<score>(.*?)</score>', text, re.DOTALL)

            if not analysis_match or not score_match:
                return {
                    "success": False,
                    "error": "Could not parse analysis/score tags",
                    "raw_output": text,
                    "prompt": prompt,
                    "image_path": image_path,
                    "testpoints": testpoints,
                    "scores": [0] * len(testpoints),
                }

            analysis_str = analysis_match.group(1).strip()
            score_str = score_match.group(1).strip()

            analysis = ast.literal_eval(analysis_str)
            scores = ast.literal_eval(score_str)

            # Validate lengths match
            if len(scores) != len(testpoints):
                return {
                    "success": False,
                    "error": f"Score count mismatch: {len(scores)} vs {len(testpoints)}",
                    "raw_output": text,
                    "prompt": prompt,
                    "image_path": image_path,
                    "testpoints": testpoints,
                    "scores": [0] * len(testpoints),
                }

            return {
                "success": True,
                "prompt": prompt,
                "image_path": image_path,
                "testpoints": testpoints,
                "scores": scores,
                "analysis": analysis,
                "raw_output": text,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "raw_output": text,
                "prompt": prompt,
                "image_path": image_path,
                "testpoints": testpoints,
                "scores": [0] * len(testpoints),
            }


# ============================================================================
# Main Scorer Class
# ============================================================================

class UniGenBenchScorer:
    """
    UniGenBench scorer for evaluating text-to-image generation.

    Loads test prompts with testpoints from CSV, evaluates generated images
    using a VLM judge, and computes accuracy across dimensions.
    """

    def __init__(
        self,
        csv_path: str,
        api_url: Optional[str] = None,
        max_workers: Optional[int] = None,
        language: str = "en",
    ):
        """
        Initialize the scorer.

        Args:
            csv_path: Path to CSV file with columns: index, prompt, sub_dims
            api_url: VLM API URL (defaults to UNIGENBENCH_API_URL env var)
            max_workers: Max concurrent workers for API calls
            language: Language for judge prompt ("en" or "zh")
        """
        self.csv_path = csv_path
        self.language = language

        # Get API URL from env if not provided
        self.api_url = api_url or os.environ.get("UNIGENBENCH_API_URL")
        if not self.api_url:
            raise ValueError(
                "UniGenBench API URL not configured. "
                "Set UNIGENBENCH_API_URL environment variable or pass api_url parameter."
            )

        # Get max workers from env or default
        self.max_workers = max_workers or int(os.environ.get("UNIGENBENCH_WORKERS", "16"))

        # Initialize VLM client
        self.client = VLMJudgeClient(api_url=self.api_url, pool_maxsize=self.max_workers)

        # Log configuration
        print(f"[UniGenBench] API URL: {self.api_url}")
        print(f"[UniGenBench] Model Name: {self.client.model_name}")
        print(f"[UniGenBench] Max Workers: {self.max_workers}")
        print(f"[UniGenBench] Language: {self.language}")

        # Load test prompts CSV
        self.prompts_data: Dict[int, Dict[str, Any]] = {}
        self._load_prompts_csv()

    def _load_prompts_csv(self) -> None:
        """
        Load test prompts CSV with unified column names.

        Expected columns: index, prompt, sub_dims
        """
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"Test prompts CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Validate required columns
        required_cols = ["index", "prompt", "sub_dims"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV missing required columns: {missing_cols}. "
                f"Expected columns: {required_cols}. "
                f"Found columns: {list(df.columns)}"
            )

        for _, row in df.iterrows():
            index = int(row["index"])
            prompt = row["prompt"]

            # Parse sub_dims JSON
            subdims_str = row["sub_dims"]
            try:
                subdims = json.loads(subdims_str) if isinstance(subdims_str, str) else subdims_str
            except json.JSONDecodeError:
                subdims = {}

            testpoints = subdims.get("Testpoints", [])
            testpoint_desc = subdims.get("Testpoint Description", [])

            self.prompts_data[index] = {
                "prompt": prompt,
                "testpoints": testpoints,
                "testpoint_description": testpoint_desc,
                "subdims": subdims,
            }

        print(f"[UniGenBench] Loaded {len(self.prompts_data)} prompts from {self.csv_path}")

    def score_images(
        self,
        image_dir: str,
        num_duplicates: int = 4,
        show_progress: bool = True,
    ) -> Dict[str, Any]:
        """
        Score all images in a directory.

        Args:
            image_dir: Directory containing generated images
            num_duplicates: Number of images per prompt (default: 4)
            show_progress: Whether to show progress bar

        Returns:
            Dict with scoring results including per-dimension accuracy
        """
        # Collect all evaluation tasks
        tasks = []
        for index, data in self.prompts_data.items():
            for dup_idx in range(num_duplicates):
                # Image naming convention: {index}.{dup_idx}.png
                img_filename = f"{index}.{dup_idx}.png"
                img_path = os.path.join(image_dir, img_filename)

                if os.path.exists(img_path):
                    tasks.append({
                        "index": index,
                        "dup_idx": dup_idx,
                        "image_path": img_path,
                        "prompt": data["prompt"],
                        "testpoints": data["testpoints"],
                        "testpoint_description": data["testpoint_description"],
                    })

        if not tasks:
            print(f"[UniGenBench] Warning: No images found in {image_dir}")
            return {"success": False, "error": "No images found"}

        print(f"[UniGenBench] Scoring {len(tasks)} images...")

        # Execute evaluations in parallel
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self.client.evaluate_single,
                    task["image_path"],
                    task["prompt"],
                    task["testpoints"],
                    task["testpoint_description"],
                    self.language,  # Pass language parameter
                ): task
                for task in tasks
            }

            progress_iter = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="[UniGenBench] Evaluating",
                disable=not show_progress,
            )

            for future in progress_iter:
                task = futures[future]
                try:
                    result = future.result()
                    result["index"] = task["index"]
                    result["dup_idx"] = task["dup_idx"]
                    results.append(result)
                except Exception as e:
                    results.append({
                        "success": False,
                        "error": str(e),
                        "index": task["index"],
                        "dup_idx": task["dup_idx"],
                        "testpoints": task["testpoints"],
                        "scores": [0] * len(task["testpoints"]),
                    })

        # Compute statistics
        return self._compute_statistics(results)

    def _compute_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute accuracy statistics from evaluation results.

        Returns:
            Dict with:
            - overall_accuracy: Overall accuracy across all testpoints
            - primary_dims: Dict of primary dimension accuracies
            - sub_dims: Dict of sub-dimension accuracies
            - results_csv: List of results suitable for saving to CSV
        """
        primary_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        sub_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        success_count = 0
        results_csv = []

        for result in results:
            if result.get("success", False):
                success_count += 1

            testpoints = result.get("testpoints", [])
            scores = result.get("scores", [])

            # Record for CSV
            results_csv.append({
                "index": result.get("index"),
                "dup_idx": result.get("dup_idx"),
                "success": result.get("success", False),
                "testpoints": str(testpoints),
                "scores": str(scores),
                "raw_output": result.get("raw_output", ""),
            })

            # Aggregate statistics
            for cp, score in zip(testpoints, scores):
                # Determine primary and sub dimension
                if " - " in cp:
                    primary = cp.split(" - ", 1)[0].strip()
                    sub = cp
                else:
                    primary = cp
                    sub = cp

                primary_stats[primary]["total"] += 1
                sub_stats[sub]["total"] += 1

                if score == 1:
                    primary_stats[primary]["correct"] += 1
                    sub_stats[sub]["correct"] += 1

        # Compute accuracies
        primary_acc = {}
        for dim, stats in primary_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            primary_acc[dim] = {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
            }

        sub_acc = {}
        for dim, stats in sub_stats.items():
            acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
            sub_acc[dim] = {
                "accuracy": acc,
                "correct": stats["correct"],
                "total": stats["total"],
            }

        # Compute overall accuracy
        total_correct = sum(s["correct"] for s in sub_stats.values())
        total_count = sum(s["total"] for s in sub_stats.values())
        overall_acc = total_correct / total_count if total_count > 0 else 0.0

        return {
            "success": True,
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_count": total_count,
            "success_rate": success_count / len(results) if results else 0.0,
            "primary_dims": primary_acc,
            "sub_dims": sub_acc,
            "results_csv": results_csv,
        }

    def format_wandb_metrics(
        self,
        stats: Dict[str, Any],
        prefix: str = "eval/unigenbench",
    ) -> Dict[str, float]:
        """
        Format statistics for wandb logging.

        Args:
            stats: Statistics from score_images()
            prefix: Metric name prefix

        Returns:
            Dict of metric_name -> value for wandb.log()

        Metric format examples:
            - eval/unigenbench_en/overall_accuracy
            - eval/unigenbench_en/Style
            - eval/unigenbench_en/Attribute.Attribute-Quantity
        """
        metrics = {}

        # Overall accuracy
        metrics[f"{prefix}/overall_accuracy"] = stats.get("overall_accuracy", 0.0)
        metrics[f"{prefix}/success_rate"] = stats.get("success_rate", 0.0)

        # Primary and sub dimension accuracies
        # Format: eval/<dataset_name>/<Primary>.<Sub> where Sub uses "-" instead of " - "
        for sub_dim, data in stats.get("sub_dims", {}).items():
            # Determine primary dimension
            if " - " in sub_dim:
                primary = sub_dim.split(" - ", 1)[0].strip()
                # Convert "Attribute - Quantity" to "Attribute-Quantity"
                sub_clean = sub_dim.replace(" - ", "-")
            else:
                primary = sub_dim
                sub_clean = sub_dim

            # Format: eval/unigenbench_en/Primary.Sub-Dim
            if primary != sub_dim:
                metric_name = f"{prefix}/{primary}.{sub_clean}"
            else:
                metric_name = f"{prefix}/{sub_clean}"

            metrics[metric_name] = data["accuracy"]

        return metrics

    def print_results(self, stats: Dict[str, Any]) -> None:
        """Print formatted evaluation results."""
        print("\n" + "=" * 70)
        print("UniGenBench Evaluation Results")
        print("=" * 70)

        print(f"\nOverall Accuracy: {stats['overall_accuracy']:.2%}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Total Testpoints: {stats['total_count']}")

        print("\n📘 Primary Dimension Results:")
        for dim, data in sorted(stats.get("primary_dims", {}).items()):
            print(f"  - {dim}: {data['correct']}/{data['total']} = {data['accuracy']:.2%}")

        print("\n📗 Sub Dimension Results:")
        for dim, data in sorted(stats.get("sub_dims", {}).items()):
            print(f"  - {dim}: {data['correct']}/{data['total']} = {data['accuracy']:.2%}")

        print("=" * 70 + "\n")


# ============================================================================
# Utility Functions
# ============================================================================

def is_unigenbench_enabled() -> bool:
    """Check if UniGenBench evaluation is enabled (API URL configured)."""
    return os.environ.get("UNIGENBENCH_API_URL") is not None


def parse_scoring_config(
    scoring_value: Any,
    config_dir: str,
) -> Optional[UniGenBenchScoringConfig]:
    """
    Parse scoring configuration from eval.yaml.

    Supports multiple formats:
        - scoring: unigenbench          # defaults to English
        - scoring: unigenbench/en       # English
        - scoring: unigenbench/zh       # Chinese
        - scoring:
            type: unigenbench
            language: zh

    Args:
        scoring_value: Scoring config value from YAML (string or dict)
        config_dir: Directory of eval.yaml for resolving relative paths

    Returns:
        UniGenBenchScoringConfig instance or None if not unigenbench type
    """
    if scoring_value is None:
        return None

    # Simple string format: scoring: unigenbench or unigenbench/en or unigenbench/zh
    if isinstance(scoring_value, str):
        # Check if it starts with "unigenbench"
        if scoring_value.startswith("unigenbench"):
            return UniGenBenchScoringConfig.from_string(scoring_value)
        return None

    # Dict format: scoring: {type: unigenbench, language: zh, ...}
    if isinstance(scoring_value, dict):
        scoring_type = scoring_value.get("type", "")
        if scoring_type == "unigenbench" or scoring_type.startswith("unigenbench/"):
            language = scoring_value.get("language", "en")
            # Also support type: unigenbench/zh format
            if "/" in scoring_type:
                language = scoring_type.split("/")[1]
            return UniGenBenchScoringConfig(type="unigenbench", language=language)
        return None

    # List format (legacy): scoring: [{type: unigenbench, ...}]
    if isinstance(scoring_value, list):
        for item in scoring_value:
            if isinstance(item, dict):
                item_type = item.get("type", "")
                if item_type == "unigenbench" or item_type.startswith("unigenbench/"):
                    language = item.get("language", "en")
                    if "/" in item_type:
                        language = item_type.split("/")[1]
                    return UniGenBenchScoringConfig(type="unigenbench", language=language)
        return None

    return None
