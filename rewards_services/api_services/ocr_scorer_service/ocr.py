from paddleocr import PaddleOCR
import torch
import numpy as np
from Levenshtein import distance
from typing import List, Union, Tuple, Any
from PIL import Image

class OcrScorer:
    def __init__(self, use_gpu: bool = False):
        """
        OCR reward calculator
        :param use_gpu: Whether to use GPU acceleration for PaddleOCR
        """
        import os
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            # use_angle_cls=True,
            lang="en",
            use_gpu=use_gpu,
            show_log=False
        )
        # Runtime config (via env vars) to avoid changing trainer/service callsites.
        # Comments in English per repo convention.
        self.mode = os.environ.get("OCR_REWARD_MODE", "loose").strip().lower()
        self.case_sensitive = os.environ.get("OCR_CASE_SENSITIVE", "1").strip() in ("1", "true", "yes", "y")
        self.ignore_punct = os.environ.get("OCR_IGNORE_PUNCT", "1").strip() in ("1", "true", "yes", "y")
        self.min_conf = float(os.environ.get("OCR_MIN_CONF", "0.0"))
        self.ngram_max = int(os.environ.get("OCR_NGRAM_MAX", "3"))
        # If enabled, confusable characters are folded to the same canonical form before distance.
        # This reduces reward noise from OCR-specific confusions (e.g., W/V, O/0, I/l/1).
        self.use_confusion_fold = os.environ.get("OCR_USE_CONFUSION_FOLD", "1").strip() in ("1", "true", "yes", "y")

    def _extract_target_texts(self, prompt: str) -> List[str]:
        """
        Extract all text segments enclosed in double quotes from prompt.
        For poster-style prompts with multiple text segments.
        :param prompt: Input prompt string
        :return: List of target text segments
        """
        import re
        # Find all text between double quotes
        matches = re.findall(r'"([^"]+)"', prompt)
        return matches if matches else []

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for reward computation.
        - Removes whitespace by default
        - Optionally removes punctuation/symbols (keeps A-Z, a-z, 0-9)
        - Optionally lowercases (when case_sensitive=False)
        """
        import re
        if text is None:
            return ""
        s = str(text)
        # Remove all whitespace (spaces, tabs, newlines)
        s = re.sub(r"\s+", "", s)
        if self.ignore_punct:
            s = re.sub(r"[^A-Za-z0-9]+", "", s)
        if not self.case_sensitive:
            s = s.lower()
        return s

    def _confusion_fold_text(self, text: str) -> str:
        """
        Fold common OCR confusions into a canonical form to reduce reward noise.
        This is intentionally conservative: it should reduce variance but not
        fully hide true spelling errors.
        """
        if not text:
            return text
        # Work in lower-case space-less domain (normalize already does this).
        # Map groups to a single representative.
        table = {
            "0": "o",
            "1": "l",
            "i": "l",
            "j": "l",  # sometimes OCR confuses j/1/l in stylized fonts
            "5": "s",
            "v": "w",
        }
        return "".join(table.get(ch, ch) for ch in text)

    def _parse_ocr_tokens(self, ocr_result: Any) -> List[Tuple[str, float]]:
        """
        Parse PaddleOCR output into a list of (text, confidence).
        Expected structure: result = [ [ [box], (text, conf) ], ... ]
        """
        tokens: List[Tuple[str, float]] = []
        if not ocr_result:
            return tokens
        try:
            lines = ocr_result[0] if isinstance(ocr_result, list) and len(ocr_result) > 0 else []
            for item in lines or []:
                # item: [box, (text, conf)]
                if not item or len(item) < 2:
                    continue
                text_conf = item[1]
                if not text_conf or len(text_conf) < 2:
                    continue
                txt = text_conf[0]
                conf = float(text_conf[1]) if text_conf[1] is not None else 0.0
                if conf < self.min_conf:
                    continue
                if txt:
                    tokens.append((str(txt), conf))
        except Exception:
            # Be robust to format changes; treat as no tokens.
            return []
        return tokens

    def _find_best_match_in_candidates(self, candidates: List[str], target: str) -> Tuple[int, str]:
        """
        Find best match among candidate strings (already normalized).
        Returns (best_distance, best_candidate_original).
        """
        target_len = len(target)
        if target_len == 0:
            return 0, ""
        if not candidates:
            return target_len, ""

        best_dist: int = target_len
        best_cand: str = ""

        # Pre-compute folded target if enabled.
        if self.use_confusion_fold:
            folded_target = self._confusion_fold_text(target)
        else:
            folded_target = target

        for cand in candidates:
            if not cand:
                continue
            # Base distance
            d0 = distance(cand, target)
            d = d0
            # Confusion-folded distance (take the min to reduce OCR noise)
            if self.use_confusion_fold:
                dc = distance(self._confusion_fold_text(cand), folded_target)
                if dc < d:
                    d = dc
            if d < best_dist:
                best_dist = d
                best_cand = cand
                if best_dist == 0:
                    break

        return best_dist, best_cand

    @torch.no_grad()
    def __call__(self,
                images: Union[List[Image.Image], List[np.ndarray]],
                prompts: List[str]) -> List[float]:
        """
        Calculate OCR reward for poster-style images with multiple text segments.
        Text comparison is configurable via env vars (see __init__).
        :param images: List of input images (PIL or numpy format)
        :param prompts: Corresponding target text list (with text in double quotes)
        :return: Reward list (CPU)
        """
        import os
        # Check if debug mode is enabled via environment variable
        _debug_ocr = os.environ.get("JONB_DEBUG_OCR", None) is not None

        rewards = []
        # Ensure input lengths are consistent
        assert len(images) == len(prompts), "Images and prompts must have the same length"

        if _debug_ocr:
            print(f"[OCR DEBUG] __call__ started with {len(images)} images and {len(prompts)} prompts")

        for idx, (img, prompt) in enumerate(zip(images, prompts)):
            if _debug_ocr:
                print(f"\n[OCR DEBUG] Processing image {idx}")
                print(f"[OCR DEBUG] Prompt: {prompt}")

            # Convert image format
            if isinstance(img, Image.Image):
                if _debug_ocr:
                    print(f"[OCR DEBUG] Image type: PIL.Image, size: {img.size}, mode: {img.mode}")
                img = np.array(img)
            if _debug_ocr:
                print(f"[OCR DEBUG] Converted to numpy array, shape: {img.shape}, dtype: {img.dtype}")

            # Extract all target text segments from prompt
            target_texts = self._extract_target_texts(prompt)
            if _debug_ocr:
                print(f"[OCR DEBUG] Extracted target texts: {target_texts}")

            if not target_texts:
                if _debug_ocr:
                    print(f"[OCR DEBUG] Warning: No target text found in prompt: {prompt[:100]}...")
                rewards.append(0.0)
                continue

            try:
                # OCR recognition
                if _debug_ocr:
                    print(f"[OCR DEBUG] Running OCR...")
                result = self.ocr.ocr(img, cls=False)
                if _debug_ocr:
                    print(f"[OCR DEBUG] OCR raw result type: {type(result)}")
                    print(f"[OCR DEBUG] OCR raw result: {result}")

                # Parse OCR tokens; then build normalized candidate strings.
                tokens = self._parse_ocr_tokens(result)
                if _debug_ocr:
                    print(f"[OCR DEBUG] Parsed OCR tokens (text, conf): {tokens}")

                token_norms: List[str] = [self._normalize_text(t) for (t, _c) in tokens]
                # Keep only non-empty normalized tokens
                token_norms = [t for t in token_norms if t]

                # Build candidates:
                # - each token
                # - concatenation of adjacent tokens up to ngram_max
                # - full concatenation fallback (similar to previous behavior)
                candidates: List[str] = []
                candidates.extend(token_norms)
                nmax = max(1, int(self.ngram_max))
                if len(token_norms) >= 2 and nmax >= 2:
                    for n in range(2, nmax + 1):
                        for i0 in range(0, len(token_norms) - n + 1):
                            candidates.append("".join(token_norms[i0:i0 + n]))
                if token_norms:
                    candidates.append("".join(token_norms))
                # De-duplicate while preserving order
                seen = set()
                deduped: List[str] = []
                for c in candidates:
                    if c not in seen:
                        seen.add(c)
                        deduped.append(c)
                candidates = deduped

                if _debug_ocr:
                    print(f"[OCR DEBUG] Candidate count: {len(candidates)}")
                    print(f"[OCR DEBUG] First 20 candidates: {candidates[:20]}")

                # Calculate reward for each target text segment
                segment_rewards = []
                total_target_len = 0

                for target in target_texts:
                    target_normalized = self._normalize_text(target)
                    target_len = len(target_normalized)
                    total_target_len += target_len
                    if _debug_ocr:
                        print(f"[OCR DEBUG] Target: '{target}', normalized: '{target_normalized}', len: {target_len}")

                    # Find best match among candidates
                    dist, best_cand = self._find_best_match_in_candidates(candidates, target_normalized)
                    if _debug_ocr:
                        print(f"[OCR DEBUG] Best candidate: '{best_cand}'")
                        print(f"[OCR DEBUG] Best Levenshtein distance: {dist}")
                    # Cap the distance to target length
                    if dist > target_len:
                        dist = target_len
                        if _debug_ocr:
                            print(f"[OCR DEBUG] Capped dist to target_len: {dist}")

                    segment_reward = 1.0 - dist / target_len if target_len > 0 else 0.0
                    segment_rewards.append((segment_reward, target_len))
                    if _debug_ocr:
                        print(f"[OCR DEBUG] Segment reward: {segment_reward:.4f} (weight: {target_len})")

                # Weighted average based on text length
                if total_target_len > 0:
                    final_reward = sum(r * l for r, l in segment_rewards) / total_target_len
                else:
                    final_reward = 0.0
                if _debug_ocr:
                    print(f"[OCR DEBUG] Final reward for image {idx}: {final_reward:.4f}")

            except Exception as e:
                # Error handling (e.g., OCR parsing failure)
                import traceback
                if _debug_ocr:
                    print(f"[OCR DEBUG] OCR processing failed: {str(e)}")
                    print(f"[OCR DEBUG] Traceback: {traceback.format_exc()}")
                final_reward = 0.0

            rewards.append(final_reward)

        if _debug_ocr:
            print(f"\n[OCR DEBUG] All rewards: {rewards}")
            print(f"[OCR DEBUG] Mean reward: {sum(rewards)/len(rewards) if rewards else 0:.4f}")
        return rewards

if __name__ == "__main__":
    # Test with poster-style prompt containing multiple text segments
    example_image_path = "media_images_eval_images_499_ef42de47b8ec98892954.jpg"
    example_image = Image.open(example_image_path)

    # Example 1: Simple single text
    simple_prompt = 'New York Skyline with "Hello World" written with fireworks on the sky'

    # Example 2: Poster-style with multiple text segments
    poster_prompt = 'A movie poster with the title "INCEPTION" at the top, the tagline "Your mind is the scene of the crime" in the middle, and "Coming Soon July 2024" at the bottom'

    # Instantiate scorer
    scorer = OcrScorer(use_gpu=False)

    # Test with simple prompt
    reward = scorer([example_image], [simple_prompt])
    print(f"Simple prompt OCR Reward: {reward}")

    # Test with poster prompt (using same image for demo)
    reward = scorer([example_image], [poster_prompt])
    print(f"Poster prompt OCR Reward: {reward}")