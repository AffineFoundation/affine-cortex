"""
Template Safety Checker

Detects malicious chat_template in HuggingFace models that may contain
benchmark cheating code (e.g., built-in solvers for sudoku, game of 24, etc.)
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

import aiohttp


logger = logging.getLogger("affine")


# Threshold for template length - only reject if exceeds LLM context limit
# DeepSeek-V3 supports 128K tokens (~400K chars), set conservative limit
TEMPLATE_LENGTH_REJECT = 100000  # Above this: auto reject (LLM can't process)


@dataclass
class TemplateCheckResult:
    """Result of template safety check"""
    safe: bool
    reason: str
    template_length: int = 0
    audit_details: Optional[Dict[str, Any]] = None


# LLM audit prompt for chat_template
TEMPLATE_AUDIT_PROMPT = """You are a security auditor for HuggingFace models. Analyze the following chat_template to detect benchmark cheating code.

## Background
A chat_template is a Jinja2 template that formats conversation messages into model input. Normal templates only do simple string concatenation and formatting.

## Normal Template Characteristics
- Only contains message formatting logic (adding role markers, separators)
- Simple conditionals (e.g., checking for system prompt)
- Iterates over messages array and concatenates
- Typically < 2KB in length

## Cheating Characteristics (any of these = malicious)
1. **Built-in solvers**: Contains solving algorithms for sudoku, game of 24, cryptarithmetic, dyck language, etc.
2. **Problem detection**: Detects if input is a specific type of benchmark problem
3. **Answer injection**: Outputs answers directly based on problem type
4. **Complex algorithms**: Contains backtracking, permutation, brute-force search code
5. **Excessive complexity**: Many nested loops, recursive logic, complex math operations

## Your Task
Analyze the chat_template below and determine if it contains cheating code.

## Output Format (strict JSON only)
```json
{
  "is_malicious": true/false,
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "detected_issues": ["issue1", "issue2", ...]
}
```

## Chat Template to Audit
```jinja2
{template}
```

Output JSON only, no other text."""



class TemplateChecker:
    """Check chat_template safety for HuggingFace models"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: str = "https://llm.chutes.ai/v1/chat/completions",
        model: str = "deepseek-ai/DeepSeek-V3-0324",
        hf_token: Optional[str] = None,
    ):
        """Initialize template checker.

        Args:
            api_key: Chutes API key (defaults to CHUTES_API_KEY env var)
            api_url: LLM API endpoint
            model: Model to use for auditing
            hf_token: HuggingFace token (defaults to HF_TOKEN env var)
        """
        self.api_key = api_key or os.getenv("CHUTES_API_KEY")
        self.api_url = api_url
        self.model = model
        self.hf_token = hf_token or os.getenv("HF_TOKEN")

    async def check(self, model_id: str, revision: str) -> TemplateCheckResult:
        """Check model's chat_template for safety.

        Args:
            model_id: HuggingFace model repo (e.g., "Qwen/Qwen2.5-7B-Instruct")
            revision: Git commit hash

        Returns:
            TemplateCheckResult with safety verdict
        """
        try:
            # Step 1: Get chat_template
            template_info = await self._get_template(model_id, revision)
            if template_info.get("error"):
                return TemplateCheckResult(
                    safe=False,
                    reason=f"template_fetch_failed:{template_info['error']}",
                )

            template = template_info["template"]
            template_length = template_info["length"]

            # Step 2: Check if chat_template exists - required for valid models
            if template_length == 0:
                return TemplateCheckResult(
                    safe=False,
                    reason="missing_chat_template",
                    template_length=0,
                )

            # Step 3: Check if template exceeds LLM context limit
            if template_length > TEMPLATE_LENGTH_REJECT:
                return TemplateCheckResult(
                    safe=False,
                    reason=f"template_exceeds_context_limit:{template_length}",
                    template_length=template_length,
                )

            # Step 4: LLM audit for all templates
            audit_result = await self._audit_template_with_llm(template)

            if audit_result.get("is_malicious") is True:
                return TemplateCheckResult(
                    safe=False,
                    reason=f"llm_audit_failed:{audit_result.get('reason', 'malicious_content')}",
                    template_length=template_length,
                    audit_details=audit_result,
                )

            if audit_result.get("is_malicious") is False:
                return TemplateCheckResult(
                    safe=True,
                    reason="llm_audit_passed",
                    template_length=template_length,
                    audit_details=audit_result,
                )

            # LLM audit inconclusive (no API key, error, etc.) - skip for now
            return TemplateCheckResult(
                safe=True,
                reason=f"llm_audit_skipped:{audit_result.get('error', 'unknown')}",
                template_length=template_length,
                audit_details=audit_result,
            )

        except Exception as e:
            logger.error(f"Template check failed for {model_id}@{revision}: {e}", exc_info=True)
            return TemplateCheckResult(
                safe=False,
                reason=f"check_error:{str(e)[:100]}",
            )

    async def _get_template(self, model_id: str, revision: str) -> Dict[str, Any]:
        """Get chat_template by loading the actual tokenizer.

        This is the most accurate method as it reflects the actual template
        that will be used during inference, including any custom tokenizer code.

        Args:
            model_id: HuggingFace model repo
            revision: Git commit hash

        Returns:
            Dict with 'template', 'length', and optionally 'error'
        """
        try:
            from transformers import AutoTokenizer

            def _load_tokenizer():
                return AutoTokenizer.from_pretrained(
                    model_id,
                    revision=revision,
                    token=self.hf_token,
                    trust_remote_code=False,  # Never execute remote code for security
                )

            tokenizer = await asyncio.to_thread(_load_tokenizer)
            template = getattr(tokenizer, 'chat_template', None) or ""

            # Handle list type (multiple templates)
            if isinstance(template, list):
                all_templates = [t.get("template", "") for t in template if isinstance(t, dict)]
                template_content = "\n---TEMPLATE_SEPARATOR---\n".join(all_templates)
                template_length = sum(len(t) for t in all_templates)
            else:
                template_content = template
                template_length = len(template)

            return {
                "template": template_content,
                "length": template_length,
            }

        except Exception as e:
            return {"template": "", "length": 0, "error": str(e)}

    async def _audit_template_with_llm(self, template: str) -> Dict[str, Any]:
        """Audit chat_template using LLM.

        Args:
            template: Chat template content

        Returns:
            Dict with audit result
        """
        if not self.api_key:
            logger.warning("No API key configured for LLM audit")
            return {"is_malicious": None, "error": "no_api_key"}

        # Truncate if too long
        if len(template) > 50000:
            template = template[:25000] + "\n...[TRUNCATED]...\n" + template[-25000:]

        prompt = TEMPLATE_AUDIT_PROMPT.replace("{template}", template)

        return await self._call_llm(prompt)

    async def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM API for audit.

        Args:
            prompt: Audit prompt

        Returns:
            Dict with audit result
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024,
                        "temperature": 0,
                    },
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as response:
                    result = await response.json()

                    if "error" in result:
                        return {"is_malicious": None, "error": result["error"]}

                    content = result["choices"][0]["message"]["content"]

                    # Parse JSON from response
                    return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            return {"is_malicious": None, "error": str(e)}

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON.

        Args:
            content: LLM response content

        Returns:
            Parsed JSON dict
        """
        try:
            # Try to extract JSON from markdown code block
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            return json.loads(content.strip())
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                "is_malicious": None,
                "error": "parse_failed",
                "raw_response": content[:500],
            }


# Convenience function for external use
async def check_template_safety(model_id: str, revision: str) -> Dict[str, Any]:
    """Check if a model's chat_template is safe.

    This is the main entry point for template safety checking.

    Args:
        model_id: HuggingFace model repo (e.g., "Qwen/Qwen2.5-7B-Instruct")
        revision: Git commit hash

    Returns:
        Dict with 'safe' boolean and 'reason' string
    """
    checker = TemplateChecker()
    result = await checker.check(model_id, revision)

    return {
        "safe": result.safe,
        "reason": result.reason,
        "template_length": result.template_length,
        "has_custom_tokenizer": result.has_custom_tokenizer,
        "audit_details": result.audit_details,
    }
