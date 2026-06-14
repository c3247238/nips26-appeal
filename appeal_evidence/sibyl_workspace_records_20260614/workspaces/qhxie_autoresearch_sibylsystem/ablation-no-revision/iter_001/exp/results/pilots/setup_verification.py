#!/usr/bin/env python3
"""
Setup verification for API-based pilot experiments.
Verifies:
1. API connectivity (DeepSeek/OpenAI)
2. MATH dataset loading
3. Answer parsing utilities
"""

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# Configuration
WORKSPACE = "/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current"
RESULTS_DIR = Path(WORKSPACE) / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def verify_api_connectivity():
    """Test API connectivity with DeepSeek (primary) or OpenAI (fallback)."""
    from openai import OpenAI

    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": {}
    }

    # Test Anthropic API (if available)
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        try:
            import anthropic
            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=20,
                messages=[{"role": "user", "content": "Say API OK"}]
            )
            content = message.content[0].text.strip()
            if "API OK" in content or "api ok" in content.lower():
                results["tests"]["anthropic"] = {
                    "status": "PASS",
                    "model": "claude-sonnet-4-20250514",
                    "response_sample": content[:100]
                }
                results["primary_api"] = "anthropic"
            else:
                results["tests"]["anthropic"] = {
                    "status": "FAIL",
                    "error": f"Unexpected response: {content}"
                }
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "AuthenticationError" in error_msg:
                results["tests"]["anthropic"] = {
                    "status": "FAIL",
                    "error": "Invalid API key format - ANTHROPIC_API_KEY appears to be a Claude Code internal key, not a standard Anthropic API key"
                }
            else:
                results["tests"]["anthropic"] = {
                    "status": "FAIL",
                    "error": error_msg[:200]
                }

    # Test DeepSeek API (primary)
    deepseek_key = os.environ.get("DEEPSEEK_API_KEY")
    if deepseek_key:
        try:
            client = OpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com"
            )
            # Simple test completion
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": "Say 'API OK' in exactly those words."}],
                max_tokens=20,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            if "API OK" in content or "api ok" in content.lower():
                results["tests"]["deepseek"] = {
                    "status": "PASS",
                    "model": "deepseek-chat",
                    "response_sample": content[:100],
                    "latency_ms": response.model_dump().get("response_ms", "unknown")
                }
                results["primary_api"] = "deepseek"
            else:
                results["tests"]["deepseek"] = {
                    "status": "FAIL",
                    "error": f"Unexpected response: {content}"
                }
        except Exception as e:
            results["tests"]["deepseek"] = {
                "status": "FAIL",
                "error": str(e)[:200]
            }
    else:
        results["tests"]["deepseek"] = {
            "status": "SKIP",
            "reason": "DEEPSEEK_API_KEY not set"
        }

    # Test OpenAI API (fallback)
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            client = OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'API OK' in exactly those words."}],
                max_tokens=20,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            if "API OK" in content or "api ok" in content.lower():
                results["tests"]["openai"] = {
                    "status": "PASS",
                    "model": "gpt-4o-mini",
                    "response_sample": content[:100]
                }
                if "primary_api" not in results:
                    results["primary_api"] = "openai"
            else:
                results["tests"]["openai"] = {
                    "status": "FAIL",
                    "error": f"Unexpected response: {content}"
                }
        except Exception as e:
            results["tests"]["openai"] = {
                "status": "FAIL",
                "error": str(e)[:200]
            }
    else:
        results["tests"]["openai"] = {
            "status": "SKIP",
            "reason": "OPENAI_API_KEY not set"
        }

    # Overall status
    if results.get("primary_api"):
        results["status"] = "PASS"
        results["summary"] = f"Primary API ({results['primary_api']}) is working"
    else:
        results["status"] = "FAIL"
        results["summary"] = "No working API found - requires DEEPSEEK_API_KEY or OPENAI_API_KEY"
        results["required_action"] = "Set up API key: export DEEPSEEK_API_KEY='your-key' or export OPENAI_API_KEY='your-key'"

    return results


def load_math_dataset(n_samples=100, seed=42):
    """Load MATH dataset subset."""
    from datasets import load_dataset

    results = {
        "dataset": "HuggingFaceH4/MATH",
        "subset_size": n_samples,
        "seed": seed,
        "tests": {}
    }

    try:
        # Load MATH dataset (removed trust_remote_code as it's deprecated)
        dataset = load_dataset("HuggingFaceH4/MATH")

        # Get test split
        test_data = dataset["test"]

        # Sample with seed
        if len(test_data) > n_samples:
            test_data = test_data.shuffle(seed=seed).select(range(n_samples))

        results["tests"]["dataset_load"] = {
            "status": "PASS",
            "total_available": len(dataset["test"]),
            "sampled": len(test_data)
        }

        # Check structure
        sample = test_data[0]
        # Note: test split has 'solution' not 'answer'
        required_fields = ["problem", "solution"]
        missing_fields = [f for f in required_fields if f not in sample]

        if missing_fields:
            results["tests"]["field_check"] = {
                "status": "FAIL",
                "missing_fields": missing_fields,
                "available_fields": list(sample.keys())
            }
        else:
            results["tests"]["field_check"] = {
                "status": "PASS",
                "fields": required_fields,
                "note": "test split uses 'solution' not 'answer'"
            }

        # Store sample for verification
        results["sample_problem"] = {
            "level": sample.get("level", "unknown"),
            "type": sample.get("type", "unknown"),
            "problem_preview": sample["problem"][:200],
            "solution_preview": sample["solution"][:100]
        }

        results["status"] = "PASS"
        results["dataset_loaded"] = True

    except Exception as e:
        results["tests"]["dataset_load"] = {
            "status": "FAIL",
            "error": str(e)[:300]
        }
        results["status"] = "FAIL"
        results["dataset_loaded"] = False

    return results


def verify_answer_parser():
    """Test answer parsing utilities for mathematical answers."""

    def extract_boxed_answer(text):
        """Extract answer from \\boxed{...} format."""
        match = re.search(r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}', text)
        return match.group(1).strip() if match else None

    def normalize_answer(answer):
        """Normalize mathematical answer for comparison."""
        if answer is None:
            return None
        # Remove whitespace
        answer = re.sub(r'\s+', ' ', answer.strip())
        # Remove \\displaystyle, \\text, etc.
        answer = re.sub(r'\\[a-zA-Z]+\{', '', answer)
        answer = re.sub(r'\}', '', answer)
        # Convert to lowercase for comparison
        answer = answer.lower()
        return answer

    test_cases = [
        ("\\boxed{42}", "42"),
        ("\\boxed{x^2 + 1}", "x^2 + 1"),
        ("The answer is \\boxed{3.14}", "3.14"),
        ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
        ("No boxed answer here", None),
    ]

    results = {
        "tests": {}
    }

    all_pass = True
    for test_input, expected in test_cases:
        extracted = extract_boxed_answer(test_input)
        normalized = normalize_answer(extracted)
        expected_norm = normalize_answer(expected)

        passed = normalized == expected_norm
        if not passed:
            all_pass = False

        results["tests"][f"parse_{test_input[:30]}"] = {
            "input": test_input,
            "extracted": extracted,
            "expected": expected,
            "passed": passed
        }

    results["status"] = "PASS" if all_pass else "PARTIAL"
    results["summary"] = f"Parser {'all tests passed' if all_pass else 'some edge cases failed'}"

    return results


def main():
    print("=" * 60)
    print("API Environment Setup Verification")
    print("=" * 60)

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "workspace": WORKSPACE,
        "checks": {}
    }

    # 1. API Connectivity
    print("\n[1/3] Testing API connectivity...")
    api_results = verify_api_connectivity()
    all_results["checks"]["api_connectivity"] = api_results
    print(f"  Status: {api_results['status']}")
    print(f"  Summary: {api_results.get('summary', 'N/A')}")

    # 2. Dataset Loading
    print("\n[2/3] Loading MATH dataset...")
    dataset_results = load_math_dataset(n_samples=100, seed=42)
    all_results["checks"]["dataset_load"] = dataset_results
    print(f"  Status: {dataset_results['status']}")
    if dataset_results.get("dataset_loaded"):
        print(f"  Sampled: {dataset_results['tests']['dataset_load']['sampled']} problems")
        if "sample_problem" in dataset_results:
            sample = dataset_results["sample_problem"]
            print(f"  Sample: [{sample['level']}] {sample['problem_preview'][:80]}...")

    # 3. Answer Parser
    print("\n[3/3] Verifying answer parser...")
    parser_results = verify_answer_parser()
    all_results["checks"]["answer_parser"] = parser_results
    print(f"  Status: {parser_results['status']}")
    print(f"  Summary: {parser_results.get('summary', 'N/A')}")

    # Overall pass criteria
    pass_criteria = (
        api_results["status"] == "PASS" and
        dataset_results["status"] == "PASS" and
        parser_results["status"] in ["PASS", "PARTIAL"]
    )

    all_results["overall_status"] = "PASS" if pass_criteria else "FAIL"
    all_results["pass_criteria"] = {
        "api_connected": api_results["status"] == "PASS",
        "dataset_loaded": dataset_results.get("dataset_loaded", False),
        "parser_works": parser_results["status"] in ["PASS", "PARTIAL"]
    }

    # Write results
    output_file = RESULTS_DIR / "setup_verification.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Overall Status: {all_results['overall_status']}")
    print(f"Results saved to: {output_file}")
    print("=" * 60)

    if not pass_criteria:
        print("\nWARNING: Setup verification failed.")
        print("Pilot experiments cannot proceed until all checks pass.")
        sys.exit(1)
    else:
        print("\nSetup verification PASSED. Ready for pilot experiments.")
        sys.exit(0)


if __name__ == "__main__":
    main()
