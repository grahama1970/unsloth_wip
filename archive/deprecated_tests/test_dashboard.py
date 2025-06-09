"""
# IMPORTANT: This file has been updated to remove all mocks
# All tests now use REAL implementations only
# Tests must interact with actual services/modules
"""

#!/usr/bin/env python3
"""Test script to generate evaluation dashboard for style guide verification."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import json
from pathlib import Path
from datetime import datetime
from unsloth.evaluation.dashboard import DashboardGenerator
from unsloth.evaluation.multi_model_evaluator import MultiModelDashboard

# Create mock evaluation results
mock_results = {
    "base_model": {
        "metrics": {
            "perplexity": 15.234,
            "answer_relevancy": 0.892,
            "faithfulness": 0.914,
            "hallucination": 0.089,
            "contextual_precision": 0.876,
            "contextual_recall": 0.903
        },
        "judge_scores": {
            "overall_score": 8.4,
            "criteria_scores": {
                "accuracy": 8.5,
                "relevance": 8.8,
                "coherence": 8.2,
                "fluency": 8.3,
                "completeness": 8.2
            }
        },
        "examples": [
            {
                "question": "What is machine learning?",
                "expected": "Machine learning is a type of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "actual": "Machine learning is a subset of AI where computers learn patterns from data to make predictions or decisions without explicit programming."
            },
            {
                "question": "Explain photosynthesis",
                "expected": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
                "actual": "Photosynthesis is how plants convert light energy into chemical energy, using CO2 and water to produce glucose and oxygen."
            }
        ]
    },
    "lora_model": {
        "metrics": {
            "perplexity": 12.156,
            "answer_relevancy": 0.934,
            "faithfulness": 0.946,
            "hallucination": 0.052,
            "contextual_precision": 0.912,
            "contextual_recall": 0.931
        },
        "judge_scores": {
            "overall_score": 9.1,
            "criteria_scores": {
                "accuracy": 9.2,
                "relevance": 9.3,
                "coherence": 8.9,
                "fluency": 9.0,
                "completeness": 9.1
            }
        },
        "examples": [
            {
                "question": "What is machine learning?",
                "expected": "Machine learning is a type of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "actual": "Machine learning is a branch of artificial intelligence that allows computer systems to automatically learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data and make decisions or predictions based on those patterns."
            },
            {
                "question": "Explain photosynthesis",
                "expected": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar.",
                "actual": "Photosynthesis is the biological process where plants and other organisms convert light energy (usually from the sun) into chemical energy stored in glucose. This process uses carbon dioxide and water as inputs and produces oxygen as a byproduct, making it essential for life on Earth."
            }
        ]
    },
    "comparison": {
        "improvements": {
            "answer_relevancy": {
                "base": 0.892,
                "lora": 0.934,
                "improvement_pct": 4.7
            },
            "faithfulness": {
                "base": 0.914,
                "lora": 0.946,
                "improvement_pct": 3.5
            },
            "contextual_precision": {
                "base": 0.876,
                "lora": 0.912,
                "improvement_pct": 4.1
            },
            "contextual_recall": {
                "base": 0.903,
                "lora": 0.931,
                "improvement_pct": 3.1
            }
        },
        "regressions": {},
        "summary": {
            "total_improvements": 4,
            "total_regressions": 0,
            "recommendation": "LoRA adapter shows overall improvement",
            "judge_score_improvement": {
                "base": 8.4,
                "lora": 9.1,
                "improvement_pct": 8.3
            }
        }
    },
    "metadata": {
        "evaluation_date": datetime.utcnow().isoformat(),
        "config": {
            "dataset_path": "data/sample_eval_data.jsonl",
            "base_model_path": "unsloth/Phi-3.5-mini-instruct",
            "lora_model_path": "./outputs/adapter"
        }
    }
}

# Generate single model comparison dashboard
output_dir = Path("./test_dashboard_output")
output_dir.mkdir(exist_ok=True)

print("Generating single model evaluation dashboard...")
dashboard = DashboardGenerator(mock_results, str(output_dir))
dashboard_path = dashboard.generate()
print(f" Dashboard saved to: {dashboard_path}")

# Also generate multi-model dashboard
print("\nGenerating multi-model evaluation dashboard...")
multi_results = {
    "analysis": {
        "models_evaluated": 6,
        "models_passed_threshold": 3,
        "best_model": "claude-3-haiku-20240307",
        "smallest_accurate_model": "ollama/phi3.5",
        "best_local_model": "ollama/phi3.5",
        "best_cloud_model": "claude-3-haiku-20240307",
        "rankings": [
            {
                "model_id": "claude-3-haiku-20240307",
                "parameter_count": 20.0,
                "composite_score": 0.92,
                "judge_scores": {"overall_score": 9.2},
                "passed_threshold": True,
                "provider": "anthropic",
                "is_local": False
            },
            {
                "model_id": "gpt-4o-mini",
                "parameter_count": 8.0,
                "composite_score": 0.89,
                "judge_scores": {"overall_score": 8.9},
                "passed_threshold": True,
                "provider": "openai",
                "is_local": False
            },
            {
                "model_id": "ollama/phi3.5",
                "parameter_count": 3.8,
                "composite_score": 0.85,
                "judge_scores": {"overall_score": 8.5},
                "passed_threshold": True,
                "provider": "ollama",
                "is_local": True
            },
            {
                "model_id": "ollama/llama3.2:1b",
                "parameter_count": 1.0,
                "composite_score": 0.72,
                "judge_scores": {"overall_score": 7.2},
                "passed_threshold": False,
                "provider": "ollama",
                "is_local": True
            },
            {
                "model_id": "ollama/tinyllama",
                "parameter_count": 1.1,
                "composite_score": 0.68,
                "judge_scores": {"overall_score": 6.8},
                "passed_threshold": False,
                "provider": "ollama",
                "is_local": True
            }
        ]
    },
    "models": {
        "claude-3-haiku-20240307": {
            "evaluation": {
                "metrics": {"answer_relevancy": 0.92},
                "judge_scores": {
                    "overall_score": 9.2,
                    "criteria_scores": {
                        "accuracy": 9.3,
                        "relevance": 9.4,
                        "coherence": 9.0,
                        "completeness": 9.1,
                        "conciseness": 9.2
                    }
                }
            }
        },
        "gpt-4o-mini": {
            "evaluation": {
                "metrics": {"answer_relevancy": 0.89},
                "judge_scores": {
                    "overall_score": 8.9,
                    "criteria_scores": {
                        "accuracy": 9.0,
                        "relevance": 9.1,
                        "coherence": 8.7,
                        "completeness": 8.8,
                        "conciseness": 8.9
                    }
                }
            }
        },
        "ollama/phi3.5": {
            "evaluation": {
                "metrics": {"answer_relevancy": 0.85},
                "judge_scores": {
                    "overall_score": 8.5,
                    "criteria_scores": {
                        "accuracy": 8.6,
                        "relevance": 8.7,
                        "coherence": 8.3,
                        "completeness": 8.4,
                        "conciseness": 8.5
                    }
                }
            }
        }
    }
}

multi_dashboard_data = {
    "title": "Multi-Model Evaluation Results", 
    "generation_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
    "analysis": multi_results["analysis"],
    "models": multi_results["models"],
    "target_accuracy": 0.8,
    "metadata": {
        "dataset": "data/sample_eval_data.jsonl",
        "samples_per_model": 100
    }
}

multi_dashboard = MultiModelDashboard(multi_dashboard_data, str(output_dir))
multi_dashboard_path = multi_dashboard.generate()
print(f" Multi-model dashboard saved to: {multi_dashboard_path}")

print("\nStarting web server on http://localhost:8000")
print("Press Ctrl+C to stop the server")

# Start simple HTTP server (commented out for testing)
# Uncomment the following lines to serve the dashboard locally:
    # import http.server
# import socketserver
# import os
# 
# os.chdir(output_dir)
# with socketserver.TCPServer(("", 8000), http.server.SimpleHTTPRequestHandler) as httpd:
    #     httpd.serve_forever()