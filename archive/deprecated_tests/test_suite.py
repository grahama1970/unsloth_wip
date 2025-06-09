"""Comprehensive test suite for inference with fine-tuned models."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))



import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd

from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

from .generate import InferenceEngine, GenerationConfig
from evaluation.litellm_evaluator import JudgeConfig
from llm_call import ask_with_retry


@dataclass
class TestCase:
    """Represents a single test case."""
    category: str
    question: str
    expected_keywords: List[str] = field(default_factory=list)
    expected_format: Optional[str] = None
    context: Optional[str] = None
    
    def to_prompt(self) -> str:
        """Convert to prompt format."""
        if self.context:
            return f"Context: {self.context}\n\nQuestion: {self.question}"
        return self.question


@dataclass
class TestResult:
    """Result of a single test."""
    test_case: TestCase
    response: str
    duration_ms: float
    tokens_per_second: float
    keywords_found: List[str]
    format_match: bool
    judge_score: Optional[float] = None
    judge_feedback: Optional[str] = None


class InferenceTestSuite:
    """Comprehensive test suite for model inference."""
    
    # Pre-defined test categories with example questions
    DEFAULT_TEST_CASES = [
        # Factual Knowledge
        TestCase(
            category="Factual Knowledge",
            question="What is the capital of France?",
            expected_keywords=["Paris"],
            expected_format="single_word_or_phrase"
        ),
        TestCase(
            category="Factual Knowledge", 
            question="Who wrote Romeo and Juliet?",
            expected_keywords=["Shakespeare", "William Shakespeare"],
            expected_format="name"
        ),
        
        # Reasoning
        TestCase(
            category="Reasoning",
            question="If all roses are flowers and all flowers need water, what do roses need?",
            expected_keywords=["water"],
            expected_format="logical_conclusion"
        ),
        TestCase(
            category="Reasoning",
            question="John is taller than Mary. Mary is taller than Sue. Who is the shortest?",
            expected_keywords=["Sue"],
            expected_format="single_name"
        ),
        
        # Math
        TestCase(
            category="Math",
            question="What is 15% of 200?",
            expected_keywords=["30"],
            expected_format="number"
        ),
        TestCase(
            category="Math",
            question="If a train travels 60 mph for 2.5 hours, how far does it go?",
            expected_keywords=["150", "miles"],
            expected_format="number_with_unit"
        ),
        
        # Code Generation
        TestCase(
            category="Code Generation",
            question="Write a Python function to calculate the factorial of a number.",
            expected_keywords=["def", "factorial", "return", "if", "else"],
            expected_format="python_code"
        ),
        TestCase(
            category="Code Generation",
            question="Write a SQL query to find all users older than 25.",
            expected_keywords=["SELECT", "FROM", "WHERE", "age", ">", "25"],
            expected_format="sql_query"
        ),
        
        # Creative Writing
        TestCase(
            category="Creative Writing",
            question="Write a haiku about spring.",
            expected_keywords=["spring", "blossom", "nature"],
            expected_format="haiku"
        ),
        TestCase(
            category="Creative Writing",
            question="Create a catchy slogan for an eco-friendly water bottle.",
            expected_keywords=["eco", "green", "sustainable", "water", "bottle"],
            expected_format="short_phrase"
        ),
        
        # Instruction Following
        TestCase(
            category="Instruction Following",
            question="List 3 benefits of regular exercise. Use bullet points.",
            expected_keywords=["•", "-", "*", "exercise", "health"],
            expected_format="bullet_list"
        ),
        TestCase(
            category="Instruction Following",
            question="Explain photosynthesis in exactly two sentences.",
            expected_keywords=["plants", "sunlight", "energy", "oxygen"],
            expected_format="two_sentences"
        ),
        
        # Domain-Specific (if trained on specific data)
        TestCase(
            category="Domain Knowledge",
            question="What are the main components of a transformer architecture?",
            expected_keywords=["attention", "encoder", "decoder", "self-attention", "feed-forward"],
            expected_format="technical_explanation"
        ),
        TestCase(
            category="Domain Knowledge",
            question="Explain the difference between supervised and unsupervised learning.",
            expected_keywords=["labeled", "unlabeled", "training", "clustering", "classification"],
            expected_format="comparison"
        ),
    ]
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./inference_test_results",
        use_judge: bool = True,
        judge_model: str = "gpt-4",
        device: str = "cuda",
        load_in_4bit: bool = True
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_judge = use_judge
        self.judge_model = judge_model
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.console = Console()
        self.results: List[TestResult] = []
        
        # Initialize inference engine
        self.engine = InferenceEngine(
            model_path=model_path,
            device=device,
            load_in_4bit=load_in_4bit
        )
    
    def add_custom_tests(self, test_file: str) -> None:
        """Load custom test cases from JSON file."""
        with open(test_file, 'r') as f:
            custom_tests = json.load(f)
        
        for test in custom_tests:
            self.DEFAULT_TEST_CASES.append(TestCase(**test))
    
    async def run_tests(
        self,
        test_cases: Optional[List[TestCase]] = None,
        categories: Optional[List[str]] = None,
        generation_config: Optional[GenerationConfig] = None
    ) -> Dict[str, Any]:
        """Run the test suite."""
        
        # Load model
        self.console.print("[bold cyan]Loading model...[/bold cyan]")
        self.engine.load_model()
        
        # Select test cases
        if test_cases is None:
            test_cases = self.DEFAULT_TEST_CASES
        
        if categories:
            test_cases = [tc for tc in test_cases if tc.category in categories]
        
        # Default generation config
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        self.console.print(f"\n[bold green]Running {len(test_cases)} test cases...[/bold green]")
        
        # Run tests with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            
            test_task = progress.add_task(
                "[cyan]Testing model...", 
                total=len(test_cases)
            )
            
            for test_case in test_cases:
                progress.update(
                    test_task,
                    description=f"[cyan]Testing: {test_case.category} - {test_case.question[:50]}..."
                )
                
                # Run inference
                result = await self._run_single_test(test_case, generation_config)
                self.results.append(result)
                
                progress.advance(test_task)
        
        # Run judge evaluation if enabled
        if self.use_judge:
            self.console.print("\n[bold cyan]Running judge evaluation...[/bold cyan]")
            await self._run_judge_evaluation()
        
        # Analyze results
        analysis = self._analyze_results()
        
        # Save results
        self._save_results(analysis)
        
        # Print summary
        self._print_summary(analysis)
        
        # Generate report
        self._generate_html_report(analysis)
        
        return analysis
    
    async def _run_single_test(
        self,
        test_case: TestCase,
        config: GenerationConfig
    ) -> TestResult:
        """Run a single test case."""
        import time
        
        # Generate response
        start_time = time.time()
        prompt = test_case.to_prompt()
        
        try:
            response = self.engine.generate(prompt, config)
            duration = (time.time() - start_time) * 1000  # ms
            
            # Estimate tokens (rough approximation)
            response_tokens = len(response.split())
            tokens_per_second = response_tokens / (duration / 1000)
            
            # Check keywords
            keywords_found = []
            if test_case.expected_keywords:
                response_lower = response.lower()
                keywords_found = [
                    kw for kw in test_case.expected_keywords 
                    if kw.lower() in response_lower
                ]
            
            # Check format
            format_match = self._check_format(response, test_case.expected_format)
            
            return TestResult(
                test_case=test_case,
                response=response,
                duration_ms=duration,
                tokens_per_second=tokens_per_second,
                keywords_found=keywords_found,
                format_match=format_match
            )
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            return TestResult(
                test_case=test_case,
                response=f"ERROR: {str(e)}",
                duration_ms=0,
                tokens_per_second=0,
                keywords_found=[],
                format_match=False
            )
    
    def _check_format(self, response: str, expected_format: Optional[str]) -> bool:
        """Check if response matches expected format."""
        if not expected_format:
            return True
        
        format_checks = {
            "single_word_or_phrase": lambda r: len(r.split()) <= 5,
            "name": lambda r: len(r.split()) <= 3 and r[0].isupper(),
            "number": lambda r: any(char.isdigit() for char in r),
            "number_with_unit": lambda r: any(char.isdigit() for char in r) and any(char.isalpha() for char in r),
            "python_code": lambda r: "def" in r or "import" in r or "class" in r,
            "sql_query": lambda r: any(kw in r.upper() for kw in ["SELECT", "INSERT", "UPDATE", "DELETE"]),
            "haiku": lambda r: len(r.split('\n')) >= 3,
            "bullet_list": lambda r: any(marker in r for marker in ["•", "-", "*", "1.", "2."]),
            "two_sentences": lambda r: r.count('.') == 2,
            "technical_explanation": lambda r: len(r.split()) > 20,
            "comparison": lambda r: any(word in r.lower() for word in ["difference", "whereas", "unlike", "however"]),
            "logical_conclusion": lambda r: len(r.split()) > 2,
            "short_phrase": lambda r: len(r.split()) <= 10,
        }
        
        check_func = format_checks.get(expected_format, lambda r: True)
        return check_func(response)
    
    async def _run_judge_evaluation(self) -> None:
        """Run judge evaluation on all results."""
        
        for result in self.results:
            if "ERROR" in result.response:
                continue
            
            # Create judge prompt
            prompt = f"""Evaluate this AI model's response on a scale of 1-10.

Question: {result.test_case.question}
Model Response: {result.response}

Evaluation criteria:
1. Accuracy: Is the answer factually correct?
2. Relevance: Does it directly answer the question?
3. Clarity: Is the response clear and well-structured?
4. Completeness: Does it provide sufficient information?

Expected elements: {', '.join(result.test_case.expected_keywords) if result.test_case.expected_keywords else 'N/A'}

Provide a score (1-10) and brief feedback. Format: "Score: X/10 - Feedback text" """
            
            try:
                judge_response = await ask_with_retry(
                    prompt=prompt,
                    model=self.judge_model,
                    temperature=0.0,
                    max_tokens=150
                )
                
                # Parse score
                if "Score:" in judge_response:
                    score_part = judge_response.split("Score:")[1].split("/")[0].strip()
                    try:
                        score = float(score_part)
                        result.judge_score = score
                        result.judge_feedback = judge_response.split("-", 1)[1].strip() if "-" in judge_response else ""
                    except:
                        result.judge_score = 5.0  # Default
                        result.judge_feedback = judge_response
                
            except Exception as e:
                logger.warning(f"Judge evaluation failed: {e}")
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze test results."""
        analysis = {
            "total_tests": len(self.results),
            "categories": {},
            "overall_metrics": {},
            "format_compliance": {},
            "keyword_accuracy": {},
            "performance": {},
            "judge_scores": {},
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Group by category
        from collections import defaultdict
        by_category = defaultdict(list)
        for result in self.results:
            by_category[result.test_case.category].append(result)
        
        # Analyze each category
        for category, results in by_category.items():
            valid_results = [r for r in results if "ERROR" not in r.response]
            
            category_stats = {
                "total": len(results),
                "successful": len(valid_results),
                "avg_duration_ms": sum(r.duration_ms for r in valid_results) / len(valid_results) if valid_results else 0,
                "avg_tokens_per_second": sum(r.tokens_per_second for r in valid_results) / len(valid_results) if valid_results else 0,
                "keyword_accuracy": 0,
                "format_compliance": 0,
                "avg_judge_score": 0
            }
            
            if valid_results:
                # Keyword accuracy
                keyword_scores = []
                for r in valid_results:
                    if r.test_case.expected_keywords:
                        score = len(r.keywords_found) / len(r.test_case.expected_keywords)
                        keyword_scores.append(score)
                category_stats["keyword_accuracy"] = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
                
                # Format compliance
                format_matches = [r.format_match for r in valid_results if r.test_case.expected_format]
                category_stats["format_compliance"] = sum(format_matches) / len(format_matches) if format_matches else 1.0
                
                # Judge scores
                judge_scores = [r.judge_score for r in valid_results if r.judge_score is not None]
                category_stats["avg_judge_score"] = sum(judge_scores) / len(judge_scores) if judge_scores else 0
            
            analysis["categories"][category] = category_stats
        
        # Overall metrics
        all_valid = [r for r in self.results if "ERROR" not in r.response]
        if all_valid:
            analysis["overall_metrics"] = {
                "success_rate": len(all_valid) / len(self.results),
                "avg_duration_ms": sum(r.duration_ms for r in all_valid) / len(all_valid),
                "avg_tokens_per_second": sum(r.tokens_per_second for r in all_valid) / len(all_valid),
                "overall_keyword_accuracy": sum(
                    len(r.keywords_found) / len(r.test_case.expected_keywords) 
                    for r in all_valid 
                    if r.test_case.expected_keywords
                ) / len([r for r in all_valid if r.test_case.expected_keywords]),
                "overall_format_compliance": sum(
                    r.format_match for r in all_valid if r.test_case.expected_format
                ) / len([r for r in all_valid if r.test_case.expected_format]),
                "overall_judge_score": sum(
                    r.judge_score for r in all_valid if r.judge_score is not None
                ) / len([r for r in all_valid if r.judge_score is not None]) if any(r.judge_score for r in all_valid) else 0
            }
        
        return analysis
    
    def _save_results(self, analysis: Dict[str, Any]) -> None:
        """Save test results."""
        # Save detailed results
        results_data = []
        for result in self.results:
            results_data.append({
                "category": result.test_case.category,
                "question": result.test_case.question,
                "response": result.response,
                "duration_ms": result.duration_ms,
                "tokens_per_second": result.tokens_per_second,
                "keywords_expected": result.test_case.expected_keywords,
                "keywords_found": result.keywords_found,
                "format_expected": result.test_case.expected_format,
                "format_match": result.format_match,
                "judge_score": result.judge_score,
                "judge_feedback": result.judge_feedback
            })
        
        # Save as JSON
        with open(self.output_dir / "test_results.json", "w") as f:
            json.dump({
                "analysis": analysis,
                "results": results_data
            }, f, indent=2)
        
        # Save as CSV
        df = pd.DataFrame(results_data)
        df.to_csv(self.output_dir / "test_results.csv", index=False)
    
    def _print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print test summary."""
        self.console.print("\n[bold green]═══ Inference Test Summary ═══[/bold green]\n")
        
        # Overall metrics
        metrics = analysis.get("overall_metrics", {})
        if metrics:
            overall_table = Table(title="Overall Performance")
            overall_table.add_column("Metric", style="cyan")
            overall_table.add_column("Value", style="magenta")
            
            overall_table.add_row("Success Rate", f"{metrics.get('success_rate', 0)*100:.1f}%")
            overall_table.add_row("Avg Response Time", f"{metrics.get('avg_duration_ms', 0):.0f} ms")
            overall_table.add_row("Avg Tokens/Second", f"{metrics.get('avg_tokens_per_second', 0):.1f}")
            overall_table.add_row("Keyword Accuracy", f"{metrics.get('overall_keyword_accuracy', 0)*100:.1f}%")
            overall_table.add_row("Format Compliance", f"{metrics.get('overall_format_compliance', 0)*100:.1f}%")
            if metrics.get('overall_judge_score', 0) > 0:
                overall_table.add_row("Judge Score", f"{metrics.get('overall_judge_score', 0):.1f}/10")
            
            self.console.print(overall_table)
        
        # Category breakdown
        category_table = Table(title="\nPerformance by Category")
        category_table.add_column("Category", style="cyan")
        category_table.add_column("Success", style="green")
        category_table.add_column("Avg Time", style="yellow")
        category_table.add_column("Keyword Acc", style="blue")
        category_table.add_column("Format", style="magenta")
        if self.use_judge:
            category_table.add_column("Judge", style="red")
        
        for category, stats in analysis["categories"].items():
            row = [
                category,
                f"{stats['successful']}/{stats['total']}",
                f"{stats['avg_duration_ms']:.0f}ms",
                f"{stats['keyword_accuracy']*100:.0f}%",
                f"{stats['format_compliance']*100:.0f}%"
            ]
            if self.use_judge and stats.get('avg_judge_score', 0) > 0:
                row.append(f"{stats['avg_judge_score']:.1f}/10")
            category_table.add_row(*row)
        
        self.console.print(category_table)
        
        # Sample responses
        self.console.print("\n[bold]Sample Responses:[/bold]")
        for i, result in enumerate(self.results[:3]):
            self.console.print(Panel(
                f"[bold]Q:[/bold] {result.test_case.question}\n\n"
                f"[bold]A:[/bold] {result.response[:200]}{'...' if len(result.response) > 200 else ''}\n\n"
                f"[dim]Time: {result.duration_ms:.0f}ms | "
                f"Keywords: {len(result.keywords_found)}/{len(result.test_case.expected_keywords)} | "
                f"Judge: {result.judge_score:.1f}/10[/dim]" if result.judge_score else "",
                title=f"Example {i+1}: {result.test_case.category}",
                border_style="blue"
            ))
        
        self.console.print(f"\n[bold]Results saved to:[/bold] {self.output_dir}")
    
    def _generate_html_report(self, analysis: Dict[str, Any]) -> None:
        """Generate HTML report following 2025 style guide."""
        # This would generate a beautiful HTML report similar to the evaluation dashboard
        # For brevity, using a simplified version here
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Inference Test Report - {self.model_path.name}</title>
    <style>
        :root {{
            --color-primary-start: #4F46E5;
            --color-primary-end: #6366F1;
            --color-secondary: #6B7280;
            --color-background: #F9FAFB;
            --color-accent: #10B981;
            --font-family-base: 'Inter', system-ui, sans-serif;
            --border-radius-base: 8px;
            --spacing-base: 8px;
        }}
        body {{
            font-family: var(--font-family-base);
            background-color: var(--color-background);
            margin: 0;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        h1 {{
            background: linear-gradient(135deg, var(--color-primary-start), var(--color-primary-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 2rem 0;
        }}
        .metric-card {{
            background: white;
            padding: 1.5rem;
            border-radius: var(--border-radius-base);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--color-primary-start);
        }}
        .metric-label {{
            color: var(--color-secondary);
            font-size: 0.875rem;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            background: white;
            border-radius: var(--border-radius-base);
            overflow: hidden;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }}
        th {{
            background: #F3F4F6;
            padding: 1rem;
            text-align: left;
            font-weight: 600;
        }}
        td {{
            padding: 1rem;
            border-top: 1px solid #E5E7EB;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Inference Test Report</h1>
        <p>Model: {self.model_path.name} | Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</p>
        
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-value">{analysis['overall_metrics'].get('success_rate', 0)*100:.0f}%</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis['overall_metrics'].get('avg_duration_ms', 0):.0f}ms</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis['overall_metrics'].get('overall_keyword_accuracy', 0)*100:.0f}%</div>
                <div class="metric-label">Keyword Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{analysis['overall_metrics'].get('overall_judge_score', 0):.1f}/10</div>
                <div class="metric-label">Judge Score</div>
            </div>
        </div>
        
        <h2>Results by Category</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Success Rate</th>
                    <th>Avg Time</th>
                    <th>Keyword Accuracy</th>
                    <th>Judge Score</th>
                </tr>
            </thead>
            <tbody>
                {''.join(f"""
                <tr>
                    <td>{cat}</td>
                    <td>{stats['successful']}/{stats['total']}</td>
                    <td>{stats['avg_duration_ms']:.0f}ms</td>
                    <td>{stats['keyword_accuracy']*100:.0f}%</td>
                    <td>{stats['avg_judge_score']:.1f}/10</td>
                </tr>""" for cat, stats in analysis['categories'].items())}
            </tbody>
        </table>
    </div>
</body>
</html>"""
        
        with open(self.output_dir / "test_report.html", "w") as f:
            f.write(html_content)


async def interactive_test_session(
    model_path: str,
    generation_config: Optional[GenerationConfig] = None
) -> None:
    """Run an interactive testing session."""
    console = Console()
    engine = InferenceEngine(model_path)
    
    console.print("[bold cyan]Loading model for interactive testing...[/bold cyan]")
    engine.load_model()
    
    config = generation_config or GenerationConfig(
        max_new_tokens=256,
        temperature=0.7,
        stream=True
    )
    
    console.print("\n[bold green]Interactive Test Mode[/bold green]")
    console.print("Type your questions to test the model. Commands:")
    console.print("  /config - Show current generation config")
    console.print("  /temp <value> - Set temperature")
    console.print("  /tokens <value> - Set max tokens")
    console.print("  /exit - Quit\n")
    
    while True:
        try:
            user_input = console.input("[bold blue]Test Question:[/bold blue] ")
            
            if user_input.startswith("/"):
                # Handle commands
                if user_input == "/exit":
                    break
                elif user_input == "/config":
                    console.print(f"Config: {config}")
                elif user_input.startswith("/temp "):
                    config.temperature = float(user_input.split()[1])
                    console.print(f"Temperature set to {config.temperature}")
                elif user_input.startswith("/tokens "):
                    config.max_new_tokens = int(user_input.split()[1])
                    console.print(f"Max tokens set to {config.max_new_tokens}")
                else:
                    console.print("[red]Unknown command[/red]")
                continue
            
            # Generate response
            console.print("\n[bold green]Model Response:[/bold green] ", end="")
            response = engine.generate(user_input, config)
            
            if not config.stream:
                console.print(response)
            
            console.print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Exiting...[/yellow]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")