"""Generate HTML dashboard following 2025 style guide."""
Module: dashboard.py

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from jinja2 import Template
from loguru import logger


class DashboardGenerator:
    """Generate beautiful HTML dashboards for evaluation results."""

    def __init__(self, results: dict[str, Any], output_dir: str):
        self.results = results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self) -> str:
        """Generate the full HTML dashboard."""
        logger.info("Generating evaluation dashboard...")

        # Generate visualizations
        charts = self._generate_charts()

        # Prepare data for template
        template_data = {
            "title": "Unsloth Model Evaluation Dashboard",
            "generation_date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            "base_results": self.results.get("base_model", {}),
            "lora_results": self.results.get("lora_model", {}),
            "comparison": self.results.get("comparison", {}),
            "charts": charts,
            "metadata": self.results.get("metadata", {}),
            "style_guide_compliant": True
        }

        # Render template
        html_content = self._render_template(template_data)

        # Save dashboard
        dashboard_path = self.output_dir / "evaluation_dashboard.html"
        with open(dashboard_path, "w") as f:
            f.write(html_content)

        logger.info(f"Dashboard saved to {dashboard_path}")
        return str(dashboard_path)

    def _generate_charts(self) -> dict[str, str]:
        """Generate Plotly charts for the dashboard."""
        charts = {}

        # 1. Metrics comparison chart
        if self.results.get("comparison"):
            charts["metrics_comparison"] = self._create_metrics_comparison_chart()

        # 2. Judge scores radar chart
        if "judge_scores" in self.results.get("base_model", {}):
            charts["judge_radar"] = self._create_judge_radar_chart()

        # 3. Performance improvement waterfall
        if self.results.get("comparison", {}).get("improvements"):
            charts["improvement_waterfall"] = self._create_improvement_waterfall()

        # 4. Example outputs comparison
        charts["examples_table"] = self._create_examples_table()

        return charts

    def _create_metrics_comparison_chart(self) -> str:
        """Create metrics comparison bar chart."""
        comparison = self.results.get("comparison", {})

        # Prepare data
        metrics = []
        base_values = []
        lora_values = []

        for metric_data in list(comparison.get("improvements", {}).values()) + list(comparison.get("regressions", {}).values()):
            if isinstance(metric_data, dict) and "base" in metric_data:
                metrics.append(metric_data.get("metric", "Unknown"))
                base_values.append(metric_data["base"])
                lora_values.append(metric_data["lora"])

        # Create figure
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Bar(
            name='Base Model',
            x=metrics,
            y=base_values,
            marker_color='#6B7280',
            marker_line_width=0
        ))

        fig.add_trace(go.Bar(
            name='LoRA Model',
            x=metrics,
            y=lora_values,
            marker_color='#4F46E5',
            marker_line_width=0
        ))

        # Update layout following style guide
        fig.update_layout(
            title={
                'text': 'Model Performance Comparison',
                'font': {'size': 24, 'family': 'Inter, system-ui, sans-serif', 'weight': 600}
            },
            xaxis_tickangle=-45,
            barmode='group',
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='white',
            font={'family': 'Inter, system-ui, sans-serif', 'size': 14},
            margin=dict(t=80, b=80, l=60, r=40),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        return fig.to_html(div_id="metrics_comparison", include_plotlyjs=False)

    def _create_judge_radar_chart(self) -> str:
        """Create radar chart for judge scores."""
        base_judge = self.results.get("base_model", {}).get("judge_scores", {})
        lora_judge = self.results.get("lora_model", {}).get("judge_scores", {})

        if not base_judge.get("criteria_scores"):
            return ""

        criteria = list(base_judge["criteria_scores"].keys())
        base_scores = [base_judge["criteria_scores"].get(c, 0) for c in criteria]
        lora_scores = [lora_judge["criteria_scores"].get(c, 0) for c in criteria] if lora_judge else base_scores

        # Create figure
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatterpolar(
            r=base_scores,
            theta=criteria,
            fill='toself',
            name='Base Model',
            line_color='#6B7280',
            fillcolor='rgba(107, 114, 128, 0.2)'
        ))

        if lora_judge:
            fig.add_trace(go.Scatterpolar(
                r=lora_scores,
                theta=criteria,
                fill='toself',
                name='LoRA Model',
                line_color='#4F46E5',
                fillcolor='rgba(79, 70, 229, 0.2)'
            ))

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10],
                    tickfont=dict(size=12)
                ),
                angularaxis=dict(
                    tickfont=dict(size=14)
                ),
                bgcolor='#F9FAFB'
            ),
            showlegend=True,
            title={
                'text': 'Judge Model Evaluation Scores',
                'font': {'size': 24, 'family': 'Inter, system-ui, sans-serif', 'weight': 600}
            },
            font={'family': 'Inter, system-ui, sans-serif'},
            paper_bgcolor='white',
            margin=dict(t=100, b=40, l=40, r=40)
        )

        return fig.to_html(div_id="judge_radar", include_plotlyjs=False)

    def _create_improvement_waterfall(self) -> str:
        """Create waterfall chart showing improvements."""
        comparison = self.results.get("comparison", {})
        improvements = comparison.get("improvements", {})
        regressions = comparison.get("regressions", {})

        # Prepare data
        x_data = []
        y_data = []
        text_data = []
        colors = []

        for metric, data in improvements.items():
            x_data.append(metric)
            y_data.append(data["improvement_pct"])
            text_data.append(f"+{data['improvement_pct']:.1f}%")
            colors.append('#10B981')  # Green for improvements

        for metric, data in regressions.items():
            x_data.append(metric)
            y_data.append(-data["regression_pct"])
            text_data.append(f"-{data['regression_pct']:.1f}%")
            colors.append('#EF4444')  # Red for regressions

        # Create figure
        fig = go.Figure(go.Waterfall(
            x=x_data,
            y=y_data,
            text=text_data,
            textposition="outside",
            connector={"line": {"color": "#E5E7EB"}},
            decreasing={"marker": {"color": "#EF4444"}},
            increasing={"marker": {"color": "#10B981"}},
            totals={"marker": {"color": "#4F46E5"}}
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': 'Performance Changes (Base → LoRA)',
                'font': {'size': 24, 'family': 'Inter, system-ui, sans-serif', 'weight': 600}
            },
            showlegend=False,
            xaxis_tickangle=-45,
            plot_bgcolor='#F9FAFB',
            paper_bgcolor='white',
            font={'family': 'Inter, system-ui, sans-serif', 'size': 14},
            margin=dict(t=80, b=100, l=60, r=40),
            yaxis_title="Change (%)",
            hovermode='x'
        )

        return fig.to_html(div_id="improvement_waterfall", include_plotlyjs=False)

    def _create_examples_table(self) -> str:
        """Create examples comparison table."""
        base_examples = self.results.get("base_model", {}).get("examples", [])
        lora_examples = self.results.get("lora_model", {}).get("examples", [])

        if not base_examples:
            return ""

        # Prepare data for table
        table_data = []
        for i, base_ex in enumerate(base_examples[:5]):
            lora_ex = lora_examples[i] if i < len(lora_examples) else {}
            table_data.append({
                "Question": base_ex.get("question", ""),
                "Expected": base_ex.get("expected", ""),
                "Base Model": base_ex.get("actual", ""),
                "LoRA Model": lora_ex.get("actual", "N/A")
            })

        # Convert to HTML table with styling
        df = pd.DataFrame(table_data)

        # Apply styling
        styled_table = df.style.set_properties(**{
            'font-family': 'Inter, system-ui, sans-serif',
            'font-size': '14px',
            'text-align': 'left',
            'padding': '12px',
            'border-bottom': '1px solid #E5E7EB'
        }).set_table_styles([
            {'selector': 'th', 'props': [
                ('background-color', '#F3F4F6'),
                ('font-weight', '600'),
                ('text-transform', 'uppercase'),
                ('font-size', '12px'),
                ('letter-spacing', '0.05em'),
                ('color', '#6B7280'),
                ('padding', '16px 12px')
            ]},
            {'selector': 'tbody tr:hover', 'props': [
                ('background-color', '#F9FAFB')
            ]},
            {'selector': '', 'props': [
                ('border-collapse', 'collapse'),
                ('width', '100%'),
                ('margin-top', '24px')
            ]}
        ])

        return styled_table.to_html(index=False, escape=False)

    def _render_template(self, data: dict[str, Any]) -> str:
        """Render the HTML template."""
        template = Template("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    
    <!-- 2025 Style Guide Compliant CSS -->
    <style>
        :root {
            --color-primary-start: #4F46E5;
            --color-primary-end: #6366F1;
            --color-secondary: #6B7280;
            --color-background: #F9FAFB;
            --color-accent: #10B981;
            --color-danger: #EF4444;
            --color-warning: #F59E0B;
            
            --font-family-base: 'Inter', system-ui, -apple-system, sans-serif;
            
            --font-weight-regular: 400;
            --font-weight-semibold: 600;
            --font-weight-bold: 700;
            
            --border-radius-base: 8px;
            --spacing-base: 8px;
            
            --transition-duration: 250ms;
            --transition-timing: cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: var(--font-family-base);
            font-size: 16px;
            line-height: 1.5;
            color: #111827;
            background-color: var(--color-background);
            font-weight: var(--font-weight-regular);
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: calc(var(--spacing-base) * 4);
        }
        
        /* Typography */
        h1, h2, h3, h4 {
            font-weight: var(--font-weight-semibold);
            line-height: 1.2;
            letter-spacing: -0.02em;
        }
        
        h1 {
            font-size: 3rem;
            margin-bottom: calc(var(--spacing-base) * 2);
            background: linear-gradient(135deg, var(--color-primary-start), var(--color-primary-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        h2 {
            font-size: 2rem;
            margin-bottom: calc(var(--spacing-base) * 3);
            margin-top: calc(var(--spacing-base) * 6);
            color: #1F2937;
        }
        
        h3 {
            font-size: 1.5rem;
            margin-bottom: calc(var(--spacing-base) * 2);
            color: #374151;
        }
        
        /* Cards */
        .card {
            background: white;
            border-radius: var(--border-radius-base);
            padding: calc(var(--spacing-base) * 3);
            margin-bottom: calc(var(--spacing-base) * 3);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1), 0 1px 2px rgba(0, 0, 0, 0.06);
            transition: all var(--transition-duration) var(--transition-timing);
        }
        
        .card:hover {
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
            transform: translateY(-2px);
        }
        
        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: calc(var(--spacing-base) * 3);
            margin-bottom: calc(var(--spacing-base) * 6);
        }
        
        .metric-card {
            background: white;
            border-radius: var(--border-radius-base);
            padding: calc(var(--spacing-base) * 3);
            text-align: center;
            border: 1px solid #E5E7EB;
            transition: all var(--transition-duration) var(--transition-timing);
        }
        
        .metric-card:hover {
            border-color: var(--color-primary-start);
            box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.1);
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: var(--font-weight-bold);
            margin: calc(var(--spacing-base) * 2) 0;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: var(--color-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: var(--font-weight-semibold);
        }
        
        .metric-delta {
            font-size: 1rem;
            font-weight: var(--font-weight-semibold);
            padding: calc(var(--spacing-base) / 2) var(--spacing-base);
            border-radius: calc(var(--border-radius-base) / 2);
            display: inline-block;
            margin-top: var(--spacing-base);
        }
        
        .metric-delta.positive {
            color: var(--color-accent);
            background-color: rgba(16, 185, 129, 0.1);
        }
        
        .metric-delta.negative {
            color: var(--color-danger);
            background-color: rgba(239, 68, 68, 0.1);
        }
        
        /* Charts */
        .chart-container {
            background: white;
            border-radius: var(--border-radius-base);
            padding: calc(var(--spacing-base) * 3);
            margin-bottom: calc(var(--spacing-base) * 4);
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        }
        
        /* Summary Box */
        .summary-box {
            background: linear-gradient(135deg, var(--color-primary-start), var(--color-primary-end));
            color: white;
            padding: calc(var(--spacing-base) * 4);
            border-radius: var(--border-radius-base);
            margin-bottom: calc(var(--spacing-base) * 6);
        }
        
        .summary-box h2 {
            color: white;
            margin-bottom: calc(var(--spacing-base) * 2);
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: calc(var(--spacing-base) * 4);
            margin-top: calc(var(--spacing-base) * 3);
        }
        
        .summary-stat {
            text-align: center;
        }
        
        .summary-stat-value {
            font-size: 2rem;
            font-weight: var(--font-weight-bold);
            display: block;
        }
        
        .summary-stat-label {
            font-size: 0.875rem;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.875rem;
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: calc(var(--spacing-base) * 4);
            color: var(--color-secondary);
            font-size: 0.875rem;
        }
        
        /* Loading States */
        @keyframes shimmer {
            0% {
                background-position: -200% 0;
            }
            100% {
                background-position: 200% 0;
            }
        }
        
        .loading {
            background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
        }
        
        /* Responsive */
        @media (max-width: 768px) {
            .container {
                padding: calc(var(--spacing-base) * 2);
            }
            
            h1 {
                font-size: 2rem;
            }
            
            h2 {
                font-size: 1.5rem;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    
    <!-- Plotly.js -->
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    
    <!-- Inter Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <header>
            <h1>{{ title }}</h1>
            <p style="color: var(--color-secondary); margin-bottom: calc(var(--spacing-base) * 4);">
                Generated on {{ generation_date }}
            </p>
        </header>
        
        {% if comparison %}
        <!-- Summary Box -->
        <div class="summary-box">
            <h2>Evaluation Summary</h2>
            <p style="font-size: 1.125rem; opacity: 0.95; margin-bottom: calc(var(--spacing-base) * 3);">
                {{ comparison.summary.recommendation }}
            </p>
            <div class="summary-grid">
                <div class="summary-stat">
                    <span class="summary-stat-value">{{ comparison.summary.total_improvements }}</span>
                    <span class="summary-stat-label">Improvements</span>
                </div>
                <div class="summary-stat">
                    <span class="summary-stat-value">{{ comparison.summary.total_regressions }}</span>
                    <span class="summary-stat-label">Regressions</span>
                </div>
                {% if comparison.summary.judge_score_improvement %}
                <div class="summary-stat">
                    <span class="summary-stat-value">{{ "%.1f"|format(comparison.summary.judge_score_improvement.improvement_pct) }}%</span>
                    <span class="summary-stat-label">Judge Score Improvement</span>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}
        
        <!-- Key Metrics -->
        <h2>Key Metrics Comparison</h2>
        <div class="metrics-grid">
            {% if base_results.metrics %}
                {% for metric, value in base_results.metrics.items() %}
                <div class="metric-card">
                    <div class="metric-label">{{ metric|replace('_', ' ')|title }}</div>
                    <div class="metric-value">{{ "%.3f"|format(value) }}</div>
                    {% if lora_results.metrics and metric in lora_results.metrics %}
                        {% set lora_value = lora_results.metrics[metric] %}
                        {% set improvement = ((lora_value - value) / value * 100) if 'perplexity' not in metric.lower() else ((value - lora_value) / value * 100) %}
                        <div class="metric-delta {{ 'positive' if improvement > 0 else 'negative' }}">
                            {{ "%.1f"|format(improvement) }}%
                        </div>
                    {% endif %}
                </div>
                {% endfor %}
            {% endif %}
        </div>
        
        <!-- Charts -->
        {% if charts %}
            {% if charts.metrics_comparison %}
            <div class="chart-container">
                <h3>Performance Metrics Comparison</h3>
                {{ charts.metrics_comparison|safe }}
            </div>
            {% endif %}
            
            {% if charts.judge_radar %}
            <div class="chart-container">
                <h3>Judge Model Evaluation</h3>
                {{ charts.judge_radar|safe }}
            </div>
            {% endif %}
            
            {% if charts.improvement_waterfall %}
            <div class="chart-container">
                <h3>Performance Changes</h3>
                {{ charts.improvement_waterfall|safe }}
            </div>
            {% endif %}
        {% endif %}
        
        <!-- Example Outputs -->
        {% if charts.examples_table %}
        <div class="card">
            <h3>Example Outputs Comparison</h3>
            {{ charts.examples_table|safe }}
        </div>
        {% endif %}
        
        <!-- Footer -->
        <footer class="footer">
            <p>Unsloth Evaluation Dashboard • Powered by DeepEval & MLflow</p>
            <p style="margin-top: var(--spacing-base);">
                <small>2025 Style Guide Compliant</small>
            </p>
        </footer>
    </div>
    
    <!-- Plotly responsive config -->
    <script>
        // Make all Plotly charts responsive
        window.addEventListener('resize', function() {
            const plots = document.querySelectorAll('[id^="metrics_"], [id^="judge_"], [id^="improvement_"]');
            plots.forEach(plot => {
                Plotly.Plots.resize(plot);
            });
        });
    </script>
</body>
</html>""")

        return template.render(**data)
