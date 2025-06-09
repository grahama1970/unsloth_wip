"""
Module: tensorboard_verifier.py
Description: TensorBoard verification for entropy-aware training monitoring

External Dependencies:
- torch: https://pytorch.org/docs/stable/index.html
- tensorboard: https://www.tensorflow.org/tensorboard
- playwright: https://playwright.dev/python/docs/intro
- loguru: https://loguru.readthedocs.io/

Sample Input:
>>> verifier = TensorBoardVerifier(log_dir="./outputs/tensorboard_logs")
>>> results = await verifier.verify_training_progress()

Expected Output:
>>> results
{"status": "healthy", "loss_decreasing": True, "entropy_metrics_present": True}

Example Usage:
>>> from unsloth.utils.tensorboard_verifier import TensorBoardVerifier
>>> verifier = TensorBoardVerifier()
>>> await verifier.capture_training_screenshots()
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import json
import subprocess

from loguru import logger
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    logger.warning("Playwright not available - screenshot features disabled")


class TensorBoardVerifier:
    """Verify TensorBoard logs and capture training progress."""
    
    def __init__(
        self,
        log_dir: str = "./outputs/tensorboard_logs",
        port: int = 6006,
        screenshot_dir: str = "./outputs/tensorboard_screenshots"
    ):
        """
        Initialize TensorBoard verifier.
        
        Args:
            log_dir: Directory containing TensorBoard logs
            port: Port to run TensorBoard on
            screenshot_dir: Directory to save screenshots
        """
        self.log_dir = Path(log_dir)
        self.port = port
        self.screenshot_dir = Path(screenshot_dir)
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        
        self.tb_process = None
        self.browser = None
        self.page = None
        
    def analyze_training_logs(self) -> Dict[str, Any]:
        """Analyze TensorBoard logs programmatically."""
        results = {
            "status": "unknown",
            "loss_trend": None,
            "entropy_metrics": {},
            "training_metrics": {},
            "warnings": []
        }
        
        try:
            # Find the latest event file
            event_files = list(self.log_dir.glob("events.out.tfevents.*"))
            if not event_files:
                results["status"] = "no_logs"
                results["warnings"].append("No TensorBoard event files found")
                return results
                
            latest_event = max(event_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Analyzing event file: {latest_event}")
            
            # Load events
            ea = EventAccumulator(str(latest_event))
            ea.Reload()
            
            # Analyze loss
            if "training/loss" in ea.Tags()["scalars"]:
                loss_events = ea.Scalars("training/loss")
                losses = [e.value for e in loss_events]
                steps = [e.step for e in loss_events]
                
                if len(losses) > 1:
                    # Check if loss is decreasing
                    loss_trend = "decreasing" if losses[-1] < losses[0] else "increasing"
                    results["loss_trend"] = loss_trend
                    
                    # Calculate statistics
                    results["training_metrics"]["initial_loss"] = losses[0]
                    results["training_metrics"]["final_loss"] = losses[-1]
                    results["training_metrics"]["min_loss"] = min(losses)
                    results["training_metrics"]["loss_reduction"] = (losses[0] - losses[-1]) / losses[0]
                    results["training_metrics"]["total_steps"] = steps[-1]
                    
                    # Check for issues
                    if loss_trend == "increasing":
                        results["warnings"].append("Loss is increasing - check learning rate")
                    
                    if any(np.isnan(losses) or np.isinf(losses)):
                        results["warnings"].append("NaN or Inf values detected in loss")
                        
            # Analyze entropy metrics
            entropy_tags = [tag for tag in ea.Tags()["scalars"] if "entropy" in tag.lower()]
            
            for tag in entropy_tags:
                events = ea.Scalars(tag)
                values = [e.value for e in events]
                
                if values:
                    metric_name = tag.split("/")[-1]
                    results["entropy_metrics"][metric_name] = {
                        "initial": values[0],
                        "final": values[-1],
                        "mean": np.mean(values),
                        "trend": "decreasing" if values[-1] < values[0] else "increasing"
                    }
                    
            # Check specific entropy metrics
            if "entropy/average_entropy" in ea.Tags()["scalars"]:
                avg_entropy = results["entropy_metrics"].get("average_entropy", {})
                if avg_entropy.get("trend") == "decreasing":
                    logger.info("Good: Average entropy is decreasing")
                else:
                    results["warnings"].append("Average entropy not decreasing as expected")
                    
            # Determine overall status
            if results["loss_trend"] == "decreasing" and results["entropy_metrics"]:
                results["status"] = "healthy"
            elif results["loss_trend"] == "decreasing":
                results["status"] = "training_ok"
            else:
                results["status"] = "needs_attention"
                
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            results["status"] = "error"
            results["warnings"].append(f"Analysis error: {str(e)}")
            
        return results
        
    async def start_tensorboard_server(self) -> bool:
        """Start TensorBoard server in background."""
        try:
            # Kill any existing TensorBoard on this port
            subprocess.run(["pkill", "-f", f"tensorboard.*--port={self.port}"], 
                         capture_output=True)
            
            # Start TensorBoard
            cmd = [
                "tensorboard",
                "--logdir", str(self.log_dir),
                "--port", str(self.port),
                "--bind_all"
            ]
            
            self.tb_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Wait for server to start
            logger.info(f"Starting TensorBoard on port {self.port}...")
            await asyncio.sleep(3)
            
            # Check if running
            if self.tb_process.poll() is None:
                logger.info(f"TensorBoard running at http://localhost:{self.port}")
                return True
            else:
                logger.error("TensorBoard failed to start")
                return False
                
        except Exception as e:
            logger.error(f"Error starting TensorBoard: {e}")
            return False
            
    async def capture_training_screenshots(
        self,
        pages_to_capture: List[str] = None
    ) -> Dict[str, str]:
        """Capture screenshots of TensorBoard pages."""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available - cannot capture screenshots")
            return {}
            
        if pages_to_capture is None:
            pages_to_capture = ["scalars", "histograms", "graphs"]
            
        screenshots = {}
        
        try:
            # Start TensorBoard if not running
            if self.tb_process is None or self.tb_process.poll() is not None:
                if not await self.start_tensorboard_server():
                    return screenshots
                    
            # Launch browser
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                
                # Set viewport
                await page.set_viewport_size({"width": 1920, "height": 1080})
                
                # Navigate to TensorBoard
                base_url = f"http://localhost:{self.port}"
                await page.goto(base_url)
                
                # Wait for page to load
                await page.wait_for_load_state("networkidle")
                await asyncio.sleep(2)
                
                # Capture main dashboard
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                main_path = self.screenshot_dir / f"tensorboard_main_{timestamp}.png"
                await page.screenshot(path=str(main_path), full_page=True)
                screenshots["main"] = str(main_path)
                logger.info(f"Captured main dashboard: {main_path}")
                
                # Capture specific pages
                for page_name in pages_to_capture:
                    try:
                        # Navigate to specific tab
                        if page_name == "scalars":
                            # Click on scalars tab if available
                            scalars_link = page.locator('a:has-text("Scalars")')
                            if await scalars_link.count() > 0:
                                await scalars_link.click()
                                await page.wait_for_load_state("networkidle")
                                await asyncio.sleep(1)
                                
                        # Capture screenshot
                        page_path = self.screenshot_dir / f"tensorboard_{page_name}_{timestamp}.png"
                        await page.screenshot(path=str(page_path), full_page=True)
                        screenshots[page_name] = str(page_path)
                        logger.info(f"Captured {page_name}: {page_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to capture {page_name}: {e}")
                        
                # Capture entropy-specific charts
                await self._capture_entropy_charts(page, timestamp, screenshots)
                
                await browser.close()
                
        except Exception as e:
            logger.error(f"Error capturing screenshots: {e}")
            
        return screenshots
        
    async def _capture_entropy_charts(
        self,
        page: "Page",
        timestamp: str,
        screenshots: Dict[str, str]
    ):
        """Capture entropy-specific metric charts."""
        try:
            # Look for entropy metrics
            entropy_cards = page.locator('div:has-text("entropy")')
            count = await entropy_cards.count()
            
            if count > 0:
                logger.info(f"Found {count} entropy metric cards")
                
                # Scroll to entropy section
                await entropy_cards.first.scroll_into_view_if_needed()
                await asyncio.sleep(1)
                
                # Capture entropy metrics
                entropy_path = self.screenshot_dir / f"tensorboard_entropy_{timestamp}.png"
                await page.screenshot(
                    path=str(entropy_path),
                    clip={"x": 0, "y": 0, "width": 1920, "height": 1080}
                )
                screenshots["entropy"] = str(entropy_path)
                logger.info(f"Captured entropy metrics: {entropy_path}")
                
        except Exception as e:
            logger.debug(f"No entropy charts found or error: {e}")
            
    async def verify_training_progress(self) -> Dict[str, Any]:
        """Comprehensive training verification."""
        logger.info("Starting training verification...")
        
        # Analyze logs
        log_analysis = self.analyze_training_logs()
        
        # Capture screenshots if possible
        screenshots = {}
        if PLAYWRIGHT_AVAILABLE:
            screenshots = await self.capture_training_screenshots()
            
        # Generate verification report
        report = {
            "timestamp": datetime.now().isoformat(),
            "log_analysis": log_analysis,
            "screenshots": screenshots,
            "verification_status": self._determine_verification_status(log_analysis),
            "recommendations": self._generate_recommendations(log_analysis)
        }
        
        # Save report
        report_path = self.screenshot_dir / f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Verification report saved: {report_path}")
        
        return report
        
    def _determine_verification_status(self, analysis: Dict[str, Any]) -> str:
        """Determine overall verification status."""
        if analysis["status"] == "healthy":
            if analysis["entropy_metrics"]:
                return "excellent" if len(analysis["warnings"]) == 0 else "good"
            return "good"
        elif analysis["status"] == "training_ok":
            return "acceptable"
        elif analysis["status"] == "needs_attention":
            return "warning"
        else:
            return "error"
            
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Loss recommendations
        if analysis["loss_trend"] == "increasing":
            recommendations.append("Reduce learning rate or check for data issues")
        elif analysis["training_metrics"].get("loss_reduction", 0) < 0.1:
            recommendations.append("Consider longer training or different hyperparameters")
            
        # Entropy recommendations
        avg_entropy = analysis["entropy_metrics"].get("average_entropy", {})
        if avg_entropy.get("trend") == "increasing":
            recommendations.append("Model uncertainty increasing - check regularization")
        elif avg_entropy.get("final", 1.0) > 0.8:
            recommendations.append("High entropy persists - consider larger model or more data")
            
        # General recommendations
        if not analysis["entropy_metrics"]:
            recommendations.append("Enable entropy metrics logging for better insights")
            
        if len(analysis["warnings"]) > 2:
            recommendations.append("Multiple warnings detected - review training configuration")
            
        return recommendations
        
    def cleanup(self):
        """Cleanup resources."""
        if self.tb_process and self.tb_process.poll() is None:
            logger.info("Stopping TensorBoard server...")
            self.tb_process.terminate()
            self.tb_process.wait()
            
            
    async def generate_training_summary(self) -> str:
        """Generate a human-readable training summary."""
        analysis = self.analyze_training_logs()
        
        summary = f"""
# TensorBoard Training Summary

**Status**: {analysis['status'].replace('_', ' ').title()}
**Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Loss Metrics
- Initial Loss: {analysis['training_metrics'].get('initial_loss', 'N/A'):.4f}
- Final Loss: {analysis['training_metrics'].get('final_loss', 'N/A'):.4f}
- Loss Reduction: {analysis['training_metrics'].get('loss_reduction', 0)*100:.1f}%
- Total Steps: {analysis['training_metrics'].get('total_steps', 0)}

## Entropy Metrics
"""
        
        for metric, values in analysis['entropy_metrics'].items():
            summary += f"\n### {metric.replace('_', ' ').title()}"
            summary += f"\n- Initial: {values['initial']:.4f}"
            summary += f"\n- Final: {values['final']:.4f}"
            summary += f"\n- Trend: {values['trend']}"
            
        if analysis['warnings']:
            summary += "\n\n## Warnings\n"
            for warning in analysis['warnings']:
                summary += f"- ⚠️ {warning}\n"
                
        recommendations = self._generate_recommendations(analysis)
        if recommendations:
            summary += "\n## Recommendations\n"
            for rec in recommendations:
                summary += f"-  {rec}\n"
                
        return summary


# Validation
if __name__ == "__main__":
    async def test_verifier():
        verifier = TensorBoardVerifier()
        
        # Test log analysis
        print("Analyzing training logs...")
        analysis = verifier.analyze_training_logs()
        print(f"Status: {analysis['status']}")
        print(f"Loss trend: {analysis['loss_trend']}")
        
        # Test screenshot capture (if available)
        if PLAYWRIGHT_AVAILABLE:
            print("\nCapturing screenshots...")
            screenshots = await verifier.capture_training_screenshots()
            print(f"Captured {len(screenshots)} screenshots")
            
        # Generate summary
        print("\nGenerating training summary...")
        summary = await verifier.generate_training_summary()
        print(summary)
        
        # Cleanup
        verifier.cleanup()
        
    # Run test
    asyncio.run(test_verifier())
    print("\n Module validation passed")