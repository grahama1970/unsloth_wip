#!/usr/bin/env python3
"""Capture screenshots of the evaluation dashboards."""

import asyncio
from playwright.async_api import async_playwright
from pathlib import Path

async def capture_dashboards():
    """Capture screenshots of the evaluation dashboards."""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1440, 'height': 900})
        
        # Set up screenshot directory
        screenshot_dir = Path("/home/graham/workspace/experiments/unsloth_wip/dashboard_screenshots")
        screenshot_dir.mkdir(exist_ok=True)
        
        # Capture index page
        print(" Capturing index page...")
        await page.goto("http://localhost:8888/index.html")
        await page.wait_for_load_state("networkidle")
        await page.screenshot(path=str(screenshot_dir / "index.png"))
        
        # Capture evaluation dashboard
        print(" Capturing evaluation dashboard...")
        await page.goto("http://localhost:8888/evaluation_dashboard.html")
        await page.wait_for_load_state("networkidle")
        await page.wait_for_timeout(1000)  # Wait for fonts to load
        
        # Full page screenshot
        await page.screenshot(path=str(screenshot_dir / "evaluation_dashboard_full.png"), full_page=True)
        
        # Capture specific sections
        # Summary box
        summary_element = await page.query_selector(".summary-box")
        if summary_element:
            await summary_element.screenshot(path=str(screenshot_dir / "summary_box.png"))
        
        # Metrics grid
        metrics_element = await page.query_selector(".metrics-grid")
        if metrics_element:
            await metrics_element.screenshot(path=str(screenshot_dir / "metrics_grid.png"))
        
        # Test hover effects
        print(" Testing hover effects...")
        metric_card = await page.query_selector(".metric-card")
        if metric_card:
            await metric_card.hover()
            await page.wait_for_timeout(300)  # Wait for transition
            await metric_card.screenshot(path=str(screenshot_dir / "metric_card_hover.png"))
        
        # Test button interaction
        button = await page.query_selector(".button")
        if button:
            await button.hover()
            await page.wait_for_timeout(300)
            await button.screenshot(path=str(screenshot_dir / "button_hover.png"))
        
        # Test responsive design
        print(" Testing responsive design...")
        await page.set_viewport_size({"width": 375, "height": 812})  # iPhone size
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(screenshot_dir / "mobile_view.png"))
        
        # Test table styling
        await page.set_viewport_size({"width": 1440, "height": 900})  # Back to desktop
        table_element = await page.query_selector("table")
        if table_element:
            await table_element.screenshot(path=str(screenshot_dir / "table_styling.png"))
        
        await browser.close()
        
        print("\n Screenshots captured successfully!")
        print(f" Screenshots saved to: {screenshot_dir}")
        
        # List all screenshots
        screenshots = list(screenshot_dir.glob("*.png"))
        print(f"\n Captured {len(screenshots)} screenshots:")
        for shot in screenshots:
            print(f"   - {shot.name}")

# Run the capture
if __name__ == "__main__":
    asyncio.run(capture_dashboards())