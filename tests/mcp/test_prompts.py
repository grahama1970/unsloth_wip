
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

    """
    Test unsloth MCP prompts implementation

    Granger standard test suite for MCP prompts.
    """

    import pytest
    import asyncio
    from unsloth.mcp.unsloth_prompts import register_all_prompts


    class TestUnslothPrompts:
        """Test prompts implementation"""

        def test_required_prompts_exist(self):
            """Verify all required prompts are registered"""
            registry = register_all_prompts()
            prompts = registry.list_prompts()
            prompt_names = [p.name for p in prompts]

            # Check required prompts (Granger standard)
            required = [
            f"unsloth:capabilities",
            f"unsloth:help",
            f"unsloth:quick-start"
            ]

            for req in required:
                assert req in prompt_names, f"Missing required prompt: {req}"

                @pytest.mark.asyncio
                @pytest.mark.asyncio
                @pytest.mark.asyncio
                async def test_capabilities_prompt(self):
                    """Test capabilities prompt execution"""
                    registry = register_all_prompts()
                    result = await registry.execute(f"unsloth:capabilities")

                    assert "unsloth" in result.lower()
                    assert "Available Prompts" in result
                    assert "Quick Start Workflow" in result

                    @pytest.mark.asyncio
                    @pytest.mark.asyncio
                    @pytest.mark.asyncio
                    async def test_help_prompt(self):
                        """Test help prompt execution"""
                        registry = register_all_prompts()

                        # Test without context
                        result = await registry.execute(f"unsloth:help")
                        assert "Common Tasks" in result

                        # Test with context
                        result = await registry.execute(f"unsloth:help", context="search")
                        assert "search" in result.lower()

                        @pytest.mark.asyncio
                        @pytest.mark.asyncio
                        @pytest.mark.asyncio
                        async def test_quick_start_prompt(self):
                            """Test quick-start prompt execution"""
                            registry = register_all_prompts()
                            result = await registry.execute(f"unsloth:quick-start")

                            assert "Quick Start" in result
                            assert "What is unsloth?" in result
                            assert "Basic Workflow" in result

                            def test_prompt_consistency(self):
                                """Test that all prompts follow Granger naming standard"""
                                registry = register_all_prompts()
                                prompts = registry.list_prompts()

                                for prompt in prompts:
                                    # All prompts should start with module name
                                    assert prompt.name.startswith(f"unsloth:"),                 f"Prompt {prompt.name} doesn't follow naming standard"

                                    # All prompts should have descriptions
                                    assert prompt.description, f"Prompt {prompt.name} missing description"

                                    # Check categories
                                    assert prompt._mcp_prompt.category in [
                                    "discovery", "research", "analysis",
                                    "integration", "export", "help"
                                    ], f"Prompt {prompt.name} has non-standard category"


                                    if __name__ == "__main__":
                                        # Quick validation
                                        print(f"Testing unsloth prompts...")
                                        registry = register_all_prompts()
                                        print(f"✅ Registered {len(registry.list_prompts())} prompts")

                                        # Run async tests
                                        import asyncio

                                        async def run_tests():
                                            test = TestUnslothPrompts()
                                            test.test_required_prompts_exist()
                                            await test.test_capabilities_prompt()
                                            await test.test_help_prompt()
                                            await test.test_quick_start_prompt()
                                            test.test_prompt_consistency()
                                            print("✅ All tests passed!")

                                            asyncio.run(run_tests())
