#!/usr/bin/env python3
"""
Simple verification script to check if the platform is correctly set up.
This doesn't require pytest or other test dependencies.
"""

import sys
import traceback

def test_import(module_name, description):
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✓ {description}")
        return True
    except Exception as e:
        print(f"✗ {description}: {str(e)}")
        return False

def test_config_creation():
    """Test creating a TrainingConfig."""
    try:
        from src.training.config import TrainingConfig
        config = TrainingConfig(model_name="gpt2")
        assert config.model_name == "gpt2"
        assert config.batch_size == 8
        print("✓ TrainingConfig creation works")
        return True
    except Exception as e:
        print(f"✗ TrainingConfig creation failed: {str(e)}")
        traceback.print_exc()
        return False

def test_config_validation():
    """Test config validation."""
    try:
        from src.training.config import TrainingConfig
        # Test invalid learning rate
        try:
            TrainingConfig(learning_rate=-1)
            print("✗ Config validation failed: accepted negative learning rate")
            return False
        except ValueError:
            print("✓ Config validation works (rejected negative learning rate)")
            return True
    except Exception as e:
        print(f"✗ Config validation test failed: {str(e)}")
        return False

def test_search_space():
    """Test search space creation."""
    try:
        from src.tuning.search_spaces import get_search_space
        space = get_search_space("quick")
        assert "learning_rate" in space
        print("✓ Search space creation works")
        return True
    except Exception as e:
        print(f"✗ Search space creation failed: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Distributed LLM Platform - Installation Verification")
    print("=" * 60)
    print()

    results = []

    print("Testing module imports...")
    print("-" * 60)
    results.append(test_import("src", "Core package"))
    results.append(test_import("src.training", "Training module"))
    results.append(test_import("src.training.config", "Training config"))
    results.append(test_import("src.tuning", "Tuning module"))
    results.append(test_import("src.evaluation", "Evaluation module"))
    results.append(test_import("src.serving", "Serving module"))
    results.append(test_import("src.optimization", "Optimization module"))
    print()

    print("Testing core functionality...")
    print("-" * 60)
    results.append(test_config_creation())
    results.append(test_config_validation())
    results.append(test_search_space())
    print()

    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("✓ All basic verification tests passed!")
        print()
        print("Next steps:")
        print("1. Install full dependencies: pip install -r requirements.txt")
        print("2. Run example: python scripts/examples/quick_start.py")
        print("3. Read documentation: docs/guides/LEARNING_GUIDE.md")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
