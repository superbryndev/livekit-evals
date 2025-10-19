#!/usr/bin/env python3
"""
Simple test script to verify livekit-evals package installation and imports.

Run this after installing the package to ensure everything works:
    python test_import.py
"""

import sys


def test_imports():
    """Test that all package imports work correctly."""
    print("Testing livekit-evals package imports...")
    
    try:
        # Test main package import
        import livekit_evals
        print(f"✓ livekit_evals imported successfully (version {livekit_evals.__version__})")
        
        # Test webhook_handler imports
        from livekit_evals import WebhookHandler, create_webhook_handler
        print("✓ WebhookHandler imported successfully")
        print("✓ create_webhook_handler imported successfully")
        
        # Test that classes are accessible
        assert WebhookHandler is not None
        assert create_webhook_handler is not None
        assert callable(create_webhook_handler)
        print("✓ All imports are callable/accessible")
        
        # Test version
        assert hasattr(livekit_evals, '__version__')
        assert isinstance(livekit_evals.__version__, str)
        print(f"✓ Version string is valid: {livekit_evals.__version__}")
        
        # Test __all__ exports
        assert hasattr(livekit_evals, '__all__')
        assert 'WebhookHandler' in livekit_evals.__all__
        assert 'create_webhook_handler' in livekit_evals.__all__
        print("✓ __all__ exports are correct")
        
        print("\n✅ All tests passed! Package is working correctly.")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\nMake sure you've installed the package:")
        print("  pip install livekit-evals")
        print("Or for development:")
        print("  pip install -e .")
        return False
        
    except AssertionError as e:
        print(f"\n❌ Assertion error: {e}")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are available."""
    print("\nTesting dependencies...")
    
    dependencies = [
        ('livekit.agents', 'livekit-agents'),
        ('aiohttp', 'aiohttp'),
    ]
    
    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"✓ {package_name} is installed")
        except ImportError:
            print(f"❌ {package_name} is NOT installed")
            print(f"   Install with: pip install {package_name}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests."""
    print("=" * 60)
    print("livekit-evals Package Test")
    print("=" * 60)
    print()
    
    imports_ok = test_imports()
    deps_ok = test_dependencies()
    
    print("\n" + "=" * 60)
    if imports_ok and deps_ok:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())

