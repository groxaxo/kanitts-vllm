"""Test script to verify BnB configuration is properly set up"""

import sys

def test_config_imports():
    """Test that config imports work correctly"""
    try:
        import config
        print("✅ Config module imported successfully")
        
        # Check BnB settings exist
        assert hasattr(config, 'USE_BNB_QUANTIZATION'), "USE_BNB_QUANTIZATION not found"
        assert hasattr(config, 'BNB_QUANTIZATION'), "BNB_QUANTIZATION not found"
        
        print(f"✅ USE_BNB_QUANTIZATION = {config.USE_BNB_QUANTIZATION}")
        print(f"✅ BNB_QUANTIZATION = {config.BNB_QUANTIZATION}")
        
        # Verify logic
        if config.USE_BNB_QUANTIZATION:
            assert config.BNB_QUANTIZATION == "bitsandbytes", "BNB_QUANTIZATION should be 'bitsandbytes' when enabled"
            print("✅ BnB quantization is enabled correctly")
        else:
            assert config.BNB_QUANTIZATION is None, "BNB_QUANTIZATION should be None when disabled"
            print("✅ BnB quantization is disabled correctly")
            
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_generator_imports():
    """Test that generator imports BnB config"""
    try:
        # Check file contents instead of importing (torch may not be available)
        with open('generation/vllm_generator.py', 'r') as f:
            content = f.read()
        
        assert 'from config import' in content and 'BNB_QUANTIZATION' in content, \
            "BNB_QUANTIZATION not imported in vllm_generator"
        print("✅ VLLMTTSGenerator imports BNB_QUANTIZATION")
        
        assert 'quantization=quantization' in content or 'quantization=' in content, \
            "quantization parameter not used in engine args"
        print("✅ VLLMTTSGenerator uses quantization parameter")
        
        assert 'if quantization is None:' in content and 'quantization = BNB_QUANTIZATION' in content, \
            "BNB_QUANTIZATION not used as default"
        print("✅ VLLMTTSGenerator defaults to BNB_QUANTIZATION from config")
        
        return True
    except Exception as e:
        print(f"❌ Generator test failed: {e}")
        return False

def test_server_imports():
    """Test that server imports BnB config"""
    try:
        # We can't fully import server as it requires async setup,
        # but we can check the file contents
        with open('server.py', 'r') as f:
            content = f.read()
        
        assert 'BNB_QUANTIZATION' in content, "BNB_QUANTIZATION not imported in server.py"
        assert 'quantization=BNB_QUANTIZATION' in content, "quantization parameter not set in server.py"
        print("✅ Server.py uses BNB_QUANTIZATION")
        
        return True
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("BitsAndBytes Configuration Test Suite")
    print("=" * 60)
    print()
    
    tests = [
        ("Config Imports", test_config_imports),
        ("Generator Imports", test_generator_imports),
        ("Server Configuration", test_server_imports),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        print(f"\n🧪 Running: {name}")
        print("-" * 60)
        if test_func():
            passed += 1
            print(f"✅ {name} PASSED")
        else:
            failed += 1
            print(f"❌ {name} FAILED")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 All tests passed! BnB integration is properly configured.")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
