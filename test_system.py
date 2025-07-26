#!/usr/bin/env python3
"""
Simple test script to demonstrate the modernized activity recognition system.
This script tests basic functionality without requiring external dependencies.
"""

import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_config():
    """Test configuration system."""
    print("Testing configuration system...")
    
    try:
        from activity_recognition.config import Config
        
        config = Config()
        
        # Test basic configuration
        assert config.get('model.lstm.n_classes') == 101
        assert config.get('processing.frames_per_segment') == 30
        
        # Test setting values
        config.set('model.lstm.batch_size', 64)
        assert config.get('model.lstm.batch_size') == 64
        
        print("✅ Configuration system works correctly")
        return True
        
    except Exception as e:
        print(f"❌ Configuration system failed: {e}")
        return False

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        'activity_recognition.config',
        'activity_recognition.lstm_model',
        'activity_recognition.utils',
        'activity_recognition.cli'
    ]
    
    failed = []
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed.append(module)
    
    if not failed:
        print("✅ All core modules import successfully")
        return True
    else:
        print(f"❌ {len(failed)} modules failed to import")
        return False

def test_project_structure():
    """Test project structure."""
    print("Testing project structure...")
    
    required_paths = [
        "src/activity_recognition/__init__.py",
        "src/activity_recognition/config.py",
        "src/activity_recognition/lstm_model.py",
        "src/activity_recognition/cli.py",
        "tests/__init__.py",
        "main.py",
        "pyproject.toml",
        "README.md",
        "config/default.yaml",
        "config/ucf101_classes.txt"
    ]
    
    missing = []
    
    for path in required_paths:
        if not Path(path).exists():
            missing.append(path)
        else:
            print(f"  ✅ {path}")
    
    if not missing:
        print("✅ All required files exist")
        return True
    else:
        print(f"❌ Missing files: {missing}")
        return False

def test_model_creation():
    """Test LSTM model creation (without TensorFlow)."""
    print("Testing LSTM model structure...")
    
    try:
        from activity_recognition.lstm_model import LSTMModel
        
        # Create model instance with configuration
        config_override = {
            'model.lstm.n_classes': 10,
            'model.lstm.chunk_size': 512,
            'model.lstm.n_chunks': 8
        }
        
        model = LSTMModel(config_override)
        
        # Test configuration
        assert model.config.get('model.lstm.n_classes') == 10
        assert model.config.get('model.lstm.chunk_size') == 512
        
        print("✅ LSTM model structure works correctly")
        return True
        
    except Exception as e:
        print(f"❌ LSTM model creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Modernized Activity Recognition System")
    print("=" * 50)
    
    tests = [
        test_project_structure,
        test_config,
        test_imports,
        test_model_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        if test():
            passed += 1
    
    print()
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The modernized system is working correctly.")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up FlowNet2 models for feature extraction")
        print("3. Run: python main.py --help")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)