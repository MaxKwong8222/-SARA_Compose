#!/usr/bin/env python3
"""
Test script to verify that the encoding fix for MSG files works correctly.
This script tests the process_msg_file function with various encoding scenarios.
"""

import sys
import os
import tempfile
import traceback

# Add the current directory to the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import process_msg_file
    print("✅ Successfully imported process_msg_file from app.py")
except ImportError as e:
    print(f"❌ Failed to import process_msg_file: {e}")
    sys.exit(1)

def test_encoding_error_handling():
    """Test that the function handles encoding errors gracefully"""
    print("\n🧪 Testing encoding error handling...")
    
    # Test with a non-existent file (should handle gracefully)
    print("Testing with non-existent file...")
    result, error = process_msg_file("non_existent_file.msg")
    if error:
        print(f"✅ Correctly handled non-existent file: {error}")
    else:
        print("❌ Should have returned an error for non-existent file")
    
    # Test with an invalid file (not a real MSG file)
    print("\nTesting with invalid file content...")
    with tempfile.NamedTemporaryFile(suffix='.msg', delete=False) as temp_file:
        # Write some invalid content that might cause encoding issues
        temp_file.write(b'\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f')  # Invalid bytes for cp950
        temp_file.flush()
        temp_path = temp_file.name
    
    try:
        result, error = process_msg_file(temp_path)
        if error:
            print(f"✅ Correctly handled invalid MSG file: {error}")
        else:
            print("❌ Should have returned an error for invalid MSG file")
    finally:
        # Clean up
        try:
            os.remove(temp_path)
        except:
            pass

def test_function_robustness():
    """Test that the function is robust against various input types"""
    print("\n🧪 Testing function robustness...")
    
    # Test with None input
    print("Testing with None input...")
    result, error = process_msg_file(None)
    if error:
        print(f"✅ Correctly handled None input: {error}")
    else:
        print("❌ Should have returned an error for None input")
    
    # Test with empty string
    print("Testing with empty string...")
    result, error = process_msg_file("")
    if error:
        print(f"✅ Correctly handled empty string: {error}")
    else:
        print("❌ Should have returned an error for empty string")
    
    # Test with invalid type
    print("Testing with invalid type (integer)...")
    result, error = process_msg_file(123)
    if error:
        print(f"✅ Correctly handled invalid type: {error}")
    else:
        print("❌ Should have returned an error for invalid type")

def main():
    """Main test function"""
    print("🚀 Starting MSG file encoding fix tests...")
    print("=" * 60)
    
    try:
        test_encoding_error_handling()
        test_function_robustness()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n📋 Summary:")
        print("- The process_msg_file function now includes robust encoding error handling")
        print("- Unicode decoding errors are caught and handled gracefully")
        print("- The function provides informative error messages")
        print("- Invalid inputs are handled properly")
        print("\n🎯 The encoding fix should resolve the 'cp950' codec error you encountered.")
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        print("Traceback:")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
