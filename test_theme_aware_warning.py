#!/usr/bin/env python3
"""
Test script to verify that the theme-aware encoding warning works correctly.
This script tests the format_email_preview function with encoding issues.
"""

import sys
import os

# Add the current directory to the path so we can import from app.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import format_email_preview
    print("✅ Successfully imported format_email_preview from app.py")
except ImportError as e:
    print(f"❌ Failed to import format_email_preview: {e}")
    sys.exit(1)

def test_encoding_warning_display():
    """Test that the encoding warning displays correctly with theme-aware styling"""
    print("\n🧪 Testing theme-aware encoding warning display...")
    
    # Create test email info with encoding issues
    email_info_with_encoding_issues = {
        'sender': 'Test Sender <test@example.com>',
        'subject': 'Test Email with Encoding Issues',
        'date': 'Monday, January 1, 2024 12:00 PM',
        'body': 'This is a test email body.',
        'html_body': '<p>This is HTML content</p>',
        'to_recipients': ['recipient@example.com'],
        'cc_recipients': [],
        'attachments': [],
        'encoding_issues': True  # This should trigger the warning
    }
    
    # Create test email info without encoding issues
    email_info_without_encoding_issues = {
        'sender': 'Test Sender <test@example.com>',
        'subject': 'Test Email without Encoding Issues',
        'date': 'Monday, January 1, 2024 12:00 PM',
        'body': 'This is a test email body.',
        'html_body': '<p>This is HTML content</p>',
        'to_recipients': ['recipient@example.com'],
        'cc_recipients': [],
        'attachments': [],
        'encoding_issues': False  # No warning should appear
    }
    
    print("Testing email with encoding issues...")
    preview_with_warning = format_email_preview(email_info_with_encoding_issues)
    
    # Check if warning is present and uses CSS classes
    if 'encoding-warning' in preview_with_warning:
        print("✅ Encoding warning container found")
    else:
        print("❌ Encoding warning container not found")
        return False
    
    if 'warning-icon' in preview_with_warning:
        print("✅ Warning icon class found")
    else:
        print("❌ Warning icon class not found")
        return False
    
    if 'Encoding Issues Detected' in preview_with_warning:
        print("✅ Warning message text found")
    else:
        print("❌ Warning message text not found")
        return False
    
    # Check that hardcoded colors are NOT present (should use CSS variables)
    if '#fff3cd' in preview_with_warning or '#ffeaa7' in preview_with_warning or '#856404' in preview_with_warning:
        print("❌ Found hardcoded colors - should use CSS variables instead")
        return False
    else:
        print("✅ No hardcoded colors found - using CSS variables correctly")
    
    print("\nTesting email without encoding issues...")
    preview_without_warning = format_email_preview(email_info_without_encoding_issues)
    
    # Check that warning is NOT present
    if 'encoding-warning' not in preview_without_warning:
        print("✅ No encoding warning shown for normal email")
    else:
        print("❌ Encoding warning incorrectly shown for normal email")
        return False
    
    return True

def test_css_variables_defined():
    """Test that the CSS variables for warnings are properly defined"""
    print("\n🧪 Testing CSS variable definitions...")
    
    # Read the app.py file to check for CSS variable definitions
    try:
        with open('app.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for warning CSS variables
        warning_vars = [
            '--warning-bg:',
            '--warning-border:',
            '--warning-text:',
            '--warning-icon:'
        ]
        
        for var in warning_vars:
            if var in content:
                print(f"✅ Found CSS variable: {var}")
            else:
                print(f"❌ Missing CSS variable: {var}")
                return False
        
        # Check for both light and dark mode definitions
        if '--warning-bg: #fff3cd' in content:  # Light mode
            print("✅ Found light mode warning colors")
        else:
            print("❌ Missing light mode warning colors")
            return False
            
        if '--warning-bg: #451a03' in content:  # Dark mode
            print("✅ Found dark mode warning colors")
        else:
            print("❌ Missing dark mode warning colors")
            return False
        
        # Check for CSS class definition
        if '.encoding-warning {' in content:
            print("✅ Found encoding-warning CSS class definition")
        else:
            print("❌ Missing encoding-warning CSS class definition")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading app.py: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting theme-aware encoding warning tests...")
    print("=" * 60)
    
    try:
        # Test the warning display functionality
        warning_test_passed = test_encoding_warning_display()
        
        # Test the CSS variable definitions
        css_test_passed = test_css_variables_defined()
        
        print("\n" + "=" * 60)
        
        if warning_test_passed and css_test_passed:
            print("✅ All tests passed successfully!")
            print("\n📋 Summary:")
            print("- Encoding warning uses theme-aware CSS variables")
            print("- Warning appears only when encoding issues are detected")
            print("- No hardcoded colors found in warning HTML")
            print("- CSS variables defined for both light and dark modes")
            print("- Proper CSS class structure implemented")
            print("\n🎯 The encoding warning is now fully theme-aware!")
            return 0
        else:
            print("❌ Some tests failed!")
            return 1
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
