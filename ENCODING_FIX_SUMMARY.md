# MSG File Encoding Error Fix

## Problem Description

The SARA Compose application was encountering a Unicode decoding error when processing certain MSG files:

```
Exception in process_msg_file: 'cp950' codec can't decode byte 0x84 in position 0: illegal multibyte sequence
```

This error occurred when the `extract_msg` library tried to decode RTF content using the Traditional Chinese encoding (cp950) but encountered invalid byte sequences.

## Root Cause

The error was happening in the `process_msg_file` function when trying to access:
- `msg.htmlBody` - which internally processes RTF content
- `msg.body` - which could also have encoding issues
- Attachment properties like filenames

The `extract_msg` library was attempting to decode content using the cp950 codec, but the email contained byte sequences that are not valid in that encoding.

## Solution Implemented

### 1. Enhanced Error Handling for MSG File Initialization

Added try-catch block around MSG file initialization to catch corruption or encoding issues early:

```python
try:
    msg = extract_msg.Message(temp_path)
except Exception as e:
    print(f"Error initializing MSG file: {e}")
    # Clean up and return informative error
    return None, f"Failed to initialize MSG file (possibly corrupted or unsupported encoding): {e}"
```

### 2. Safe HTML Body Extraction

Wrapped `msg.htmlBody` access in comprehensive error handling:

```python
try:
    html_body = getattr(msg, 'htmlBody', None)
except UnicodeDecodeError as e:
    encoding_issues_detected = True
    print(f"Unicode decoding error when extracting HTML body: {e}")
    # Fallback to None and continue processing
    html_body = None
```

### 3. Safe Plain Text Body Extraction

Added similar protection for plain text body extraction:

```python
try:
    body = msg.body or ""
except UnicodeDecodeError as e:
    print(f"Unicode decoding error when extracting plain text body: {e}")
    body = ""  # Fallback to empty string
```

### 4. Safe Attachment Processing

Enhanced attachment processing to handle encoding issues in filenames:

```python
try:
    filename = attachment.longFilename or attachment.shortFilename
except UnicodeDecodeError as e:
    print(f"Unicode error extracting attachment filename: {e}")
    filename = "attachment_with_encoding_issue"
```

### 5. User-Friendly Warning Display

Added a theme-aware visual warning in the email preview when encoding issues are detected:

```python
if encoding_issues:
    encoding_warning = '''
    <div class="encoding-warning">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span class="warning-icon">⚠️</span>
            <div>
                <strong>Encoding Issues Detected</strong>
                <p>This email contains characters that couldn't be decoded properly...</p>
            </div>
        </div>
    </div>
    '''
```

### 6. Theme-Aware CSS Variables

Added CSS variables for warning colors that adapt to light and dark themes:

```css
/* Light Mode Warning Colors */
--warning-bg: #fff3cd;
--warning-border: #ffeaa7;
--warning-text: #856404;
--warning-icon: #d97706;

/* Dark Mode Warning Colors */
--warning-bg: #451a03;
--warning-border: #92400e;
--warning-text: #fbbf24;
--warning-icon: #f59e0b;
```

## Benefits of the Fix

1. **Graceful Degradation**: The application continues to work even when encoding issues occur
2. **Informative Feedback**: Users are notified when encoding problems are detected
3. **Content Preservation**: As much content as possible is extracted and displayed
4. **Robust Error Handling**: Multiple fallback strategies ensure the application doesn't crash
5. **Detailed Logging**: Console output helps with debugging encoding issues
6. **Theme-Aware Design**: Warning messages adapt to both light and dark themes
7. **Professional Appearance**: Consistent with HKMA purple theme and design standards

## Testing

Created comprehensive test suites that verify:

**Encoding Fix Tests (`test_encoding_fix.py`)**:
- Handling of invalid MSG files
- Graceful error handling for various input types
- Proper error message generation
- Function robustness

**Theme-Aware Warning Tests (`test_theme_aware_warning.py`)**:
- Warning displays correctly when encoding issues are detected
- No warning shown for normal emails
- CSS variables used instead of hardcoded colors
- Both light and dark mode color definitions present
- Proper CSS class structure implementation

## Files Modified

1. **app.py**: Enhanced `process_msg_file` and `format_email_preview` functions, added theme-aware CSS variables
2. **test_encoding_fix.py**: Test suite for the encoding fix
3. **test_theme_aware_warning.py**: Test suite for theme-aware warning functionality
4. **ENCODING_FIX_SUMMARY.md**: This documentation

## Usage

The fix is automatically applied when processing MSG files. If encoding issues are detected:

1. The application will continue processing the email
2. A warning message will be displayed to the user
3. Available content will be extracted using fallback methods
4. The email can still be processed for reply generation

## Future Considerations

- Consider adding support for additional encoding detection libraries
- Implement more sophisticated RTF to HTML conversion for problematic files
- Add user options for handling encoding issues (strict vs. lenient mode)
