import gradio as gr
import os
import extract_msg
import html2text
from bs4 import BeautifulSoup
import tempfile
import markdown
import re
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import fastapi_poe as fp
from dotenv import load_dotenv
import time
import threading
import json
import lxml
from concurrent.futures import ThreadPoolExecutor
import queue

# Load environment variables from .env file
load_dotenv()

# HTML Parser Cache - Global cache for parser availability and performance
PARSER_CACHE = {
    'lxml_available': None,
    'preferred_parser': None,
    'initialized': False,
    'test_results': {}
}

def initialize_parser_cache():
    """Initialize parser cache at startup to avoid repeated availability checks"""
    global PARSER_CACHE

    if PARSER_CACHE['initialized']:
        return

    print("Initializing HTML parser cache...")

    try:
        # Test lxml availability with actual parsing
        test_html = "<p>Parser test content</p>"
        start_time = time.time()
        soup = BeautifulSoup(test_html, "lxml")
        lxml_time = time.time() - start_time

        if soup and soup.find('p'):
            PARSER_CACHE['lxml_available'] = True
            PARSER_CACHE['preferred_parser'] = 'lxml'
            PARSER_CACHE['test_results']['lxml'] = lxml_time
            print(f"✅ lxml parser available and tested ({lxml_time:.4f}s)")
        else:
            raise Exception("lxml parsing test failed")

    except Exception as e:
        print(f"⚠️ lxml parser not available: {e}")
        PARSER_CACHE['lxml_available'] = False
        PARSER_CACHE['preferred_parser'] = 'html.parser'

    # Test html.parser as fallback
    try:
        start_time = time.time()
        soup = BeautifulSoup(test_html, "html.parser")
        html_parser_time = time.time() - start_time
        PARSER_CACHE['test_results']['html.parser'] = html_parser_time
        print(f"✅ html.parser available and tested ({html_parser_time:.4f}s)")
    except Exception as e:
        print(f"❌ html.parser failed: {e}")

    PARSER_CACHE['initialized'] = True
    print(f"Parser cache initialized. Preferred parser: {PARSER_CACHE['preferred_parser']}")

# HKMA Purple Color Theme Configuration
hkma_purple_color = gr.themes.Color(
    name="HKMA_purple",
    c50="#faf5ff",
    c100="#e9d5ff",
    c200="#c084fc",
    c300="#a855f7",
    c400="#7e22ce",
    c500="#6b21a8",
    c600="#581c87",
    c700="#4c1a73",
    c800="#461964",
    c900="#43185D",
    c950="#42185A",
)

# Create HKMA-themed Gradio theme
hkma_theme = gr.themes.Soft(
    primary_hue=hkma_purple_color,
    secondary_hue=hkma_purple_color,
    neutral_hue=gr.themes.colors.gray,
).set(
    # Button styling
    button_primary_background_fill=hkma_purple_color.c600,
    button_primary_background_fill_hover=hkma_purple_color.c700,
    button_primary_text_color="white",
    button_primary_border_color=hkma_purple_color.c600,

    # Input focus colors
    input_border_color_focus=hkma_purple_color.c500,

    # Block and label styling
    block_label_text_color=hkma_purple_color.c700,
)

MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ['.msg']

# POE API Configuration - Use environment variable for security
POE_API_KEY = os.getenv("POE_API_KEY", "")

# Available Models
POE_MODELS = ["GPT-4o", "DeepSeek-R1-Distill"]

# Model validation cache settings
MODEL_CACHE_DURATION = 300 # 5 minutes in seconds
model_validation_cache = {
"poe": {
"models": [],
"last_updated": 0,
"is_valid": False
}
}

# HTML Parser configuration
HTML_PARSER_OPTIONS = [
    "Auto (lxml with html.parser fallback)",
    "Force html.parser",
    "Force lxml"
]
DEFAULT_PARSER_CHOICE = "Auto (lxml with html.parser fallback)"

# Global parser preference (can be updated by UI)
current_parser_preference = DEFAULT_PARSER_CHOICE

# Model validation functions



def fetch_poe_models():
    """Fetch available models from POE API"""
    try:
        if not POE_API_KEY:
            print("POE API key not configured")
            return []

        # For POE, we'll use a simplified validation approach to avoid async generator cleanup issues
        # Since POE test calls can cause async generator cleanup warnings, we'll do a basic API key
        # format validation and assume our static model list is valid if the API key is properly configured

        # Basic API key validation - check if it's not empty and has reasonable format
        if POE_API_KEY and len(POE_API_KEY.strip()) > 10:
            # API key appears to be configured, return our known models
            # The actual validation will happen during real usage
            print(f"POE API key configured, returning {len(POE_MODELS)} models")
            return POE_MODELS
        else:
            print("POE API key appears to be invalid or too short")
            return []

    except ImportError:
        print("POE backend requires 'fastapi_poe' package")
        return []
    except Exception as e:
        print(f"Error validating POE API: {e}")
        return []

def validate_poe_models():
    """Validate POE models and update cache"""
    current_time = time.time()
    cache = model_validation_cache["poe"]

    # Check if cache is still valid
    if (cache["is_valid"] and
        current_time - cache["last_updated"] < MODEL_CACHE_DURATION):
        return cache["models"]

    # Fetch fresh model list
    available_models = fetch_poe_models()

    # Update cache
    cache["models"] = available_models
    cache["last_updated"] = current_time
    cache["is_valid"] = len(available_models) > 0

    print(f"Validated {len(available_models)} POE models")
    return available_models

def get_default_model():
    """Get default model for POE"""
    return "GPT-4o"

def validate_model_selection(model):
    """Validate that a model is available for POE"""
    available_models = validate_poe_models()
    return model in available_models

def get_fallback_model(unavailable_model):
    """Get a fallback model when the selected model becomes unavailable"""
    print(f"Model {unavailable_model} is no longer available for POE")

    available_models = validate_poe_models()
    if available_models:
        # Try to return the default model if available
        default_model = get_default_model()
        if default_model in available_models:
            return default_model
        # Otherwise return the first available model
        return available_models[0]

    return None
def initialize_model_validation():
    """Initialize model validation on startup"""
    print("Initializing model validation...")

    # Pre-warm the POE model cache if API key is available
    if POE_API_KEY:
        try:
            # Run validation in background thread to avoid blocking startup
            def background_poe_validation():
                validate_poe_models()

            validation_thread = threading.Thread(target=background_poe_validation, daemon=True)
            validation_thread.start()
            print("Started background POE model validation")
        except Exception as e:
            print(f"Failed to start background POE model validation: {e}")
    else:
        print("POE API key not configured, skipping POE model validation")

# HTML Parser selection functions
def get_parser_from_choice(parser_choice):
    """Convert UI choice to BeautifulSoup parser string"""
    if "html.parser" in parser_choice:
        return "html.parser"
    elif "lxml" in parser_choice:
        return "lxml"
    else:  # Auto mode
        return "auto"

def create_soup_with_parser(html_content, parser_choice, context=""):
    """Optimized BeautifulSoup creation with parser caching and performance tracking"""
    # Initialize cache if not already done
    if not PARSER_CACHE['initialized']:
        initialize_parser_cache()

    start_time = time.time()
    parser_used = None
    error_message = None

    try:
        parser_type = get_parser_from_choice(parser_choice)

        if parser_type == "auto":
            # Use cached preferred parser instead of trying both
            parser = PARSER_CACHE['preferred_parser']
            if parser == "lxml" and not PARSER_CACHE['lxml_available']:
                # Fallback if cache is inconsistent
                parser = "html.parser"
            soup = BeautifulSoup(html_content, parser)
            parser_used = parser

        elif parser_type == "html.parser":
            # Force html.parser
            soup = BeautifulSoup(html_content, "html.parser")
            parser_used = "html.parser"

        elif parser_type == "lxml":
            # Force lxml with cache check
            if not PARSER_CACHE['lxml_available']:
                error_message = "lxml parser not available. Please install with 'pip install lxml' or switch to html.parser"
                raise Exception(error_message)

            soup = BeautifulSoup(html_content, "lxml")
            parser_used = "lxml"

        parse_time = time.time() - start_time

        # Only log detailed timing in development mode or for slow operations
        if parse_time > 0.1 or context == "parser_test":
            print(f"HTML parsing in {context}: {parser_used} parser, {parse_time:.3f}s")

        return soup, parser_used, parse_time, None

    except Exception as e:
        parse_time = time.time() - start_time
        final_error = error_message or str(e)
        print(f"HTML parsing failed in {context}: {final_error}")
        return None, parser_used, parse_time, final_error

def get_parser_performance_info(parser_used, parse_time, error=None):
    """Generate performance information string for UI feedback"""
    if error:
        return f"❌ Parser Error: {error}"

    performance_icon = "🚀" if parser_used == "lxml" else "🐌"
    time_str = f"{parse_time:.3f}s"

    return f"{performance_icon} Parsed with {parser_used} ({time_str})"





# Default AI Instructions - comprehensive but concise
DEFAULT_AI_INSTRUCTIONS = """You are an expert email assistant. Write professional email replies that incorporate the provided key messages.

FORMATTING REQUIREMENTS:
- Use British English spelling and grammar throughout
- Write only the email body content (no subject line)
- Use professional greeting and closing with the sender's name
- Include proper paragraph spacing for readability
- Bold critical or actionable parts of the reply to draw attention effectively

CONTENT REQUIREMENTS:
- Address all key messages naturally in the email body
- Maintain professional tone appropriate to the context
- Do not repeat or quote the original email content
- Ensure the response reflects the sender's professional identity

STRUCTURE:
- Professional greeting addressing the recipient by name
- Clear, well-organized body paragraphs covering key messages
- Appropriate professional closing followed by sender's name"""

# Abstract AI Backend Interface
class AIBackend(ABC):
    """Abstract base class for AI backends"""

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the backend is available and healthy"""
        pass

    @abstractmethod
    def stream_response(self, prompt: str, model: str, conversation_history: list = None) -> Iterator[Tuple[str, bool]]:
        """Stream response from the AI backend

        Args:
            prompt: The input prompt
            model: Model identifier

        Yields:
            Tuple of (chunk_text, is_done)
        """
        pass


class POEBackend(AIBackend):
    """POE API backend implementation using fastapi_poe"""

    def __init__(self):
        self.api_key = POE_API_KEY

    def is_healthy(self) -> bool:
        """Check if POE API is available"""
        return bool(self.api_key)

    def stream_response(self, prompt: str, model: str, conversation_history: list = None) -> Iterator[Tuple[str, bool]]:
        """Stream response from POE API"""
        try:
            if not self.api_key:
                yield "POE API key not configured", True
                return

            # Validate model availability dynamically
            available_models = validate_poe_models()
            if model not in available_models:
                yield f"Model {model} not available in POE", True
                return

            # Prepare messages for POE API - use conversation history if provided
            if conversation_history:
                messages = []
                for msg in conversation_history:
                    # Map OpenAI role format to POE role format
                    poe_role = msg["role"]
                    if poe_role == "assistant":
                        poe_role = "bot" # POE uses 'bot' instead of 'assistant'
                    messages.append(fp.ProtocolMessage(role=poe_role, content=msg["content"]))
            else:
                messages = [fp.ProtocolMessage(role="user", content=prompt)]

            # Stream the response using POE API
            try:
                # Use the synchronous wrapper for easier integration
                response_generator = fp.get_bot_response_sync(
                    messages=messages,
                    bot_name=model,
                    api_key=self.api_key,
                    temperature=0.3
                )

                full_response = ""
                for partial_response in response_generator:
                    # POE API returns PartialResponse objects with text attribute
                    if hasattr(partial_response, 'text') and partial_response.text:
                        chunk_text = partial_response.text
                        full_response += chunk_text
                        yield chunk_text, False

                # Signal completion
                yield "", True
                return

            except Exception as e:
                yield f"POE API Error: {str(e)}", True
                return

        except Exception as e:
            yield f"POE Backend Error: {str(e)}", True
            return




# Simplified backend manager for POE only
class BackendManager:
    """Manages POE AI backend"""

    def __init__(self):
        self.poe_backend = POEBackend()

    def get_current_backend(self) -> AIBackend:
        """Get the POE backend"""
        return self.poe_backend

    def get_healthy_backend(self) -> AIBackend:
        """Get the POE backend if healthy"""
        if self.poe_backend.is_healthy():
            return self.poe_backend

        # If POE is not healthy, return it anyway (will show error)
        print("Warning: POE backend is not healthy")
        return self.poe_backend

    def is_backend_healthy(self) -> bool:
        """Check if POE backend is healthy"""
        return self.poe_backend.is_healthy()

    def is_any_backend_healthy(self) -> bool:
        """Check if POE backend is healthy"""
        return self.poe_backend.is_healthy()

    def get_available_models(self) -> list:
        """Get available models for POE"""
        validated_models = validate_poe_models()
        if not validated_models:
            print("No validated POE models available, falling back to static list")
            return POE_MODELS # Fallback to static list if validation fails
        return validated_models

    def get_backend_status(self) -> dict:
        """Get status of POE backend"""
        return {
            "poe": {
                "healthy": self.poe_backend.is_healthy(),
                "models": POE_MODELS
            },
            "current": "poe"
        }

# Initialize backend manager
backend_manager = BackendManager()

# Global thread pool for asynchronous AI generation
AI_THREAD_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai_gen")

def ai_generation_worker(result_queue, prompt, model, updated_conversation_history, info, key_msgs, user_name, ai_instructions, email_token_limit):
    """Worker function for AI generation that runs in background thread"""
    try:
        # Get healthy backend for AI generation
        healthy_backend = backend_manager.get_healthy_backend()
        full_response = ""

        # Stream the response using the backend
        for chunk, done in healthy_backend.stream_response(prompt, model, updated_conversation_history):
            full_response += chunk
            # Use thread-safe queue to communicate with main thread
            result_queue.put(('chunk', full_response, done))
            if done:
                break

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Exception in ai_generation_worker: {e}\n{tb}")
        result_queue.put(('error', str(e), True))



def create_bouncing_dots_html(text="Processing", model=None):
    """Create bouncing dots loading animation HTML with optional model information"""

    # Create detailed text with model info if provided
    if model:
        # POE models are already clean (e.g., "GPT-4o")
        detailed_text = f"{text} using {model} via POE"
    else:
        detailed_text = text

    return f"""
    <div style="display: flex; align-items: center; justify-content: center; padding: 20px;">
        <span style="margin-right: 12px; font-weight: 500; color: var(--text-primary);">{detailed_text}</span>
        <div class="bouncing-dots">
            <div class="dot"></div>
            <div class="dot"></div>
            <div class="dot"></div>
        </div>
    </div>
    """

def create_loading_overlay_html(text="Processing", model=None, background_content=""):
    """Create loading overlay that preserves background content while showing loading message"""

    # Create detailed text with model info if provided
    if model:
        # POE models are already clean (e.g., "GPT-4o")
        detailed_text = f"{text} using {model} via POE"
    else:
        detailed_text = text

    # If no background content provided, show placeholder
    if not background_content.strip():
        background_content = """
        <div class='thread-placeholder'>
            <div class='placeholder-content'>
                <div class='placeholder-icon'>📧</div>
                <h3>Draft Email Preview Will Appear Here</h3>
                <p>Upload an email and generate a response to see your draft email exactly as you'll download it</p>
            </div>
        </div>
        """

    return f"""
    <div style="position: relative; min-height: 200px;">
        <!-- Background content (dimmed) -->
        <div style="opacity: 0.3; pointer-events: none;">
            {background_content}
        </div>

        <!-- Loading overlay - positioned at top for better visibility -->
        <div style="
            position: absolute;
            top: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px 24px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(4px);
            z-index: 10;
            min-width: 280px;
            text-align: center;
        ">
            <div style="display: flex; align-items: center; justify-content: center;">
                <span style="margin-right: 12px; font-weight: 500; color: #374151; font-size: 13px;">{detailed_text}</span>
                <div class="bouncing-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
            </div>
        </div>
    </div>
    """







# SARA Framework CSS - Clean, professional design with HKMA purple color scheme

custom_css = """
/* ===== BANNER-ONLY CSS - Minimal styling for workflow banner ===== */

/* Essential CSS variables for banner colors and dark mode support */
:root {
    /* Primary Brand Colors - HKMA Purple Theme */
    --primary-color: #6b21a8;
    --primary-hover: #581c87;
    --primary-light: #a855f7;
    --primary-gradient: linear-gradient(135deg, #6b21a8 0%, #a855f7 100%);

    /* Light Mode Colors */
    --bg-primary: #ffffff;
    --bg-secondary: #f8fafc;
    --bg-tertiary: #f1f5f9;
    --text-primary: #000000;
    --text-secondary: #374151;
    --text-muted: #64748b;
    --border-light: #e5e7eb;
    --border-medium: #d1d5db;
    --scrollbar-track: #f1f5f9;
    --scrollbar-thumb: #cbd5e1;
    --scrollbar-thumb-hover: #94a3b8;

    /* Warning/Alert Colors - Light Mode */
    --warning-bg: #fff3cd;
    --warning-border: #ffeaa7;
    --warning-text: #856404;
    --warning-icon: #d97706;
}

/* Dark Mode Colors - Multiple selectors to catch Gradio's dark theme */
[data-theme="dark"],
.dark,
.gradio-container.dark,
body.dark,
html.dark,
.gradio-app.dark,
:root:has(.dark) {
    --bg-primary: #1f2937;
    --bg-secondary: #111827;
    --bg-tertiary: #374151;
    --text-primary: #f9fafb;
    --text-secondary: #e5e7eb;
    --text-muted: #9ca3af;
    --border-light: #374151;
    --border-medium: #4b5563;
    --scrollbar-track: #374151;
    --scrollbar-thumb: #6b7280;
    --scrollbar-thumb-hover: #9ca3af;

    /* Warning/Alert Colors - Dark Mode */
    --warning-bg: #451a03;
    --warning-border: #92400e;
    --warning-text: #fbbf24;
    --warning-icon: #f59e0b;
}

/* Auto-detect system dark mode preference and Gradio dark theme URL parameter */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #1f2937;
        --bg-secondary: #111827;
        --bg-tertiary: #374151;
        --text-primary: #f9fafb;
        --text-secondary: #e5e7eb;
        --text-muted: #9ca3af;
        --border-light: #374151;
        --border-medium: #4b5563;
        --scrollbar-track: #374151;
        --scrollbar-thumb: #6b7280;
        --scrollbar-thumb-hover: #9ca3af;

        /* Warning/Alert Colors - Dark Mode */
        --warning-bg: #451a03;
        --warning-border: #92400e;
        --warning-text: #fbbf24;
        --warning-icon: #f59e0b;
    }
}

/* Force dark mode when URL contains __theme=dark */
body:has([data-testid*="dark"]),
.gradio-container:has([data-testid*="dark"]),
.gradio-app:has([data-testid*="dark"]) {
    --bg-primary: #1f2937 !important;
    --bg-secondary: #111827 !important;
    --bg-tertiary: #374151 !important;
    --text-primary: #f9fafb !important;
    --text-secondary: #e5e7eb !important;
    --text-muted: #9ca3af !important;
    --border-light: #374151 !important;
    --border-medium: #4b5563 !important;
    --scrollbar-track: #374151 !important;
    --scrollbar-thumb: #6b7280 !important;
    --scrollbar-thumb-hover: #9ca3af !important;

    /* Warning/Alert Colors - Dark Mode */
    --warning-bg: #451a03 !important;
    --warning-border: #92400e !important;
    --warning-text: #fbbf24 !important;
    --warning-icon: #f59e0b !important;
}

/* ===== GRADIO NATIVE STYLING OVERRIDES ===== */
/* Borderless Group - Removes default Group styling (grey background and borders) */
.borderless-group {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

/* Comprehensive targeting for status-instructions Group to make it transparent */
#status-instructions,
#status-instructions > div,
#status-instructions > div > div,
#status-instructions .gradio-group,
#status-instructions .gradio-container,
#status-instructions .block,
.gradio-group#status-instructions,
.block#status-instructions,
.gradio-container#status-instructions {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

/* Additional comprehensive targeting for status-instructions Gradio elements */
.gradio-app #status-instructions,
.gradio-app #status-instructions > div,
.gradio-app #status-instructions .gradio-group {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

/* ===== DYNAMIC STATUS INSTRUCTIONS PANEL ===== */
.status-instructions-panel {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    position: relative;
    z-index: 100;
}

/* Workflow stages layout */
.workflow-stages {
    display: flex;
    gap: 16px;
    align-items: stretch;
}

.stage {
    flex: 1;
    padding: 12px;
    border-radius: 10px;
    border: 2px solid transparent;
    background: rgba(248, 250, 252, 0.6);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    opacity: 0.5;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.stage::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #6b21a8 0%, #a855f7 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.stage.active {
    background: linear-gradient(135deg, #faf5ff 0%, #e9d5ff 100%);
    border-color: #6b21a8;
    opacity: 1;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(107, 33, 168, 0.15);
}

.stage.active::before {
    opacity: 1;
}

.stage-header {
    display: flex;
    align-items: center;
    margin-bottom: 8px;
}

.stage-number {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background: #e2e8f0;
    color: #64748b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 1.4rem;
    margin-right: 12px;
    transition: all 0.3s ease;
    flex-shrink: 0;
}

.stage.active .stage-number {
    background: #6b21a8;
    color: white;
    box-shadow: 0 6px 20px rgba(107, 33, 168, 0.4);
}

.stage-icon {
    font-size: 4rem;
    margin-right: 12px;
    transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
    filter: grayscale(100%);
    display: inline-block;
    line-height: 1;
}

.stage.active .stage-icon {
    filter: grayscale(0%);
    animation: enhanced-wiggle-dance 1.2s ease-in-out infinite;
}

.stage-title {
    font-weight: 700;
    color: #1e293b;
    font-size: 1.4rem;
    margin: 0;
    line-height: 1.2;
}

.stage.active .stage-title {
    color: #1e293b;
    font-weight: 800;
}

.stage-description {
    color: #64748b;
    font-size: 1.1rem;
    line-height: 1.4;
    margin-top: 6px;
    font-weight: 500;
}

.stage.active .stage-description {
    color: #374151;
    font-weight: 600;
}

/* ===== CLICKABLE STAGE NAVIGATION STYLES ===== */
.stage.clickable {
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

/* ===== DISABLED STAGE NAVIGATION STYLES ===== */
.stage.disabled {
    opacity: 0.4;
    cursor: not-allowed;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    background: rgba(248, 250, 252, 0.3) !important;
    border-color: transparent !important;
}

.stage.disabled .stage-number {
    background: #f1f5f9 !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
}

.stage.disabled .stage-icon {
    filter: grayscale(100%) !important;
    opacity: 0.5;
}

.stage.disabled .stage-title {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

.stage.disabled .stage-description {
    color: #cbd5e1 !important;
    font-weight: 400 !important;
}

/* Prevent hover effects on disabled stages */
.stage.disabled:hover {
    background: rgba(248, 250, 252, 0.3) !important;
    border-color: transparent !important;
    transform: none !important;
    box-shadow: none !important;
    cursor: not-allowed;
}

.stage.disabled:hover::before {
    opacity: 0 !important;
}

.stage.disabled:hover .stage-number {
    background: #f1f5f9 !important;
    color: #94a3b8 !important;
    box-shadow: none !important;
}

.stage.disabled:hover .stage-title {
    color: #94a3b8 !important;
    font-weight: 500 !important;
}

.stage.disabled:hover .stage-description {
    color: #cbd5e1 !important;
    font-weight: 400 !important;
}

.stage.disabled:hover .stage-icon {
    filter: grayscale(100%) !important;
    transform: none !important;
    opacity: 0.5;
}

.stage.clickable:hover {
    background: linear-gradient(135deg, #faf5ff 0%, #e9d5ff 100%);
    border-color: #6b21a8;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(107, 33, 168, 0.15);
}

.stage.clickable:hover::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #6b21a8 0%, #581c87 100%);
    opacity: 1;
    transition: opacity 0.3s ease;
}

.stage.clickable:hover .stage-number {
    background: linear-gradient(135deg, #6b21a8 0%, #581c87 100%);
    color: white;
    box-shadow: 0 4px 12px rgba(107, 33, 168, 0.3);
}

.stage.clickable:hover .stage-title {
    color: #581c87;
    font-weight: 700;
}

.stage.clickable:hover .stage-description {
    color: #4c1a73;
    font-weight: 600;
}

.stage.clickable:hover .stage-icon {
    filter: grayscale(0%);
    transform: scale(1.05);
}

/* Focus styles for accessibility */
.stage.clickable:focus {
    outline: 2px solid #6b21a8;
    outline-offset: 2px;
    background: linear-gradient(135deg, #faf5ff 0%, #e9d5ff 100%);
    border-color: #6b21a8;
}

/* Button press animation with reduced motion support */
.stage.clickable:active {
    transform: translateY(0px);
    box-shadow: 0 2px 6px rgba(107, 33, 168, 0.2);
}

/* Respect user's motion preferences */
@media (prefers-reduced-motion: reduce) {
    .stage.clickable {
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }

    .stage.clickable:hover {
        transform: none;
    }

    .stage.clickable:hover .stage-icon {
        transform: none;
    }

    .stage.clickable:active {
        transform: none;
    }
}

/* Enhanced Wiggle Dance Animation for active stage icons */
@keyframes enhanced-wiggle-dance {
    0%, 100% {
        transform: rotate(0deg) scale(1);
        filter: grayscale(0%) drop-shadow(0 0 8px rgba(107, 33, 168, 0.4));
    }
    8% {
        transform: rotate(-5deg) scale(1.05);
        filter: grayscale(0%) drop-shadow(0 0 15px rgba(107, 33, 168, 0.6));
    }
    16% {
        transform: rotate(5deg) scale(1.08);
        filter: grayscale(0%) drop-shadow(0 0 20px rgba(107, 33, 168, 0.7));
    }
    24% {
        transform: rotate(-4deg) scale(1.06);
        filter: grayscale(0%) drop-shadow(0 0 18px rgba(107, 33, 168, 0.65));
    }
    32% {
        transform: rotate(4deg) scale(1.07);
        filter: grayscale(0%) drop-shadow(0 0 22px rgba(107, 33, 168, 0.75));
    }
    40% {
        transform: rotate(-3deg) scale(1.09);
        filter: grayscale(0%) drop-shadow(0 0 25px rgba(107, 33, 168, 0.8));
    }
    48% {
        transform: rotate(3deg) scale(1.07);
        filter: grayscale(0%) drop-shadow(0 0 22px rgba(107, 33, 168, 0.75));
    }
    56% {
        transform: rotate(-4deg) scale(1.05);
        filter: grayscale(0%) drop-shadow(0 0 18px rgba(107, 33, 168, 0.65));
    }
    64% {
        transform: rotate(4deg) scale(1.06);
        filter: grayscale(0%) drop-shadow(0 0 20px rgba(107, 33, 168, 0.7));
    }
    72% {
        transform: rotate(-3deg) scale(1.04);
        filter: grayscale(0%) drop-shadow(0 0 16px rgba(107, 33, 168, 0.6));
    }
    80% {
        transform: rotate(3deg) scale(1.03);
        filter: grayscale(0%) drop-shadow(0 0 14px rgba(107, 33, 168, 0.55));
    }
    88% {
        transform: rotate(-2deg) scale(1.02);
        filter: grayscale(0%) drop-shadow(0 0 12px rgba(107, 33, 168, 0.5));
    }
    96% {
        transform: rotate(1deg) scale(1.01);
        filter: grayscale(0%) drop-shadow(0 0 10px rgba(107, 33, 168, 0.45));
    }
}

/* Responsive design for workflow stages */
@media (max-width: 768px) {
    .workflow-stages {
        flex-direction: column;
        gap: 10px;
    }

    .stage {
        padding: 10px;
    }

    .status-instructions-panel {
        padding: 10px;
    }

    .stage-icon {
        font-size: 3.2rem;
        margin-right: 10px;
    }

    .stage-number {
        width: 40px;
        height: 40px;
        font-size: 1.2rem;
        margin-right: 10px;
    }

    .stage-title {
        font-size: 1.2rem;
    }

    .stage-description {
        font-size: 1rem;
        margin-top: 4px;
    }

    .stage-header {
        margin-bottom: 6px;
    }
}

/* ===== UPLOAD PANEL STYLING ===== */
/* Full-width upload panel styling with HKMA purple theme consistency and dark mode support */
.full-width-upload-panel {
    border: 2px solid var(--border-light) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    padding: 16px !important;
    margin: 16px 0 !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.full-width-upload-panel:hover {
    border-color: #6b21a8 !important;
    box-shadow: 0 4px 15px rgba(107, 33, 168, 0.1) !important;
}

/* File input styling within upload panel */
.full-width-file-input {
    border: 2px dashed var(--border-medium) !important;
    border-radius: 8px !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    transition: all 0.3s ease !important;
}

.full-width-file-input:hover {
    border-color: #6b21a8 !important;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
}

/* Additional comprehensive targeting for upload panel Gradio elements */
.gradio-app .full-width-upload-panel,
.gradio-app .full-width-upload-panel > div,
.gradio-app .full-width-upload-panel .gradio-group {
    border: 2px solid var(--border-light) !important;
    border-radius: 12px !important;
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-primary) 100%) !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    padding: 16px !important;
    margin: 16px 0 !important;
}

/* ===== EMAIL PREVIEW COMPONENTS ===== */
/* Email Panel Container Styling - Theme-aware styling with dark mode support */
.email-panel-container {
    border: 1px solid var(--border-medium);
    border-top: none;
    background: var(--bg-primary);
    border-radius: 0 0 8px 8px;
    overflow: hidden;
    position: relative;
    z-index: 1;
}

.email-panel-content {
    padding: 0;
    margin: 0;
    background: var(--bg-primary);
    border: none;
}

/* ===== GLOBAL TEXTBOX STYLING - Remove grey rectangles ===== */
/* Fix for unwanted grey rectangles at bottom of input boxes */
.gradio-textbox {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

.gradio-textbox > div {
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
    background: transparent !important;
}

/* Remove any grey container backgrounds from all textbox elements */
.gradio-textbox,
.gradio-textbox > div,
.gradio-textbox > div > div,
.gradio-textbox > div > div > div {
    background: transparent !important;
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}

/* ===== PLACEHOLDER CONTENT STYLING ===== */
/* Theme-aware placeholder content with proper dark mode support */
.thread-placeholder,
.email-placeholder {
    background: var(--bg-primary);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 40px 20px;
    text-align: center;
    color: var(--text-muted);
}

/* ===== WARNING/ALERT STYLING ===== */
/* Theme-aware encoding warning with proper dark mode support */
.encoding-warning {
    background: var(--warning-bg) !important;
    border: 1px solid var(--warning-border) !important;
    border-radius: 4px;
    padding: 12px;
    margin-bottom: 16px;
    color: var(--warning-text) !important;
}

.encoding-warning strong {
    color: var(--warning-text) !important;
}

.encoding-warning p {
    color: var(--warning-text) !important;
    margin: 4px 0 0 0;
    font-size: 0.9em;
}

.encoding-warning .warning-icon {
    color: var(--warning-icon) !important;
    font-size: 16px;
}

.placeholder-content {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
}

.placeholder-content .placeholder-icon {
    font-size: 2.5rem;
    margin-bottom: 8px;
}

.placeholder-content h3 {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
}

.placeholder-content p {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.4;
}

.placeholder-content .placeholder-hint {
    color: var(--text-muted);
    font-size: 0.8rem;
    font-style: italic;
    margin-top: 8px;
}

/* ===== EMAIL CONTENT STYLING ===== */
/* Theme-aware email content containers */
.email-content-container {
    font-family: Calibri, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.0;
    background: var(--bg-primary);
    border: 2px solid var(--border-light);
    border-radius: 8px;
    overflow: hidden;
    margin: 0;
}

.email-header-section {
    background: var(--bg-secondary);
    border-bottom: 2px solid var(--border-light);
    padding: 16px;
}

.email-body-section {
    padding: 16px;
    font-family: 'Microsoft Sans Serif', sans-serif;
    font-size: 11pt;
    line-height: 1.0;
    color: var(--text-primary);
    background: var(--bg-primary);
}

.email-scroll-container {
    max-height: 550px;
    overflow-y: auto;
    overflow-x: hidden;
    scrollbar-width: thin;
    scrollbar-color: var(--scrollbar-thumb) var(--scrollbar-track);
}

/* Custom scrollbar styling for email containers */
.email-scroll-container::-webkit-scrollbar {
    width: 6px;
}

.email-scroll-container::-webkit-scrollbar-track {
    background: var(--scrollbar-track);
    border-radius: 3px;
}

.email-scroll-container::-webkit-scrollbar-thumb {
    background: var(--scrollbar-thumb);
    border-radius: 3px;
}

.email-scroll-container::-webkit-scrollbar-thumb:hover {
    background: var(--scrollbar-thumb-hover);
}

/* Email header field styling */
.email-header-field {
    margin: 3px 0;
    font-size: 11pt;
    display: flex;
}

.email-header-label {
    font-weight: bold;
    color: var(--text-secondary);
    min-width: 80px;
    display: inline-block;
}

.email-header-value {
    color: var(--text-secondary);
}

/* Email thread content styling */
.email-thread-content {
    font-family: 'Microsoft Sans Serif', sans-serif;
    font-size: 11pt;
    line-height: 1.0;
    color: var(--text-primary);
}

/* Email paragraph styling - Outlook compatible */
.email-paragraph {
    margin: 0;
    padding: 0;
    margin-bottom: 0pt;
    line-height: 1.0;
    font-family: 'Microsoft Sans Serif', sans-serif;
    font-size: 11pt;
    color: var(--text-primary);
}

.email-paragraph-calibri {
    margin: 0;
    padding: 0;
    margin-bottom: 0pt;
    font-family: Calibri, Arial, sans-serif;
    font-size: 11pt;
    color: var(--text-primary);
    line-height: 1.0;
}

/* Original email content styling */
.original-email-content {
    margin-top: 16px;
    border-top: 1px solid var(--border-light);
    padding-top: 8px;
    font-family: Calibri, Arial, sans-serif;
}

.original-email-body {
    margin-top: 0px;
    font-family: Calibri, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.0;
    color: var(--text-primary);
}

/* Error and fallback content styling */
.error-content {
    padding: 20px;
    background: var(--bg-primary);
    border-radius: 8px;
    margin: 20px;
    border: 1px solid #ef4444;
    color: #dc2626;
    text-align: center;
    font-family: 'Microsoft Sans Serif', sans-serif;
    line-height: 1.6;
    font-size: 11pt;
}

.empty-state {
    color: var(--text-muted);
    text-align: center;
    padding: 20px;
    font-style: italic;
}

/* ===== SIDEBAR DESCRIPTION STYLING ===== */
/* Theme-aware description box styling */
.description-box {
    background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-light);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.description-text {
    margin: 0;
    color: var(--text-secondary);
    line-height: 1.6;
    font-size: 0.9rem;
    text-align: left;
}

/* Sidebar section headers */
.sidebar-section-header {
    margin: 0 0 12px 0;
    color: var(--text-muted);
    font-size: 0.9rem;
}

.sidebar-section-header.with-top-margin {
    margin: 16px 0 12px 0;
}

/* Disclaimer and contact information styling */
.disclaimer-text {
    margin: 0;
    color: var(--text-muted);
    line-height: 1.6;
    font-size: 0.85rem;
}

.contact-text {
    margin-bottom: 16px;
    color: var(--text-secondary);
    line-height: 1.6;
}

.info-table {
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    overflow: hidden;
}

.info-table th {
    background-color: var(--bg-tertiary);
    border: 1px solid var(--border-light);
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    color: var(--text-secondary);
}

.info-table td {
    border: 1px solid var(--border-light);
    padding: 12px;
    color: var(--text-secondary);
}

/* Parser info styling */
.parser-info {
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 8px;
}

.parser-info.error {
    color: #ef4444;
}

.parser-info.success {
    color: #10b981;
}

/* ===== HYPERLINK STYLING ===== */
/* Consistent hyperlink colors across light and dark themes */
a,
.email-content-container a,
.email-body-section a,
.email-thread-content a,
.original-email-body a {
    color: #2563eb !important; /* Blue color for light mode */
    text-decoration: underline;
}

/* Dark mode hyperlink styling - maintain blue but with better contrast */
[data-theme="dark"] a,
.dark a,
.gradio-container.dark a,
body.dark a,
html.dark a,
.gradio-app.dark a,
:root:has(.dark) a,
.dark-mode a,
[data-theme="dark"] .email-content-container a,
.dark .email-content-container a,
.dark-mode .email-content-container a,
[data-theme="dark"] .email-body-section a,
.dark .email-body-section a,
.dark-mode .email-body-section a,
[data-theme="dark"] .email-thread-content a,
.dark .email-thread-content a,
.dark-mode .email-thread-content a,
[data-theme="dark"] .original-email-body a,
.dark .original-email-body a,
.dark-mode .original-email-body a {
    color: #60a5fa !important; /* Lighter blue for dark mode with better contrast */
    text-decoration: underline;
}

/* Visited link styling */
a:visited,
.email-content-container a:visited,
.email-body-section a:visited,
.email-thread-content a:visited,
.original-email-body a:visited {
    color: #7c3aed !important; /* Purple for visited links in light mode */
}

/* Dark mode visited link styling */
[data-theme="dark"] a:visited,
.dark a:visited,
.dark-mode a:visited,
[data-theme="dark"] .email-content-container a:visited,
.dark .email-content-container a:visited,
.dark-mode .email-content-container a:visited,
[data-theme="dark"] .email-body-section a:visited,
.dark .email-body-section a:visited,
.dark-mode .email-body-section a:visited,
[data-theme="dark"] .email-thread-content a:visited,
.dark .email-thread-content a:visited,
.dark-mode .email-thread-content a:visited,
[data-theme="dark"] .original-email-body a:visited,
.dark .original-email-body a:visited,
.dark-mode .original-email-body a:visited {
    color: #a78bfa !important; /* Lighter purple for visited links in dark mode */
}

/* JavaScript-based dark mode detection and application */
</style>

<script>
// Dark mode detection and application for SARA Compose
(function() {
    function applyDarkMode() {
        // Check URL parameter for dark theme
        const urlParams = new URLSearchParams(window.location.search);
        const isDarkTheme = urlParams.get('__theme') === 'dark';

        // Check system preference
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;

        // Check if Gradio has applied dark theme classes
        const hasGradioDark = document.body.classList.contains('dark') ||
                             document.documentElement.classList.contains('dark') ||
                             document.querySelector('.gradio-container.dark') !== null;

        if (isDarkTheme || prefersDark || hasGradioDark) {
            document.documentElement.classList.add('dark-mode');
            document.body.classList.add('dark-mode');

            // Apply dark mode CSS variables
            const root = document.documentElement;
            root.style.setProperty('--bg-primary', '#1f2937');
            root.style.setProperty('--bg-secondary', '#111827');
            root.style.setProperty('--bg-tertiary', '#374151');
            root.style.setProperty('--text-primary', '#f9fafb');
            root.style.setProperty('--text-secondary', '#e5e7eb');
            root.style.setProperty('--text-muted', '#9ca3af');
            root.style.setProperty('--border-light', '#374151');
            root.style.setProperty('--border-medium', '#4b5563');
            root.style.setProperty('--scrollbar-track', '#374151');
            root.style.setProperty('--scrollbar-thumb', '#6b7280');
            root.style.setProperty('--scrollbar-thumb-hover', '#9ca3af');

            // Warning/Alert Colors - Dark Mode
            root.style.setProperty('--warning-bg', '#451a03');
            root.style.setProperty('--warning-border', '#92400e');
            root.style.setProperty('--warning-text', '#fbbf24');
            root.style.setProperty('--warning-icon', '#f59e0b');
        }
    }

    // Apply on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyDarkMode);
    } else {
        applyDarkMode();
    }

    // Re-apply when URL changes (for SPA navigation)
    window.addEventListener('popstate', applyDarkMode);

    // Watch for system theme changes
    if (window.matchMedia) {
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyDarkMode);
    }

    // Watch for Gradio theme changes
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'attributes' &&
                (mutation.attributeName === 'class' || mutation.attributeName === 'data-theme')) {
                applyDarkMode();
            }
        });
    });

    observer.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-theme'] });
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-theme'] });
})();
</script>

<style>

/* Ensure textarea elements have proper theme-aware background */
.gradio-textbox textarea {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    margin-bottom: 0 !important;
    padding-bottom: 16px !important;
}

/* ===== EMAIL PREVIEW COLUMN STYLING ===== */
.email-preview-column {
    overflow: visible;
    margin-right: 16px;
    flex: 4;
    min-width: 800px;
    width: 80%;
}

.email-preview-column .gradio-accordion {
    overflow: visible;
}

.email-preview-column .gradio-accordion .accordion-content {
    overflow: visible;
    max-height: none;
}

/* ===== THREAD DISPLAY AREA STYLING ===== */
.thread-display-area {
    background: var(--bg-primary);
    border: none;
    padding: 0;
    margin: 0;
}

/* Remove any default Gradio container styling from thread display */
.thread-display-area > div,
.thread-display-area .gradio-html,
.thread-display-area .gradio-html > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ===== ORIGINAL REFERENCE DISPLAY AREA STYLING ===== */
.original-reference-display-area {
    background: var(--bg-primary);
    border: none;
    padding: 0;
    margin: 0;
}

/* Remove any default Gradio container styling from original reference display */
.original-reference-display-area > div,
.original-reference-display-area .gradio-html,
.original-reference-display-area .gradio-html > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}


"""

# Initialize backend manager on app start
print("Initializing AI backends...")
backend_status = backend_manager.get_backend_status()
print(f"POE backend healthy: {backend_status['poe']['healthy']}")
print(f"Current backend: {backend_status['current']}")
print("Backend initialization complete.")

# ===== FILE PROCESSING FUNCTIONS =====

def validate_file(file):
    try:
        if file is None:
            return None, "No file uploaded."
        # If file is a dict (Gradio >= 3.41), use its keys
        if isinstance(file, dict):
            filename = file.get('name', 'uploaded_file')
            size_mb = file.get('size', 0) / (1024 * 1024)
        elif hasattr(file, 'name') and hasattr(file, 'seek') and hasattr(file, 'tell'):
            filename = file.name
            file.seek(0, os.SEEK_END)
            size_mb = file.tell() / (1024 * 1024)
            file.seek(0)
        elif isinstance(file, bytes):
            filename = "uploaded_file"
            size_mb = len(file) / (1024 * 1024)
        else:
            # fallback: skip size check, just check extension
            filename = str(file)
            size_mb = 0
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return None, f"Invalid file type: {ext}. Only .msg files are allowed."
        if size_mb > MAX_FILE_SIZE_MB:
            return None, f"File size {size_mb:.2f}MB exceeds the 10MB limit."
        return file, None
    except Exception as e:
        print(f"Validation error: {e}")
        return None, f"Validation error: {e}"



def html_to_text(html):
    """
    Convert HTML content to plain text using html2text library.

    Uses configurable parser selection with performance tracking.

    Args:
        html (str): HTML content to convert

    Returns:
        str: Plain text representation of the HTML
    """
    if not html:
        return ""

    # Use the configured parser preference
    soup, parser_used, parse_time, error = create_soup_with_parser(
        html, current_parser_preference, "html_to_text"
    )

    if error:
        print(f"HTML parsing failed in html_to_text: {error}")
        return html  # Return original HTML if parsing fails

    if soup is None:
        return html

    clean_html = soup.prettify()
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    h.ignore_images = True
    h.ignore_emphasis = False
    h.ignore_tables = False
    return h.handle(clean_html)

# ===== EMAIL FORMATTING FUNCTIONS =====

def standardize_date_format(date_input):
    """Standardize date format to match Microsoft Outlook exactly: 'Day, Month DD, YYYY H:MM AM/PM'"""
    if not date_input or date_input == 'Unknown':
        return 'Unknown'

    try:
        from datetime import datetime
        import re

        # Handle datetime objects directly
        if isinstance(date_input, datetime):
            # Format exactly like Outlook: "Tuesday, June 3, 2025 6:26 PM"
            # Always use Windows-compatible format and remove leading zeros manually
            formatted = date_input.strftime('%A, %B %d, %Y %I:%M %p')
            # Remove leading zeros manually for cross-platform compatibility
            formatted = re.sub(r' 0(\d,)', r' \1', formatted)  # Remove leading zero from day
            formatted = re.sub(r' 0(\d:\d{2} [AP]M)', r' \1', formatted)  # Remove leading zero from hour
            return formatted

        # Convert to string if not already
        date_str = str(date_input).strip()

        # Enhanced date patterns to handle more formats
        date_patterns = [
            # ISO format with timezone: 2025-06-03 18:25:59+08:00
            (r'(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2})', '%Y-%m-%d %H:%M:%S'),
            # ISO format with T: 2025-06-03T18:25:59
            (r'(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})', '%Y-%m-%dT%H:%M:%S'),
            # US format: Tuesday, June 3, 2025 12:05 PM
            (r'(\w+),\s+(\w+)\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}:\d{2})\s+(AM|PM)', '%A, %B %d, %Y %I:%M %p'),
            # Short format: June 3, 2025 12:05 PM
            (r'(\w+)\s+(\d{1,2}),\s+(\d{4})\s+(\d{1,2}:\d{2})\s+(AM|PM)', '%B %d, %Y %I:%M %p'),
            # Date only: May 30, 2025
            (r'(\w+)\s+(\d{1,2}),\s+(\d{4})', '%B %d, %Y'),
            # RFC 2822 format: Mon, 03 Jun 2025 18:26:00 +0800
            (r'(\w+),\s+(\d{1,2})\s+(\w+)\s+(\d{4})\s+(\d{2}:\d{2}:\d{2})', '%a, %d %b %Y %H:%M:%S'),
        ]

        # Try to parse with different patterns
        for pattern, format_str in date_patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                try:
                    # Extract the matched part for parsing
                    matched_text = match.group(0)

                    # Handle timezone info by removing it
                    if '+' in matched_text:
                        matched_text = matched_text.split('+')[0].strip()
                    if '-' in matched_text and 'T' not in matched_text:
                        # Only split on timezone minus, not date minus
                        parts = matched_text.split('-')
                        if len(parts) > 3:  # Has timezone
                            matched_text = '-'.join(parts[:-1]).strip()

                    # Parse the date
                    dt = datetime.strptime(matched_text, format_str)

                    # Return in Outlook format: "Tuesday, June 3, 2025 6:26 PM"
                    # Always use Windows-compatible format and remove leading zeros manually
                    formatted = dt.strftime('%A, %B %d, %Y %I:%M %p')
                    # Remove leading zeros manually for cross-platform compatibility
                    formatted = re.sub(r' 0(\d,)', r' \1', formatted)  # Remove leading zero from day
                    formatted = re.sub(r' 0(\d:\d{2} [AP]M)', r' \1', formatted)  # Remove leading zero from hour

                    return formatted
                except Exception as parse_error:
                    print(f"Parse error for '{matched_text}': {parse_error}")
                    continue

        # If no pattern matches, return original string
        return date_str

    except Exception as e:
        print(f"Date formatting error: {e}")
        return str(date_input) if date_input else 'Unknown'

# ===== EMAIL DISPLAY FUNCTIONS =====

def format_email_preview(email_info):
    """Format email content directly as Outlook-style display without thread parsing"""
    if not email_info:
        return "<div class='empty-state'>No email content to display</div>"

    # Get email details directly from email_info
    sender = email_info.get('sender', 'Unknown')
    subject = email_info.get('subject', '(No Subject)')
    date = standardize_date_format(email_info.get('date', 'Unknown'))
    body = email_info.get('body', '')
    to_recipients = email_info.get('to_recipients', [])
    cc_recipients = email_info.get('cc_recipients', [])
    html_body = email_info.get('html_body', '')
    attachments = email_info.get('attachments', [])
    encoding_issues = email_info.get('encoding_issues', False)

    # Process email body content with Outlook-compatible formatting
    if html_body:
        try:
            from bs4 import BeautifulSoup
            # Parse and clean HTML content while preserving formatting
            # Use configurable parser selection
            soup, parser_used, parse_time, error = create_soup_with_parser(
                html_body, current_parser_preference, "format_email_preview"
            )

            if error or soup is None:
                print(f"HTML parsing failed in format_email_preview: {error}")
                # Fallback to plain text processing
                if body:
                    text_lines = body.split('\n')
                    formatted_lines = []
                    for line in text_lines:
                        if line.strip():
                            formatted_lines.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;">{line}</p>')
                        else:
                            formatted_lines.append('<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')
                    body_html = ''.join(formatted_lines)
                else:
                    body_html = '<p style="margin: 0; padding: 0; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;"><em>No content</em></p>'
            else:
                # Remove script tags for security
                for script in soup.find_all('script'):
                    script.decompose()

                # Apply Outlook-compatible styling to all paragraphs
                for p in soup.find_all('p'):
                    p['style'] = 'margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;'

                # Apply Outlook-compatible styling to lists
                for ul in soup.find_all('ul'):
                    ul['style'] = 'margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;'

                for ol in soup.find_all('ol'):
                    ol['style'] = 'margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;'

                for li in soup.find_all('li'):
                    li['style'] = 'margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;'

                # Handle embedded images with cid: references
                for img in soup.find_all('img'):
                    src = img.get('src', '')
                    if src.startswith('cid:'):
                        content_id = src.replace('cid:', '')
                        # Find matching attachment
                        for attachment in attachments:
                            if attachment.get('content_id') == content_id:
                                # Convert to base64 data URL
                                import base64
                                file_ext = attachment.get('filename', '').split('.')[-1].lower()
                                mime_type = {
                                    'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                                    'png': 'image/png', 'gif': 'image/gif',
                                    'bmp': 'image/bmp', 'svg': 'image/svg+xml'
                                }.get(file_ext, 'image/jpeg')

                                base64_data = base64.b64encode(attachment.get('data', b'')).decode('utf-8')
                                data_url = f"data:{mime_type};base64,{base64_data}"
                                img['src'] = data_url
                                break

                    # Ensure images are responsive
                    current_style = img.get('style', '')
                    if 'max-width' not in current_style:
                        img['style'] = current_style + '; max-width: 100%; height: auto;'

                # Get the cleaned HTML content
                body_html = str(soup)

        except Exception as e:
            print(f"Error processing HTML body: {e}")
            # Fallback to plain text processing with Outlook-compatible formatting
            if body:
                text_lines = body.split('\n')
                formatted_lines = []
                for line in text_lines:
                    if line.strip():
                        formatted_lines.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;">{line}</p>')
                    else:
                        formatted_lines.append('<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')
                body_html = ''.join(formatted_lines)
            else:
                body_html = '<p style="margin: 0; padding: 0; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;"><em>No content</em></p>'
    elif body:
        # Convert plain text to HTML with Outlook-compatible formatting
        text_lines = body.split('\n')
        formatted_lines = []
        for line in text_lines:
            if line.strip():
                # Convert **text** to <strong>text</strong>
                line_formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', line)
                formatted_lines.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;">{line_formatted}</p>')
            else:
                formatted_lines.append('<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')
        body_html = ''.join(formatted_lines)
    else:
        body_html = '<p style="margin: 0; padding: 0; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt;"><em>No content</em></p>'

    # Format recipient lists for display with clickable mailto links
    def format_recipients(recipients):
        """Format recipient list for display with clickable mailto links"""
        if not recipients:
            return "(None)"

        formatted_recipients = []
        for recipient in recipients:
            # Extract email from "Name <email>" format or use as-is
            email_match = re.search(r'<([^>]+)>', recipient)
            if email_match:
                email = email_match.group(1)
                name = recipient.replace(f'<{email}>', '').strip().strip('"')
                formatted_recipients.append(f'<a href="mailto:{email}">{name} &lt;{email}&gt;</a>')
            else:
                # Assume it's just an email address
                formatted_recipients.append(f'<a href="mailto:{recipient}">{recipient}</a>')

        if len(formatted_recipients) == 1:
            return formatted_recipients[0]
        elif len(formatted_recipients) <= 3:
            return ", ".join(formatted_recipients)
        else:
            # For more than 3 recipients, show first 2 and count
            return f"{formatted_recipients[0]}, {formatted_recipients[1]}, and {len(formatted_recipients) - 2} more"

    to_display = format_recipients(to_recipients)
    cc_display = format_recipients(cc_recipients)

    # Build header lines in standardized order: Sent, To, Cc, Subject (From removed since it will always be the user)
    header_lines = [
        f'<div class="email-header-field"><span class="email-header-label">Sent:</span><span class="email-header-value">{date}</span></div>'
    ]

    # Add To: field if recipients exist
    if to_recipients:
        header_lines.append(f'<div class="email-header-field"><span class="email-header-label">To:</span><span class="email-header-value">{to_display}</span></div>')

    # Add Cc: field if recipients exist
    if cc_recipients:
        header_lines.append(f'<div class="email-header-field"><span class="email-header-label">Cc:</span><span class="email-header-value">{cc_display}</span></div>')

    # Add Subject last
    header_lines.append(f'<div class="email-header-field"><span class="email-header-label">Subject:</span><span class="email-header-value">{subject}</span></div>')

    # Add encoding warning if issues were detected
    encoding_warning = ""
    if encoding_issues:
        encoding_warning = '''
        <div class="encoding-warning">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span class="warning-icon">⚠️</span>
                <div>
                    <strong>Encoding Issues Detected</strong>
                    <p>
                        This email contains characters that couldn't be decoded properly. The content has been processed using fallback methods, but some formatting or special characters may not display correctly.
                    </p>
                </div>
            </div>
        </div>
        '''

    # Create beautiful email preview with theme-aware styling and scroll functionality - 10% height increase
    email_preview = f'''
    <div class="email-scroll-container">
        <div class="email-content-container">
            {encoding_warning}
            <!-- Email Header Section -->
            <div class="email-header-section">
                {"".join(header_lines)}
            </div>

            <!-- Email Body Section -->
            <div class="email-body-section">
                {body_html}
            </div>
        </div>
    </div>
    '''

    return email_preview

def extract_and_separate_think_content(text):
    """Extract <think> content and return both parts separately"""
    if not text or '<think>' not in text:
        return text, None
    
    # Extract think content
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        think_content = think_match.group(1).strip()
        # Remove think tags from main content
        main_content = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        return main_content, think_content
    
    return text, None

def truncate_email_content(text, token_limit=2000):
    """Truncate email content to specified token limit (approximate)"""
    if not text or not token_limit:
        return text

    # Rough approximation: 1 token ≈ 4 characters for English text
    char_limit = int(token_limit * 4)

    if len(text) <= char_limit:
        return text

    # Truncate at word boundary near the limit
    truncated = text[:char_limit]
    last_space = truncated.rfind(' ')
    if last_space > char_limit * 0.8:  # If we can find a space in the last 20%
        truncated = truncated[:last_space]

    return truncated + "...[content truncated]"

def validate_and_restore_ai_instructions(ai_instructions):
    """Validate AI instructions and restore defaults if empty or insufficient"""
    if not ai_instructions or not ai_instructions.strip():
        print("AI instructions empty, restoring defaults")
        return DEFAULT_AI_INSTRUCTIONS

    return ai_instructions.strip()

def format_reply_content_simple(text):
    """Format reply content with Outlook-compatible spacing using empty paragraphs instead of CSS margins"""
    if not text:
        return ""

    # Clean the text
    clean_text = text.strip()

    # Remove any subject line patterns at the beginning
    clean_text = re.sub(r'^Subject:\s*.*?\n\s*', '', clean_text, flags=re.IGNORECASE | re.MULTILINE)
    clean_text = re.sub(r'^RE:\s*.*?\n\s*', '', clean_text, flags=re.IGNORECASE | re.MULTILINE)

    # Convert markdown to HTML properly with enhanced formatting
    try:
        # Handle bold text and other markdown
        html_content = markdown.markdown(clean_text, extensions=['nl2br'])
        # Remove any subject line patterns that might be in HTML
        html_content = re.sub(r'<p[^>]*>Subject:\s*.*?</p>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<p[^>]*>RE:\s*.*?</p>', '', html_content, flags=re.IGNORECASE)

        # Apply Outlook-compatible paragraph styling (no bottom margin, use empty paragraphs for spacing)
        html_content = re.sub(r'<p>', '<p class="email-paragraph">', html_content)

        # Insert empty paragraphs between content paragraphs for Outlook-compatible spacing
        html_content = re.sub(r'</p>\s*<p', '</p><p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p><p', html_content)
    except:
        # Fallback: enhanced formatting with proper paragraph breaks using empty paragraphs
        html_content = clean_text

        # Split into paragraphs (double line breaks for proper Outlook-style spacing)
        paragraphs = html_content.split('\n\n')
        formatted_paragraphs = []

        for i, para in enumerate(paragraphs):
            if para.strip():
                # Skip paragraphs that look like subject lines
                if re.match(r'^\s*(Subject|RE):\s*', para.strip(), re.IGNORECASE):
                    continue

                # Convert single line breaks to <br> within paragraphs
                para_formatted = para.replace('\n', '<br>')
                # Convert **text** to <strong>text</strong>
                para_formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para_formatted)
                # Wrap in paragraph tags with Outlook-compatible styling (no bottom margin)
                formatted_paragraphs.append(f'<p class="email-paragraph">{para_formatted}</p>')

                # Add empty paragraph for spacing between content paragraphs (except for the last one)
                if i < len([p for p in paragraphs if p.strip()]) - 1:
                    formatted_paragraphs.append('<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')

        html_content = ''.join(formatted_paragraphs)

    # Return content directly without wrapper div to avoid double spacing
    return html_content

def format_reply_content(text):
    """Format reply content with Outlook-compatible spacing using empty paragraphs instead of CSS margins"""
    if not text:
        return "<div class='empty-state'>No content to display</div>"

    # Clean the text and remove any subject line that might have been included
    clean_text = text.strip()

    # Remove any subject line patterns at the beginning
    clean_text = re.sub(r'^Subject:\s*.*?\n\s*', '', clean_text, flags=re.IGNORECASE | re.MULTILINE)
    clean_text = re.sub(r'^RE:\s*.*?\n\s*', '', clean_text, flags=re.IGNORECASE | re.MULTILINE)

    # Convert markdown to HTML properly with enhanced formatting
    try:
        # Handle bold text and other markdown
        html_content = markdown.markdown(clean_text, extensions=['nl2br'])
        # Remove any subject line patterns that might be in HTML
        html_content = re.sub(r'<p[^>]*>Subject:\s*.*?</p>', '', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<p[^>]*>RE:\s*.*?</p>', '', html_content, flags=re.IGNORECASE)

        # Apply Outlook-compatible paragraph styling (no bottom margin, use empty paragraphs for spacing)
        html_content = re.sub(r'<p>', '<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0;">', html_content)

        # Insert empty paragraphs between content paragraphs for Outlook-compatible spacing
        html_content = re.sub(r'</p>\s*<p', '</p><p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p><p', html_content)
    except:
        # Fallback: enhanced formatting with proper paragraph breaks using empty paragraphs
        html_content = clean_text

        # Split into paragraphs (single line breaks for proper Outlook-style spacing)
        paragraphs = html_content.split('\n\n')
        formatted_paragraphs = []

        for i, para in enumerate(paragraphs):
            if para.strip():
                # Skip paragraphs that look like subject lines
                if re.match(r'^\s*(Subject|RE):\s*', para.strip(), re.IGNORECASE):
                    continue

                # Convert single line breaks to <br> within paragraphs
                para_formatted = para.replace('\n', '<br>')
                # Convert **text** to <strong>text</strong>
                para_formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para_formatted)
                # Wrap in paragraph tags with Outlook-compatible styling (no bottom margin)
                formatted_paragraphs.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0;">{para_formatted}</p>')

                # Add empty paragraph for spacing between content paragraphs (except for the last one)
                if i < len([p for p in paragraphs if p.strip()]) - 1:
                    formatted_paragraphs.append('<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')

        html_content = ''.join(formatted_paragraphs)

    # Ensure proper styling for the reply content using draft-content class
    return f'<div class="draft-content">{html_content}</div>'

def normalize_email_address(email_str):
    """Extract and normalize email address from various formats for comparison"""
    if not email_str:
        return ""

    # Extract email from "Name <email>" format
    import re
    email_match = re.search(r'<([^>]+)>', email_str)
    if email_match:
        email = email_match.group(1).strip()
    else:
        # Assume it's just an email address, clean it up
        email = email_str.strip().strip('"').strip("'")

    # Return lowercase for case-insensitive comparison
    return email.lower()

def is_same_email_address(email1, email2):
    """Compare two email addresses, handling different formats and case sensitivity"""
    if not email1 or not email2:
        return False

    normalized_email1 = normalize_email_address(email1)
    normalized_email2 = normalize_email_address(email2)

    return normalized_email1 == normalized_email2



def format_complete_email_thread_preview(reply_text, original_email_info, user_email="", user_name=""):
    """Format complete email thread preview that matches exactly what gets downloaded"""
    try:
        # Get email details
        original_sender = original_email_info.get('sender', 'Unknown')
        original_subject = original_email_info.get('subject', '(No Subject)')
        to_recipients = original_email_info.get('to_recipients', [])
        cc_recipients = original_email_info.get('cc_recipients', [])

        # Create reply subject
        reply_subject = original_subject
        if not reply_subject.lower().startswith('re:'):
            reply_subject = f"RE: {reply_subject}"

        # Build comprehensive CC list: all original recipients (To + CC) except sender and user
        all_original_recipients = to_recipients + cc_recipients
        reply_cc_recipients = []

        for recipient in all_original_recipients:
            # Skip the original sender (they become the To recipient)
            if is_same_email_address(recipient, original_sender):
                continue
            # Skip the user's email if provided - improved logic with robust email comparison
            if user_email and is_same_email_address(recipient, user_email):
                continue
            # Add to CC if not already present (check for duplicates using normalized comparison)
            is_duplicate = False
            for existing_recipient in reply_cc_recipients:
                if is_same_email_address(recipient, existing_recipient):
                    is_duplicate = True
                    break
            if not is_duplicate:
                reply_cc_recipients.append(recipient)

        # Format CC recipients with clickable mailto links using authentic Outlook blue
        def format_email_links(recipients):
            if not recipients:
                return 'None'
            formatted = []
            for recipient in recipients:
                # Extract email from "Name <email>" format or use as-is
                email_match = re.search(r'<([^>]+)>', recipient)
                if email_match:
                    email = email_match.group(1)
                    name = recipient.replace(f'<{email}>', '').strip().strip('"')
                    formatted.append(f'<a href="mailto:{email}">{name} &lt;{email}&gt;</a>')
                else:
                    # Assume it's just an email address
                    formatted.append(f'<a href="mailto:{recipient}">{recipient}</a>')
            return '; '.join(formatted)

        cc_display = format_email_links(reply_cc_recipients)

        # Format To recipient (original sender) with mailto link
        to_display = format_email_links([original_sender]) if original_sender != 'Unknown' else 'Unknown'

        # Create threaded content for preview (use theme-aware colors)
        threaded_html, _ = create_threaded_email_content(reply_text, original_email_info, for_email_client=False)

        # Use the properly formatted threaded_html content from create_threaded_email_content
        # This ensures proper HTML rendering like Stage 2
        
        # Create complete email thread preview with theme-aware styling - 10% height increase
        thread_preview = f"""
<div class="email-scroll-container">
    <div class="email-content-container">
        <!-- Email Header Section -->
        <div class="email-header-section">
            <div class="email-header-field">
                <span class="email-header-label">From:</span>
                <span class="email-header-value">{user_name + ' <' + user_email + '>' if user_name and user_email else 'SARA Compose <sara.compose@example.com>'}</span>
            </div>
            <div class="email-header-field">
                <span class="email-header-label">To:</span>
                <span class="email-header-value">{to_display}</span>
            </div>
            <div class="email-header-field">
                <span class="email-header-label">Cc:</span>
                <span class="email-header-value">{cc_display}</span>
            </div>
            <div class="email-header-field">
                <span class="email-header-label">Subject:</span>
                <span class="email-header-value">{reply_subject}</span>
            </div>
        </div>

        <!-- Email Body Section with proper single line spacing -->
        <div class="email-body-section">
            <div class="email-thread-content">
                {threaded_html}
            </div>
        </div>
    </div>
</div>"""

        return thread_preview

    except Exception as e:
        print(f"Error creating email thread preview: {e}")
        return format_reply_content(reply_text)

def create_threaded_email_content(reply_text, original_email_info, for_email_client=False):
    """Create a complete threaded email with reply and original content

    Args:
        reply_text: The reply content
        original_email_info: Original email information
        for_email_client: If True, use hardcoded colors for email client compatibility.
                         If False, use CSS variables for theme-aware preview.
    """
    try:
        from bs4 import BeautifulSoup

        # Clean the reply text for both HTML and plain text
        try:
            # Try lxml parser first for better performance
            soup = BeautifulSoup(reply_text, 'lxml')
        except Exception as e:
            print(f"lxml parsing failed, falling back to html.parser: {e}")
            # Fallback to html.parser if lxml fails
            soup = BeautifulSoup(reply_text, 'html.parser')
        reply_plain_text = soup.get_text().strip()

        # Remove any subject line from the plain text reply
        reply_plain_text = re.sub(r'^Subject:\s*.*?\n\s*', '', reply_plain_text, flags=re.IGNORECASE | re.MULTILINE)
        reply_plain_text = re.sub(r'^RE:\s*.*?\n\s*', '', reply_plain_text, flags=re.IGNORECASE | re.MULTILINE)

        # Ensure proper paragraph formatting for plain text version (single line breaks for Outlook compatibility)
        # Split by double line breaks and rejoin with single line breaks
        paragraphs = reply_plain_text.split('\n\n')
        formatted_paragraphs = []
        for para in paragraphs:
            if para.strip():
                # Clean up single line breaks within paragraphs
                para_clean = ' '.join(para.split())
                formatted_paragraphs.append(para_clean)

        # Join paragraphs with single line breaks for proper Outlook-style spacing
        reply_plain_text = '\n'.join(formatted_paragraphs)

        # Determine color based on context (define early so it's available throughout the function)
        text_color = "#000000" if for_email_client else "var(--text-primary)"

        # Get original email details with complete content preservation
        original_sender = original_email_info.get('sender', 'Unknown')
        original_date = standardize_date_format(original_email_info.get('date', 'Unknown'))
        original_subject = original_email_info.get('subject', '(No Subject)')
        original_body = original_email_info.get('body', '')
        original_html_body = original_email_info.get('html_body', '')
        to_recipients = original_email_info.get('to_recipients', [])
        cc_recipients = original_email_info.get('cc_recipients', [])

        # Use HTML body if available for complete content preservation
        if original_html_body and original_html_body.strip():
            # Clean up HTML for email threading while preserving all content
            try:
                # Try lxml parser first for better performance
                soup = BeautifulSoup(original_html_body, 'lxml')
            except Exception as e:
                print(f"lxml parsing failed, falling back to html.parser: {e}")
                # Fallback to html.parser if lxml fails
                soup = BeautifulSoup(original_html_body, 'html.parser')

            # Remove any script tags for security
            for script in soup.find_all('script'):
                script.decompose()

            # Apply Outlook-compatible styling to all paragraphs in original content
            for p in soup.find_all('p'):
                p['style'] = f'margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt; color: {text_color};'

            # Apply Outlook-compatible styling to lists in original content
            for ul in soup.find_all('ul'):
                ul['style'] = 'margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;'

            for ol in soup.find_all('ol'):
                ol['style'] = 'margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;'

            for li in soup.find_all('li'):
                li['style'] = f'margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt; color: {text_color};'

            # Get the body content if it exists, otherwise use the entire soup
            body_content = soup.find('body')
            if body_content:
                original_body_for_threading = str(body_content.decode_contents())
            else:
                # Preserve all other HTML elements including images, tables, formatting
                original_body_for_threading = str(soup)

            # If the result is empty or just whitespace, fall back to plain text
            if not original_body_for_threading.strip():
                original_body_for_threading = original_body.replace('\n', '<br>')
        else:
            # Fallback to plain text with basic HTML formatting and proper styling
            if original_body:
                # Convert plain text to HTML with Outlook-compatible formatting
                text_lines = original_body.split('\n')
                formatted_lines = []
                for line in text_lines:
                    if line.strip():
                        formatted_lines.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt; color: {text_color};">{line}</p>')
                    else:
                        formatted_lines.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')
                original_body_for_threading = ''.join(formatted_lines)
            else:
                original_body_for_threading = f'<p style="margin: 0; padding: 0; line-height: 1.0; font-family: Calibri, sans-serif; font-size: 11pt; color: {text_color};"><em>No content</em></p>'

        # Format recipients for display (use comma separation like Outlook)
        to_display = ', '.join(to_recipients) if to_recipients else ''
        cc_display = ', '.join(cc_recipients) if cc_recipients else ''

        # Preserve HTML formatting while cleaning unwanted elements
        clean_reply_html = reply_text

        # Remove only subject lines, preserve all other formatting
        clean_reply_html = re.sub(r'<p[^>]*>Subject:\s*.*?</p>', '', clean_reply_html, flags=re.IGNORECASE)
        clean_reply_html = re.sub(r'<p[^>]*>RE:\s*.*?</p>', '', clean_reply_html, flags=re.IGNORECASE)

        # Check if the reply text contains HTML formatting
        has_html = bool(re.search(r'<[^>]+>', clean_reply_html))

        if has_html:
            # Already HTML - preserve all formatting including lists, colors, styles
            formatted_reply_html = clean_reply_html

            # Clean up any wrapper divs but preserve inner content formatting
            try:
                # Try lxml parser first for better performance
                soup = BeautifulSoup(formatted_reply_html, 'lxml')
            except Exception as e:
                print(f"lxml parsing failed, falling back to html.parser: {e}")
                # Fallback to html.parser if lxml fails
                soup = BeautifulSoup(formatted_reply_html, 'html.parser')

            # If wrapped in a single div, extract contents but preserve all inner HTML
            if soup.find('div', class_='reply-content') or soup.find('div', class_='draft-content'):
                content_div = soup.find('div', class_='reply-content') or soup.find('div', class_='draft-content')
                if content_div:
                    formatted_reply_html = str(content_div.decode_contents())

            # Ensure proper styling for email clients while preserving original formatting
            # Add email-safe CSS for lists and formatting elements using Outlook-compatible spacing
            formatted_reply_html = formatted_reply_html.replace('<ul>', '<ul style="margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;">')
            formatted_reply_html = formatted_reply_html.replace('<ol>', '<ol style="margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;">')
            formatted_reply_html = formatted_reply_html.replace('<li>', '<li style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: \'Microsoft Sans Serif\', sans-serif; font-size: 11pt;">')

            # Ensure all paragraphs have consistent Outlook-compatible styling (no bottom margin)
            formatted_reply_html = re.sub(r'<p(?![^>]*style=)', f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: \'Microsoft Sans Serif\', sans-serif; font-size: 11pt; color: {text_color};"', formatted_reply_html)
            formatted_reply_html = re.sub(r'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1\.0;"', f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: \'Microsoft Sans Serif\', sans-serif; font-size: 11pt; color: {text_color};"', formatted_reply_html)

            # Insert empty paragraphs between content paragraphs for Outlook-compatible spacing (only if not already present)
            # Check if empty paragraphs are already present to avoid double-spacing
            if '>&nbsp;</p>' not in formatted_reply_html:
                formatted_reply_html = re.sub(r'</p>\s*<p', '</p><p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p><p', formatted_reply_html)

        else:
            # Plain text - convert to HTML while preserving line breaks and structure using Outlook-compatible spacing
            paragraphs = clean_reply_html.split('\n\n')
            formatted_paragraphs = []

            for i, para in enumerate(paragraphs):
                if para.strip():
                    # Convert single line breaks to <br> within paragraphs
                    para_formatted = para.strip().replace('\n', '<br>')
                    # Convert **text** to <strong>text</strong>
                    para_formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para_formatted)
                    # Convert bullet points to proper lists
                    if para_formatted.startswith('•') or para_formatted.startswith('-') or para_formatted.startswith('*'):
                        # Handle bullet lists
                        lines = para_formatted.split('<br>')
                        list_items = []
                        for line in lines:
                            if line.strip():
                                # Remove bullet characters and create list item
                                clean_line = re.sub(r'^[•\-\*]\s*', '', line.strip())
                                list_items.append(f'<li style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: \'Microsoft Sans Serif\', sans-serif; font-size: 11pt;">{clean_line}</li>')
                        if list_items:
                            formatted_paragraphs.append(f'<ul style="margin: 0; padding: 0; margin-left: 18pt; line-height: 1.0;">{"".join(list_items)}</ul>')
                    else:
                        # Regular paragraph with Microsoft Sans Serif for AI-generated content (no bottom margin)
                        formatted_paragraphs.append(f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-family: \'Microsoft Sans Serif\', sans-serif; font-size: 11pt; color: {text_color};">{para_formatted}</p>')

                    # Add empty paragraph for spacing between content paragraphs (except for the last one)
                    if i < len([p for p in paragraphs if p.strip()]) - 1:
                        formatted_paragraphs.append('<p style="margin: 0; padding: 0; margin-bottom: 0pt; line-height: 1.0; font-size: 11pt;">&nbsp;</p>')

            formatted_reply_html = ''.join(formatted_paragraphs)

        # Determine colors and border based on context
        border_color = "#E1E1E1" if for_email_client else "var(--border-light)"

        # Create threaded content with Microsoft Sans Serif for AI reply using tight Outlook-compatible spacing (like Stage 2)
        threaded_html = f"""<div style="font-family: 'Microsoft Sans Serif', sans-serif; font-size: 11pt; line-height: 1.0; color: {text_color};">
{formatted_reply_html}
</div>
<div style="margin-top: 16px; border-top: 1px solid {border_color}; padding-top: 8px; font-family: Calibri, Arial, sans-serif;">
<p style="margin: 0; padding: 0; margin-bottom: 0pt; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: {text_color}; line-height: 1.0;"><strong>From:</strong> {original_sender}</p>
<p style="margin: 0; padding: 0; margin-bottom: 0pt; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: {text_color}; line-height: 1.0;"><strong>Sent:</strong> {original_date}</p>"""

        if to_recipients:
            threaded_html += f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: {text_color}; line-height: 1.0;"><strong>To:</strong> {to_display}</p>'

        if cc_recipients:
            threaded_html += f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: {text_color}; line-height: 1.0;"><strong>Cc:</strong> {cc_display}</p>'

        # Add subject line to match Outlook format
        threaded_html += f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: {text_color}; line-height: 1.0;"><strong>Subject:</strong> {original_subject}</p>'

        # Add blank line after Subject (matching Outlook format)
        threaded_html += f'<p style="margin: 0; padding: 0; margin-bottom: 0pt; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: {text_color}; line-height: 1.0;">&nbsp;</p>'

        # Add original email body content with Calibri font for authentic Outlook look
        threaded_html += f"""
<div style="margin-top: 0px; font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.0; color: {text_color};">
{original_body_for_threading}
</div>
</div>"""

        # Create plain text version (Outlook style with proper separator)
        threaded_plain = f"""{reply_plain_text}

________________________________
From: {original_sender}
Sent: {original_date}"""

        if to_recipients:
            threaded_plain += f"\nTo: {to_display}"

        if cc_recipients:
            threaded_plain += f"\nCc: {cc_display}"

        # Add subject line to match Outlook format
        threaded_plain += f"\nSubject: {original_subject}"

        # Add blank line after Subject (matching Outlook format) and original email body content
        threaded_plain += f"""

{original_body}"""

        return threaded_html, threaded_plain

    except Exception as e:
        print(f"Error creating threaded content: {e}")
        return reply_text, reply_text

def create_msg_file(reply_text, original_email_info, output_path, user_email="", user_name=""):
    """Create a draft email file with threading and proper CC recipients"""
    try:
        from datetime import datetime

        # Create threaded email content for email client (use hardcoded colors)
        threaded_html, threaded_plain = create_threaded_email_content(reply_text, original_email_info, for_email_client=True)

        # Get original email info
        original_sender = original_email_info.get('sender', 'Unknown')
        original_subject = original_email_info.get('subject', '(No Subject)')
        to_recipients = original_email_info.get('to_recipients', [])
        cc_recipients = original_email_info.get('cc_recipients', [])

        # Create reply subject
        reply_subject = original_subject
        if not reply_subject.lower().startswith('re:'):
            reply_subject = f"RE: {reply_subject}"

        # Build comprehensive CC list: all original recipients (To + CC) except sender and user
        all_original_recipients = to_recipients + cc_recipients
        reply_cc_recipients = []

        for recipient in all_original_recipients:
            # Skip the original sender (they become the To recipient)
            if is_same_email_address(recipient, original_sender):
                continue
            # Skip the user's email if provided - improved logic with robust email comparison
            if user_email and is_same_email_address(recipient, user_email):
                continue
            # Add to CC if not already present (check for duplicates using normalized comparison)
            is_duplicate = False
            for existing_recipient in reply_cc_recipients:
                if is_same_email_address(recipient, existing_recipient):
                    is_duplicate = True
                    break
            if not is_duplicate:
                reply_cc_recipients.append(recipient)

        # Format CC recipients for email header
        cc_header = ""
        if reply_cc_recipients:
            cc_header = f"Cc: {'; '.join(reply_cc_recipients)}\n"

        # Create draft EML content with user identity or default
        from_field = "SARA Compose <sara.compose@example.com>"
        if user_email:
            if user_name:
                from_field = f"{user_name} <{user_email}>"
            else:
                from_field = user_email

        # Create draft EML content with original sender as default To recipient
        eml_content = f"""From: {from_field}
To: {original_sender}
{cc_header}Subject: {reply_subject}
Date: {datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')}
MIME-Version: 1.0
Content-Type: multipart/alternative; boundary="boundary123"
X-Unsent: 1

--boundary123
Content-Type: text/plain; charset=utf-8

{threaded_plain}

--boundary123
Content-Type: text/html; charset=utf-8

<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="Generator" content="Microsoft Outlook">
    <title>Email Reply Draft</title>
    <style>
        body {{
            font-family: 'Microsoft Sans Serif', sans-serif;
            font-size: 11pt;
            line-height: 1.0;
            color: #000000; /* Keep black for email compatibility */
            margin: 0;
            padding: 0;
        }}
        a {{ color: #0563C1; text-decoration: underline; }}
        p {{ margin: 0; padding: 0; margin-bottom: 0pt; }}
        .original-email {{
            margin-top: 16px;
            border-top: 1px solid #E1E1E1;
            padding-top: 8px;
            font-family: Calibri, Arial, sans-serif;
        }}
        .quoted-header {{
            font-family: Calibri, Arial, sans-serif;
            font-size: 11pt;
            color: #000000; /* Keep black for email compatibility */
            line-height: 1.0;
        }}
    </style>
</head>
<body>
{threaded_html}
</body>
</html>

--boundary123--
"""

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(eml_content)

        return True, None

    except Exception as e:
        return False, f"Failed to create draft email file: {str(e)}"

def export_reply_to_msg(reply_text, original_email_info, user_email="", user_name=""):
    """Export reply as downloadable draft email file"""
    try:
        import tempfile
        import os
        from datetime import datetime

        if not reply_text or not reply_text.strip():
            return None, "No reply content to export"

        # Create descriptive filename based on subject
        subject = original_email_info.get('subject', 'Email_Reply')
        # Clean subject for filename
        clean_subject = re.sub(r'[^\w\s-]', '', subject).strip()
        clean_subject = re.sub(r'[-\s]+', '_', clean_subject)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"SARA_Draft_{clean_subject}_{timestamp}.eml"
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, filename)

        # Create the draft email file
        success, error = create_msg_file(reply_text, original_email_info, file_path, user_email, user_name)

        if success:
            return file_path, None
        else:
            return None, error

    except Exception as e:
        return None, f"Export failed: {str(e)}"

def process_msg_file(file):
    try:
        print(f"process_msg_file received: {type(file)} {file}")
        with tempfile.NamedTemporaryFile(delete=False, suffix='.msg') as temp:
            temp_path = temp.name
            if isinstance(file, dict):
                file_bytes = file.get('file') or file.get('data')
                print(f"dict file_bytes type: {type(file_bytes)}")
                if hasattr(file_bytes, 'read'):
                    file_bytes = file_bytes.read()
                temp.write(file_bytes)
            elif isinstance(file, bytes):
                print("file is bytes")
                temp.write(file)
            elif hasattr(file, 'read'):
                print("file is file-like object")
                file.seek(0)
                temp.write(file.read())
            elif isinstance(file, str) and os.path.exists(file):
                print(f"file is file path: {file}")
                with open(file, "rb") as fsrc:
                    temp.write(fsrc.read())
            else:
                print(f"Unsupported file type: {type(file)}")
                return None, f"Unsupported file type for processing: {type(file)} {file}"
        print(f"temp_path: {temp_path}")

        # Initialize MSG file with encoding error handling
        try:
            msg = extract_msg.Message(temp_path)
        except Exception as e:
            print(f"Error initializing MSG file: {e}")
            # Clean up temp file before returning error
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return None, f"Failed to initialize MSG file (possibly corrupted or unsupported encoding): {e}"

        # Extract sender with proper name and email formatting
        raw_sender = msg.sender or "Unknown"

        # Try multiple ways to get sender email address
        sender_email = None

        # Method 1: Try senderEmailAddress property
        sender_email = getattr(msg, 'senderEmailAddress', None)

        # Method 2: Try senderEmailAddress with different casing
        if not sender_email:
            sender_email = getattr(msg, 'senderemailaddress', None)

        # Method 3: Try to extract from sender name if it contains email
        if not sender_email and raw_sender and '<' in raw_sender and '>' in raw_sender:
            email_match = re.search(r'<([^>]+)>', raw_sender)
            if email_match:
                sender_email = email_match.group(1)

        # Method 4: Try to get from message properties
        if not sender_email:
            try:
                # Try different property names that might contain sender email
                for prop_name in ['senderEmailAddress', 'senderEmail', 'fromEmail', 'from']:
                    prop_value = getattr(msg, prop_name, None)
                    if prop_value and '@' in str(prop_value):
                        sender_email = str(prop_value)
                        break
            except:
                pass

        # Method 5: Look in the original email body for sender email patterns
        if not sender_email and hasattr(msg, 'body') and msg.body:
            # Look for email patterns in the body that might be the sender's email
            email_patterns = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', msg.body)
            if email_patterns:
                # Use the first email found (often the sender's email in signatures)
                sender_email = email_patterns[0]

        print(f"Debug: raw_sender='{raw_sender}', sender_email='{sender_email}'")

        # Clean up sender formatting - ensure proper "Name <email>" format
        if raw_sender != "Unknown":
            # Remove quotes around the name part if present
            if raw_sender.startswith('"') and '"' in raw_sender[1:]:
                # Extract name from quotes and email if present
                quote_end = raw_sender.find('"', 1)
                name_part = raw_sender[1:quote_end]
                email_part = raw_sender[quote_end+1:].strip()
                if email_part.startswith(' <') and email_part.endswith('>'):
                    sender = f"{name_part} {email_part}"
                else:
                    # Try to get email separately and format properly
                    if sender_email:
                        sender = f"{name_part} <{sender_email}>"
                    else:
                        sender = name_part
            elif '<' not in raw_sender:
                # Name only - try to get sender email separately and format properly
                if sender_email:
                    sender = f"{raw_sender} <{sender_email}>"
                else:
                    # If no email available, keep just the name (this shouldn't happen in normal cases)
                    sender = raw_sender
            else:
                # Already contains < and > - use as is
                sender = raw_sender
        else:
            # Unknown sender - try to get email if available
            if sender_email:
                sender = f"Unknown <{sender_email}>"
            else:
                sender = raw_sender

        subject = msg.subject or "(No Subject)"
        # Apply standardize_date_format to ensure consistent Outlook-style formatting
        raw_date = msg.date or "Unknown"
        date = standardize_date_format(raw_date)

        # Safely extract plain text body with encoding error handling
        body = ""
        try:
            body = msg.body or ""
        except UnicodeDecodeError as e:
            print(f"Unicode decoding error when extracting plain text body: {e}")
            # Try alternative encoding approaches for plain text
            try:
                # Try to access raw body data with different encodings
                print("Attempting to extract plain text body with encoding fallbacks...")
                body = ""  # Fallback to empty string if all methods fail
            except Exception as fallback_error:
                print(f"Fallback plain text extraction also failed: {fallback_error}")
                body = ""
        except Exception as e:
            print(f"Unexpected error when extracting plain text body: {e}")
            body = ""

        # Safely extract HTML body with encoding error handling
        html_body = None
        encoding_issues_detected = False
        try:
            html_body = getattr(msg, 'htmlBody', None)
        except UnicodeDecodeError as e:
            encoding_issues_detected = True
            print(f"Unicode decoding error when extracting HTML body: {e}")
            print("This is likely due to non-standard character encoding in the email.")
            # Try alternative methods to extract HTML content
            try:
                # Try to get RTF body and convert it manually
                rtf_body = getattr(msg, 'rtfBody', None)
                if rtf_body:
                    print("Attempting to extract HTML from RTF body with encoding fallbacks...")
                    # Try different encoding approaches
                    for encoding in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
                        try:
                            # This is a simplified approach - in practice, RTF to HTML conversion is complex
                            # For now, we'll fall back to plain text body
                            print(f"RTF body available but HTML extraction failed, using plain text body")
                            html_body = None
                            break
                        except Exception:
                            continue
                else:
                    print("No RTF body available, using plain text body")
            except Exception as fallback_error:
                print(f"Fallback HTML extraction also failed: {fallback_error}")
                html_body = None
        except Exception as e:
            print(f"Unexpected error when extracting HTML body: {e}")
            html_body = None

        # Extract To: and Cc: recipients using the recipients array for better accuracy
        to_recipients = []
        cc_recipients = []

        # Method 1: Use recipients array (most reliable)
        if hasattr(msg, 'recipients') and msg.recipients:
            for recipient in msg.recipients:
                try:
                    recipient_type = getattr(recipient, 'type', None)
                    email = getattr(recipient, 'email', None)
                    name = getattr(recipient, 'name', None)

                    if email:
                        # Format as "Name <email>" if name exists and is different from email
                        if name and name != email and not email in name:
                            formatted_recipient = f"{name} <{email}>"
                        else:
                            formatted_recipient = email

                        # Type 1 = To, Type 2 = Cc, Type 3 = Bcc
                        if recipient_type == 1:  # To
                            to_recipients.append(formatted_recipient)
                        elif recipient_type == 2:  # Cc
                            cc_recipients.append(formatted_recipient)
                except Exception as e:
                    print(f"Warning: Error processing recipient: {e}")
                    continue

        # Method 2: Fallback to string parsing if recipients array failed
        if not to_recipients and hasattr(msg, 'to') and msg.to:
            to_str = str(msg.to).strip()
            if to_str:
                # Clean up and ensure proper formatting
                to_str = to_str.strip('"').strip()  # Remove quotes if present
                to_recipients = [to_str]

        if not cc_recipients and hasattr(msg, 'cc') and msg.cc:
            cc_str = str(msg.cc).strip()
            if cc_str:
                # Parse Cc string - split by semicolon and clean up
                cc_parts = []
                for part in cc_str.split(';'):
                    part = part.strip().strip('"').strip()  # Remove quotes and whitespace
                    if part:
                        cc_parts.append(part)
                cc_recipients = cc_parts

        # Preserve complete HTML content including images and complex structures
        preserved_html_body = html_body
        if html_body:
            # Extract and preserve embedded images and attachments
            try:
                # Get all attachments from the message with encoding error handling
                attachments = []
                if hasattr(msg, 'attachments') and msg.attachments:
                    for attachment in msg.attachments:
                        try:
                            if hasattr(attachment, 'data') and hasattr(attachment, 'longFilename'):
                                # Safely extract attachment properties
                                filename = None
                                try:
                                    filename = attachment.longFilename or attachment.shortFilename
                                except UnicodeDecodeError as e:
                                    print(f"Unicode error extracting attachment filename: {e}")
                                    filename = "attachment_with_encoding_issue"
                                except Exception as e:
                                    print(f"Error extracting attachment filename: {e}")
                                    filename = "unknown_attachment"

                                # Safely extract content ID
                                content_id = None
                                try:
                                    content_id = getattr(attachment, 'contentId', None)
                                except Exception as e:
                                    print(f"Error extracting attachment content ID: {e}")

                                attachments.append({
                                    'filename': filename,
                                    'data': attachment.data,
                                    'content_id': content_id
                                })
                        except Exception as e:
                            print(f"Warning: Could not process attachment: {e}")
                            continue

                # Process HTML to preserve embedded images
                if attachments:
                    # Use configurable parser selection with performance tracking
                    soup, parser_used, parse_time, error = create_soup_with_parser(
                        html_body, current_parser_preference, "MSG embedded images processing"
                    )

                    if error or soup is None:
                        print(f"HTML parsing failed in MSG processing: {error}")
                        # Skip image processing if parsing fails
                        preserved_html_body = html_body
                    else:
                        # Find all img tags with cid: references
                        for img in soup.find_all('img'):
                            src = img.get('src', '')
                            if src.startswith('cid:'):
                                content_id = src.replace('cid:', '')
                                # Find matching attachment
                                for attachment in attachments:
                                    if attachment['content_id'] == content_id:
                                        # Convert to base64 data URL
                                        import base64
                                        file_ext = attachment['filename'].split('.')[-1].lower()
                                        mime_type = {
                                            'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                                            'png': 'image/png', 'gif': 'image/gif',
                                            'bmp': 'image/bmp', 'svg': 'image/svg+xml'
                                        }.get(file_ext, 'image/jpeg')

                                        base64_data = base64.b64encode(attachment['data']).decode('utf-8')
                                        data_url = f"data:{mime_type};base64,{base64_data}"
                                        img['src'] = data_url
                                        break

                        preserved_html_body = str(soup)

            except Exception as e:
                print(f"Warning: Could not process embedded images: {e}")
                preserved_html_body = html_body

        # Store both text and HTML versions with complete content preservation
        result = {
            "sender": sender,
            "subject": subject,
            "date": date,
            "body": html_to_text(preserved_html_body) if preserved_html_body else body,
            "html_body": preserved_html_body,
            "original_html_body": html_body,  # Keep original for reference
            "to_recipients": to_recipients,
            "cc_recipients": cc_recipients,
            "attachments": attachments if 'attachments' in locals() else [],
            "encoding_issues": encoding_issues_detected  # Flag for UI to show warning
        }
        
        msg.close()
        os.remove(temp_path)
        return result, None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Exception in process_msg_file: {e}\n{tb}")
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass
        return None, f"Failed to process .msg file: {e}\n{tb}"

def create_upload_panel():
    """Create the full-width upload panel section with expanded interactive area"""
    with gr.Group(elem_classes=["full-width-upload-panel"], visible=True) as upload_panel:
        # Single file input that spans the entire panel area
        file_input = gr.File(
            label="📧 Drag and drop your email here, or click to upload",
            file_types=[".msg"],
            elem_classes=["full-width-file-input"],
            height=560,  # Doubled height for much more prominent upload area
            show_label=True
        )

    return upload_panel, file_input

def create_status_section():
    """Create the workflow status banner section with separate clickable components"""

    def get_stage_html(stage_num, title, icon, description, is_active=False, is_clickable=False, is_disabled=False):
        """Generate HTML for a single stage with support for disabled state"""
        stage_classes = ["stage"]

        if is_active:
            stage_classes.append("active")
        elif is_disabled:
            stage_classes.append("disabled")
        elif is_clickable:
            stage_classes.append("clickable")

        stage_class = " ".join(stage_classes)

        # Only add interactive attributes if clickable and not disabled
        attrs = ""
        if is_clickable and not is_disabled:
            attrs = f'role="button" tabindex="0" aria-label="Go back to {title} stage"'
        elif is_disabled:
            attrs = f'aria-disabled="true" aria-label="{title} stage - Complete previous stages to unlock"'

        return f"""
        <div class="{stage_class}" {attrs}>
            <div class="stage-header">
                <div class="stage-number">{stage_num}</div>
                <div class="stage-icon">{icon}</div>
                <h3 class="stage-title">{title}</h3>
            </div>
            <div class="stage-description">
                {description}
            </div>
        </div>
        """

    # Create the banner container with separate stage components arranged horizontally
    with gr.Group(elem_id="status-instructions", elem_classes=["borderless-group"]) as status_banner:
        gr.HTML("""<div class="status-instructions-panel"><div class="workflow-stages">""")

        # Arrange the three stage buttons horizontally in a single row
        with gr.Row():
            # Stage 1 - Upload Email
            stage1_html = gr.HTML(
                value=get_stage_html(1, "Upload Email", "📧", "Upload your email to begin", is_active=True),
                elem_id="stage1-banner"
            )

            # Stage 2 - Add Key Messages
            stage2_html = gr.HTML(
                value=get_stage_html(2, "Add Key Messages", "✍️", "Add key points for your reply"),
                elem_id="stage2-banner"
            )

            # Stage 3 - Review & Revise
            stage3_html = gr.HTML(
                value=get_stage_html(3, "Review & Revise", "📋", "Review, revise, and download"),
                elem_id="stage3-banner"
            )

        gr.HTML("""</div></div>""")

    def update_stage_banners(active_stage, unlocked_stages=None):
        """Update all stage banners based on current active stage and unlocked stages"""
        # Default unlocked stages if not provided (for backward compatibility)
        if unlocked_stages is None:
            unlocked_stages = [1]  # Only Stage 1 unlocked by default

        # Determine clickability and disabled state based on workflow progression
        stage1_clickable = active_stage > 1 and 1 in unlocked_stages
        stage1_disabled = 1 not in unlocked_stages

        stage2_clickable = active_stage > 2 and 2 in unlocked_stages
        stage2_disabled = 2 not in unlocked_stages

        stage3_clickable = False  # Stage 3 is never clickable (end of workflow)
        stage3_disabled = 3 not in unlocked_stages

        # Generate updated HTML for each stage
        stage1_update = gr.update(value=get_stage_html(
            1, "Upload Email", "📧", "Upload your email to begin",
            is_active=(active_stage == 1),
            is_clickable=stage1_clickable,
            is_disabled=stage1_disabled
        ))

        stage2_update = gr.update(value=get_stage_html(
            2, "Add Key Messages", "✍️", "Add key points for your reply",
            is_active=(active_stage == 2),
            is_clickable=stage2_clickable,
            is_disabled=stage2_disabled
        ))

        stage3_update = gr.update(value=get_stage_html(
            3, "Review & Revise", "📋", "Review, revise, and download",
            is_active=(active_stage == 3),
            is_clickable=stage3_clickable,
            is_disabled=stage3_disabled
        ))

        return stage1_update, stage2_update, stage3_update

    def get_banner_updates_for_stage(stage, unlocked_stages=None):
        """Helper function to get banner updates for a specific stage"""
        return update_stage_banners(stage, unlocked_stages)

    # Compatibility function for existing code that expects single banner HTML
    def get_workflow_banner_html(active_stage=1, unlocked_stages=None):
        """Compatibility function - returns banner updates for the new system"""
        return update_stage_banners(active_stage, unlocked_stages)

    return status_banner, stage1_html, stage2_html, stage3_html, update_stage_banners, get_banner_updates_for_stage, get_workflow_banner_html

def create_left_column():
    """Create the left column components for email preview, thinking process, and draft response sections"""
    # SARA Thinking Process Accordion - Moved between Upload and Draft Response
    with gr.Accordion("💭 SARA Thinking Process", open=False, visible=False) as think_accordion:
        think_output = gr.Markdown(value="")

    # Draft Email Preview Container - Non-collapsible without header
    with gr.Group(visible=False) as thread_preview_accordion:
        thread_preview = gr.HTML(
            value="""
            <div class='thread-placeholder'>
                <div class='placeholder-content'>
                    <div class='placeholder-icon'>📧</div>
                    <h3>Draft Email Preview Will Appear Here</h3>
                    <p>Upload an email and generate a response to see your draft email exactly as you'll download it</p>
                </div>
            </div>
            """,
            elem_classes=["thread-display-area", "email-panel-container"]
        )

    # Uploaded Email Display Container - Non-collapsible without header
    with gr.Group(visible=False) as original_reference_accordion:
        original_reference_display = gr.HTML(
            value="""
            <div class='email-placeholder'>
                <div class='placeholder-content'>
                    <div class='placeholder-icon'>📧</div>
                    <h3>Upload Email File Above</h3>
                    <p>Select your .msg email file to view the original email content</p>
                </div>
            </div>
            """,
            elem_classes=["original-reference-display-area", "email-panel-container"]
        )

    return {
        'think_accordion': think_accordion,
        'think_output': think_output,
        'thread_preview_accordion': thread_preview_accordion,
        'thread_preview': thread_preview,
        'original_reference_accordion': original_reference_accordion,
        'original_reference_display': original_reference_display
    }

def create_sidebar():
    """Create the left sidebar with description, preferences, and information sections"""

    # Description Header Section
    gr.HTML("""
    <div class="description-box">
        <p class="description-text">
            <strong>SARA Compose</strong> is an email drafting assistant developed by <strong>RD</strong>, designed to handle confidential materials and protect user privacy.
        </p>
    </div>
    """)

    # Personal Preferences Section
    with gr.Accordion("👤 Personal Preferences", open=False):
        with gr.Group():
            gr.HTML("<h4 class='sidebar-section-header'>User Identity</h4>")
            user_name = gr.Textbox(
                label="Your Name",
                placeholder="Enter your full name...",
                value="Max Kwong",
                interactive=True,
                elem_id="user-name-input"
            )
            user_email = gr.Textbox(
                label="Your Email Address",
                placeholder="Enter your email address...",
                value="mwmkwong@hkma.gov.hk",
                interactive=True,
                elem_id="user-email-input"
            )

        with gr.Group():
            gr.HTML("<h4 class='sidebar-section-header with-top-margin'>AI Instructions</h4>")

            ai_instructions = gr.Textbox(
                label="AI Instructions",
                value=DEFAULT_AI_INSTRUCTIONS,
                placeholder="Enter the complete instructions that will be sent to the AI model...",
                lines=12,
                max_lines=20,
                interactive=True,
                info="These are the exact instructions sent to the AI model. You have complete control over how the AI behaves. Only your identity context is added automatically.",
                elem_id="ai-instructions-textarea"
            )

            # Restore Default Instructions button
            restore_default_btn = gr.Button(
                "🔄 Restore Default Instructions",
                size="sm",
                variant="primary"
            )

            # Hidden HTML component for visual feedback
            restore_feedback = gr.HTML(visible=False)

    # Disclaimer Section
    with gr.Accordion("⚠️ Disclaimer", open=False):
        gr.HTML("""
        <div style="padding: 16px;">
            <p class="disclaimer-text">
                Please be advised that all responses generated by SARA are provided in good faith and designed solely for the purpose of offering general information. While we strive for accuracy, SARA does not guarantee or warrant the completeness, reliability, or precision of the information provided. It is strongly recommended that users independently verify the information generated by SARA before utilizing it in any further capacity. Any actions or decisions made based on SARA's responses are undertaken entirely at the user's own risk.
            </p>
        </div>
        """)

    # Support & Feedback Section
    with gr.Accordion("📞 Support & Feedback", open=False):
        gr.HTML("""
        <div style="padding: 16px;">
            <p class="contact-text">
                If you have any comments or need assistance regarding the tool, please don't hesitate to contact us:
            </p>
            <table class="info-table" style="width: 100%; font-size: 0.85rem;">
                <thead>
                    <tr>
                        <th>Name</th>
                        <th>Post</th>
                        <th>Extension</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>Max Kwong</td>
                        <td>SM(RD)</td>
                        <td>1673</td>
                    </tr>
                    <tr>
                        <td>Oscar So</td>
                        <td>M(RD)1</td>
                        <td>0858</td>
                    </tr>
                    <tr>
                        <td>Maggie Poon</td>
                        <td>AM(RD)1</td>
                        <td>0746</td>
                    </tr>
                    <tr>
                        <td>Cynwell Lau</td>
                        <td>AM(RD)3</td>
                        <td>0460</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """)

    # Changelog Section with corrected content
    with gr.Accordion("📌 Changelog", open=False):
        gr.HTML("""
        <div style="padding: 16px;">
            <table class="info-table" style="width: 100%;">
                <thead>
                    <tr>
                        <th>Date</th>
                        <th>Version</th>
                        <th>Notes</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>2025-07-07</td>
                        <td>1.0</td>
                        <td>• First release</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """)

    # Development Settings Section
    with gr.Accordion("⚙️ Development Settings", open=False):
        with gr.Group():
            gr.HTML("<h4 class='sidebar-section-header'>AI Model Configuration</h4>")

            # Model selection - POE models only
            model_selector = gr.Dropdown(
                label="AI Model",
                choices=POE_MODELS,
                value="GPT-4o",  # Default to GPT-4o
                interactive=True,
                info="Select the AI model for email generation (powered by POE API)"
            )

            email_token_limit = gr.Number(
                label="Email Content Limit (tokens)",
                value=2000,
                minimum=100,
                maximum=32000,
                step=100,
                interactive=True,
                info="Maximum tokens for email content sent to AI (longer emails will be truncated)"
            )

            gr.HTML("<h4 class='sidebar-section-header with-top-margin'>Email Parser Configuration</h4>")

            # HTML Parser selection
            parser_selector = gr.Dropdown(
                label="HTML Parser",
                choices=HTML_PARSER_OPTIONS,
                value=DEFAULT_PARSER_CHOICE,
                interactive=True,
                info="Select HTML parser for email processing. Auto mode tries lxml first, then falls back to html.parser."
            )

            # Parser performance display
            parser_info = gr.HTML(
                value="<div class='parser-info'>Parser performance information will appear here after processing emails.</div>",
                visible=True
            )

    return {
        'user_name': user_name,
        'user_email': user_email,
        'ai_instructions': ai_instructions,
        'restore_default_btn': restore_default_btn,
        'restore_feedback': restore_feedback,
        'model_selector': model_selector,
        'email_token_limit': email_token_limit,
        'parser_selector': parser_selector,
        'parser_info': parser_info
    }

def create_right_column():
    """Create the right column components for controls, preferences, and development settings"""
    # Key Messages Input - Using built-in label feature
    key_messages = gr.Textbox(
        label="📝 Key Messages",
        placeholder="Enter the key messages you want to include in your reply...\n\nExample:\n• Thank them for their inquiry\n• Confirm the meeting time\n• Provide additional resources",
        lines=21.5,
        max_lines=30,
        show_label=True,
        visible=False
    )

    # Action buttons outside accordion for cleaner visual hierarchy
    with gr.Row():
        generate_btn = gr.Button(
            "🚀 Generate Reply",
            interactive=True,
            size="lg",
            variant="primary",
            visible=False,  # Hidden until key messages accordion is visible
            elem_classes=["action-button"]
        )

        download_button = gr.DownloadButton(
            label="📥 Download Draft",
            visible=False,
            variant="primary",
            size="lg",
            elem_classes=["download-button"]
        )

    return {
        'key_messages': key_messages,
        'generate_btn': generate_btn,
        'download_button': download_button
    }

with gr.Blocks(theme=hkma_theme, css=custom_css, title="SARA Compose") as demo:
    # Create status section components
    status_banner, stage1_html, stage2_html, stage3_html, update_stage_banners, get_banner_updates_for_stage, get_workflow_banner_html = create_status_section()

    # Simplified localStorage persistence using Gradio BrowserState
    # This is more reliable than complex JavaScript
    preferences_state = gr.BrowserState({
        "user_name": "Max Kwong",
        "user_email": "mwmkwong@hkma.gov.hk",
        "ai_instructions": DEFAULT_AI_INSTRUCTIONS
    })







    # Status instructions panel - spans full width above main content
    status_banner

    # Full-width upload panel - spans full width below status banner
    upload_panel, file_input = create_upload_panel()

    with gr.Row():
        # Left sidebar - Information sections, preferences, and settings
        with gr.Sidebar(position="left", width=400, open=True):
            sidebar_components = create_sidebar()
            user_name = sidebar_components['user_name']
            user_email = sidebar_components['user_email']
            ai_instructions = sidebar_components['ai_instructions']
            restore_default_btn = sidebar_components['restore_default_btn']
            restore_feedback = sidebar_components['restore_feedback']
            model_selector = sidebar_components['model_selector']
            email_token_limit = sidebar_components['email_token_limit']
            parser_selector = sidebar_components['parser_selector']
            parser_info = sidebar_components['parser_info']

        # Middle column - Email preview and draft display
        with gr.Column(scale=2):
            left_components = create_left_column()
            think_accordion = left_components['think_accordion']
            think_output = left_components['think_output']
            thread_preview_accordion = left_components['thread_preview_accordion']
            thread_preview = left_components['thread_preview']
            original_reference_accordion = left_components['original_reference_accordion']
            original_reference_display = left_components['original_reference_display']

        # Right column - Key messages and generation controls
        with gr.Column(scale=1):
            right_components = create_right_column()
            key_messages = right_components['key_messages']
            generate_btn = right_components['generate_btn']
            download_button = right_components['download_button']





    # State management
    current_reply = gr.State("")
    current_think = gr.State("")
    current_email_info = gr.State({})
    current_stage = gr.State(1)  # Track current workflow stage (1, 2, or 3)
    unlocked_stages = gr.State([1])  # Track which stages are unlocked (starts with only Stage 1)

    # Multi-turn conversation state
    conversation_history = gr.State([])
    is_revision_mode = gr.State(False)
    initial_key_messages = gr.State("")





    # Function to restore default AI instructions
    def restore_default_instructions():
        """Restore AI instructions to default values"""
        # Simplified for Hugging Face Spaces deployment
        return [
            gr.update(value=DEFAULT_AI_INSTRUCTIONS),
            "<div style='color: green; font-size: 0.9rem; margin-top: 8px;'>✅ Default instructions restored!</div>"
        ]

    # Function to save custom instructions when text area changes (simplified)
    def save_custom_instructions_on_change(instructions):
        """Save custom instructions - simplified for Hugging Face Spaces"""
        # Return empty string to avoid JavaScript issues
        _ = instructions  # Acknowledge parameter to avoid warning
        return ""

    # Preferences persistence functions using BrowserState
    def save_user_name(name, prefs_state):
        """Save user name to preferences state"""
        prefs = prefs_state.copy()
        prefs["user_name"] = name
        return prefs

    def save_user_email(email, prefs_state):
        """Save user email to preferences state"""
        prefs = prefs_state.copy()
        prefs["user_email"] = email
        return prefs

    def save_ai_instructions(instructions, prefs_state):
        """Save AI instructions to preferences state"""
        prefs = prefs_state.copy()
        prefs["ai_instructions"] = instructions
        return prefs





    def on_model_change(model):
        """Handle model selection change with validation and fallback"""
        try:
            # Validate the selected model for POE
            if not validate_model_selection(model):
                print(f"Model {model} is not available for POE")

                # Get fallback model
                fallback_model = get_fallback_model(model)
                if fallback_model:
                    print(f"Falling back to {fallback_model}")
                    return gr.update(
                        value=fallback_model,
                        info=f"Model {model} unavailable - switched to {fallback_model}"
                    )
                else:
                    print(f"No fallback model available for POE")
                    return gr.update(
                        info=f"Model {model} is not available"
                    )
            else:
                # Model is valid
                return gr.update(
                    info=f"Model {model} is ready for use (powered by POE API)"
                )
        except Exception as e:
            print(f"Error validating model selection: {e}")
            return gr.update(
                info="Error validating model selection"
            )

    def on_parser_change(parser_choice):
        """Handle HTML parser selection change"""
        global current_parser_preference
        try:
            current_parser_preference = parser_choice
            print(f"Parser preference changed to: {parser_choice}")

            # Test the parser selection with a simple HTML snippet
            test_html = "<p>Test HTML parsing</p>"
            soup, parser_used, parse_time, error = create_soup_with_parser(
                test_html, parser_choice, "parser_test"
            )

            if error:
                return gr.update(
                    value=f"<div class='parser-info error'>❌ {error}</div>"
                )
            else:
                performance_info = get_parser_performance_info(parser_used, parse_time)
                return gr.update(
                    value=f"<div class='parser-info success'>{performance_info} - Parser ready for use</div>"
                )
        except Exception as e:
            print(f"Error changing parser: {e}")
            return gr.update(
                value=f"<div class='parser-info error'>❌ Error: {str(e)}</div>"
            )

    def get_backend_health_info():
        """Get POE backend health information for UI display"""
        status = backend_manager.get_backend_status()

        poe_info = status['poe']
        status_icon = "✅" if poe_info['healthy'] else "❌"
        model_count = len(poe_info['models'])

        return f"POE: {status_icon} ({model_count} models)"

    def load_preferences_on_startup(prefs_state):
        """Load saved preferences on application startup"""
        prefs = prefs_state
        return [
            gr.update(value=prefs.get("user_name", "Max Kwong")),
            gr.update(value=prefs.get("user_email", "mwmkwong@hkma.gov.hk")),
            gr.update(value=prefs.get("ai_instructions", DEFAULT_AI_INSTRUCTIONS))
        ]



    def extract_and_display_email(file):
        if not file:
            # Reset status banners to initial state (Stage 1) - only Stage 1 unlocked
            stage1_update, stage2_update, stage3_update = update_stage_banners(1, [1])

            return (
                gr.update(visible=True),  # Keep upload panel visible
                """
                <div class='email-placeholder'>
                    <div class='placeholder-content'>
                        <div class='placeholder-icon'>📧</div>
                        <h3>Upload Email File Above</h3>
                        <p>Select your .msg email file to view the original email content</p>
                        <div class='placeholder-hint'>Supported format: .msg files</div>
                    </div>
                </div>
                """,
                gr.update(visible=False),  # Keep original reference group hidden
                                    gr.update(visible=False),  # Hide key messages container when no file
                {},
                stage1_update,    # Update stage 1 banner
                stage2_update,    # Update stage 2 banner
                stage3_update,    # Update stage 3 banner
                gr.update(visible=False),  # Hide generate button when no file
                1,  # Reset to Stage 1
                [1]  # Only Stage 1 unlocked
            )

        info, error = process_msg_file(file)
        if error:
            # Error status - stay on Stage 1, only Stage 1 unlocked
            stage1_update, stage2_update, stage3_update = update_stage_banners(1, [1])

            return (
                gr.update(visible=True),  # Keep upload panel visible for retry
                """
                <div class='email-placeholder'>
                    <div class='placeholder-content'>
                        <div class='placeholder-icon'>❌</div>
                        <h3>Error Processing Email</h3>
                        <p>There was an error processing the uploaded email file</p>
                        <div class='placeholder-hint'>Please try uploading a different .msg file</div>
                    </div>
                </div>
                """,
                gr.update(visible=False),  # Keep original reference group hidden
                                    gr.update(visible=False),  # Hide key messages container on error
                {},
                stage1_update,      # Update stage 1 banner
                stage2_update,      # Update stage 2 banner
                stage3_update,      # Update stage 3 banner
                gr.update(visible=False),  # Hide generate button on error
                1,  # Stay on Stage 1
                [1]  # Only Stage 1 unlocked
            )

        # Success status - move to Stage 2, unlock Stages 1 and 2
        stage1_update, stage2_update, stage3_update = update_stage_banners(2, [1, 2])

        preview_html = format_email_preview(info)
        # After successful email upload: display email content and auto-expand email panel for immediate preview
        return (
            gr.update(visible=False),   # Hide upload panel after successful upload to reduce clutter at Stage 2
            preview_html,               # Original email display
            gr.update(visible=True),    # Make original reference group visible for immediate preview
            gr.update(visible=True),    # Make key messages container visible after successful upload
            info,                       # Current email info state
            stage1_update,              # Update stage 1 banner
            stage2_update,              # Update stage 2 banner
            stage3_update,              # Update stage 3 banner
            gr.update(visible=True),    # Show generate button when key messages accordion is visible
            2,  # Move to Stage 2
            [1, 2]  # Stages 1 and 2 unlocked
        )



    def generate_download_file(reply_text, email_info, user_email="", user_name=""):
        """Generate downloadable .eml file automatically when reply is complete"""
        try:
            if not reply_text or not reply_text.strip():
                return gr.update(visible=False, value=None)

            file_path, error = export_reply_to_msg(reply_text, email_info, user_email, user_name)

            if error:
                print(f"Auto-export error: {error}")
                return gr.update(visible=False, value=None)

            if file_path and os.path.exists(file_path):
                print(f"Auto-export successful: {file_path}")
                # For DownloadButton, we need to return the file path as the value
                return gr.update(visible=True, value=file_path)
            else:
                print("Auto-export failed: No file created")
                return gr.update(visible=False, value=None)

        except Exception as e:
            print(f"Auto-export error: {str(e)}")
            return gr.update(visible=False, value=None)

    def handle_download_click(reply_text, email_info, user_email="", user_name=""):
        """Handle download button click - generate and return file for download"""
        try:
            if not reply_text or not reply_text.strip():
                return None

            file_path, error = export_reply_to_msg(reply_text, email_info, user_email, user_name)

            if error:
                print(f"Download error: {error}")
                return None

            if file_path and os.path.exists(file_path):
                print(f"Download successful: {file_path}")
                return file_path
            else:
                print("Download failed: No file created")
                return None

        except Exception as e:
            print(f"Download error: {str(e)}")
            return None

    def copy_to_clipboard_js(reply_text):
        """Simplified for Hugging Face Spaces deployment"""
        # Return empty string to avoid JavaScript issues
        _ = reply_text # Acknowledge parameter to avoid warning
        return ""


    def validate_inputs(file, key_msgs, model):
        """Validate inputs including model availability"""
        if not file or not key_msgs or not model:
            return gr.update(interactive=False)

        # Additional validation: check if model is available for POE
        if not validate_model_selection(model):
            print(f"Model {model} not available for POE backend")
            return gr.update(interactive=False)

        return gr.update(interactive=True)

    def validate_revision_inputs(file, key_msgs, model, is_revision_mode):
        """Validate inputs specifically for revision mode - button should be disabled when revision text is empty"""
        if not file or not model:
            return gr.update(interactive=False)

        # In revision mode, key_msgs should not be empty (revision instructions required)
        # In initial mode, use the standard validation
        if is_revision_mode:
            if not key_msgs or not key_msgs.strip():
                return gr.update(interactive=False)
        else:
            if not key_msgs:
                return gr.update(interactive=False)

        # Additional validation: check if model is available for POE
        if not validate_model_selection(model):
            print(f"Model {model} not available for POE backend")
            return gr.update(interactive=False)

        return gr.update(interactive=True)



    def update_ui_for_revision_mode(is_revision):
        """Update UI components based on revision mode state"""
        if is_revision:
            # Revision mode: change label and button text
            label_text = "✏️ Revision Instructions"
            button_text = "🔄 Revise Draft"
            placeholder_text = "Enter your revision instructions...\n\nExample:\n• Make the tone more formal\n• Add information about next steps\n• Shorten the response"
        else:
            # Initial mode: original key messages setup
            label_text = "📝 Key Messages"
            button_text = "🚀 Generate Reply"
            placeholder_text = "Enter the key messages you want to include in your reply...\n\nExample:\n• Thank them for their inquiry\n• Confirm the meeting time\n• Provide additional resources"

        return label_text, button_text, placeholder_text

    def build_conversation_history(email_info, conversation_history, new_user_input, user_name="", ai_instructions="", email_token_limit=2000):
        """Build conversation history for multi-turn interactions"""

        # If this is the first turn, create initial conversation
        if not conversation_history:
            # Build system message with email context
            recipient_info = ""
            to_recipients = email_info.get('to_recipients', [])
            cc_recipients = email_info.get('cc_recipients', [])

            if to_recipients:
                recipient_info += f"To: {', '.join(to_recipients)}\n"
            if cc_recipients:
                recipient_info += f"Cc: {', '.join(cc_recipients)}\n"

            # Truncate email content if needed
            email_body = truncate_email_content(email_info['body'], email_token_limit)

            # User identity context
            user_identity_context = ""
            if user_name:
                user_identity_context = f"You are responding as: {user_name}\n\n"

            # System message with email context and instructions
            system_message = f"""{user_identity_context}{ai_instructions}

Original Email:
From: {email_info['sender']}
{recipient_info}Subject: {email_info['subject']}
Date: {email_info['date']}

{email_body}"""

            return [
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Key Messages to Include:\n{new_user_input}"}
            ]
        else:
            # Add new user message to existing conversation
            updated_history = conversation_history.copy()
            updated_history.append({"role": "user", "content": f"Please revise the draft with these instructions:\n{new_user_input}"})
            return updated_history

    def build_prompt(email_info, key_msgs, user_name="", ai_instructions="", email_token_limit=2000):
        """Build the prompt for the LLM with transparent instructions and fallback protection"""

        # Always use provided instructions with fallback protection
        final_instructions = validate_and_restore_ai_instructions(ai_instructions)

        # Build recipient information for context
        recipient_info = ""
        to_recipients = email_info.get('to_recipients', [])
        cc_recipients = email_info.get('cc_recipients', [])

        if to_recipients:
            recipient_info += f"To: {', '.join(to_recipients)}\n"
        if cc_recipients:
            recipient_info += f"Cc: {', '.join(cc_recipients)}\n"

        # Truncate email content if needed
        email_body = truncate_email_content(email_info['body'], email_token_limit)

        # Only add user identity context automatically (hidden from user)
        user_identity_context = ""
        if user_name:
            user_identity_context = f"You are responding as: {user_name}\n\n"

        # Build the transparent prompt with selected instructions
        prompt = f"""{user_identity_context}{final_instructions}

Original Email:
From: {email_info['sender']}
{recipient_info}Subject: {email_info['subject']}
Date: {email_info['date']}

{email_body}

Key Messages to Include:
{key_msgs}"""

        return prompt

    # AI generation worker function is now handled in backend.py module

    def on_generate_stream(file, key_msgs, model, user_name, user_email, ai_instructions, email_token_limit, conversation_history, is_revision_mode, initial_key_messages):
        try:
            print(f"on_generate_stream called with file: {type(file)} {file}")
            if not file:
                # No file - back to Stage 1
                stage1_update, stage2_update, stage3_update = get_workflow_banner_html(1)

                yield (
                    gr.update(visible=True),  # Show upload panel when no file
                    """
                    <div class='thread-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>❌</div>
                            <h3>No File Uploaded</h3>
                            <p>Please upload an email file first</p>
                        </div>
                    </div>
                    """,
                    """
                    <div class='email-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>📧</div>
                            <h3>Upload Email File Above</h3>
                            <p>Select your .msg email file to view the original email content</p>
                        </div>
                    </div>
                    """,
                    gr.update(visible=False),  # Hide thinking accordion
                    gr.update(visible=False),  # Hide thread preview group
                    gr.update(visible=False),  # Hide original reference group
                    gr.update(visible=False),  # Hide key messages container
                    "",  # Clear thinking content
                    gr.update(visible=False, value=None),  # Hide download file
                    "",  # Clear current_reply state
                    "",  # Clear current_think state
                    stage1_update,  # Update stage 1 banner
                    stage2_update,  # Update stage 2 banner
                    stage3_update,  # Update stage 3 banner
                    gr.update(interactive=True, value="🚀 Generate Reply"),  # Re-enable button
                    [],  # Clear conversation history
                    False,  # Reset revision mode
                    "",  # Clear initial key messages
                    1,  # Back to Stage 1
                    [1]  # Only Stage 1 unlocked
                )
                return
            # Check if any backend is healthy (with automatic fallback)
            if not backend_manager.is_any_backend_healthy():
                # No APIs available - stay on Stage 2
                stage1_update, stage2_update, stage3_update = get_workflow_banner_html(2)

                # Get backend status for detailed error message
                backend_status = backend_manager.get_backend_status()
                poe_status = "✅ Healthy" if backend_status['poe']['healthy'] else "❌ Unavailable"

                yield (
                    gr.update(visible=True),   # Show upload panel for retry
                    f"""
                    <div class='thread-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>❌</div>
                            <h3>POE AI Backend Unavailable</h3>
                            <p><strong>Backend Status:</strong></p>
                            <ul style='text-align: left; margin: 10px 0;'>
                                <li>POE API: {poe_status}</li>
                            </ul>
                            <p>Please check your POE API key and try again.</p>
                        </div>
                    </div>
                    """,
                    format_email_preview({}),
                    gr.update(visible=False),  # Hide thinking accordion
                    gr.update(visible=False),  # Hide thread preview group
                    gr.update(visible=False),  # Hide original reference group
                    gr.update(visible=False),  # Hide key messages container
                    "",  # Clear thinking content
                    gr.update(visible=False, value=None),  # Hide download file
                    "",  # Clear current_reply state
                    "",  # Clear current_think state
                    stage1_update,  # Update stage 1 banner
                    stage2_update,  # Update stage 2 banner
                    stage3_update,  # Update stage 3 banner
                    gr.update(interactive=True, value="🚀 Generate Reply"),  # Re-enable button
                    [],  # Clear conversation history
                    False,  # Reset revision mode
                    "",  # Clear initial key messages
                    2,  # Stay on Stage 2
                    [1, 2]  # Stages 1 and 2 unlocked (error in Stage 2)
                )
                return

            info, error = process_msg_file(file)
            print(f"process_msg_file returned info: {info}, error: {error}")
            if error:
                # Processing error - back to Stage 1
                stage1_update, stage2_update, stage3_update = get_workflow_banner_html(1)

                yield (
                    gr.update(visible=True),   # Show upload panel for retry
                    """
                    <div class='thread-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>❌</div>
                            <h3>Processing Error</h3>
                            <p>Error processing email file</p>
                        </div>
                    </div>
                    """,
                    format_email_preview({}),
                    gr.update(visible=False),  # Hide thinking accordion
                    gr.update(visible=False),  # Hide thread preview group
                    gr.update(visible=False),  # Hide original reference group
                    gr.update(visible=False),  # Hide key messages container
                    "",  # Clear thinking content
                    gr.update(visible=False, value=None),  # Hide download file
                    "",  # Clear current_reply state
                    "",  # Clear current_think state
                    stage1_update,  # Update stage 1 banner
                    stage2_update,  # Update stage 2 banner
                    stage3_update,  # Update stage 3 banner
                    gr.update(interactive=True, value="🚀 Generate Reply"),  # Re-enable button
                    [],  # Clear conversation history
                    False,  # Reset revision mode
                    "",  # Clear initial key messages
                    1,  # Back to Stage 1
                    [1]  # Only Stage 1 unlocked
                )
                return

            # Show original email in bottom section, hide file upload
            original_email_preview = format_email_preview(info)

            # Determine if this is initial generation or revision
            if not conversation_history or len(conversation_history) == 0:
                # Initial generation - build conversation history
                updated_conversation_history = build_conversation_history(
                    info, [], key_msgs, user_name, ai_instructions, email_token_limit
                )
                updated_initial_key_messages = key_msgs
                updated_is_revision_mode = False
                prompt = build_prompt(info, key_msgs, user_name, ai_instructions, email_token_limit)  # Fallback for non-conversation backends
            else:
                # Revision mode - add new user message to conversation
                updated_conversation_history = build_conversation_history(
                    info, conversation_history, key_msgs, user_name, ai_instructions, email_token_limit
                )
                updated_initial_key_messages = initial_key_messages
                updated_is_revision_mode = True
                prompt = ""  # Not used in conversation mode

            # Initialize streaming - open draft accordion and show generation status
            full_response = ""

            # Initial draft area - loading overlay that preserves background content
            initial_draft_status = create_loading_overlay_html(
                "Generating your email response",
                model,
                ""  # No background content initially, will show placeholder
            )

            # Generation status - stay on Stage 2 during generation
            stage1_update, stage2_update, stage3_update = get_workflow_banner_html(2)

            # After Generate Reply: make thread preview accordion visible and open to show streaming content
            # Clear key_messages field if this is a revision submission
            key_messages_update = gr.update(value="") if updated_is_revision_mode else gr.update()

            yield (
                gr.update(visible=False),       # Hide upload panel after successful upload - Stage 2 and beyond
                initial_draft_status,           # Show generation status in thread preview area
                original_email_preview,         # Keep original email visible in reference section
                gr.update(visible=False),       # Hide thinking accordion initially
                gr.update(visible=True),  # Make thread preview group visible to show generation
                gr.update(visible=False),  # Hide original reference group when draft preview becomes available
                key_messages_update,            # Clear key messages field if revision, otherwise keep as is
                "",                             # Clear thinking content initially
                gr.update(visible=False, value=None),  # Hide download file
                "",                             # Clear current_reply state
                "",                             # Clear current_think state
                stage1_update,                  # Update stage 1 banner
                stage2_update,                  # Update stage 2 banner
                stage3_update,                  # Update stage 3 banner
                gr.update(interactive=False, value="⏳ Generating..."),  # Disable button and show generating text
                updated_conversation_history,   # Update conversation history state
                updated_is_revision_mode,       # Update revision mode state
                updated_initial_key_messages,   # Update initial key messages state
                2,                              # Stay on Stage 2 during generation
                [1, 2]                          # Stages 1 and 2 unlocked
            )

            # Initialize result queue for thread-safe communication
            result_queue = queue.Queue()

            # Start AI generation in background thread
            future = AI_THREAD_POOL.submit(
                ai_generation_worker,
                result_queue,
                prompt,
                model,
                updated_conversation_history,
                info,
                key_msgs,
                user_name,
                ai_instructions,
                email_token_limit
            )

            # Non-blocking UI updates with responsive streaming
            full_response = ""
            while True:
                try:
                    # Check for results with timeout to keep UI responsive
                    msg_type, content, is_done = result_queue.get(timeout=0.1)

                    if msg_type == 'chunk':
                        full_response = content

                        if not is_done:
                            # Extract think content and main reply for streaming
                            # Note: No progress update here - users can see real-time streaming content
                            main_reply, think_content = extract_and_separate_think_content(full_response)

                            # During streaming: show real-time content in thread preview - complete email thread
                            if main_reply.strip():
                                # Show streaming thread preview with partial content
                                try:
                                    partial_thread_preview = format_complete_email_thread_preview(
                                        main_reply, info, user_email, user_name
                                    )
                                    draft_content = partial_thread_preview
                                except Exception as e:
                                    print(f"Error creating partial thread preview: {e}")
                                    # Fallback to simple content display
                                    formatted_content = format_reply_content_simple(main_reply)
                                    draft_content = f"""
                                    <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 2px solid #6b21a8;'>
                                        <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.0; color: #374151; font-size: 11pt;'>
                                            {formatted_content}
                                        </div>
                                    </div>
                                    """

                            else:
                                # Still processing - show loading overlay that preserves any existing content
                                draft_content = create_loading_overlay_html(
                                    "Processing your request",
                                    model,
                                    ""  # No background content during processing
                                )

                            # Show/hide think accordion based on content with auto-scroll - ONLY thinking content
                            think_visible = think_content is not None and len(think_content.strip()) > 0
                            think_display = think_content if think_visible else ""

                            # Update status instructions during streaming - stay on Stage 2
                            stage1_update, stage2_update, stage3_update = get_workflow_banner_html(2)

                            # Stream to thread preview area, keep thread accordion open, show thinking if available
                            # Keep key_messages field cleared if this is a revision
                            key_messages_update = gr.update(value="") if updated_is_revision_mode else gr.update()

                            yield (
                                gr.update(visible=False),           # Keep upload panel hidden after successful upload - Stage 2 and beyond
                                draft_content,                      # Show streaming content in thread preview area
                                original_email_preview,             # Keep original email visible in reference section
                                gr.update(visible=think_visible, open=think_visible),  # Show/hide thinking accordion
                                gr.update(visible=True), # Keep thread preview group visible during streaming
                                gr.update(visible=False), # Hide original reference group when draft preview is available
                                key_messages_update,                # Keep key messages field cleared if revision
                                think_display,                      # Show thinking content if available
                                gr.update(visible=False, value=None),  # Hide download file during generation
                                main_reply,                         # Update current_reply state
                                think_content or "",                # Update current_think state
                                stage1_update,                      # Update stage 1 banner
                                stage2_update,                      # Update stage 2 banner
                                stage3_update,                      # Update stage 3 banner
                                gr.update(interactive=False, value="⏳ Generating..."),  # Keep button disabled during streaming
                                updated_conversation_history,       # Update conversation history state
                                updated_is_revision_mode,           # Update revision mode state
                                updated_initial_key_messages,       # Update initial key messages state
                                2,                                  # Stay on Stage 2 during streaming
                                [1, 2]                              # Stages 1 and 2 unlocked
                            )

                        elif is_done:
                            # Final response - show complete thread preview in main section, original email reference available
                            # Note: No progress update here - users can see the final content being displayed
                            main_reply, think_content = extract_and_separate_think_content(full_response)

                            # Format the final complete email thread preview - exactly like download
                            try:
                                final_thread_preview = format_complete_email_thread_preview(
                                    main_reply, info, user_email, user_name
                                )
                                final_draft_content = final_thread_preview
                            except Exception as e:
                                print(f"Error creating final thread preview: {e}")
                                # Fallback to simple content display
                                formatted_content = format_reply_content_simple(main_reply)
                                final_draft_content = f"""
                                <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 2px solid #6b21a8;'>
                                    <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.0; color: #374151; font-size: 11pt;'>
                                        {formatted_content}
                                    </div>
                                </div>
                                """

                            # Show/hide think accordion based on content - automatically collapse after completion
                            think_visible = think_content is not None and len(think_content.strip()) > 0
                            think_display = think_content if think_visible else ""

                            # Completion status instructions - all completion status moved here
                            # Completion status - move to Stage 3, unlock all stages
                            stage1_update, stage2_update, stage3_update = get_workflow_banner_html(3, [1, 2, 3])

                            # Automatically generate download file when generation completes
                            download_file_update = generate_download_file(main_reply, info, user_email, user_name)

                            # Update conversation history with assistant response
                            updated_conversation_history.append({"role": "assistant", "content": main_reply})

                            # Update UI for revision mode after first generation
                            label_text, button_text, _ = update_ui_for_revision_mode(True)

                            # Final state: complete thread preview in main section, original email reference available
                            # Clear key_messages field and update label for revision mode
                            key_messages_final_update = gr.update(visible=True, label=label_text, value="")

                            # Button should be disabled initially in revision mode since key_messages is empty
                            # The validation will be triggered by the key_messages.change event when user types
                            button_update = gr.update(interactive=False, value=button_text)

                            # Note: No completion progress update - users can see the completed response

                            yield (
                                gr.update(visible=False),           # Keep upload panel hidden after successful upload - Stage 3 completion
                                final_draft_content,                # Show final complete thread preview
                                original_email_preview,             # Keep original email visible in reference section
                                gr.update(visible=think_visible, open=False),  # Show thinking accordion but collapsed
                                gr.update(visible=True), # Keep thread preview group visible to show final result
                                gr.update(visible=False), # Hide original reference group when draft preview is complete
                                key_messages_final_update,          # Clear key messages field and update label for revision mode
                                think_display,                      # Show thinking content if available
                                download_file_update,               # Show download file
                                main_reply,                         # Update current_reply state
                                think_content or "",                # Update current_think state
                                stage1_update,                      # Update stage 1 banner
                                stage2_update,                      # Update stage 2 banner
                                stage3_update,                      # Update stage 3 banner
                                button_update,                      # Button disabled initially in revision mode
                                updated_conversation_history,       # Update conversation history state
                                True,                               # Set revision mode to True
                                updated_initial_key_messages,       # Update initial key messages state
                                3,                                  # Move to Stage 3
                                [1, 2, 3]                           # All stages unlocked
                            )
                            return

                    elif msg_type == 'error':
                        # Handle error from worker thread
                        error_draft = f"""
                        <div class='error-content'>
                            <div style='font-size: 1.1em;'>Generation failed: {content}</div>
                        </div>
                        """

                        # Error generation status - stay on Stage 2
                        stage1_update, stage2_update, stage3_update = get_workflow_banner_html(2)

                        yield (
                            gr.update(visible=True),            # Show upload panel for retry
                            error_draft,                        # Show error in thread preview area
                            format_email_preview({}),           # Clear original email area
                            gr.update(visible=False),           # Hide thinking accordion
                            gr.update(visible=False),  # Hide thread preview group
                            gr.update(visible=False),  # Hide original reference group
                            gr.update(visible=False),  # Hide key messages container
                            "",                                 # Clear thinking content
                            gr.update(visible=False, value=None),  # Hide download file
                            "",                                 # Clear current_reply state
                            "",                                 # Clear current_think state
                            stage1_update,                      # Update stage 1 banner
                            stage2_update,                      # Update stage 2 banner
                            stage3_update,                      # Update stage 3 banner
                            gr.update(interactive=True, value="🚀 Generate Reply"),  # Re-enable button on error
                            [],                                 # Clear conversation history
                            False,                              # Reset revision mode
                            "",                                 # Clear initial key messages
                            2,                                  # Stay on Stage 2
                            [1, 2]                              # Stages 1 and 2 unlocked
                        )
                        return

                except queue.Empty:
                    # No new data, yield progress indicator to keep UI responsive
                    # Use overlay to preserve any existing content users might want to reference
                    progress_content = create_loading_overlay_html(
                        "Connecting to AI service",
                        model,
                        ""  # No specific background content - will show placeholder
                    )

                    # Update status instructions during connection - stay on Stage 2
                    stage1_update, stage2_update, stage3_update = get_workflow_banner_html(2)

                    yield (
                        gr.update(visible=False),           # Keep upload panel hidden
                        progress_content,                   # Show progress indicator
                        original_email_preview,             # Keep original email visible
                        gr.update(visible=False),           # Hide thinking accordion
                        gr.update(visible=True), # Keep thread preview group visible
                        gr.update(visible=True), # Keep original reference group visible so users can read original email
                        gr.update(visible=True), # Keep key messages container visible
                        "",                                 # Clear thinking content
                        gr.update(visible=False, value=None),  # Hide download file
                        "",                                 # Clear current_reply state
                        "",                                 # Clear current_think state
                        stage1_update,                      # Update stage 1 banner
                        stage2_update,                      # Update stage 2 banner
                        stage3_update,                      # Update stage 3 banner
                        gr.update(interactive=False, value="⏳ Generating..."),  # Keep button disabled
                        updated_conversation_history,       # Update conversation history state
                        updated_is_revision_mode,           # Update revision mode state
                        updated_initial_key_messages,       # Update initial key messages state
                        2,                                  # Stay on Stage 2 during connection
                        [1, 2]                              # Stages 1 and 2 unlocked
                    )
                    continue
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Exception in on_generate_stream: {e}\n{tb}")

            error_draft = """
            <div class='error-content'>
                <div style='font-size: 1.1em;'>Generation failed. Please try again.</div>
            </div>
            """

            # Error generation status - stay on Stage 2
            stage1_update, stage2_update, stage3_update = get_workflow_banner_html(2)

            yield (
                gr.update(visible=True),            # Show upload panel for retry
                error_draft,                        # Show error in thread preview area
                format_email_preview({}),           # Clear original email area
                gr.update(visible=False),           # Hide thinking accordion
                gr.update(visible=False),  # Hide thread preview group
                gr.update(visible=False),  # Hide original reference group
                gr.update(visible=False),  # Hide key messages container
                "",                                 # Clear thinking content
                gr.update(visible=False, value=None),  # Hide download file
                "",                                 # Clear current_reply state
                "",                                 # Clear current_think state
                stage1_update,                      # Update stage 1 banner
                stage2_update,                      # Update stage 2 banner
                stage3_update,                      # Update stage 3 banner
                gr.update(interactive=True, value="🚀 Generate Reply"),  # Re-enable button on error
                [],                                 # Clear conversation history
                False,                              # Reset revision mode
                "",                                 # Clear initial key messages
                2,                                  # Stay on Stage 2
                [1, 2]                              # Stages 1 and 2 unlocked
            )

    def reset_conversation_state():
        """Reset conversation state when a new email is uploaded"""
        return [], False, ""  # Clear conversation_history, reset is_revision_mode, clear initial_key_messages

    def clear_key_messages_for_revision(is_revision):
        """Clear key messages field when entering revision mode and update label"""
        if is_revision:
            return gr.update(
                label="✏️ Revision Instructions",
                value="",
                placeholder="Enter your revision instructions...\n\nExample:\n• Make the tone more formal\n• Add information about next steps\n• Shorten the response"
            )
        else:
            return gr.update(
                label="📝 Key Messages",
                placeholder="Enter the key messages you want to include in your reply...\n\nExample:\n• Thank them for their inquiry\n• Confirm the meeting time\n• Provide additional resources"
            )

    # ===== STAGE NAVIGATION FUNCTIONS =====

    def navigate_to_stage_1():
        """Navigate back to Stage 1 - Reset all application state"""
        # Get banner updates for Stage 1 - reset to only Stage 1 unlocked
        stage1_update, stage2_update, stage3_update = update_stage_banners(1, [1])

        return (
            gr.update(visible=True),  # Show upload panel
            """
            <div class='thread-placeholder'>
                <div class='placeholder-content'>
                    <div class='placeholder-icon'>📧</div>
                    <h3>Upload Email File Above</h3>
                    <p>Select your .msg email file to get started</p>
                    <div class='placeholder-hint'>Supported format: .msg files</div>
                </div>
            </div>
            """,  # Clear thread preview
            """
            <div class='email-placeholder'>
                <div class='placeholder-content'>
                    <div class='placeholder-icon'>📧</div>
                    <h3>Upload Email File Above</h3>
                    <p>Select your .msg email file to view the original email content</p>
                    <div class='placeholder-hint'>Supported format: .msg files</div>
                </div>
            </div>
            """,  # Clear original reference display
            gr.update(visible=False),  # Hide thinking accordion
            gr.update(visible=False),  # Hide thread preview group
            gr.update(visible=False),  # Hide original reference group
            gr.update(visible=False),  # Hide key messages container
            "",  # Clear thinking content
            gr.update(visible=False, value=None),  # Hide download button
            "",  # Clear current_reply state
            "",  # Clear current_think state
            stage1_update,  # Update stage 1 banner
            stage2_update,  # Update stage 2 banner
            stage3_update,  # Update stage 3 banner
            gr.update(interactive=False, value="🚀 Generate Reply"),  # Reset generate button
            [],  # Clear conversation history
            False,  # Reset revision mode
            "",  # Clear initial key messages
            {},  # Clear current email info
            None,  # Clear file input
            1,  # Set current stage to 1
            [1]  # Reset to only Stage 1 unlocked
        )

    def navigate_to_stage_2(current_email_info, current_unlocked_stages):
        """Navigate back to Stage 2 - Preserve email, clear draft content"""
        # Check if Stage 2 is unlocked
        if 2 not in current_unlocked_stages:
            # Stage 2 is not unlocked, stay on current stage
            return [gr.update() for _ in range(22)]  # Return no-change updates for all outputs

        # Get banner updates for Stage 2
        stage1_update, stage2_update, stage3_update = update_stage_banners(2, current_unlocked_stages)

        # Format the preserved email preview
        email_preview = format_email_preview(current_email_info) if current_email_info else """
        <div class='email-placeholder'>
            <div class='placeholder-content'>
                <div class='placeholder-icon'>📧</div>
                <h3>Upload Email File Above</h3>
                <p>Select your .msg email file to view the original email content</p>
                <div class='placeholder-hint'>Supported format: .msg files</div>
            </div>
        </div>
        """

        return (
            gr.update(visible=False),  # Hide upload panel (email already uploaded)
            """
            <div class='thread-placeholder'>
                <div class='placeholder-content'>
                    <div class='placeholder-icon'>✍️</div>
                    <h3>Ready to Generate Reply</h3>
                    <p>Enter your key messages and click Generate Reply</p>
                </div>
            </div>
            """,  # Clear thread preview but show ready state
            email_preview,  # Preserve original email display
            gr.update(visible=False),  # Hide thinking accordion
            gr.update(visible=False),  # Hide thread preview group
            gr.update(visible=True),   # Show original reference group
            gr.update(
                visible=True,
                label="📝 Key Messages",
                value="",
                placeholder="Enter the key messages you want to include in your reply...\n\nExample:\n• Thank them for their inquiry\n• Confirm the meeting time\n• Provide additional resources"
            ),   # Show key messages container and reset to initial state
            "",  # Clear thinking content
            gr.update(visible=False, value=None),  # Hide download button
            "",  # Clear current_reply state
            "",  # Clear current_think state
            stage1_update,  # Update stage 1 banner
            stage2_update,  # Update stage 2 banner
            stage3_update,  # Update stage 3 banner
            gr.update(interactive=False, value="🚀 Generate Reply"),  # Reset generate button (will be enabled by validation)
            [],  # Clear conversation history
            False,  # Reset revision mode
            "",  # Clear initial key messages
            current_email_info,  # Preserve current email info
            gr.update(),  # Keep file input as is
            2,  # Set current stage to 2
            current_unlocked_stages  # Maintain current unlocked stages
        )

    # Event handlers - Updated for full-width upload panel
    file_input.change(extract_and_display_email, inputs=file_input, outputs=[upload_panel, original_reference_display, original_reference_accordion, key_messages, current_email_info, stage1_html, stage2_html, stage3_html, generate_btn, current_stage, unlocked_stages])
    file_input.change(reset_conversation_state, outputs=[conversation_history, is_revision_mode, initial_key_messages])
    file_input.change(validate_revision_inputs, inputs=[file_input, key_messages, model_selector, is_revision_mode], outputs=generate_btn)
    key_messages.change(validate_revision_inputs, inputs=[file_input, key_messages, model_selector, is_revision_mode], outputs=generate_btn)
    model_selector.change(validate_revision_inputs, inputs=[file_input, key_messages, model_selector, is_revision_mode], outputs=generate_btn)



    # Restore default instructions button handler
    restore_default_btn.click(restore_default_instructions, outputs=[ai_instructions, restore_feedback])

    # AI instructions text area change handler for saving custom instructions
    ai_instructions.change(save_custom_instructions_on_change, inputs=ai_instructions, outputs=restore_feedback)

    # Model selector change handler for validation
    model_selector.change(on_model_change, inputs=[model_selector], outputs=[model_selector])

    # Parser selector change handler
    parser_selector.change(on_parser_change, inputs=[parser_selector], outputs=[parser_info])

    generate_btn.click(on_generate_stream, inputs=[file_input, key_messages, model_selector, user_name, user_email, ai_instructions, email_token_limit, conversation_history, is_revision_mode, initial_key_messages], outputs=[upload_panel, thread_preview, original_reference_display, think_accordion, thread_preview_accordion, original_reference_accordion, key_messages, think_output, download_button, current_reply, current_think, stage1_html, stage2_html, stage3_html, generate_btn, conversation_history, is_revision_mode, initial_key_messages, current_stage, unlocked_stages])

    # Update key messages field when revision mode changes
    is_revision_mode.change(clear_key_messages_for_revision, inputs=[is_revision_mode], outputs=[key_messages])
    # Update button validation when revision mode changes
    is_revision_mode.change(validate_revision_inputs, inputs=[file_input, key_messages, model_selector, is_revision_mode], outputs=generate_btn)

    # Stage navigation click handlers - Gradio-native approach
    stage1_html.click(
        navigate_to_stage_1,
        outputs=[
            upload_panel, thread_preview, original_reference_display, think_accordion,
            thread_preview_accordion, original_reference_accordion, key_messages,
            think_output, download_button, current_reply, current_think,
            stage1_html, stage2_html, stage3_html,  # Update all stage banners
            generate_btn, conversation_history, is_revision_mode, initial_key_messages,
            current_email_info, file_input, current_stage, unlocked_stages
        ]
    )

    stage2_html.click(
        navigate_to_stage_2,
        inputs=[current_email_info, unlocked_stages],
        outputs=[
            upload_panel, thread_preview, original_reference_display, think_accordion,
            thread_preview_accordion, original_reference_accordion, key_messages,
            think_output, download_button, current_reply, current_think,
            stage1_html, stage2_html, stage3_html,  # Update all stage banners
            generate_btn, conversation_history, is_revision_mode, initial_key_messages,
            current_email_info, file_input, current_stage, unlocked_stages
        ]
    )

    # Preference persistence using BrowserState - reliable localStorage alternative
    user_name.change(save_user_name, inputs=[user_name, preferences_state], outputs=preferences_state)
    user_email.change(save_user_email, inputs=[user_email, preferences_state], outputs=preferences_state)
    ai_instructions.change(save_ai_instructions, inputs=[ai_instructions, preferences_state], outputs=preferences_state)



    # Load preferences on startup
    demo.load(
        load_preferences_on_startup,
        inputs=preferences_state,
        outputs=[user_name, user_email, ai_instructions]
    )

if __name__ == "__main__":
    # Initialize parser cache for optimal performance
    print("🚀 Starting SARA Compose with performance optimizations...")
    initialize_parser_cache()

    # Initialize model validation on startup
    print("🔄 Initializing dynamic model validation...")
    initialize_model_validation()

    # Check if running on Hugging Face Spaces
    is_hf_spaces = os.getenv("SPACE_ID") is not None

    if is_hf_spaces:
        # Hugging Face Spaces configuration
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True
        )
    else:
        # Local development configuration
        demo.launch(
            share=True,
            server_name="localhost",
            show_error=True
        )