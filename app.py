import gradio as gr
import os
import extract_msg
import html2text
from bs4 import BeautifulSoup
import tempfile
import markdown
import re
import base64
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import fastapi_poe as fp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ['.msg']

# POE API Configuration - Use environment variable for security
POE_API_KEY = os.getenv("POE_API_KEY", "")

# Available Models
POE_MODELS = ["GPT-4o", "DeepSeek-R1-Distill"]

# Default AI Instructions - comprehensive but concise
DEFAULT_AI_INSTRUCTIONS = """You are an expert email assistant. Write professional email replies that incorporate the provided key messages.

FORMATTING REQUIREMENTS:
- Use British English spelling and grammar throughout
- Write only the email body content (no subject line)
- Use professional greeting and closing with the sender's name
- Include proper paragraph spacing for readability

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
    def stream_response(self, prompt: str, model: str) -> Iterator[Tuple[str, bool]]:
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

    def stream_response(self, prompt: str, model: str) -> Iterator[Tuple[str, bool]]:
        """Stream response from POE API"""
        try:
            if not self.api_key:
                yield "POE API key not configured", True
                return

            if model not in POE_MODELS:
                yield f"Model {model} not available in POE", True
                return

            # Prepare messages for POE API
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
    """Manages POE AI backend for cloud deployment"""

    def __init__(self):
        self.poe_backend = POEBackend()

    def get_current_backend(self) -> AIBackend:
        """Get the POE backend instance"""
        return self.poe_backend

    def is_backend_healthy(self) -> bool:
        """Check if POE backend is healthy"""
        return self.poe_backend.is_healthy()

    def get_available_models(self) -> list:
        """Get available POE models"""
        return POE_MODELS

# Initialize backend manager
backend_manager = BackendManager()

# SARA Framework Integration - CSS and JavaScript
# Load the working copy button implementation from main SARA app
def load_copy_button_js():
    """Load the working copy button JavaScript from the main SARA framework"""
    try:
        with open('src/overrideCopyButton.js', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print("Warning: overrideCopyButton.js not found. Copy functionality may not work properly.")
        return ""

def load_sara_icon_base64():
    """Load SARA icon as base64 data URL for reliable display with transparency fix"""
    try:
        with open('icons/SARA_no_word.png', 'rb') as f:
            icon_data = f.read()
            icon_base64 = base64.b64encode(icon_data).decode('utf-8')
            return f"data:image/png;base64,{icon_base64}"
    except FileNotFoundError:
        print("Warning: SARA icon not found. Banner will display without icon.")
        return ""

copy_button_js = load_copy_button_js()
sara_icon_data_url = load_sara_icon_base64()

# JavaScript for localStorage persistence - Enhanced version with better Gradio integration
preferences_js = """
// SARA Compose Preferences Management - Enhanced Version
window.saraPreferences = {
    storageKey: 'sara_compose_preferences',

    // Default values
    defaults: {
        user_name: 'Max Kwong',
        user_email: 'mwmkwong@hkma.gov.hk',
        ai_instructions: ''
    },

    // Find preference components using multiple targeting strategies
    findComponents: function() {
        const components = {};

        // Strategy 1: Direct element ID targeting with multiple selectors
        const selectors = {
            user_name: [
                '#user-name-input input',
                '#user-name-input textarea',
                '[data-testid="textbox"]:has(label:contains("Your Name"))',
                'input[placeholder*="full name"]'
            ],
            user_email: [
                '#user-email-input input',
                '#user-email-input textarea',
                '[data-testid="textbox"]:has(label:contains("Email"))',
                'input[placeholder*="email"]'
            ],

            ai_instructions: [
                '#ai-instructions-textarea textarea',
                '#ai-instructions-textarea input',
                '[data-testid="textbox"]:has(label:contains("AI Instructions"))',
                'textarea[placeholder*="complete instructions"]'
            ]
        };

        // Try each selector strategy for each component
        Object.keys(selectors).forEach(key => {
            for (const selector of selectors[key]) {
                try {
                    const element = document.querySelector(selector);
                    if (element) {
                        components[key] = element;
                        console.log(`✅ Found ${key} using selector: ${selector}`);
                        break;
                    }
                } catch (e) {
                    // Skip invalid selectors
                    console.log(`⚠️ Invalid selector for ${key}: ${selector}`);
                }
            }

            if (!components[key]) {
                console.log(`❌ Could not find ${key} with any selector`);
            }
        });

        // Strategy 2: Fallback - search by label text within Personal Preferences section
        if (Object.keys(components).length < 3) {
            console.log('🔄 Trying fallback strategy - searching by label text');
            const personalPrefsSection = document.querySelector('.personal-preferences');
            if (personalPrefsSection) {
                // Find inputs/selects by their labels - more comprehensive search
                const allInputs = personalPrefsSection.querySelectorAll('input, select, textarea');
                console.log(`Found ${allInputs.length} total input elements in Personal Preferences`);

                allInputs.forEach((input, index) => {
                    // Get the label text from various possible locations
                    const parentLabel = input.closest('label');
                    const siblingLabel = input.parentElement?.querySelector('label');
                    const nearbyLabel = input.parentElement?.parentElement?.querySelector('label');

                    const labelTexts = [
                        parentLabel?.textContent || '',
                        siblingLabel?.textContent || '',
                        nearbyLabel?.textContent || '',
                        input.placeholder || '',
                        input.value || '',
                        input.getAttribute('aria-label') || ''
                    ].map(text => text.toLowerCase());

                    const allLabelText = labelTexts.join(' ');
                    console.log(`Input ${index}: type=${input.type || input.tagName}, value="${input.value}", labels="${allLabelText}"`);

                    if (allLabelText.includes('name') && !components.user_name) {
                        components.user_name = input;
                        console.log('✅ Found user_name via label fallback');
                    } else if (allLabelText.includes('email') && !components.user_email) {
                        components.user_email = input;
                        console.log('✅ Found user_email via label fallback');
                    } else if (allLabelText.includes('instructions') && !components.ai_instructions) {
                        components.ai_instructions = input;
                        console.log('✅ Found ai_instructions via label fallback');
                    }
                });
            }
        }


        const foundCount = Object.keys(components).length;
        console.log(`🎯 Final component detection: ${foundCount}/3 components found`);
        console.log('Components:', Object.entries(components).map(([key, comp]) => `${key}: ${comp ? '✅' : '❌'}`).join(', '));

        return foundCount > 0 ? components : null;
    },

    // Save preferences to localStorage
    save: function() {
        const components = this.findComponents();
        if (!components) {
            console.log('Components not ready for saving');
            return;
        }

        const preferences = {};
        Object.keys(this.defaults).forEach(key => {
            if (components[key]) {
                preferences[key] = components[key].value || this.defaults[key];
            } else {
                preferences[key] = this.defaults[key];
            }
        });

        localStorage.setItem(this.storageKey, JSON.stringify(preferences));
        console.log('✅ Preferences saved:', preferences);
    },

    // Load preferences from localStorage
    load: function() {
        try {
            const saved = localStorage.getItem(this.storageKey);
            if (!saved) {
                console.log('No saved preferences found, using defaults');
                return this.defaults;
            }

            const preferences = JSON.parse(saved);
            console.log('📥 Loading saved preferences:', preferences);

            const components = this.findComponents();
            if (!components) {
                console.log('Components not ready for loading, will retry');
                return preferences;
            }

            // Apply saved values to components with enhanced Gradio dropdown handling
            Object.keys(preferences).forEach(key => {
                if (components[key] && preferences[key] !== undefined) {
                    const element = components[key];
                    const oldValue = element.value;
                    const newValue = preferences[key];

                    console.log(`🔄 Setting ${key}: "${oldValue}" → "${newValue}"`);

                    // Enhanced value setting for Gradio components
                    const setValueWithRetry = (retryCount = 0) => {
                        try {
                            // Method 1: Direct value assignment
                            element.value = newValue;

                            // Method 2: For select elements, set selectedIndex
                            if (element.tagName === 'SELECT') {
                                const options = Array.from(element.options);
                                const targetOption = options.find(opt => opt.value === newValue || opt.text === newValue);
                                if (targetOption) {
                                    element.selectedIndex = targetOption.index;
                                    console.log(`📋 Set dropdown ${key} selectedIndex to ${targetOption.index}`);
                                }
                            }

                            // Method 3: For Gradio dropdowns, try setting via setAttribute
                            element.setAttribute('value', newValue);

                            // Method 4: Trigger comprehensive event sequence for Gradio
                            const events = [
                                'focus',
                                'input',
                                'change',
                                'blur',
                                'keyup',
                                'keydown'
                            ];

                            events.forEach((eventType, index) => {
                                setTimeout(() => {
                                    const event = new Event(eventType, {
                                        bubbles: true,
                                        cancelable: true,
                                        composed: true
                                    });

                                    // Add additional properties for input events
                                    if (eventType === 'input' || eventType === 'change') {
                                        Object.defineProperty(event, 'target', {
                                            value: element,
                                            enumerable: true
                                        });
                                    }

                                    element.dispatchEvent(event);
                                    console.log(`📡 Triggered ${eventType} event for ${key}`);
                                }, index * 50); // Stagger events
                            });

                            // Method 5: Additional Gradio-specific triggers
                            setTimeout(() => {
                                // Try to trigger Gradio's internal update mechanism
                                if (element._gradio_change) {
                                    element._gradio_change(newValue);
                                }

                                // Force a final input event
                                const finalEvent = new Event('input', { bubbles: true });
                                element.dispatchEvent(finalEvent);

                                // Verify the value was set
                                if (element.value === newValue) {
                                    console.log(`✅ Successfully set ${key}: "${oldValue}" → "${newValue}"`);
                                } else if (retryCount < 3) {
                                    console.log(`⚠️ Value not set correctly for ${key}, retrying... (attempt ${retryCount + 1})`);
                                    setTimeout(() => setValueWithRetry(retryCount + 1), 500);
                                } else {
                                    console.log(`❌ Failed to set ${key} after ${retryCount + 1} attempts`);
                                }
                            }, 300);

                        } catch (error) {
                            console.error(`❌ Error setting ${key}:`, error);
                            if (retryCount < 3) {
                                setTimeout(() => setValueWithRetry(retryCount + 1), 500);
                            }
                        }
                    };

                    // Start the value setting process
                    setValueWithRetry();
                }
            });

            return preferences;
        } catch (e) {
            console.error('❌ Error loading preferences:', e);
            return this.defaults;
        }
    },

    // Debounce function to limit save frequency
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // Setup auto-save listeners
    setupAutoSave: function() {
        const components = this.findComponents();
        if (!components) {
            console.log('Components not ready for auto-save setup');
            return false;
        }

        Object.entries(components).forEach(([key, component]) => {
            if (component) {
                // Create a debounced save function to avoid excessive saves
                const debouncedSave = this.debounce(() => {
                    console.log(`💾 Auto-saving due to ${key} change`);
                    this.save();
                }, 500);

                // Remove existing listeners to avoid duplicates
                const events = ['change', 'input', 'blur'];
                events.forEach(eventType => {
                    component.removeEventListener(eventType, debouncedSave);
                });

                // Add new listeners for multiple events
                events.forEach(eventType => {
                    component.addEventListener(eventType, debouncedSave);
                });

                console.log(`🔗 Setup auto-save listeners for ${key} (${component.tagName})`);
            }
        });

        console.log('✅ Auto-save listeners setup complete');
        return true;
    },

    // Initialize preferences system with robust retry logic
    init: function() {
        console.log('🚀 Initializing SARA Preferences System');

        const attemptInitialization = (attempt = 1, maxAttempts = 10) => {
            console.log(`🔄 Initialization attempt ${attempt}/${maxAttempts}`);

            const components = this.findComponents();
            if (components && Object.keys(components).length >= 2) {
                console.log('✅ Sufficient components found, proceeding with initialization');
                this.load();
                this.setupAutoSave();
                return true;
            } else {
                console.log(`⏳ Only ${components ? Object.keys(components).length : 0} components found, retrying...`);
                if (attempt < maxAttempts) {
                    const delay = Math.min(1000 * attempt, 5000); // Exponential backoff, max 5s
                    setTimeout(() => attemptInitialization(attempt + 1, maxAttempts), delay);
                } else {
                    console.log('❌ Max initialization attempts reached');
                }
                return false;
            }
        };

        // Start initialization attempts
        attemptInitialization();
    }
};

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.saraPreferences.init();
    });
} else {
    window.saraPreferences.init();
}

// Also initialize after delays to handle Gradio's dynamic loading
setTimeout(() => {
    console.log('🔄 Retrying preferences initialization (2s delay)');
    window.saraPreferences.init();
}, 2000);

setTimeout(() => {
    console.log('🔄 Retrying preferences initialization (5s delay)');
    window.saraPreferences.init();
}, 5000);

// Expose functions globally for debugging
window.savePreferences = () => window.saraPreferences.save();
window.loadPreferences = () => window.saraPreferences.load();
window.debugPreferences = () => {
    console.log('🔍 Debug Preferences System');
    const components = window.saraPreferences.findComponents();
    console.log('Components:', components);

    // Detailed component analysis
    if (components) {
        Object.entries(components).forEach(([key, element]) => {
            if (element) {
                console.log(`${key}:`, {
                    tagName: element.tagName,
                    type: element.type,
                    value: element.value,
                    id: element.id,
                    className: element.className,
                    placeholder: element.placeholder,
                    'data-testid': element.getAttribute('data-testid')
                });
            } else {
                console.log(`${key}: NOT FOUND`);
            }
        });
    }

    console.log('Current localStorage:', localStorage.getItem('sara_compose_preferences'));
    return window.saraPreferences;
};

window.debugPersonalPreferences = () => {
    console.log('🔍 Debug Personal Preferences Components');

    // Check Personal Preferences section specifically
    const personalPrefs = document.querySelector('.personal-preferences');
    if (personalPrefs) {
        const prefsInputs = personalPrefs.querySelectorAll('input, select, textarea');
        console.log(`Found ${prefsInputs.length} inputs in Personal Preferences section`);

        prefsInputs.forEach((input, index) => {
            console.log(`Personal Prefs Input ${index}:`, {
                tagName: input.tagName,
                type: input.type,
                value: input.value,
                placeholder: input.placeholder,
                id: input.id,
                className: input.className
            });
        });
    } else {
        console.log('❌ Personal Preferences section not found');
    }
};
window.forceInitPreferences = () => {
    console.log('🔧 Force initializing preferences');
    window.saraPreferences.init();
};
"""

# SARA Framework CSS - Clean, professional design with consistent color scheme
custom_css = """
/* ===== UNIFIED COLOR SYSTEM ===== */
:root {
    /* Primary Brand Colors - Updated to Orange Theme to Match Gradio */
    --primary-color: #f97316;
    --primary-hover: #ea580c;
    --primary-light: #fb923c;
    --primary-gradient: linear-gradient(135deg, #f97316 0%, #fb923c 100%);

    /* Secondary Colors */
    --secondary-color: #6366f1;
    --secondary-hover: #5b21b6;
    --secondary-light: #8b5cf6;

    /* Accent Colors - Consistent Orange Theme */
    --accent-color: #f97316;
    --accent-hover: #ea580c;
    --accent-light: #fb923c;
    --accent-gradient: linear-gradient(135deg, #f97316 0%, #fb923c 100%);

    /* Status Colors */
    --success-color: #10b981;
    --success-hover: #059669;
    --success-light: #34d399;
    --success-gradient: linear-gradient(135deg, #10b981 0%, #34d399 100%);

    --error-color: #ef4444;
    --error-hover: #dc2626;
    --error-light: #f87171;

    --warning-color: #f97316;
    --warning-hover: #ea580c;
    --warning-light: #fb923c;

    --info-color: #f97316;
    --info-hover: #ea580c;
    --info-light: #fb923c;

    /* Text Colors */
    --text-primary: #111827;
    --text-secondary: #374151;
    --text-muted: #6b7280;
    --text-light: #9ca3af;
    --text-inverse: #ffffff;

    /* Background Colors */
    --bg-primary: #ffffff;
    --bg-secondary: #f9fafb;
    --bg-tertiary: #f3f4f6;
    --bg-accent: #eff6ff;

    /* Border Colors */
    --border-primary: #e5e7eb;
    --border-secondary: #d1d5db;
    --border-accent: #bae6fd;

    /* Shadow System */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);

    /* Border Radius System */
    --radius-sm: 4px;
    --radius-md: 6px;
    --radius-lg: 8px;
    --radius-xl: 12px;

    /* Transition System */
    --transition-fast: all 0.15s ease;
    --transition-normal: all 0.2s ease;
    --transition-slow: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== LAYOUT & CONTAINER STYLES ===== */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    background: var(--bg-secondary) !important;
    max-width: 1800px !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 20px !important;
    min-height: 100vh !important;
    box-sizing: border-box !important;
}

/* Ensure consistent dashboard width */
.gradio-container > div {
    width: 100% !important;
    min-width: 1200px !important;
}

/* Main row container for stable layout */
.main-layout-row {
    display: flex !important;
    width: 100% !important;
    gap: 20px !important;
    min-height: 600px !important;
}

/* ===== HEADER STYLES ===== */
.app-header {
    background: var(--primary-gradient);
    border-radius: var(--radius-xl);
    padding: 24px;
    text-align: center;
    margin-bottom: 20px;
    box-shadow: var(--shadow-lg);
}

.header-content {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 20px;
    max-width: 600px;
    margin: 0 auto;
}

.sara-icon {
    width: 64px;
    height: 64px;
    object-fit: contain;
    filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.3)) brightness(1.1) contrast(1.1);
    background: rgba(255, 255, 255, 0.1);
    border-radius: var(--radius-lg);
    padding: 4px;
}

.header-text {
    text-align: left;
}

.main-title {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--text-inverse);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    line-height: 1.1;
}

.subtitle {
    margin: 4px 0 0 0;
    font-size: 1.1rem;
    font-weight: 400;
    color: var(--text-inverse);
    opacity: 0.9;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

/* ===== DYNAMIC STATUS INSTRUCTIONS PANEL ===== */
.status-instructions-panel {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 24px;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Workflow stages layout */
.workflow-stages {
    display: flex;
    gap: 16px;
    align-items: stretch;
}

.stage {
    flex: 1;
    padding: 20px;
    border-radius: 10px;
    border: 2px solid transparent;
    background: rgba(248, 250, 252, 0.6);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    opacity: 0.5;
}

.stage::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #f97316 0%, #fb923c 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.stage.active {
    background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
    border-color: #f97316;
    opacity: 1;
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(249, 115, 22, 0.15);
}

.stage.active::before {
    opacity: 1;
}

.stage.active .stage-icon {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.05); }
}

.stage-header {
    display: flex;
    align-items: center;
    margin-bottom: 12px;
}

.stage-number {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #e2e8f0;
    color: #64748b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.9rem;
    margin-right: 12px;
    transition: all 0.3s ease;
}

.stage.active .stage-number {
    background: #f97316;
    color: white;
    box-shadow: 0 4px 12px rgba(249, 115, 22, 0.3);
}

.stage-icon {
    font-size: 1.5rem;
    margin-right: 8px;
    transition: transform 0.3s ease;
}

.stage-title {
    font-weight: 600;
    color: #1e293b;
    font-size: 1rem;
    margin: 0;
}

.stage.active .stage-title {
    color: #1e293b;
    font-weight: 700;
}

.stage-description {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.4;
    margin-top: 8px;
}

.stage.active .stage-description {
    color: #374151;
    font-weight: 600;
}

/* Responsive design for workflow stages */
@media (max-width: 768px) {
    .workflow-stages {
        flex-direction: column;
        gap: 12px;
    }

    .stage {
        padding: 16px;
    }

    .status-instructions-panel {
        padding: 16px;
    }
}



/* ===== RESPONSIVE DESIGN ===== */
@media (max-width: 768px) {
    .header-content {
        flex-direction: column;
        gap: 12px;
    }

    .header-text {
        text-align: center;
    }

    .main-title {
        font-size: 2rem;
    }

    .subtitle {
        font-size: 1rem;
    }

    .sara-icon {
        width: 48px;
        height: 48px;
    }
}

/* ===== CARD COMPONENTS ===== */
.card {
    background: var(--bg-primary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: var(--shadow-sm);
    transition: var(--transition-normal);
    position: relative;
    overflow: hidden;
}

.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--primary-gradient);
    opacity: 0;
    transition: var(--transition-normal);
}

.card:hover::before {
    opacity: 1;
}



/* ===== EMAIL PREVIEW COMPONENTS ===== */
.email-preview-container {
    max-height: 500px;
    overflow-y: auto;
    overflow-x: hidden;
}

.email-thread {
    padding: 0;
    margin: 0;
    overflow: visible;
}

.email-item {
    padding: 20px;
    position: relative;
    background: var(--bg-primary);
    margin: 0;
}

.email-item:last-child {
    border-bottom: none;
}

.email-item:not(:last-child) {
    margin-bottom: 32px;
    padding-bottom: 24px;
    position: relative;
    border-bottom: none;
}

.email-item:not(:last-child)::after {
    content: '';
    position: absolute;
    bottom: -16px;
    left: 0;
    right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--border-secondary) 15%, var(--text-light) 50%, var(--border-secondary) 85%, transparent 100%);
    box-shadow: var(--shadow-sm);
}

.email-item:not(:last-child)::before {
    content: '';
    position: absolute;
    bottom: -20px;
    left: -20px;
    right: -20px;
    height: 8px;
    background: linear-gradient(180deg, transparent 0%, var(--bg-secondary) 50%, transparent 100%);
    border-radius: var(--radius-sm);
    z-index: -1;
}

/* ===== EMAIL HEADER & BODY STYLES ===== */
.email-header {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    padding: 12px;
    margin-bottom: 12px;
    font-size: 0.875rem;
}

.email-header-line {
    margin: 4px 0;
    display: flex;
}

.email-header-label {
    font-weight: 600;
    color: var(--text-primary);
    min-width: 60px;
    margin-right: 8px;
}

.email-header-value {
    color: var(--text-secondary);
    flex: 1;
}

.email-body {
    line-height: 1.6;
    color: var(--text-secondary);
    font-size: 0.9rem;
}

.email-body p {
    margin: 8px 0;
}

.email-body a {
    color: #0066cc !important;
    text-decoration: none;
    transition: var(--transition-fast);
}

.email-body a:hover {
    color: #004499 !important;
    text-decoration: underline;
}

/* ===== ACCORDION COMPONENTS ===== */
.think-section {
    margin: 16px 0;
    border: 1px solid var(--warning-color);
    border-radius: var(--radius-md);
    background: var(--bg-accent);
}

/* Accordion Content Styling */
.gradio-accordion .accordion-content,
.gradio-accordion [data-testid="accordion-content"] {
    max-height: 400px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 12px !important;
    scrollbar-width: thin !important;
    scrollbar-color: var(--border-secondary) var(--bg-secondary) !important;
}

/* Accordion Scrollbar Styling */
.gradio-accordion .accordion-content::-webkit-scrollbar,
.gradio-accordion [data-testid="accordion-content"]::-webkit-scrollbar {
    width: 6px !important;
}

.gradio-accordion .accordion-content::-webkit-scrollbar-track,
.gradio-accordion [data-testid="accordion-content"]::-webkit-scrollbar-track {
    background: var(--bg-secondary) !important;
    border-radius: var(--radius-sm) !important;
}

.gradio-accordion .accordion-content::-webkit-scrollbar-thumb,
.gradio-accordion [data-testid="accordion-content"]::-webkit-scrollbar-thumb {
    background: var(--border-secondary) !important;
    border-radius: var(--radius-sm) !important;
}

.gradio-accordion .accordion-content::-webkit-scrollbar-thumb:hover,
.gradio-accordion [data-testid="accordion-content"]::-webkit-scrollbar-thumb:hover {
    background: var(--text-light) !important;
}

/* Accordion Container Fixes */
.gradio-accordion {
    overflow: visible !important;
    transition: var(--transition-slow) !important;
}

.email-preview-column {
    overflow: visible !important;
}

.email-preview-column .gradio-accordion {
    overflow: visible !important;
}

.email-preview-column .gradio-accordion .accordion-content {
    overflow: visible !important;
    max-height: none !important;
}

/* Accordion Animations */
.think-streaming {
    animation: scroll-to-bottom 0.3s ease-out;
}

@keyframes scroll-to-bottom {
    to {
        scroll-behavior: smooth;
    }
}

/* Accordion Header Styling - Consolidated */
.gradio-accordion .label-wrap,
.gradio-accordion .accordion-header,
.gradio-accordion button,
.gradio-accordion [data-testid="accordion-header"],
.gradio-accordion .accordion-trigger,
div[data-testid="accordion"] button,
div[data-testid="accordion"] .label-wrap {
    font-weight: 700 !important;
}

.gradio-accordion .label-wrap span,
.gradio-accordion button span,
div[data-testid="accordion"] .label-wrap span {
    font-weight: 700 !important;
    font-size: 1rem !important;
}

/* Thinking Content Styling */
.gradio-accordion .accordion-content,
.gradio-accordion [data-testid="accordion-content"] {
    color: var(--text-muted) !important;
    font-family: 'Microsoft Sans Serif', sans-serif !important;
    font-size: 11pt !important;
}

.gradio-accordion .accordion-content p,
.gradio-accordion [data-testid="accordion-content"] p,
.gradio-accordion .accordion-content div,
.gradio-accordion [data-testid="accordion-content"] div {
    color: var(--text-muted) !important;
    font-family: 'Microsoft Sans Serif', sans-serif !important;
    font-size: 11pt !important;
}

/* Key Messages Accordion Styling */
.key-messages-accordion {
    margin-bottom: 0px !important;
}

.key-messages-accordion .gradio-accordion {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-secondary) !important;
    box-shadow: var(--shadow-sm) !important;
}

.key-messages-accordion .gradio-accordion-header {
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.key-messages-accordion .gradio-accordion-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.key-messages-accordion .gradio-accordion-header:hover::before {
    opacity: 1;
}

/* Remove grey padding/margin from key messages card container */
.key-messages-accordion .gradio-accordion > div {
    padding-bottom: 0px !important;
    margin-bottom: 0px !important;
}

/* Inline Generate Button Styling */
.generate-button-inline {
    margin-top: 16px !important;
    margin-bottom: 0px !important;
    width: 100% !important;
}

/* Remove extra spacing from card container for key messages */
.card {
    margin-bottom: 8px !important;
    padding-bottom: 12px !important;
}

/* Generate Button Section Styling */
.generate-button-section {
    margin-top: 16px !important;
    margin-bottom: 16px !important;
    text-align: center;
}

/* ===== FILE UPLOAD & DROP ZONE COMPONENTS ===== */
.download-file-section {
    margin: 16px 0;
    padding: 12px;
    border: 2px dashed var(--primary-color);
    border-radius: var(--radius-lg);
    background: var(--bg-accent);
}

/* Custom Download Button Styling */
.custom-download-button {
    margin: 16px 0 !important;
    width: 100% !important;
}

.custom-download-button button {
    width: 100% !important;
    padding: 16px 24px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    border: none !important;
    border-radius: var(--radius-lg) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

.custom-download-button button:hover {
    background: linear-gradient(135deg, #059669 0%, #047857 100%) !important;
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    transform: translateY(-2px) !important;
}

.custom-download-button button:active {
    transform: translateY(0px) !important;
    box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3) !important;
}

.custom-download-button button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.custom-download-button button:hover::before {
    left: 100%;
}

/* Subtle pulse animation when button first appears */
@keyframes downloadButtonPulse {
    0% {
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
    50% {
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.5);
    }
    100% {
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }
}

.custom-download-button button {
    animation: downloadButtonPulse 2s ease-in-out 3 !important;
}

/* Download button within draft accordion styling */
.draft-download-button {
    margin-top: 8px !important;
    margin-bottom: 8px !important;
}



/* Enhanced Upload Panel with Email Preview */
.upload-panel-collapsed {
    background: var(--bg-secondary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    padding: 12px 16px;
    margin-bottom: 16px;
    cursor: pointer;
    transition: var(--transition-normal);
}

.upload-panel-collapsed:hover {
    background: var(--bg-tertiary);
    box-shadow: var(--shadow-sm);
}

.upload-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}

.upload-panel-title {
    font-weight: 600;
    color: var(--text-primary);
    font-size: 0.95em;
}

.upload-panel-toggle {
    color: var(--text-muted);
    font-size: 0.8em;
    transition: var(--transition-fast);
}

.upload-panel-preview {
    font-size: 0.85em;
    color: var(--text-secondary);
    line-height: 1.4;
    max-height: 60px;
    overflow: hidden;
    text-overflow: ellipsis;
}

.upload-panel-meta {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 0.8em;
    color: var(--text-muted);
}

.upload-panel-meta span {
    display: flex;
    align-items: center;
    gap: 4px;
}

.email-drop-zone {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 300px;
    border: 2px dashed var(--border-secondary);
    border-radius: var(--radius-xl);
    background: linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
    transition: var(--transition-normal);
    cursor: pointer;
}

.email-drop-zone:hover {
    border-color: var(--primary-color);
    background: linear-gradient(135deg, var(--bg-accent) 0%, var(--bg-secondary) 100%);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.drop-zone-content {
    text-align: center;
    padding: 40px 20px;
}

.drop-zone-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.7;
    color: var(--text-muted);
}

.drop-zone-content h3 {
    margin: 0 0 8px 0;
    color: var(--text-primary);
    font-size: 1.25rem;
    font-weight: 600;
}

.drop-zone-content p {
    margin: 0 0 12px 0;
    color: var(--text-secondary);
    font-size: 0.95rem;
}

.drop-zone-hint {
    font-size: 0.85rem;
    color: var(--text-muted);
    font-style: italic;
}

/* Status indicators */
.status-healthy {
    color: var(--success-color);
    font-weight: 600;
}

.status-error {
    color: var(--error-color);
    font-weight: 600;
}

/* ===== STATUS INDICATORS ===== */

/* ===== SECTION COMPONENTS ===== */
.generate-section {
    background: linear-gradient(135deg, var(--bg-accent) 0%, var(--bg-secondary) 100%);
    border: 1px solid var(--border-accent);
    border-radius: var(--radius-lg);
    padding: 16px;
    text-align: center;
}

.thinking-section {
    background: var(--bg-primary);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    padding: 0;
    margin-bottom: 6px;
    box-shadow: var(--shadow-sm);
}



/* Output panel container - matching email preview structure */
.output-panel-container {
    max-height: 500px;
    overflow-y: auto;
    background: var(--background);
    border-radius: 8px;
}

.output-panel-container::-webkit-scrollbar {
    width: 6px;
}

.output-panel-container::-webkit-scrollbar-track {
    background: #f1f5f9;
}

.output-panel-container::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

/* Draft preview styling - consistent with email items */
.draft-preview-item {
    padding: 20px;
    background: var(--background);
    margin: 0;
    border-radius: 8px;
}

/* Draft header styling - matching email headers */
.draft-header {
    background: #f0f9ff;
    border: 1px solid #bae6fd;
    border-radius: 6px;
    padding: 12px;
    margin-bottom: 12px;
    font-size: 0.875rem;
}

.draft-header-line {
    margin: 4px 0;
    display: flex;
}

.draft-header-label {
    font-weight: 600;
    color: var(--text-color);
    min-width: 60px;
    margin-right: 8px;
}

.draft-header-value {
    color: var(--text-secondary);
    flex: 1;
}

/* Draft content styling - matching email body */
.draft-content {
    line-height: 1.6;
    color: var(--text-color);
    font-size: 11pt;
    font-family: 'Microsoft Sans Serif', sans-serif;
}

.draft-content p {
    margin: 12px 0;
}

.draft-content a {
    color: #0066cc !important;
    text-decoration: none;
}

.draft-content a:hover {
    text-decoration: underline;
}



/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--text-secondary);
    font-size: 0.9rem;
}

/* Scrollbar */
.email-preview-container::-webkit-scrollbar {
    width: 6px;
}

.email-preview-container::-webkit-scrollbar-track {
    background: #f1f5f9;
}

.email-preview-container::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 3px;
}

/* Reply formatting */
.reply-content {
    line-height: 1.6;
    color: var(--text-color);
    padding: 16px;
    background: var(--background);
    border-radius: 6px;
    border: 1px solid var(--border-color);
}

.reply-content p {
    margin: 12px 0;
}

.reply-content strong {
    font-weight: 600;
}

/* Enhanced streaming and generation indicators */
.streaming {
    opacity: 0.8;
    animation: pulse 1.5s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

.streaming::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent 0%, rgba(99, 102, 241, 0.1) 50%, transparent 100%);
    animation: shimmer 2s infinite;
    z-index: 1;
}

@keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

@keyframes shimmer {
    0% { left: -100%; }
    100% { left: 100%; }
}

/* Generation state indicators */
.generating-state {
    background: linear-gradient(90deg, #f0f9ff 0%, #e0f2fe 50%, #f0f9ff 100%);
    background-size: 200% 100%;
    animation: shimmer-bg 2s infinite;
    border: 2px solid #f97316;
    border-radius: 8px;
    padding: 16px;
    position: relative;
}

@keyframes shimmer-bg {
    0% { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}



@keyframes success-glow {
    0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(16, 185, 129, 0); }
    100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
}

/* Copy notification */
.copy-notification {
    position: fixed;
    top: 20px;
    right: 20px;
    background: var(--primary-color);
    color: white;
    padding: 12px 20px;
    border-radius: 6px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 1000;
    animation: slideIn 0.3s ease-out;
}

@keyframes slideIn {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

/* 2-Column Layout Optimization - Left column for email preview/draft display */
.email-preview-column {
    margin-right: 16px; /* Right margin for spacing from controls column */
    flex: 1.5; /* Larger proportion for maximum content visibility */
    min-width: 600px !important; /* Ensure stable minimum width */
    width: 60% !important; /* Fixed percentage width for consistency */
}

/* Right column for all controls - optimized workflow */
.rhs-controls-column {
    margin-left: 8px; /* Left margin for spacing from preview column */
    flex: 1; /* Smaller proportion to maximize preview space */
    min-width: 400px !important; /* Ensure adequate space for controls */
    width: 40% !important; /* Fixed percentage width for consistency */
    max-width: 500px !important; /* Prevent excessive expansion */
}

/* Stable accordion containers */
.gradio-accordion {
    min-height: 60px !important; /* Minimum height to prevent layout shifts */
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* Ensure stable content areas */
.draft-display-area, .original-email-display-area {
    min-height: 200px !important;
    transition: all 0.3s ease !important;
}

/* Integrated file upload styling */
.integrated-file-upload {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 2px dashed var(--primary-color);
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    transition: all 0.3s ease;
}

.integrated-file-upload:hover {
    background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
    border-color: var(--accent-color);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
}

/* Enhanced spacing for 2-column layout with improved transitions */
.rhs-controls-column .gradio-group {
    margin-bottom: 16px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    transform-origin: top !important;
}

.rhs-controls-column .gradio-group:last-child {
    margin-bottom: 0 !important;
}

/* Remove margin from actions section to prevent gaps when hidden */
.rhs-controls-column .actions-section {
    margin-bottom: 0 !important;
}

/* Ensure clean layout when hidden elements don't create gaps */
.rhs-controls-column .gradio-group[style*="display: none"],
.rhs-controls-column .gradio-group[style*="visibility: hidden"] {
    margin-bottom: 0 !important;
    margin-top: 0 !important;
}

/* Section transition animations */
.section-transition {
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    transform-origin: top !important;
}





/* Action buttons styling - matching Generate Reply button with royal blue theme */
.action-button,
button.action-button,
.gradio-button.action-button {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: var(--text-inverse) !important;
    padding: 12px 24px !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: var(--radius-lg) !important;
    margin: 0 !important;
    min-width: 160px !important;
    transition: var(--transition-slow) !important;
    box-shadow: var(--shadow-sm) !important;
    /* Override any parent container backgrounds */
    position: relative !important;
    z-index: 10 !important;
}

.action-button:hover,
button.action-button:hover,
.gradio-button.action-button:hover {
    background: linear-gradient(135deg, var(--accent-hover) 0%, var(--accent-light) 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-lg) !important;
    border: none !important;
    color: var(--text-inverse) !important;
}

/* Widescreen layout optimizations for 2-column design */
@media (min-width: 1440px) {
    .gradio-container {
        max-width: 2000px !important; /* Even wider for large monitors */
        padding: 24px !important;
    }

    .email-preview-column {
        margin-right: 20px;
        flex: 1.6; /* Increased proportion for better content display */
    }

    .rhs-controls-column {
        margin-left: 12px;
        min-width: 450px; /* Slightly wider for better control spacing */
    }
}

/* Ultra-wide monitor support for 2-column design */
@media (min-width: 1920px) {
    .gradio-container {
        max-width: 2400px !important;
        padding: 32px !important;
    }

    .email-preview-column {
        margin-right: 24px;
        flex: 1.7; /* Maximum proportion for ultra-wide displays */
    }

    .rhs-controls-column {
        margin-left: 16px;
        min-width: 500px; /* Optimal width for ultra-wide displays */
    }
}

/* ===== LEGACY STYLES CLEANUP ===== */
.card:hover {
    box-shadow: var(--shadow-medium);
    transform: translateY(-1px);
}

/* ===== BUTTON COMPONENTS ===== */
/* Override Gradio's primary button styling for action buttons */
.primary.action-button,
button.primary.action-button {
    background: var(--accent-gradient) !important;
    border: none !important;
    color: var(--text-inverse) !important;
    box-shadow: var(--shadow-sm) !important;
}

.primary.action-button:hover,
button.primary.action-button:hover {
    background: linear-gradient(135deg, var(--accent-hover) 0%, var(--accent-light) 100%) !important;
    border: none !important;
    color: var(--text-inverse) !important;
    box-shadow: var(--shadow-lg) !important;
}

/* ===== PERSONAL PREFERENCES ===== */
.personal-preferences {
    margin-top: 0;
    opacity: 0.95;
}

.personal-preferences .gradio-accordion {
    background: linear-gradient(135deg, #fef7ff 0%, #f3e8ff 100%) !important;
    border: 1px solid #d8b4fe !important;
    box-shadow: var(--shadow-sm) !important;
}

.personal-preferences .gradio-accordion .label-wrap,
.personal-preferences .gradio-accordion button {
    background: linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%) !important;
    color: #7c3aed !important;
}

/* ===== AI INSTRUCTIONS STYLING ===== */
/* Normal editable styling for AI instructions */

/* Preference textarea styling */
.preference-textarea {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    background-color: #f8fafc;
}
    font-weight: 600 !important;
}

/* ===== DEVELOPMENT SETTINGS ===== */
.dev-settings {
    margin-top: 16px;
    opacity: 0.9;
}

/* 2-Column Layout Enhancements */
/* ===== FILE UPLOAD COMPONENTS ===== */

/* Optimize email preview container - single scroll container */
.email-preview-column .email-preview-container {
    max-height: 600px;
    overflow-y: auto;
    overflow-x: hidden;
}

/* Right column controls optimization */
.rhs-controls-column .thinking-section {
    margin: 16px 0 !important;
}

/* Draft placeholder styling */
.draft-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 250px;
    padding: 20px;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    border-radius: 0 0 10px 10px;
}

.draft-placeholder .placeholder-content {
    text-align: center;
    max-width: 400px;
}

.draft-placeholder .placeholder-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.7;
}

.draft-placeholder h3 {
    margin: 0 0 8px 0;
    color: #1f2937;
    font-size: 1.25rem;
    font-weight: 600;
}

.draft-placeholder p {
    margin: 0 0 12px 0;
    color: #6b7280;
    font-size: 0.95rem;
}

.draft-placeholder .placeholder-hint {
    font-size: 0.85rem;
    color: #9ca3af;
    font-style: italic;
}

/* Email placeholder styling */
.email-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 200px;
    padding: 20px;
    background: #fafafa;
    border-radius: 0 0 10px 10px;
}

.email-placeholder .placeholder-content {
    text-align: center;
    max-width: 400px;
}

.email-placeholder .placeholder-icon {
    font-size: 48px;
    margin-bottom: 16px;
    opacity: 0.6;
}

.email-placeholder h3 {
    margin: 0 0 8px 0;
    color: #374151;
    font-size: 1.25rem;
    font-weight: 600;
}

.email-placeholder p {
    margin: 0 0 12px 0;
    color: #6b7280;
    font-size: 0.95rem;
}

.email-placeholder .placeholder-hint {
    font-size: 0.85rem;
    color: #9ca3af;
    font-style: italic;
}

/* Real-time streaming content styling */
@keyframes draft-appear {
    0% {
        opacity: 0;
        transform: translateY(10px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Improved content display areas - fixed scroll bar conflicts */
.draft-display-area {
    min-height: 300px;
    overflow: visible;
}

.original-email-display-area {
    min-height: 250px;
    overflow: visible;
}

/* Remove redundant scrollbar styling to prevent conflicts */
/* Scrolling will be handled by parent containers only */

/* Accordion styling for progressive disclosure */
.upload-accordion {
    margin-bottom: 16px !important;
}

.draft-accordion {
    margin-bottom: 16px !important;
}

.original-accordion {
    margin-bottom: 16px !important;
}

/* Accordion header styling */
.upload-accordion .gradio-accordion-header {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%) !important;
    border: 2px solid #bae6fd !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.upload-accordion .gradio-accordion-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9 0%, #0284c7 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.upload-accordion .gradio-accordion-header:hover::before {
    opacity: 1;
}

.draft-accordion .gradio-accordion-header {
    background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%) !important;
    border: 2px solid #a7f3d0 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.draft-accordion .gradio-accordion-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #10b981 0%, #059669 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.draft-accordion .gradio-accordion-header:hover::before {
    opacity: 1;
}

.original-accordion .gradio-accordion-header {
    background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%) !important;
    border: 2px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.original-accordion .gradio-accordion-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6b7280 0%, #4b5563 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.original-accordion .gradio-accordion-header:hover::before {
    opacity: 1;
}

.thinking-accordion .gradio-accordion-header {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
    border: 2px solid #fbbf24 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    position: relative !important;
    overflow: hidden !important;
}

.thinking-accordion .gradio-accordion-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.thinking-accordion .gradio-accordion-header:hover::before {
    opacity: 1;
}

/* Accordion content styling */
.upload-accordion .gradio-accordion-content,
.draft-accordion .gradio-accordion-content,
.original-accordion .gradio-accordion-content,
.thinking-accordion .gradio-accordion-content {
    border: none !important;
    border-radius: 0 0 8px 8px !important;
    padding: 0 !important;
}



/* Enhanced visual hierarchy improvements */
.card {
    position: relative;
    overflow: hidden;
}

.card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.card:hover::before {
    opacity: 1;
}

/* Improved empty state styling */
.empty-state {
    transition: all 0.3s ease;
    border-radius: 8px;
}

.empty-state:hover {
    background: rgba(249, 250, 251, 0.8);
    transform: translateY(-1px);
}




"""

# Initialize backend manager on app start
print("Initializing AI backend...")
print(f"POE backend healthy: {backend_manager.poe_backend.is_healthy()}")
print("Backend initialization complete.")

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

def upload_file(file):
    _, error = validate_file(file)
    if error:
        return error
    return "File uploaded successfully."

def html_to_text(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    clean_html = soup.prettify()
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    h.ignore_images = True
    h.ignore_emphasis = False
    h.ignore_tables = False
    return h.handle(clean_html)

def html_to_markdown(html):
    """Convert HTML to markdown for better display"""
    if not html:
        return ""
    
    # Clean up the HTML first
    soup = BeautifulSoup(html, "html.parser")
    
    # Convert to markdown
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.body_width = 0
    h.ignore_images = False
    h.ignore_emphasis = False
    h.ignore_tables = False
    h.mark_code = True
    
    markdown_text = h.handle(str(soup))
    return markdown_text

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

def parse_email_thread(email_info):
    """Parse email content to extract individual emails in a thread"""
    body = email_info.get('body', '')
    html_body = email_info.get('html_body', '')
    to_recipients = email_info.get('to_recipients', [])
    cc_recipients = email_info.get('cc_recipients', [])

    emails = []

    # Enhanced parsing to handle different email thread formats
    # Look for patterns like "**From:** " or "From: " that indicate email headers
    from_patterns = [
        r'\*\*\s*From:\s*\*\*\s*(.+?)(?:\s*<br\s*/?>|\n)',  # **From:** pattern
        r'From:\s*(.+?)(?:\s*<br\s*/?>|\n)',                # From: pattern
        r'<strong>\s*From:\s*</strong>\s*(.+?)(?:\s*<br\s*/?>|\n)'  # <strong>From:</strong> pattern
    ]

    sent_patterns = [
        r'\*\*\s*Sent:\s*\*\*\s*(.+?)(?:\s*<br\s*/?>|\n)',  # **Sent:** pattern
        r'Sent:\s*(.+?)(?:\s*<br\s*/?>|\n)',                # Sent: pattern
        r'<strong>\s*Sent:\s*</strong>\s*(.+?)(?:\s*<br\s*/?>|\n)'  # <strong>Sent:</strong> pattern
    ]

    # Add patterns for To: and Cc: fields
    to_patterns = [
        r'\*\*\s*To:\s*\*\*\s*(.+?)(?:\s*<br\s*/?>|\n)',    # **To:** pattern
        r'To:\s*(.+?)(?:\s*<br\s*/?>|\n)',                  # To: pattern
        r'<strong>\s*To:\s*</strong>\s*(.+?)(?:\s*<br\s*/?>|\n)'  # <strong>To:</strong> pattern
    ]

    cc_patterns = [
        r'\*\*\s*Cc:\s*\*\*\s*(.+?)(?:\s*<br\s*/?>|\n)',    # **Cc:** pattern
        r'Cc:\s*(.+?)(?:\s*<br\s*/?>|\n)',                  # Cc: pattern
        r'<strong>\s*Cc:\s*</strong>\s*(.+?)(?:\s*<br\s*/?>|\n)'  # <strong>Cc:</strong> pattern
    ]

    # Try to find email separators in the body
    email_separators = []

    # Look for From: patterns that indicate start of new emails
    for pattern in from_patterns:
        matches = list(re.finditer(pattern, body, re.IGNORECASE | re.MULTILINE))
        for match in matches:
            email_separators.append((match.start(), match.group(1).strip()))

    if email_separators:
        # Sort separators by position
        email_separators.sort(key=lambda x: x[0])

        # First email is the main one (before first separator)
        first_separator_pos = email_separators[0][0]
        main_body = body[:first_separator_pos].strip()

        emails.append({
            'sender': email_info.get('sender', 'Unknown'),
            'subject': email_info.get('subject', '(No Subject)'),
            'date': standardize_date_format(email_info.get('date', 'Unknown')),
            'body': main_body,
            'html_body': email_info.get('html_body', ''),
            'attachments': email_info.get('attachments', []),
            'to_recipients': to_recipients,
            'cc_recipients': cc_recipients,
            'is_main': True
        })

        # Parse each subsequent email in the thread
        for i, (sep_pos, sender) in enumerate(email_separators):
            # Find the end position for this email (start of next email or end of body)
            if i + 1 < len(email_separators):
                end_pos = email_separators[i + 1][0]
                email_content = body[sep_pos:end_pos].strip()
            else:
                email_content = body[sep_pos:].strip()

            # Extract date from the email content
            date = 'Unknown'
            for pattern in sent_patterns:
                date_match = re.search(pattern, email_content, re.IGNORECASE | re.MULTILINE)
                if date_match:
                    date = standardize_date_format(date_match.group(1).strip())
                    break

            # Extract To: recipients from the email content
            email_to_recipients = []
            for pattern in to_patterns:
                to_match = re.search(pattern, email_content, re.IGNORECASE | re.MULTILINE)
                if to_match:
                    to_text = to_match.group(1).strip()
                    # Parse recipients (handle multiple formats)
                    if ';' in to_text:
                        email_to_recipients = [r.strip() for r in to_text.split(';') if r.strip()]
                    elif ',' in to_text:
                        email_to_recipients = [r.strip() for r in to_text.split(',') if r.strip()]
                    else:
                        email_to_recipients = [to_text] if to_text else []
                    break

            # Extract Cc: recipients from the email content
            email_cc_recipients = []
            for pattern in cc_patterns:
                cc_match = re.search(pattern, email_content, re.IGNORECASE | re.MULTILINE)
                if cc_match:
                    cc_text = cc_match.group(1).strip()
                    # Parse recipients (handle multiple formats)
                    if ';' in cc_text:
                        email_cc_recipients = [r.strip() for r in cc_text.split(';') if r.strip()]
                    elif ',' in cc_text:
                        email_cc_recipients = [r.strip() for r in cc_text.split(',') if r.strip()]
                    else:
                        email_cc_recipients = [cc_text] if cc_text else []
                    break

            # Extract the actual email body (skip the header lines)
            lines = email_content.split('\n')
            body_start = 0

            # Skip header lines to find actual content
            for j, line in enumerate(lines):
                if any(header in line.lower() for header in ['from:', 'sent:', 'to:', 'cc:', 'subject:']):
                    continue
                elif line.strip() and not any(header in line.lower() for header in ['from:', 'sent:', 'to:', 'cc:', 'subject:']):
                    body_start = j
                    break

            email_body = '\n'.join(lines[body_start:]).strip()

            # Clean up sender name (remove extra ** characters)
            clean_sender = sender.replace('**', '').strip()

            # Only add emails with meaningful content (not just "**" or empty)
            if email_body and email_body != '**' and len(email_body) > 10:
                emails.append({
                    'sender': clean_sender,
                    'subject': email_info.get('subject', '(No Subject)'),  # Use main subject for thread
                    'date': date,
                    'body': email_body,
                    'html_body': '',  # Threaded emails typically don't have separate HTML
                    'attachments': [],
                    'to_recipients': email_to_recipients,
                    'cc_recipients': email_cc_recipients,
                    'is_main': False
                })
    else:
        # Single email - include recipient information and HTML content
        emails.append({
            'sender': email_info.get('sender', 'Unknown'),
            'subject': email_info.get('subject', '(No Subject)'),
            'date': standardize_date_format(email_info.get('date', 'Unknown')),
            'body': html_to_markdown(html_body) if html_body else body,
            'html_body': email_info.get('html_body', ''),
            'attachments': email_info.get('attachments', []),
            'to_recipients': to_recipients,
            'cc_recipients': cc_recipients,
            'is_main': True
        })

    return emails

def format_email_preview(email_info):
    """Format email content as Outlook-style thread with proper dividers"""
    emails = parse_email_thread(email_info)
    
    if not emails:
        return "<div class='empty-state'>No email content to display</div>"
    
    html_parts = ['<div class="email-preview-container"><div class="email-thread">']
    
    for email in emails:
        sender = email.get('sender', 'Unknown')
        subject = email.get('subject', '(No Subject)')
        date = email.get('date', 'Unknown')
        body = email.get('body', '')
        to_recipients = email.get('to_recipients', [])
        cc_recipients = email.get('cc_recipients', [])

        # Use HTML body if available for rich content rendering, otherwise format plain text
        html_body = email.get('html_body', '')
        if html_body:
            try:
                from bs4 import BeautifulSoup
                # Parse and clean HTML content while preserving formatting
                soup = BeautifulSoup(html_body, 'html.parser')

                # Remove script tags for security
                for script in soup.find_all('script'):
                    script.decompose()

                # Handle embedded images with cid: references
                attachments = email.get('attachments', [])
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

                # Get the cleaned HTML content
                body_html = str(soup)

                # If the HTML is just a wrapper around plain text, extract and format it
                if not soup.find_all(['img', 'table', 'div', 'span', 'strong', 'em', 'ul', 'ol']):
                    body_text = soup.get_text()
                    if body_text.strip():
                        body_html = body_text.replace('\n', '<br>')
                        body_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', body_html)

            except Exception as e:
                print(f"Error processing HTML body: {e}")
                # Fallback to plain text processing
                body_html = body.replace('\n', '<br>') if body else '<em>No content</em>'
                body_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', body_html)
        elif body:
            try:
                # Convert markdown to HTML, but first fix any **text** that should be bold
                body_clean = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', body)
                body_html = markdown.markdown(body_clean, extensions=['nl2br'])
            except:
                # Fallback: simple line break conversion and bold formatting
                body_html = body.replace('\n', '<br>')
                body_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', body_html)
        else:
            body_html = '<em>No content</em>'

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
                    formatted_recipients.append(f'<a href="mailto:{email}" style="color: #0078d4; text-decoration: none;">{name} &lt;{email}&gt;</a>')
                else:
                    # Assume it's just an email address
                    formatted_recipients.append(f'<a href="mailto:{recipient}" style="color: #0078d4; text-decoration: none;">{recipient}</a>')

            if len(formatted_recipients) == 1:
                return formatted_recipients[0]
            elif len(formatted_recipients) <= 3:
                return ", ".join(formatted_recipients)
            else:
                # For more than 3 recipients, show first 2 and count
                return f"{formatted_recipients[0]}, {formatted_recipients[1]}, and {len(formatted_recipients) - 2} more"

        to_display = format_recipients(to_recipients)
        cc_display = format_recipients(cc_recipients)

        # Build header lines in standardized order: From, Sent, To, Cc, Subject
        header_lines = [
            f'<div class="email-header-line"><span class="email-header-label">From:</span><span class="email-header-value">{sender}</span></div>',
            f'<div class="email-header-line"><span class="email-header-label">Sent:</span><span class="email-header-value">{date}</span></div>'
        ]

        # Add To: field if recipients exist
        if to_recipients:
            header_lines.append(f'<div class="email-header-line"><span class="email-header-label">To:</span><span class="email-header-value">{to_display}</span></div>')

        # Add Cc: field if recipients exist
        if cc_recipients:
            header_lines.append(f'<div class="email-header-line"><span class="email-header-label">Cc:</span><span class="email-header-value">{cc_display}</span></div>')

        # Add Subject last
        header_lines.append(f'<div class="email-header-line"><span class="email-header-label">Subject:</span><span class="email-header-value">{subject}</span></div>')

        # Create email item with proper divider and enhanced header
        html_parts.append(f'''
        <div class="email-item">
            <div class="email-header">
                {"".join(header_lines)}
            </div>
            <div class="email-body">
                {body_html}
            </div>
        </div>
        ''')
    
    html_parts.append('</div></div>')
    return ''.join(html_parts)

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

def format_reply_content(text):
    """Format reply content as clean HTML with proper spacing and paragraph breaks"""
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

        # Apply custom paragraph styling to markdown-generated paragraphs (single line break spacing for Outlook compatibility)
        html_content = re.sub(r'<p>', '<p style="margin: 0 0 11pt 0; line-height: 1.15;">', html_content)
    except:
        # Fallback: enhanced formatting with proper paragraph breaks
        html_content = clean_text

        # Split into paragraphs (single line breaks for proper Outlook-style spacing)
        paragraphs = html_content.split('\n\n')
        formatted_paragraphs = []

        for para in paragraphs:
            if para.strip():
                # Skip paragraphs that look like subject lines
                if re.match(r'^\s*(Subject|RE):\s*', para.strip(), re.IGNORECASE):
                    continue

                # Convert single line breaks to <br> within paragraphs
                para_formatted = para.replace('\n', '<br>')
                # Convert **text** to <strong>text</strong>
                para_formatted = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', para_formatted)
                # Wrap in paragraph tags with single line break spacing (11pt margin for Outlook compatibility)
                formatted_paragraphs.append(f'<p style="margin: 0 0 11pt 0; line-height: 1.15;">{para_formatted}</p>')

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

def format_draft_preview(reply_text, original_email_info, user_email=""):
    """Format reply as a draft email preview showing the complete email structure"""
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
                    formatted.append(f'<a href="mailto:{email}" style="color: #0563C1; text-decoration: underline;">{name} &lt;{email}&gt;</a>')
                else:
                    # Assume it's just an email address
                    formatted.append(f'<a href="mailto:{recipient}" style="color: #0563C1; text-decoration: underline;">{recipient}</a>')
            return '; '.join(formatted)

        cc_display = format_email_links(reply_cc_recipients)

        # Format To recipient (original sender) with mailto link
        to_display = format_email_links([original_sender]) if original_sender != 'Unknown' else 'Unknown'

        # Create threaded content
        threaded_html, _ = create_threaded_email_content(reply_text, original_email_info)

        # Create draft preview with authentic Outlook styling
        draft_preview = f"""
<div class="output-panel-container">
    <div class="draft-preview-item">
        <div style="margin-bottom: 8px;">
            <strong style="color: #0078d4; font-size: 1.1em;">📧 Email Draft Preview</strong>
        </div>
        <div style="font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.15; background: white; border: 1px solid #d1d5db; border-radius: 4px; padding: 16px;">
            <div style="margin-bottom: 12px; border-bottom: 1px solid #e5e7eb; padding-bottom: 8px;">
                <div style="margin: 2px 0; font-size: 11pt;">
                    <span style="font-weight: bold; color: #000; min-width: 60px; display: inline-block;">To:</span>
                    <span style="color: #000;">{to_display}</span>
                </div>
                <div style="margin: 2px 0; font-size: 11pt;">
                    <span style="font-weight: bold; color: #000; min-width: 60px; display: inline-block;">Cc:</span>
                    <span style="color: #000;">{cc_display}</span>
                </div>
                <div style="margin: 2px 0; font-size: 11pt;">
                    <span style="font-weight: bold; color: #000; min-width: 60px; display: inline-block;">Subject:</span>
                    <span style="color: #000;">{reply_subject}</span>
                </div>
            </div>
            <div style="font-family: 'Microsoft Sans Serif', sans-serif; font-size: 11pt; line-height: 1.15; color: #000;">
                {threaded_html}
            </div>
        </div>
        <div style="margin-top: 16px; padding: 8px; background: #e7f3ff; border-radius: 4px; font-size: 0.85em; color: #0066cc;">
            💡 <strong>Ready to send:</strong> Click "📧 Export Draft Email" to download this as a draft email file that you can open in any email client and send directly.
        </div>
    </div>
</div>"""

        return draft_preview

    except Exception as e:
        print(f"Error creating draft preview: {e}")
        return format_reply_content(reply_text)

def create_threaded_email_content(reply_text, original_email_info):
    """Create a complete threaded email with reply and original content"""
    try:
        from bs4 import BeautifulSoup

        # Clean the reply text for both HTML and plain text
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

        # Get original email details with complete content preservation
        original_sender = original_email_info.get('sender', 'Unknown')
        original_date = standardize_date_format(original_email_info.get('date', 'Unknown'))
        original_subject = original_email_info.get('subject', '(No Subject)')
        original_body = original_email_info.get('body', '')
        original_html_body = original_email_info.get('html_body', '')
        to_recipients = original_email_info.get('to_recipients', [])
        cc_recipients = original_email_info.get('cc_recipients', [])

        # Use HTML body if available for complete content preservation
        if original_html_body:
            # Clean up HTML for email threading while preserving all content
            soup = BeautifulSoup(original_html_body, 'html.parser')

            # Remove any script tags for security
            for script in soup.find_all('script'):
                script.decompose()

            # Preserve all other HTML elements including images, tables, formatting
            original_body_for_threading = str(soup)
        else:
            # Fallback to plain text with basic HTML formatting
            original_body_for_threading = original_body.replace('\n', '<br>')

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
            soup = BeautifulSoup(formatted_reply_html, 'html.parser')

            # If wrapped in a single div, extract contents but preserve all inner HTML
            if soup.find('div', class_='reply-content') or soup.find('div', class_='draft-content'):
                content_div = soup.find('div', class_='reply-content') or soup.find('div', class_='draft-content')
                if content_div:
                    formatted_reply_html = str(content_div.decode_contents())

            # Ensure proper styling for email clients while preserving original formatting
            # Add email-safe CSS for lists and formatting elements
            formatted_reply_html = formatted_reply_html.replace('<ul>', '<ul style="margin: 16px 0; padding-left: 20px;">')
            formatted_reply_html = formatted_reply_html.replace('<ol>', '<ol style="margin: 16px 0; padding-left: 20px;">')
            formatted_reply_html = formatted_reply_html.replace('<li>', '<li style="margin: 4px 0;">')

        else:
            # Plain text - convert to HTML while preserving line breaks and structure
            paragraphs = clean_reply_html.split('\n\n')
            formatted_paragraphs = []

            for para in paragraphs:
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
                                list_items.append(f'<li style="margin: 4px 0;">{clean_line}</li>')
                        if list_items:
                            formatted_paragraphs.append(f'<ul style="margin: 0 0 11pt 0; padding-left: 20px;">{"".join(list_items)}</ul>')
                    else:
                        # Regular paragraph with Microsoft Sans Serif for AI-generated content
                        formatted_paragraphs.append(f'<p style="margin: 0 0 11pt 0; line-height: 1.15; font-family: \'Microsoft Sans Serif\', sans-serif; font-size: 11pt; color: #000;">{para_formatted}</p>')

            formatted_reply_html = ''.join(formatted_paragraphs)

        # Create threaded content with Microsoft Sans Serif for AI reply (single line break spacing)
        threaded_html = f"""
<div style="font-family: 'Microsoft Sans Serif', sans-serif; font-size: 11pt; line-height: 1.15; color: #000;">
{formatted_reply_html}
<div style="margin: 16px 0;">
<hr style="border: none; border-top: 1px solid #b5b5b5; margin: 16px 0;">
</div>

<div style="margin: 0; padding: 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000;">
<p style="margin: 0 0 0 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000; line-height: 1.15;"><strong>From:</strong> {original_sender}</p>
<p style="margin: 0 0 0 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000; line-height: 1.15;"><strong>Sent:</strong> {original_date}</p>"""

        if to_recipients:
            threaded_html += f'<p style="margin: 0 0 0 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000; line-height: 1.15;"><strong>To:</strong> {to_display}</p>'

        if cc_recipients:
            threaded_html += f'<p style="margin: 0 0 0 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000; line-height: 1.15;"><strong>Cc:</strong> {cc_display}</p>'

        # Add subject line to match Outlook format
        threaded_html += f'<p style="margin: 0 0 0 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000; line-height: 1.15;"><strong>Subject:</strong> {original_subject}</p>'

        # Add blank line after Subject (matching Outlook format)
        threaded_html += '<p style="margin: 0 0 0 0; font-family: Calibri, Arial, sans-serif; font-size: 11pt; color: #000000; line-height: 1.15;">&nbsp;</p>'

        # Add original email body content with Calibri font for authentic Outlook look
        threaded_html += f"""
<div style="margin-top: 0px; font-family: Calibri, Arial, sans-serif; font-size: 11pt; line-height: 1.15; color: #000000;">
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

        # Create threaded email content
        threaded_html, threaded_plain = create_threaded_email_content(reply_text, original_email_info)

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
            line-height: 1.15;
            color: #000000;
            margin: 0;
            padding: 0;
        }}
        a {{ color: #0563C1; text-decoration: underline; }}
        p {{ margin: 0 0 11pt 0; }}
        .original-email {{
            margin-top: 16px;
            border-top: 1px solid #b5b5b5;
            padding-top: 8px;
            font-family: Calibri, Arial, sans-serif;
        }}
        .quoted-header {{
            font-family: Calibri, Arial, sans-serif;
            font-size: 11pt;
            color: #000000;
            line-height: 1.15;
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
        msg = extract_msg.Message(temp_path)

        # Extract sender with proper name and email formatting
        raw_sender = msg.sender or "Unknown"

        # Clean up sender formatting - remove unnecessary quotes
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
                    # Try to get email separately
                    sender_email = getattr(msg, 'senderEmailAddress', None)
                    if sender_email:
                        sender = f"{name_part} <{sender_email}>"
                    else:
                        sender = name_part
            elif '<' not in raw_sender:
                # Try to get sender email separately if not included in sender name
                sender_email = getattr(msg, 'senderEmailAddress', None)
                if sender_email:
                    sender = f"{raw_sender} <{sender_email}>"
                else:
                    sender = raw_sender
            else:
                sender = raw_sender
        else:
            sender = raw_sender

        subject = msg.subject or "(No Subject)"
        # Apply standardize_date_format to ensure consistent Outlook-style formatting
        raw_date = msg.date or "Unknown"
        date = standardize_date_format(raw_date)
        body = msg.body or ""
        html_body = getattr(msg, 'htmlBody', None)

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
                # Get all attachments from the message
                attachments = []
                if hasattr(msg, 'attachments') and msg.attachments:
                    for attachment in msg.attachments:
                        if hasattr(attachment, 'data') and hasattr(attachment, 'longFilename'):
                            attachments.append({
                                'filename': attachment.longFilename or attachment.shortFilename,
                                'data': attachment.data,
                                'content_id': getattr(attachment, 'contentId', None)
                            })

                # Process HTML to preserve embedded images
                if attachments:
                    soup = BeautifulSoup(html_body, 'html.parser')

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
            "attachments": attachments if 'attachments' in locals() else []
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

with gr.Blocks(css=custom_css, title="SARA Compose - A prototype by Max Kwong") as demo:
    # Create banner HTML with embedded base64 icon (temporarily hidden)
    banner_html = f"""
    <div class="app-header" style="display: none;">
        <div class="header-content">
            {f'<img src="{sara_icon_data_url}" alt="SARA Logo" class="sara-icon">' if sara_icon_data_url else ''}
            <div class="header-text">
                <h1 class="main-title">SARA Compose</h1>
                <p class="subtitle">A prototype by Max Kwong</p>
            </div>
        </div>
    </div>
    """

    # Banner temporarily hidden as requested
    # gr.HTML(banner_html)

    # Dynamic Status Instructions Panel with 3-Stage Workflow
    def get_workflow_banner_html(active_stage=1):
        """Generate the 3-stage workflow banner HTML"""
        return f"""
        <div class="status-instructions-panel">
            <div class="workflow-stages">
                <div class="stage {'active' if active_stage == 1 else ''}">
                    <div class="stage-header">
                        <div class="stage-number">1</div>
                        <div class="stage-icon">📧</div>
                        <h3 class="stage-title">Upload Email</h3>
                    </div>
                    <div class="stage-description">
                        Upload your email to get started
                    </div>
                </div>

                <div class="stage {'active' if active_stage == 2 else ''}">
                    <div class="stage-header">
                        <div class="stage-number">2</div>
                        <div class="stage-icon">✍️</div>
                        <h3 class="stage-title">Add Key Messages</h3>
                    </div>
                    <div class="stage-description">
                        Enter your key points and let SARA craft your reply
                    </div>
                </div>

                <div class="stage {'active' if active_stage == 3 else ''}">
                    <div class="stage-header">
                        <div class="stage-number">3</div>
                        <div class="stage-icon">📋</div>
                        <h3 class="stage-title">Review & Download</h3>
                    </div>
                    <div class="stage-description">
                        Review, download, or refine your draft with SARA
                    </div>
                </div>
            </div>
        </div>
        """

    status_instructions = gr.HTML(
        value=get_workflow_banner_html(1),
        elem_id="status-instructions"
    )

    # Simplified localStorage persistence using Gradio BrowserState
    # This is more reliable than complex JavaScript
    preferences_state = gr.BrowserState({
        "user_name": "Max Kwong",
        "user_email": "mwmkwong@hkma.gov.hk",
        "ai_instructions": DEFAULT_AI_INSTRUCTIONS
    })







    with gr.Row(elem_classes=["main-layout-row"]):
        # Left column - Redesigned with collapsible accordions for progressive disclosure
        with gr.Column(scale=3, elem_classes=["email-preview-column"]):
            # File Upload Accordion - Open by default
            with gr.Accordion("📧 Upload Email File", open=True, elem_classes=["upload-accordion"]) as upload_accordion:
                file_input = gr.File(
                    label="Select .msg email file or drag & drop here",
                    file_types=[".msg"],
                    elem_classes=["integrated-file-upload"]
                )

            # SARA Thinking Process Accordion - Moved between Upload and Draft Response
            with gr.Accordion("💭 SARA Thinking Process", open=False, visible=False, elem_classes=["thinking-accordion"]) as think_accordion:
                think_output = gr.Markdown(value="")

            # Draft Preview Accordion - Closed by default, opens during generation
            with gr.Accordion("📝 SARA Draft Response", open=False, visible=False, elem_classes=["draft-accordion"]) as draft_accordion:
                draft_preview = gr.HTML(
                    value="""
                    <div class='draft-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>✍️</div>
                            <h3>SARA Draft Will Appear Here</h3>
                            <p>Upload an email and generate a response to see your SARA-composed draft</p>
                        </div>
                    </div>
                    """,
                    elem_classes=["draft-display-area"]
                )

                # Custom download button component - appears when email is generated
                download_button = gr.DownloadButton(
                    label="📥 Download Draft Email",
                    visible=False,
                    variant="primary",
                    size="lg",
                    elem_classes=["custom-download-button", "draft-download-button"]
                )

            # Original Email Accordion - Hidden by default, becomes visible after email upload
            with gr.Accordion("📧 Original Email", open=False, visible=False, elem_classes=["original-accordion"]) as original_accordion:
                original_email_display = gr.HTML(
                    value="""
                    <div class='email-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>📧</div>
                            <h3>Upload Email File Above</h3>
                            <p>Select your .msg email file to view the original email content</p>
                        </div>
                    </div>
                    """,
                    elem_classes=["original-email-display-area"]
                )

        # Right column - All controls and output (optimized workflow)
        with gr.Column(scale=2, elem_classes=["rhs-controls-column"]):
            # Top Section - Input Controls with Key Messages Accordion (Hidden by default, entire container)
            with gr.Group(elem_classes=["card"], visible=False) as key_messages_group:
                # Key Messages Accordion - Hidden by default, becomes visible after email upload
                with gr.Accordion("📝 Key Messages", open=False, elem_classes=["key-messages-accordion"]) as key_messages_accordion:
                    key_messages = gr.Textbox(
                        label="Enter your key messages",
                        placeholder="Enter the key messages you want to include in your reply...\n\nExample:\n• Thank them for their inquiry\n• Confirm the meeting time\n• Provide additional resources",
                        lines=6,
                        max_lines=12
                    )

                    # Generate button inside accordion - closer to input
                    generate_btn = gr.Button(
                        "🚀 Generate Reply",
                        interactive=True,
                        size="lg",
                        variant="primary",
                        elem_classes=["action-button", "generate-button-inline"]
                    )



            # SARA Processing Section - Removed (thinking process moved to left column)
            # Actions Section - Removed (download button moved to Draft Response accordion)

            # Personal Preferences Section - New section above Development Settings
            with gr.Accordion("👤 Personal Preferences", open=False, elem_classes=["personal-preferences"], elem_id="personal-preferences-accordion"):
                with gr.Group():
                    gr.HTML("<h4 style='margin: 0 0 12px 0; color: #64748b; font-size: 0.9rem;'>User Identity</h4>")
                    user_name = gr.Textbox(
                        label="Your Name",
                        placeholder="Enter your full name...",
                        value="Max Kwong",
                        interactive=True,
                        elem_id="user-name-input",
                        elem_classes=["preference-input"]
                    )
                    user_email = gr.Textbox(
                        label="Your Email Address",
                        placeholder="Enter your email address...",
                        value="mwmkwong@hkma.gov.hk",
                        interactive=True,
                        elem_id="user-email-input",
                        elem_classes=["preference-input"]
                    )



                with gr.Group():
                    gr.HTML("<h4 style='margin: 16px 0 12px 0; color: #64748b; font-size: 0.9rem;'>AI Instructions</h4>")

                    ai_instructions = gr.Textbox(
                        label="AI Instructions",
                        value=DEFAULT_AI_INSTRUCTIONS,
                        placeholder="Enter the complete instructions that will be sent to the AI model...",
                        lines=12,
                        max_lines=20,
                        interactive=True,  # Always editable
                        info="These are the exact instructions sent to the AI model. You have complete control over how the AI behaves. Only your identity context is added automatically.",
                        elem_id="ai-instructions-textarea",
                        elem_classes=["preference-textarea"]
                    )

                    # Restore Default Instructions button
                    restore_default_btn = gr.Button(
                        "🔄 Restore Default Instructions",
                        elem_classes=["action-button"],
                        size="sm",
                        variant="primary"
                    )

                    # Hidden HTML component for visual feedback
                    restore_feedback = gr.HTML(visible=False)

            # Bottom Section - Development Settings (moved to bottom for better UX hierarchy)
            with gr.Accordion("⚙️ Development Settings", open=False, elem_classes=["dev-settings"]):
                with gr.Group():
                    gr.HTML("<h4 style='margin: 0 0 12px 0; color: #64748b; font-size: 0.9rem;'>AI Model Configuration</h4>")

                    # Model selection - POE only for cloud deployment
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



    # State management
    current_reply = gr.State("")
    current_think = gr.State("")
    current_email_info = gr.State({})



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

    def load_preferences_on_startup(prefs_state):
        """Load saved preferences on application startup"""
        prefs = prefs_state
        return [
            gr.update(value=prefs.get("user_name", "Max Kwong")),
            gr.update(value=prefs.get("user_email", "mwmkwong@hkma.gov.hk")),
            gr.update(value=prefs.get("ai_instructions", DEFAULT_AI_INSTRUCTIONS))
        ]

    # Add a component to trigger preference loading on page load - temporarily disabled for debugging
    # preference_loader = gr.HTML(
    #     value="""
    #     <script>
    #     // Initialize localStorage persistence for AI instructions
    #     setTimeout(() => {
    #         console.log('🔄 Initializing AI instruction persistence');
    #
    #         // Load saved custom instructions
    #         const savedInstructions = loadCustomInstructions();
    #         if (savedInstructions) {
    #             const textarea = document.querySelector('#ai-instructions-textarea textarea');
    #             if (textarea) {
    #                 textarea.value = savedInstructions;
    #                 // Trigger change event to update Gradio state
    #                 textarea.dispatchEvent(new Event('input', { bubbles: true }));
    #                 console.log('Loaded saved custom instructions');
    #             }
    #         }
    #     }, 1000);
    #     </script>
    #     """,
    #     visible=False
    # )

    def extract_and_display_email(file):
        if not file:
            # Reset status instructions to initial state (Stage 1)
            initial_status_html = get_workflow_banner_html(1)

            return (
                gr.update(visible=True),  # Keep file input visible
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
                gr.update(open=True),   # Keep upload accordion open
                gr.update(open=False, visible=False),  # Keep original accordion closed and hidden
                gr.update(open=False, visible=False),  # Keep key messages accordion closed and hidden
                gr.update(visible=False),  # Hide entire key messages group container when no file
                {},
                initial_status_html     # Reset status instructions
            )

        info, error = process_msg_file(file)
        if error:
            # Error status - stay on Stage 1
            error_status_html = get_workflow_banner_html(1)

            return (
                gr.update(visible=True),  # Keep file input visible for retry
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
                gr.update(open=True),   # Keep upload accordion open for retry
                gr.update(open=False, visible=False),  # Keep original accordion closed and hidden
                gr.update(open=False, visible=False),  # Keep key messages accordion closed and hidden
                gr.update(visible=False),  # Hide entire key messages group container on error
                {},
                error_status_html       # Show error status
            )

        # Success status - move to Stage 2
        success_status_html = get_workflow_banner_html(2)

        preview_html = format_email_preview(info)
        # After successful email upload: display email content and hide upload section
        return (
            gr.update(visible=True),    # Keep file input visible
            preview_html,               # Original email display
            gr.update(open=False, visible=False),  # Hide upload accordion after successful upload
            gr.update(open=True, visible=True),    # Make original email accordion visible and open to show content
            gr.update(open=True, visible=True),    # Make key messages accordion visible and auto-expand
            gr.update(visible=True),    # Make entire key messages group container visible after successful upload
            info,
            success_status_html         # Show success status
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

    def copy_to_clipboard_js(reply_text):
        """Simplified for Hugging Face Spaces deployment"""
        # Return empty string to avoid JavaScript issues
        _ = reply_text  # Acknowledge parameter to avoid warning
        return ""

    def validate_inputs(file, key_msgs, model):
        if not file or not key_msgs or not model:
            return gr.update(interactive=False)
        return gr.update(interactive=True)



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

    def on_generate_stream(file, key_msgs, model, user_name, user_email, ai_instructions, email_token_limit):
        try:
            print(f"on_generate_stream called with file: {type(file)} {file}")
            if not file:
                # No file - back to Stage 1
                no_file_status_html = get_workflow_banner_html(1)

                yield (
                    gr.update(visible=True),  # Keep file input visible
                    """
                    <div class='draft-placeholder'>
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
                    gr.update(open=False, visible=False),  # Hide and close draft accordion
                    gr.update(open=False, visible=False),  # Close and hide original accordion
                    gr.update(open=False, visible=False),   # Close and hide key messages accordion
                    gr.update(visible=False),  # Hide entire key messages group container
                    "",  # Clear thinking content
                    gr.update(visible=False, value=None),  # Hide download file
                    "",  # Clear current_reply state
                    "",  # Clear current_think state
                    no_file_status_html  # Show no file status
                )
                return
            # Check if POE API is healthy
            current_backend = backend_manager.get_current_backend()
            if not current_backend.is_healthy():
                # API unavailable - stay on Stage 2
                api_unavailable_status_html = get_workflow_banner_html(2)

                yield (
                    gr.update(visible=True),   # Keep file input visible for retry
                    """
                    <div class='draft-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>❌</div>
                            <h3>API Unavailable</h3>
                            <p>POE API service is not healthy</p>
                        </div>
                    </div>
                    """,
                    format_email_preview({}),
                    gr.update(visible=False),  # Hide thinking accordion
                    gr.update(open=False, visible=False),  # Hide and close draft accordion
                    gr.update(open=False, visible=False),  # Close and hide original accordion
                    gr.update(open=False, visible=False),   # Close and hide key messages accordion
                    gr.update(visible=False),  # Hide entire key messages group container
                    "",  # Clear thinking content
                    gr.update(visible=False, value=None),  # Hide download file
                    "",  # Clear current_reply state
                    "",  # Clear current_think state
                    api_unavailable_status_html  # Show API unavailable status
                )
                return

            info, error = process_msg_file(file)
            print(f"process_msg_file returned info: {info}, error: {error}")
            if error:
                # Processing error - back to Stage 1
                processing_error_status_html = get_workflow_banner_html(1)

                yield (
                    gr.update(visible=True),   # Keep file input visible for retry
                    """
                    <div class='draft-placeholder'>
                        <div class='placeholder-content'>
                            <div class='placeholder-icon'>❌</div>
                            <h3>Processing Error</h3>
                            <p>Error processing email file</p>
                        </div>
                    </div>
                    """,
                    format_email_preview({}),
                    gr.update(visible=False),  # Hide thinking accordion
                    gr.update(open=False, visible=False),  # Hide and close draft accordion
                    gr.update(open=False, visible=False),  # Close and hide original accordion
                    gr.update(open=False, visible=False),   # Close and hide key messages accordion
                    gr.update(visible=False),  # Hide entire key messages group container
                    "",  # Clear thinking content
                    gr.update(visible=False, value=None),  # Hide download file
                    "",  # Clear current_reply state
                    "",  # Clear current_think state
                    processing_error_status_html  # Show processing error status
                )
                return

            # Show original email in bottom section, hide file upload
            original_email_preview = format_email_preview(info)

            prompt = build_prompt(info, key_msgs, user_name, ai_instructions, email_token_limit)

            # Initialize streaming - open draft accordion and show generation status
            full_response = ""

            # Initial draft area - pure content placeholder, no status messages
            initial_draft_status = """
            <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 1px solid #e5e7eb;'>
                <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.6; color: #9ca3af; text-align: center; padding: 40px 20px; font-size: 11pt;'>
                    <div style='font-size: 1.1em;'>Generating your email response...</div>
                </div>
            </div>
            """

            # Generation status - stay on Stage 2 during generation
            generating_status_html = get_workflow_banner_html(2)

            # After Generate Reply: make draft accordion visible and open to show streaming content
            yield (
                gr.update(visible=True),        # Keep file input visible for persistent email access
                initial_draft_status,           # Show generation status in draft area
                original_email_preview,         # Keep original email visible
                gr.update(visible=False),       # Hide thinking accordion initially
                gr.update(open=True, visible=True),  # Make draft accordion visible and open to show generation
                gr.update(open=True, visible=True),  # Make original email accordion visible and open
                gr.update(open=True, visible=True),  # Make key messages accordion visible and open
                gr.update(visible=True),  # Make entire key messages group container visible
                "",                             # Clear thinking content initially
                gr.update(visible=False, value=None),  # Hide download file
                "",                             # Clear current_reply state
                "",                             # Clear current_think state
                generating_status_html          # Show generating status
            )

            # Stream the response using the current backend
            for chunk, done in current_backend.stream_response(prompt, model):
                full_response += chunk

                if not done:
                    # Extract think content and main reply for streaming
                    main_reply, think_content = extract_and_separate_think_content(full_response)

                    # During streaming: show real-time content in draft area - pure content only
                    if main_reply.strip():
                        # Show actual streaming content in draft area - clean, no status headers
                        draft_content = f"""
                        <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 2px solid #f97316;'>
                            <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.6; color: #374151; font-size: 11pt;'>
                                {format_reply_content(main_reply)}
                            </div>
                        </div>
                        """


                    else:
                        # Still processing - show clean loading state
                        draft_content = """
                        <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 1px solid #e5e7eb;'>
                            <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.6; color: #9ca3af; text-align: center; padding: 40px 20px; font-size: 11pt;'>
                                <div style='font-size: 1.1em;'>Processing your request...</div>
                            </div>
                        </div>
                        """



                    # Show/hide think accordion based on content with auto-scroll - ONLY thinking content
                    think_visible = think_content is not None and len(think_content.strip()) > 0
                    think_display = think_content if think_visible else ""

                    # Auto-scroll JavaScript removed for Hugging Face Spaces compatibility

                    # Update status instructions during streaming - stay on Stage 2
                    streaming_status_html = get_workflow_banner_html(2)

                    # Stream to draft area, keep draft accordion open, show thinking if available
                    yield (
                        gr.update(visible=True),            # Keep file input visible for persistent email access
                        draft_content,                      # Show streaming content in draft area
                        original_email_preview,             # Keep original email visible
                        gr.update(visible=think_visible, open=think_visible),  # Show/hide thinking accordion
                        gr.update(open=True, visible=True), # Keep draft accordion visible and open during streaming
                        gr.update(open=True, visible=True), # Keep original email accordion visible and open
                        gr.update(open=True, visible=True), # Keep key messages accordion visible and open
                        gr.update(visible=True), # Keep entire key messages group container visible
                        think_display,                      # Show thinking content if available
                        gr.update(visible=False, value=None),  # Hide download file during generation
                        main_reply,                         # Update current_reply state
                        think_content or "",                # Update current_think state
                        streaming_status_html               # Show streaming status with hints
                    )
                else:
                    # Final response - show complete draft in draft area, keep original email in bottom section
                    main_reply, think_content = extract_and_separate_think_content(full_response)

                    # Format the final draft for the draft preview area - pure content only
                    final_draft_content = f"""
                    <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 2px solid #f97316;'>
                        <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.6; color: #374151; font-size: 11pt;'>
                            {format_reply_content(main_reply)}
                        </div>
                    </div>
                    """



                    # Show/hide think accordion based on content - automatically collapse after completion
                    think_visible = think_content is not None and len(think_content.strip()) > 0
                    think_display = think_content if think_visible else ""

                    # Completion status instructions - all completion status moved here
                    # Completion status - move to Stage 3
                    completion_status_html = get_workflow_banner_html(3)

                    # Automatically generate download file when generation completes
                    download_file_update = generate_download_file(main_reply, info, user_email, user_name)

                    # Final state: draft in top section with accordion open, original email in bottom section
                    yield (
                        gr.update(visible=True),            # Keep file input visible for persistent email access
                        final_draft_content,                # Show final draft content
                        original_email_preview,             # Keep original email visible
                        gr.update(visible=think_visible, open=False),  # Show thinking accordion but collapsed
                        gr.update(open=True, visible=True), # Keep draft accordion visible and open to show final result
                        gr.update(open=True, visible=True), # Keep original email accordion visible and open
                        gr.update(open=True, visible=True), # Keep key messages accordion visible and open
                        gr.update(visible=True), # Keep entire key messages group container visible
                        think_display,                      # Show thinking content if available
                        download_file_update,               # Show download file
                        main_reply,                         # Update current_reply state
                        think_content or "",                # Update current_think state
                        completion_status_html              # Show completion status
                    )
                    return
            
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"Exception in on_generate_stream: {e}\n{tb}")

            error_draft = """
            <div style='padding: 20px; background: white; border-radius: 8px; margin: 20px; border: 1px solid #ef4444;'>
                <div style='font-family: "Microsoft Sans Serif", sans-serif; line-height: 1.6; color: #dc2626; text-align: center; padding: 40px 20px; font-size: 11pt;'>
                    <div style='font-size: 1.1em;'>Generation failed. Please try again.</div>
                </div>
            </div>
            """

            # Error generation status - stay on Stage 2
            error_generation_status_html = get_workflow_banner_html(2)

            yield (
                gr.update(visible=True),            # Show file input for retry
                error_draft,                        # Show error in draft area
                format_email_preview({}),           # Clear original email area
                gr.update(visible=False),           # Hide thinking accordion
                gr.update(open=False, visible=False),  # Hide and close draft accordion
                gr.update(open=False, visible=False),  # Close and hide original accordion
                gr.update(open=False, visible=False),  # Close and hide key messages accordion
                gr.update(visible=False),  # Hide entire key messages group container
                "",                                 # Clear thinking content
                gr.update(visible=False, value=None),  # Hide download file
                "",                                 # Clear current_reply state
                "",                                 # Clear current_think state
                error_generation_status_html        # Show error status
            )

    # Event handlers - Updated for new component order: Upload → Thinking → Draft → Original
    file_input.change(extract_and_display_email, inputs=file_input, outputs=[file_input, original_email_display, upload_accordion, original_accordion, key_messages_accordion, key_messages_group, current_email_info, status_instructions])
    file_input.change(validate_inputs, inputs=[file_input, key_messages, model_selector], outputs=generate_btn)
    key_messages.change(validate_inputs, inputs=[file_input, key_messages, model_selector], outputs=generate_btn)
    model_selector.change(validate_inputs, inputs=[file_input, key_messages, model_selector], outputs=generate_btn)



    # Restore default instructions button handler
    restore_default_btn.click(restore_default_instructions, outputs=[ai_instructions, restore_feedback])

    # AI instructions text area change handler for saving custom instructions
    ai_instructions.change(save_custom_instructions_on_change, inputs=ai_instructions, outputs=restore_feedback)

    generate_btn.click(on_generate_stream, inputs=[file_input, key_messages, model_selector, user_name, user_email, ai_instructions, email_token_limit], outputs=[file_input, draft_preview, original_email_display, think_accordion, draft_accordion, original_accordion, key_messages_accordion, key_messages_group, think_output, download_button, current_reply, current_think, status_instructions])

    # Preference persistence using BrowserState - reliable localStorage alternative
    user_name.change(save_user_name, inputs=[user_name, preferences_state], outputs=preferences_state)
    user_email.change(save_user_email, inputs=[user_email, preferences_state], outputs=preferences_state)
    ai_instructions.change(save_ai_instructions, inputs=[ai_instructions, preferences_state], outputs=preferences_state)

    # Load preferences on startup
    demo.load(load_preferences_on_startup, inputs=preferences_state, outputs=[user_name, user_email, ai_instructions])

if __name__ == "__main__":
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