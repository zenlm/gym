# Copyright 2025 Zoo Labs Foundation Inc.
#
# Zoo Labs Foundation branded CSS for Gym WebUI
# Black monochromatic theme with Zoo branding
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

CSS = r"""
/* Zoo Labs Foundation Black Monochromatic Theme */

/* Base styles */
.gradio-container {
    background: #000000 !important;
    color: #ffffff !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Dark background for all components */
* {
    --body-background-fill: #000000 !important;
    --background-fill-primary: #0a0a0a !important;
    --background-fill-secondary: #111111 !important;
    --panel-background-fill: #0a0a0a !important;
    --input-background-fill: #0a0a0a !important;
    --input-background-fill-hover: #1a1a1a !important;
    --input-background-fill-focus: #1a1a1a !important;
    --button-secondary-background-fill: #1a1a1a !important;
    --button-secondary-background-fill-hover: #2a2a2a !important;
    --neutral-50: #f5f5f5 !important;
    --neutral-100: #e5e5e5 !important;
    --neutral-200: #d4d4d4 !important;
    --neutral-300: #a3a3a3 !important;
    --neutral-400: #737373 !important;
    --neutral-500: #525252 !important;
    --neutral-600: #404040 !important;
    --neutral-700: #262626 !important;
    --neutral-800: #171717 !important;
    --neutral-900: #0a0a0a !important;
    --neutral-950: #000000 !important;
    --body-text-color: #ffffff !important;
    --body-text-color-subdued: #a0a0a0 !important;
    --border-color-primary: #2a2a2a !important;
    --border-color-accent: #3a3a3a !important;
    --link-text-color: #ffffff !important;
    --link-text-color-hover: #cccccc !important;
    --block-label-text-color: #ffffff !important;
    --block-title-text-color: #ffffff !important;
    --loader-color: #ffffff !important;
    --input-placeholder-color: #666666 !important;
}

/* Zoo Labs Foundation Header */
#title {
    font-size: 48px !important;
    font-weight: 900 !important;
    background: linear-gradient(90deg, #ffffff 0%, #cccccc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    padding: 20px 0 !important;
    letter-spacing: -1px !important;
    text-shadow: 0 0 20px rgba(255, 255, 255, 0.1) !important;
}

#title::before {
    content: "🦁 " !important;
    -webkit-text-fill-color: white !important;
}

#subtitle {
    font-size: 18px !important;
    color: #808080 !important;
    text-align: center !important;
    padding-bottom: 20px !important;
    font-weight: 300 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

#subtitle::after {
    content: " | zoo.ngo" !important;
    color: #606060 !important;
}

/* Tabs styling */
.tabs {
    background: #000000 !important;
    border: none !important;
}

.tabs > .tab-nav {
    background: #0a0a0a !important;
    border-bottom: 1px solid #2a2a2a !important;
}

.tabs > .tab-nav > button {
    color: #808080 !important;
    background: transparent !important;
    border: none !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 12px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
}

.tabs > .tab-nav > button:hover {
    color: #ffffff !important;
    background: #1a1a1a !important;
}

.tabs > .tab-nav > button.selected {
    color: #ffffff !important;
    background: #000000 !important;
    border-bottom: 2px solid #ffffff !important;
}

/* Buttons */
.primary, .gr-button-primary {
    background: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}

.primary:hover, .gr-button-primary:hover {
    background: #cccccc !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 20px rgba(255, 255, 255, 0.2) !important;
}

.secondary, .gr-button-secondary {
    background: transparent !important;
    color: #ffffff !important;
    border: 1px solid #333333 !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    transition: all 0.3s ease !important;
}

.secondary:hover, .gr-button-secondary:hover {
    background: #1a1a1a !important;
    border-color: #666666 !important;
}

/* Input fields */
input, textarea, select {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #ffffff !important;
    padding: 10px !important;
    border-radius: 4px !important;
    transition: all 0.3s ease !important;
}

input:focus, textarea:focus, select:focus {
    background: #1a1a1a !important;
    border-color: #4a4a4a !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(255, 255, 255, 0.1) !important;
}

/* Cards and panels */
.gr-box, .gr-panel {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.5) !important;
}

/* Dropdown */
.dropdown {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #ffffff !important;
}

.dropdown ul {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
}

.dropdown li:hover {
    background: #1a1a1a !important;
}

/* Progress bars */
.progress {
    background: #1a1a1a !important;
}

.progress-bar {
    background: #ffffff !important;
}

/* Sliders */
input[type="range"] {
    background: transparent !important;
}

input[type="range"]::-webkit-slider-track {
    background: #1a1a1a !important;
    border-radius: 4px !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: #ffffff !important;
    border: none !important;
    border-radius: 50% !important;
}

/* Labels */
label {
    color: #a0a0a0 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-weight: 500 !important;
}

/* Footer */
#footer {
    text-align: center !important;
    padding: 40px 0 20px 0 !important;
    color: #606060 !important;
    font-size: 14px !important;
    border-top: 1px solid #1a1a1a !important;
    margin-top: 40px !important;
}

#footer a {
    color: #808080 !important;
    text-decoration: none !important;
    transition: color 0.3s ease !important;
}

#footer a:hover {
    color: #ffffff !important;
}

/* Chat interface */
.message {
    background: #0a0a0a !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 8px !important;
    padding: 12px !important;
    margin: 8px 0 !important;
}

.message.user {
    background: #1a1a1a !important;
    border-color: #2a2a2a !important;
}

.message.bot {
    background: #0a0a0a !important;
    border-left: 3px solid #ffffff !important;
}

/* Code blocks */
pre, code {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #ffffff !important;
    border-radius: 4px !important;
    font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Fira Code', monospace !important;
}

/* Scrollbars */
::-webkit-scrollbar {
    width: 10px !important;
    height: 10px !important;
}

::-webkit-scrollbar-track {
    background: #0a0a0a !important;
}

::-webkit-scrollbar-thumb {
    background: #2a2a2a !important;
    border-radius: 5px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: #3a3a3a !important;
}

/* Zoo Logo placeholder */
.zoo-logo {
    width: 60px !important;
    height: 60px !important;
    display: inline-block !important;
    background: url('@zooai/logo') no-repeat center !important;
    background-size: contain !important;
    filter: invert(1) !important;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.gradio-container > * {
    animation: fadeIn 0.5s ease-out !important;
}

/* Duplicate button styling */
.duplicate-button {
    background: #1a1a1a !important;
    color: #ffffff !important;
    border: 1px solid #2a2a2a !important;
    margin: 20px auto !important;
    display: block !important;
}

.duplicate-button:hover {
    background: #2a2a2a !important;
    border-color: #3a3a3a !important;
}

/* Error and warning messages */
.error {
    background: #1a0000 !important;
    border: 1px solid #330000 !important;
    color: #ff6666 !important;
}

.warning {
    background: #1a1a00 !important;
    border: 1px solid #333300 !important;
    color: #ffff66 !important;
}

.success {
    background: #001a00 !important;
    border: 1px solid #003300 !important;
    color: #66ff66 !important;
}
"""