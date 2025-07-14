# Omani-Voice-Only-Chatbot
A voice-only conversational AI designed to provide empathetic and culturally-aware mental health support in the Omani Arabic dialect. It leverages a dual-model architecture using GPT-4o for primary responses and Claude Opus 4 for real-time validation to ensure therapeutic and cultural accuracy. The entire pipeline, from speech-to-text to response synthesis, is handled by Azure Speech Services and presented in a Gradio web interface.
## Features

**Voice-Only Interaction:** Real-time, human like conversation.

**Authentic Omani Dialect:** Utilizes Azure's `ar-OM` and `en-US` for multi-language support and speech recognition for a natural feel.

**Dual-Model Architecture:** **GPT-4o** for initial response generation. **Claude Opus 4** for parallel validation of gpt's response, ensuring cultural and therapeutic safety.

**Context-Aware Memory:** Implements a rolling summary mechanism to maintain long-term context in conversations.

**Gradio Web Interface:** Simple and accessible UI for demonstration.

## Tech Stack

**Backend:** Python

**Web UI:** Gradio

**LLMs:** OpenAI (GPT-4o), Anthropic (Claude Opus 4)

**Speech Services:** Microsoft Azure Cognitive Services (STT & TTS)

For detailed documentation of the project, please refer to the report uploaded in the repository.

## Setup and Installation

Follow these steps to run the project locally.

### 1. Prerequisites

Python 3.9+

An account and API keys for:

OpenAI

Anthropic

Microsoft Azure (with access to Speech Services)

### 2. Clone the Repository

git clone https://github.com/your-username/Omani-Voice-Only-Chatbot.git

cd Omani-Voice-Only-Chatbot

### 3. Install Dependencies

#### Create and activate a virtual environment (optional but recommended)

python -m venv venv

source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

#### Install required packages

pip install -r requirements.txt

### 4. Configure Environment Variables

Create a file named .env in the root of the project directory. Add your API keys to this file:

OPENAI_API_KEY="sk-..."

ANTHROPIC_API_KEY="sk-ant-..."

AZURE_SPEECH_KEY="your_azure_key"

AZURE_SPEECH_REGION="your_azure_region"

### 5. Run the Application

python app.py
