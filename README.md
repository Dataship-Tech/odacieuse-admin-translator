# Blaxel OpenAI Agent

<p align="center">
  <img src="https://blaxel.ai/logo.png" alt="Blaxel" width="200"/>
</p>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI](https://img.shields.io/badge/OpenAI-powered-brightgreen.svg)](https://openai.com/)
[![GPT-4](https://img.shields.io/badge/GPT--4-enabled-orange.svg)](https://openai.com/gpt-4)

</div>

A template implementation of a conversational agent using OpenAI Assistants API and GPT-4. This agent demonstrates the power of OpenAI Assistants for building interactive AI agents with advanced tool integration capabilities and streaming responses.

## 📑 Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Running Locally](#running-the-server-locally)
  - [Testing](#testing-your-agent)
  - [Deployment](#deploying-to-blaxel)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Support](#support)
- [License](#license)

## ✨ Features

- Interactive conversational interface using OpenAI Assistants API
- Tool integration support (including weather and search capabilities)
- Streaming responses for real-time interaction
- Built on OpenAI Assistants for sophisticated agent orchestration
- Function calling capabilities with custom tools
- Easy deployment and integration with Blaxel platform

## 🚀 Quick Start

For those who want to get up and running quickly:

```bash
# Clone the repository
git clone https://github.com/blaxel-ai/template-openai-py.git

# Navigate to the project directory
cd template-openai-py

# Install dependencies
uv sync

# Start the server
bl serve --hotreload

# In another terminal, test the agent
bl chat --local blaxel-agent
```

## 📋 Prerequisites

- **Python:** 3.10 or later
- **[UV](https://github.com/astral-sh/uv):** An extremely fast Python package and project manager, written in Rust
- **Blaxel Platform Setup:** Complete Blaxel setup by following the [quickstart guide](https://docs.blaxel.ai/Get-started#quickstart)
  - **[Blaxel CLI](https://docs.blaxel.ai/Get-started):** Ensure you have the Blaxel CLI installed. If not, install it globally:
    ```bash
    curl -fsSL https://raw.githubusercontent.com/blaxel-ai/toolkit/main/install.sh | BINDIR=/usr/local/bin sudo -E sh
    ```
  - **Blaxel login:** Login to Blaxel platform
    ```bash
    bl login YOUR-WORKSPACE
    ```

## 💻 Installation

**Clone the repository and install dependencies:**

```bash
git clone https://github.com/blaxel-ai/template-openai-py.git
cd template-openai-py
uv sync
```
## 🔧 Usage

### Running Locally

Start the development server with hot reloading:

```bash
bl serve --hotreload
```

For production run:

```bash
bl serve
```

_Note:_ The development server automatically restarts when you make changes to the source code.

### Testing

You can test your OpenAI agent locally:

```bash
# Using the Blaxel CLI chat interface
bl chat --local blaxel-agent
```

Example queries you can test:

```
What is the weather in Paris?
```

```
Help me analyze some data
```

```
Can you assist with writing a summary?
```

You can also run it directly with specific input:

```bash
bl run agent blaxel-agent --local --data '{"input": "What is the weather in Paris?"}'
```

### Deployment

When you are ready to deploy your agent:

```bash
bl deploy
```

This command uses your code and the configuration files under the `.blaxel` directory to deploy your OpenAI agent on the Blaxel platform.

## 📁 Project Structure

- **src/main.py** - Application entry point and FastAPI server setup
- **src/agent.py** - Core agent implementation with OpenAI Assistants integration
- **src/server/** - Server implementation and routing
  - **router.py** - API route definitions
  - **middleware.py** - Request/response middleware
- **pyproject.toml** - UV package manager configuration with dependencies
- **blaxel.toml** - Blaxel deployment configuration
- **.env.example** - Environment variables template
- **LICENSE** - MIT license file

## ❓ Troubleshooting

### Common Issues

1. **Blaxel Platform Issues**:
   - Ensure you're logged in to your workspace: `bl login MY-WORKSPACE`
   - Verify models are available: `bl get models`
   - Check that functions exist: `bl get functions`

2. **OpenAI API Connection Issues**:
   - Verify your OpenAI API key is valid and has sufficient credits
   - Check API rate limits and usage quotas
   - Ensure network connectivity to OpenAI endpoints
   - Test API access with a simple request

3. **Assistant Configuration Issues**:
   - Verify assistant creation and configuration in the code
   - Check assistant permissions and capabilities
   - Ensure tools are properly registered with the assistant
   - Monitor assistant usage and performance

4. **Function Calling Problems**:
   - Verify tool function definitions and schemas
   - Check function parameter validation
   - Ensure function execution permissions
   - Review function response formats

5. **Python Environment Issues**:
   - Make sure you have Python 3.10+
   - Try `uv sync --upgrade` to update dependencies
   - Check for conflicting package versions
   - Verify virtual environment activation with UV

For more help, please [submit an issue](https://github.com/blaxel-templates/template-openai-py/issues) on GitHub.

## 👥 Contributing

Contributions are welcome! Here's how you can contribute:

1. **Fork** the repository
2. **Create** a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit** your changes:
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push** to the branch:
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Submit** a Pull Request

Please make sure to update tests as appropriate and follow the code style of the project.

## 🆘 Support

If you need help with this template:

- [Submit an issue](https://github.com/blaxel-templates/template-openai-py/issues) for bug reports or feature requests
- Visit the [Blaxel Documentation](https://docs.blaxel.ai) for platform guidance
- Join our [Discord Community](https://discord.gg/G3NqzUPcHP) for real-time assistance

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
