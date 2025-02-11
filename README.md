# AI-Powered Faceless Video Production System

A Python-based system for automatically creating faceless YouTube videos using AI services. The system uses DeepSeek AI for script generation, ElevenLabs for voice synthesis, and Flux AI for visual content generation.

## Features

- 🤖 AI-powered script generation optimized for video content
- 🗣️ Natural-sounding voiceovers using ElevenLabs
- 🎨 Dynamic visual content generation with Flux AI
- 🎬 Automatic video assembly with transitions and effects
- 📊 Detailed metadata and logging
- 🎯 Customizable styles, durations, and target audiences

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/faceless-video-producer.git
cd faceless-video-producer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your API keys:
   - Copy `.env.example` to `.env`
   - Add your API keys for DeepSeek, ElevenLabs, and Flux AI
   - Adjust other configuration settings as needed

## Usage

### Command Line Interface

Create a video using the command-line interface:

```bash
python -m src "Your Video Topic" \
    --style educational \
    --duration 5 \
    --audience beginners \
    --visual-style "modern minimalist"
```

Optional arguments:
- `--style`: Content style (informative, entertaining, educational)
- `--duration`: Target video duration in minutes
- `--audience`: Target audience
- `--voice-id`: ElevenLabs voice ID
- `--visual-style`: Style for generated visuals
- `--output`: Custom output path
- `--context`: Path to JSON file with additional context

### Python API

```python
from src.workflow import create_video

video_path = create_video(
    topic="Introduction to AI",
    style="educational",
    duration_minutes=5,
    target_audience="beginners",
    visual_style="modern tech",
    additional_context={
        "key_points": ["Point 1", "Point 2"],
        "tone": "friendly"
    }
)
```

## Project Structure

```
faceless-video-producer/
├── src/
│   ├── __init__.py
│   ├── __main__.py           # CLI entry point
│   ├── config.py             # Configuration management
│   ├── script_generator.py   # DeepSeek integration
│   ├── audio_generator.py    # ElevenLabs integration
│   ├── visual_generator.py   # Flux AI integration
│   ├── video_assembler.py    # Video assembly
│   └── workflow.py           # Main orchestrator
├── examples/
│   └── create_video.py       # Example usage
├── output/                   # Generated content
├── temp/                     # Temporary files
├── .env.example             # Environment template
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Configuration

The system uses environment variables for configuration. Copy `.env.example` to `.env` and set the following variables:

```
DEEPSEEK_API_KEY=your_deepseek_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key
FLUX_API_KEY=your_flux_api_key
```

Additional settings can be configured in `.env`:
- `OUTPUT_DIR`: Directory for generated content
- `TEMP_DIR`: Directory for temporary files
- `LOG_LEVEL`: Logging verbosity

## Output

The system generates:
1. A final video file (MP4 format)
2. A JSON metadata file containing:
   - Video title and sections
   - Duration information
   - Generated content paths
   - Additional metadata

## Logging

Logs are stored in `output/workflow.log` with the following features:
- Daily rotation
- 7-day retention
- Configurable log level
- Detailed error tracking

## Error Handling

The system includes comprehensive error handling:
- API error recovery
- Resource cleanup
- Detailed error logging
- User-friendly error messages

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DeepSeek AI for script generation
- ElevenLabs for voice synthesis
- Flux AI for visual content generation
- MoviePy for video processing 