# Video Generation System

A comprehensive system for automated video production with advanced features and multiple service providers.

## Features

- **Script Generation**: Generate engaging video scripts using multiple LLM providers
  - Google Gemini
  - DeepSeek
  - OpenAI GPT
  - Local HuggingFace models
  
- **Audio Generation**: Convert scripts to high-quality speech using multiple TTS providers
  - Google Cloud TTS
  - ElevenLabs
  - Coqui TTS (offline)
  
- **Visual Generation**: Create compelling visuals using state-of-the-art models
  - FLUX.1
  - Stable Diffusion
  - DALL·E
  
- **Video Assembly**: Combine all elements into polished videos
  - Automated synchronization
  - Caption generation
  - Background music
  - Professional transitions

## Project Structure

```
src/
├── core/                    # Core system components
│   ├── config.py           # Configuration management
│   ├── errors.py           # Error handling and logging
│   └── cache.py            # Caching functionality
├── providers/              # Service providers
│   ├── base.py            # Base provider classes
│   ├── audio.py           # Audio providers
│   ├── image.py           # Image generation providers
│   └── llm.py             # Language model providers
├── generation/             # Content generation
│   ├── audio.py           # Unified audio generation
│   ├── visual.py          # Visual content generation
│   └── script.py          # Script generation
└── features/              # Higher-level features
    ├── captions.py        # Caption generation
    ├── preferences.py     # Video preferences
    ├── workflow.py        # Workflow orchestration
    ├── music.py           # Background music
    ├── assembler.py       # Video assembly
    ├── synchronizer.py    # Video synchronization
    ├── concepts.py        # Concept extraction
    └── uploader.py        # YouTube upload
```

## Setup

1. Create a Python virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

4. Run tests:
```bash
pytest
```

## Usage

Basic example:
```python
from video_gen import VideoGenerator, VideoPreferences

# Initialize video generator
generator = VideoGenerator()

# Generate a video
video = await generator.create_video(
    topic="Python Programming Tips",
    style="educational",
    duration_minutes=5.0
)

# Upload to YouTube
await video.upload_to_youtube(
    title="Top Python Tips for 2024",
    description="Learn essential Python programming tips...",
    privacy="private"
)
```

## Development

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Write unit tests for new features
- Use type hints
- Handle errors appropriately
- Add proper logging
- Implement caching where beneficial

## Testing

Run tests with coverage:
```bash
pytest --cov=src tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details 