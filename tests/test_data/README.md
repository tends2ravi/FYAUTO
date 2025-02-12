# Test Data

This directory contains test data files used in unit tests.

## Files

- `OpenSans-Regular.ttf`: Font file for testing caption system
- `test_video.mp4`: Sample video file for testing video processing
- `test_image.png`: Sample image file for testing image processing
- `test_audio.mp3`: Sample audio file for testing audio processing

## Usage

These files are used by the test fixtures in `conftest.py` to provide consistent test data across all test suites.

## Adding New Test Files

When adding new test files:
1. Keep files small and minimal
2. Document the file's purpose in this README
3. Add appropriate fixtures in `conftest.py`
4. Use descriptive filenames 