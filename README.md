# Speech-to-Text Conversion using Whisper

This project leverages the Whisper speech recognition model from OpenAI to convert audio files into text. It utilizes the Hugging Face Transformers library and the Gradio library for creating a simple user interface.

## Prerequisites

Before running the code, make sure you have the following dependencies installed:

- Python 3.6 or higher
- PyTorch
- Transformers
- Datasets
- Gradio

You can install the required Python packages using pip:
```pip install requirements.txt```

## Usage

1. Clone the repository or copy the provided code into a Python file (e.g., `speech_to_text.py`).

2. Run the Python script:
`python speech_to_text.py`

3. A Gradio interface will open in your default web browser.

4. Click the "Choose File" button and select an audio file (e.g., .mp3, .wav) you want to convert to text.

5. Click the "Convert to text!" button.

6. The transcribed text will appear in the text box below.

## Code Explanation

The code performs the following steps:

1. Import necessary libraries and modules.
2. Set the device (CPU or GPU) and data type for PyTorch operations.
3. Load the Whisper-large-v3 model from the Hugging Face Model Hub.
4. Create a pipeline for automatic speech recognition using the loaded model and processor.
5. Load a sample audio file from the LibriSpeech dataset for demonstration purposes.
6. Define a function `speech_to_text` that takes an audio file as input and returns the transcribed text using the Whisper pipeline.
7. Create a Gradio interface with an audio file input, a text output, and a button to trigger the speech-to-text conversion.
8. Launch the Gradio interface, which will open in your default web browser.


## Customization

You can customize the code by modifying the following parameters:

- `model_id`: Change the Whisper model size (e.g., "openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small").
- `torch_dtype`: Adjust the data type for PyTorch operations based on your system's capabilities.
- `max_new_tokens`, `chunk_length_s`, `batch_size`: Modify these parameters to control the transcription process.

Additionally, you can explore the Whisper pipeline's documentation for more advanced options and configurations.
