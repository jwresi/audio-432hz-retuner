import os
from retune_432 import AudioRetuner432

# Test with a sample file
input_file = "test_audio/sample.wav"
output_file = "output/sample_432hz.wav"

if os.path.exists(input_file):
    print("Testing audio retuner...")
    retuner = AudioRetuner432(input_file, output_file)
    retuner.run(use_separation=False)  # Start without separation for speed
    print(f"\n✓ Output saved to: {output_file}")
else:
    print(f"Place a test audio file at: {input_file}")
