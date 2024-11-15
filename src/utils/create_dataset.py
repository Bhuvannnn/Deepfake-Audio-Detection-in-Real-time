import torch
import torchaudio
from gtts import gTTS
import os
from pathlib import Path
from pydub import AudioSegment
import numpy as np

class DatasetCreator:
    def __init__(self, output_dir: str = 'data', sample_rate: int = 16000):
        self.output_dir = Path(output_dir)
        self.real_dir = self.output_dir / 'real'
        self.fake_dir = self.output_dir / 'fake'
        self.sample_rate = sample_rate
        
        # Create directories
        self.real_dir.mkdir(parents=True, exist_ok=True)
        self.fake_dir.mkdir(parents=True, exist_ok=True)
    
    def create_real_sample(self, text: str, filename: str):
        """Create a real audio sample using gTTS"""
        tts = gTTS(text=text, lang='en')
        output_path = self.real_dir / f"{filename}.wav"
        
        # Save as MP3 first
        temp_mp3 = self.real_dir / f"{filename}_temp.mp3"
        tts.save(str(temp_mp3))
        
        # Convert to WAV with specific sample rate
        audio = AudioSegment.from_mp3(str(temp_mp3))
        audio = audio.set_frame_rate(self.sample_rate)
        audio.export(str(output_path), format="wav")
        
        # Remove temporary MP3
        temp_mp3.unlink()
        
        return str(output_path)
    
    def create_fake_sample(self, audio_path: str, filename: str):
        """Create a fake sample by applying modifications"""
        audio = AudioSegment.from_wav(audio_path)
        
        # Ensure consistent sample rate
        audio = audio.set_frame_rate(self.sample_rate)
        
        # Apply modifications
        modified = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 1.2)
        })
        modified = modified.set_frame_rate(self.sample_rate)
        
        # Save fake sample
        output_path = self.fake_dir / f"{filename}.wav"
        modified.export(str(output_path), format="wav")
        return str(output_path)

def create_sample_dataset(size=50):
    """Create a small dataset of real and fake audio samples"""
    creator = DatasetCreator()
    
    # Sample texts (expanded for variety)
    texts = [
        "Hello, how are you today?",
        "This is a test of the audio system.",
        "Artificial intelligence is fascinating.",
        "I'm working on a deep learning project.",
        "The weather is beautiful today.",
        "Machine learning is transforming technology.",
        "Please verify your identity.",
        "This is a secure system check.",
        "Welcome to the voice recognition system.",
        "Thank you for participating in this test."
    ]
    
    print("Creating dataset...")
    for i in range(size):
        # Select random text
        text = np.random.choice(texts)
        
        # Create real sample
        print(f"Creating sample pair {i+1}/{size}")
        real_path = creator.create_real_sample(text, f"real_{i}")
        
        # Create fake version
        creator.create_fake_sample(real_path, f"fake_{i}")
    
    print(f"\nDataset creation completed!")
    print(f"Created {size} real and {size} fake samples")
    print(f"Total samples: {size * 2}")
    print(f"\nLocation: {creator.output_dir}")

if __name__ == "__main__":
    create_sample_dataset(50)  # Create 50 pairs (50 real + 50 fake = 100 total samples)