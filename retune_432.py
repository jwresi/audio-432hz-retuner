"""
Audio Instrument Separator & 432Hz Retuner
Separates instruments from a mix and retunes from 440Hz to 432Hz

Required packages:
pip install demucs librosa soundfile numpy scipy pydub

Usage:
python retune_432.py input_audio.wav output_audio.wav
"""

import os
import sys
import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from pathlib import Path
import subprocess
import tempfile
import shutil

class AudioRetuner432:
    def __init__(self, input_file, output_file):
        self.input_file = input_file
        self.output_file = output_file
        self.temp_dir = tempfile.mkdtemp()
        
        # Frequency ratio for 440Hz -> 432Hz conversion
        self.freq_ratio = 432.0 / 440.0  # ≈ 0.9818
        
    def separate_sources(self):
        """Separate audio into stems using Demucs"""
        print("🎵 Separating audio sources with Demucs...")
        print("This may take a few minutes depending on audio length...")
        
        try:
            # Run Demucs to separate sources
            # Using htdemucs model (high quality, includes vocals, drums, bass, other)
            cmd = [
                'demucs',
                '--two-stems=vocals',  # Can also use full separation
                '-o', self.temp_dir,
                self.input_file
            ]
            
            # For full separation (vocals, drums, bass, other):
            cmd = ['demucs', '-o', self.temp_dir, self.input_file]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                print("⚠️  Demucs not found or failed. Falling back to full mix retuning...")
                return None
                
            # Find the separated files
            model_name = 'htdemucs'
            audio_name = Path(self.input_file).stem
            separated_dir = Path(self.temp_dir) / model_name / audio_name
            
            if separated_dir.exists():
                stems = list(separated_dir.glob('*.wav'))
                print(f"✓ Separated into {len(stems)} stems: {[s.name for s in stems]}")
                return stems
            else:
                return None
                
        except Exception as e:
            print(f"⚠️  Separation failed: {e}")
            print("Continuing with full mix retuning...")
            return None
    
    def pitch_shift_432(self, audio, sr):
        """
        Shift pitch from 440Hz to 432Hz using high-quality time-stretching
        """
        print(f"   Retuning to 432Hz (ratio: {self.freq_ratio:.4f})...")
        
        # Calculate semitone shift
        # 432/440 = 0.9818, which is about -0.318 semitones
        n_steps = 12 * np.log2(self.freq_ratio)
        
        # Use librosa's pitch shift with high quality
        shifted = librosa.effects.pitch_shift(
            audio, 
            sr=sr, 
            n_steps=n_steps,
            bins_per_octave=12 * 4  # Higher resolution for better quality
        )
        
        return shifted
    
    def time_stretch_method(self, audio, sr):
        """
        Alternative method: time-stretch then resample
        This can sometimes preserve quality better
        """
        print(f"   Time-stretching method...")
        
        # Stretch time by inverse of frequency ratio
        stretched = librosa.effects.time_stretch(audio, rate=1.0/self.freq_ratio)
        
        # Resample back to original sample rate
        # This effectively changes pitch while maintaining duration
        target_length = len(audio)
        if len(stretched) != target_length:
            stretched = librosa.resample(
                stretched, 
                orig_sr=len(stretched), 
                target_sr=target_length
            )
        
        return stretched
    
    def process_stem(self, stem_file, method='pitch_shift'):
        """Process individual stem file"""
        print(f"📼 Processing: {Path(stem_file).name}")
        
        # Load audio
        audio, sr = librosa.load(stem_file, sr=None, mono=False)
        
        # Handle stereo
        if len(audio.shape) == 2:
            processed = np.array([
                self.pitch_shift_432(audio[0], sr),
                self.pitch_shift_432(audio[1], sr)
            ])
        else:
            processed = self.pitch_shift_432(audio, sr)
        
        return processed, sr
    
    def mix_stems(self, processed_stems):
        """Mix all processed stems back together"""
        print("🎚️  Mixing processed stems...")
        
        # Ensure all stems have same length
        max_length = max(stem.shape[-1] for stem, _ in processed_stems)
        
        mixed = None
        for stem_audio, sr in processed_stems:
            # Pad if necessary
            if stem_audio.shape[-1] < max_length:
                if len(stem_audio.shape) == 2:
                    pad_width = ((0, 0), (0, max_length - stem_audio.shape[-1]))
                else:
                    pad_width = (0, max_length - stem_audio.shape[-1])
                stem_audio = np.pad(stem_audio, pad_width, mode='constant')
            
            # Mix
            if mixed is None:
                mixed = stem_audio
            else:
                mixed += stem_audio
        
        # Normalize to prevent clipping
        max_val = np.abs(mixed).max()
        if max_val > 0:
            mixed = mixed * (0.95 / max_val)
        
        return mixed, sr
    
    def process_full_mix(self):
        """Process entire mix without separation"""
        print("📼 Processing full mix...")
        
        # Load audio
        audio, sr = librosa.load(self.input_file, sr=None, mono=False)
        
        # Handle stereo
        if len(audio.shape) == 2:
            processed = np.array([
                self.pitch_shift_432(audio[0], sr),
                self.pitch_shift_432(audio[1], sr)
            ])
        else:
            processed = self.pitch_shift_432(audio, sr)
        
        return processed, sr
    
    def run(self, use_separation=True):
        """Main processing pipeline"""
        print(f"\n{'='*60}")
        print("🎼 Audio Retuner: 440Hz → 432Hz")
        print(f"{'='*60}\n")
        print(f"Input:  {self.input_file}")
        print(f"Output: {self.output_file}\n")
        
        try:
            if use_separation:
                # Separate sources
                stems = self.separate_sources()
                
                if stems:
                    # Process each stem
                    processed_stems = []
                    for stem in stems:
                        processed, sr = self.process_stem(stem)
                        processed_stems.append((processed, sr))
                    
                    # Mix back together
                    final_audio, sr = self.mix_stems(processed_stems)
                else:
                    # Fallback to full mix
                    final_audio, sr = self.process_full_mix()
            else:
                # Process full mix directly
                final_audio, sr = self.process_full_mix()
            
            # Save output
            print(f"💾 Saving to: {self.output_file}")
            sf.write(self.output_file, final_audio.T, sr)
            
            print(f"\n{'='*60}")
            print("✓ Processing complete!")
            print(f"{'='*60}\n")
            
        finally:
            # Cleanup temp directory
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🗑️  Cleaned up temporary files")


def main():
    if len(sys.argv) < 2:
        print("Usage: python retune_432.py <input_file> [output_file] [--no-separation]")
        print("\nExample:")
        print("  python retune_432.py song.wav song_432hz.wav")
        print("  python retune_432.py song.wav song_432hz.wav --no-separation")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    use_separation = '--no-separation' not in sys.argv
    
    if not output_file:
        base = Path(input_file).stem
        ext = Path(input_file).suffix
        output_file = f"{base}_432hz{ext}"
    
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found!")
        sys.exit(1)
    
    # Create retuner and process
    retuner = AudioRetuner432(input_file, output_file)
    retuner.run(use_separation=use_separation)


if __name__ == "__main__":
    main()
