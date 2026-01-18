from imslib.audio import Audio
from imslib.mixer import Mixer
from imslib.note import NoteGenerator, Envelope
from imslib.wavegen import WaveGenerator, SpeedModulator
from imslib.wavesrc import WaveBuffer, WaveFile, make_wave_buffers
from envelope import Envelope2
import numpy as np

class LoopManager:
    def __init__(self, mixer):
        self.mixer = mixer
        self.recording = False
        self.current_frames = []
        self.loops = []
        self.sample_rate = Audio.sample_rate  # Assuming consistent sample rate

    def start_recording(self):
        """Start capturing audio data."""
        print("LoopManager: Starting recording...")
        self.recording = True
        self.current_frames = []  # Clear previous recording buffer

    def stop_recording(self):
        """Stop capturing audio data and create a new loop."""
        print("LoopManager: Stopping recording...")
        self.recording = False

        if not self.current_frames:
            print("LoopManager: No frames recorded. Skipping loop creation.")
            return

        # Combine all recorded frames
        recorded_data = np.concatenate(self.current_frames)

        # Process the recorded data to prevent distortion
        recorded_data = self.process_recorded_data(recorded_data)

        # Create a WaveBuffer and looping WaveGenerator
        num_frames = len(recorded_data) // 2  # Assuming stereo (2 channels)
        wave_buffer = WaveBuffer.from_data(recorded_data, num_frames)

        loop_gen = WaveGenerator(wave_buffer, loop=True)
        self.loops.append(loop_gen)

        # Add to mixer and play
        self.mixer.add(loop_gen)
        loop_gen.play()
        print("LoopManager: Loop added and playing.")

    def process_audio_frame(self, audio_frame):
        """Process each incoming audio frame during recording."""
        if not self.recording:
            return

        # Normalize the incoming frame to prevent clipping
        max_val = np.max(np.abs(audio_frame))
        if max_val > 1.0:
            audio_frame = audio_frame / max_val

        # Remove DC offset from the frame
        audio_frame = self.remove_dc_offset(audio_frame)

        # Append the processed frame to the buffer
        self.current_frames.append(audio_frame)

    def process_recorded_data(self, recorded_data):
        """Post-process the recorded data before creating a loop."""
        # Normalize the entire buffer to prevent clipping
        max_val = np.max(np.abs(recorded_data))
        if max_val > 1.0:
            recorded_data = recorded_data / max_val

        # Apply a crossfade to avoid clicks/pops at loop points
        recorded_data = self.crossfade_audio(recorded_data)

        # Remove DC offset
        recorded_data = self.remove_dc_offset(recorded_data)

        return recorded_data

    def remove_dc_offset(self, audio_data):
        """Remove DC offset from audio data."""
        return audio_data - np.mean(audio_data)
    
    def add_frames(self, frames):
        """Add audio frames to the current recording buffer."""
        if self.recording:
            self.current_frames.append(frames)

    def clear_loops(self):
        """Stop and remove all active loops."""
        print("LoopManager: Clearing all loops...")
        for loop in self.loops:
            self.mixer.remove(loop)
        self.loops = []
        print("LoopManager: All loops cleared.")

    def crossfade_audio(self, audio_data, crossfade_length=1024):
        """Apply crossfade at the loop start and end to prevent clicks/pops."""
        if len(audio_data) < crossfade_length * 2:
            return audio_data  # Skip crossfade for short audio

        fade_in = np.linspace(0, 1, crossfade_length)
        fade_out = np.linspace(1, 0, crossfade_length)

        # Apply fade-in and fade-out
        audio_data[:crossfade_length] *= fade_in
        audio_data[-crossfade_length:] *= fade_out

        return audio_data
