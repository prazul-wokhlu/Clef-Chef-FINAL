import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd
import mido
import UI as UI
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.graphics import Ellipse, Color, Line
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.button import Button
import numpy as np
import librosa
from scipy.signal import stft, istft
from scipy.ndimage import gaussian_filter
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from kivy.uix.popup import Popup
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from scipy.signal import lfilter

from kivy.uix.boxlayout import BoxLayout
from kivy.uix.slider import Slider
from kivy.uix.label import Label

from imslib.audio import Audio
from imslib.mixer import Mixer
from imslib.note import NoteGenerator, Envelope
from imslib.wavegen import WaveGenerator, SpeedModulator
from imslib.wavesrc import WaveBuffer, WaveFile, make_wave_buffers
from envelope import Envelope2

# Audio loading function
def load_audio(file_path, sample_rate=44100):
    """Load audio file and return audio data and sample rate."""
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr


def spectral_blend3(audio_list, sr, alpha1=0.5, alpha2=0.5):
    # Step 1: Perform STFT for each audio signal
    print(audio_list)
    _, _, Zxx1 = stft(audio_list[0], fs=sr, nperseg=1024)
    _, _, Zxx2 = stft(audio_list[1], fs=sr, nperseg=1024)
    _, _, Zxx3 = stft(audio_list[2], fs=sr, nperseg=1024)
    _, _, Zxx4 = stft(audio_list[3], fs=sr, nperseg=1024)
    _, _, Zxx5 = stft(audio_list[4], fs=sr, nperseg=1024)
    _, _, Zxx6 = stft(audio_list[5], fs=sr, nperseg=1024)
    _, _, Zxx7 = stft(audio_list[6], fs=sr, nperseg=1024)
    _, _, Zxx8 = stft(audio_list[7], fs=sr, nperseg=1024)
    _, _, Zxx9 = stft(audio_list[8], fs=sr, nperseg=1024)
    _, _, Zxx10 = stft(audio_list[9], fs=sr, nperseg=1024)

    # Ensure all spectrograms have the same shape by matching time bins
    min_time_bins = min(Zxx1.shape[1], Zxx2.shape[1], Zxx3.shape[1])
    Zxx1, Zxx2, Zxx3, Zxx4, Zxx5 = Zxx1[:, :min_time_bins], Zxx2[:, :min_time_bins], Zxx3[:, :min_time_bins], Zxx4[:, :min_time_bins], Zxx5[:, :min_time_bins]
    Zxx6, Zxx7, Zxx8, Zxx9, Zxx10 = Zxx6[:, :min_time_bins], Zxx7[:, :min_time_bins], Zxx8[:, :min_time_bins], Zxx9[:, :min_time_bins], Zxx10[:, :min_time_bins]

    # Step 2: Blend audio1 and audio2 first
    # mag1, mag2 = np.abs(Zxx1), np.abs(Zxx2)
    # phase_weight1 = mag1 / (mag1 + mag2 + 1e-10)  # Avoid division by zero

    # blended_magnitude1 = (1 - alpha1) * mag1 + alpha1 * mag2
    # blended_phase1 = phase_weight1 * np.angle(Zxx1) + (1 - phase_weight1) * np.angle(Zxx2)
    # blended_Zxx1_2 = blended_magnitude1 * np.exp(1j * blended_phase1)

    # # Step 3: Blend the result of (audio1 + audio2) with audio3
    # mag_blended1_2, mag3 = np.abs(blended_Zxx1_2), np.abs(Zxx3)
    # phase_weight2 = mag_blended1_2 / (mag_blended1_2 + mag3 + 1e-10)  # Avoid division by zero

    # # blended_magnitude_final = (1 - alpha2) * mag_blended1_2 + alpha2 * mag3
    # # blended_phase_final = phase_weight2 * np.angle(blended_Zxx1_2) + (1 - phase_weight2) * np.angle(Zxx3)
    # # blended_Zxx_final = blended_magnitude_final * np.exp(1j * blended_phase_final)
    
    # ############################################################    testing more sounds
    # blended_magnitude2 = (1 - alpha1) * mag_blended1_2 + alpha1 * mag3
    # blended_phase2 = phase_weight2 * np.angle(blended_Zxx1_2) + (1 - phase_weight2) * np.angle(Zxx3)
    # blended_Zxx1_2_3 = blended_magnitude2 * np.exp(1j * blended_phase2)
    
    # # # Step 4: Blend (audio1 + audio2 + audio3) with audio4
    # mag1_2_3, mag4 = np.abs(blended_Zxx1_2_3), np.abs(Zxx4)
    # phase_weight3 = mag1_2_3 / (mag1_2_3 + mag4 + 1e-10)

    # blended_magnitude3 = (1 - alpha1) * mag1_2_3 + alpha1 * mag4
    # blended_phase3 = phase_weight3 * np.angle(blended_Zxx1_2_3) + (1 - phase_weight3) * np.angle(Zxx4)
    # blended_Zxx1_2_3_4 = blended_magnitude3 * np.exp(1j * blended_phase3)
    
    # # # Step 5: Blend (audio1 + audio2 + audio3 + audio4) with audio5
    # mag1_2_3_4, mag5 = np.abs(blended_Zxx1_2_3_4), np.abs(Zxx5)
    # phase_weight4 = mag1_2_3_4 / (mag1_2_3_4 + mag5 + 1e-10)

    # blended_magnitude4 = (1 - alpha1) * mag1_2_3_4 + alpha1 * mag5
    # blended_phase4 = phase_weight4 * np.angle(blended_Zxx1_2_3_4) + (1 - phase_weight4) * np.angle(Zxx5)
    # blended_Zxx1_2_3_4_5 = blended_magnitude4 * np.exp(1j * blended_phase4)
    # Step 1: Start with the first two audio signals (audio1 and audio2)
    mag1, mag2 = np.abs(Zxx1), np.abs(Zxx2)
    phase_weight1 = mag1 / (mag1 + mag2 + 1e-10)  # Avoid division by zero
    blended_magnitude1 = (1 - alpha1) * mag1 + alpha1 * mag2
    blended_phase1 = phase_weight1 * np.angle(Zxx1) + (1 - phase_weight1) * np.angle(Zxx2)
    blended_Zxx1_2 = blended_magnitude1 * np.exp(1j * blended_phase1)

    if np.array_equal(audio_list[2],np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3 = blended_Zxx1_2
    # Step 2: Blend (audio1 + audio2) with audio3
    else:
        mag_blended1_2, mag3 = np.abs(blended_Zxx1_2), np.abs(Zxx3)
        phase_weight2 = mag_blended1_2 / (mag_blended1_2 + mag3 + 1e-10)  # Avoid division by zero
        blended_magnitude2 = (1 - alpha1) * mag_blended1_2 + alpha1 * mag3
        blended_phase2 = phase_weight2 * np.angle(blended_Zxx1_2) + (1 - phase_weight2) * np.angle(Zxx3)
        blended_Zxx1_2_3 = blended_magnitude2 * np.exp(1j * blended_phase2)
    if np.array_equal(audio_list[3], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4 = blended_Zxx1_2_3
    # Step 3: Blend (audio1 + audio2 + audio3) with audio4
    else:
        mag1_2_3, mag4 = np.abs(blended_Zxx1_2_3), np.abs(Zxx4)
        phase_weight3 = mag1_2_3 / (mag1_2_3 + mag4 + 1e-10)
        blended_magnitude3 = (1 - alpha1) * mag1_2_3 + alpha1 * mag4
        blended_phase3 = phase_weight3 * np.angle(blended_Zxx1_2_3) + (1 - phase_weight3) * np.angle(Zxx4)
        blended_Zxx1_2_3_4 = blended_magnitude3 * np.exp(1j * blended_phase3)
    if np.array_equal(audio_list[4], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4_5 = blended_Zxx1_2_3_4
    # Step 4: Blend (audio1 + audio2 + audio3 + audio4) with audio5
    else:
        mag1_2_3_4, mag5 = np.abs(blended_Zxx1_2_3_4), np.abs(Zxx5)
        phase_weight4 = mag1_2_3_4 / (mag1_2_3_4 + mag5 + 1e-10)
        blended_magnitude4 = (1 - alpha1) * mag1_2_3_4 + alpha1 * mag5
        blended_phase4 = phase_weight4 * np.angle(blended_Zxx1_2_3_4) + (1 - phase_weight4) * np.angle(Zxx5)
        blended_Zxx1_2_3_4_5 = blended_magnitude4 * np.exp(1j * blended_phase4)

    if np.array_equal(audio_list[5], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4_5_6 = blended_Zxx1_2_3_4_5
    # Step 5: Blend (audio1 + audio2 + audio3 + audio4 + audio5) with audio6
    else:
        mag1_2_3_4_5, mag6 = np.abs(blended_Zxx1_2_3_4_5), np.abs(Zxx6)
        phase_weight5 = mag1_2_3_4_5 / (mag1_2_3_4_5 + mag6 + 1e-10)
        blended_magnitude5 = (1 - alpha1) * mag1_2_3_4_5 + alpha1 * mag6
        blended_phase5 = phase_weight5 * np.angle(blended_Zxx1_2_3_4_5) + (1 - phase_weight5) * np.angle(Zxx6)
        blended_Zxx1_2_3_4_5_6 = blended_magnitude5 * np.exp(1j * blended_phase5)

    if np.array_equal(audio_list[6], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4_5_6_7 = blended_Zxx1_2_3_4_5_6
    # Step 6: Blend (audio1 + audio2 + audio3 + audio4 + audio5 + audio6) with audio7
    else:
        mag1_2_3_4_5_6, mag7 = np.abs(blended_Zxx1_2_3_4_5_6), np.abs(Zxx7)
        phase_weight6 = mag1_2_3_4_5_6 / (mag1_2_3_4_5_6 + mag7 + 1e-10)
        blended_magnitude6 = (1 - alpha1) * mag1_2_3_4_5_6 + alpha1 * mag7
        blended_phase6 = phase_weight6 * np.angle(blended_Zxx1_2_3_4_5_6) + (1 - phase_weight6) * np.angle(Zxx7)
        blended_Zxx1_2_3_4_5_6_7 = blended_magnitude6 * np.exp(1j * blended_phase6)

    if np.array_equal(audio_list[7], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4_5_6_7_8 = blended_Zxx1_2_3_4_5_6_7
    # Step 7: Blend (audio1 + audio2 + audio3 + audio4 + audio5 + audio6 + audio7) with audio8
    else:
        mag1_2_3_4_5_6_7, mag8 = np.abs(blended_Zxx1_2_3_4_5_6_7), np.abs(Zxx8)
        phase_weight7 = mag1_2_3_4_5_6_7 / (mag1_2_3_4_5_6_7 + mag8 + 1e-10)
        blended_magnitude7 = (1 - alpha1) * mag1_2_3_4_5_6_7 + alpha1 * mag8
        blended_phase7 = phase_weight7 * np.angle(blended_Zxx1_2_3_4_5_6_7) + (1 - phase_weight7) * np.angle(Zxx8)
        blended_Zxx1_2_3_4_5_6_7_8 = blended_magnitude7 * np.exp(1j * blended_phase7)

    if np.array_equal(audio_list[8], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4_5_6_7_8_9 = blended_Zxx1_2_3_4_5_6_7_8
    # Step 8: Blend (audio1 + audio2 + audio3 + audio4 + audio5 + audio6 + audio7 + audio8) with audio9
    else:
        mag1_2_3_4_5_6_7_8, mag9 = np.abs(blended_Zxx1_2_3_4_5_6_7_8), np.abs(Zxx9)
        phase_weight8 = mag1_2_3_4_5_6_7_8 / (mag1_2_3_4_5_6_7_8 + mag9 + 1e-10)
        blended_magnitude8 = (1 - alpha1) * mag1_2_3_4_5_6_7_8 + alpha1 * mag9
        blended_phase8 = phase_weight8 * np.angle(blended_Zxx1_2_3_4_5_6_7_8) + (1 - phase_weight8) * np.angle(Zxx9)
        blended_Zxx1_2_3_4_5_6_7_8_9 = blended_magnitude8 * np.exp(1j * blended_phase8)

    if np.array_equal(audio_list[9], np.zeros_like(audio_list[0])):
        blended_Zxx1_2_3_4_5_6_7_8_9_10 = blended_Zxx1_2_3_4_5_6_7_8_9
    else:
    # Step 9: Blend (audio1 + audio2 + audio3 + audio4 + audio5 + audio6 + audio7 + audio8 + audio9) with audio10
        mag1_2_3_4_5_6_7_8_9, mag10 = np.abs(blended_Zxx1_2_3_4_5_6_7_8_9), np.abs(Zxx10)
        phase_weight9 = mag1_2_3_4_5_6_7_8_9 / (mag1_2_3_4_5_6_7_8_9 + mag10 + 1e-10)
        blended_magnitude9 = (1 - alpha1) * mag1_2_3_4_5_6_7_8_9 + alpha1 * mag10
        blended_phase9 = phase_weight9 * np.angle(blended_Zxx1_2_3_4_5_6_7_8_9) + (1 - phase_weight9) * np.angle(Zxx10)
        blended_Zxx1_2_3_4_5_6_7_8_9_10 = blended_magnitude9 * np.exp(1j * blended_phase9)



    
    # Inverse STFT to convert back to time domain
    # _, blended_audio = istft(blended_Zxx_final, fs=sr)
    _, blended_audio = istft(blended_Zxx1_2_3_4_5_6_7_8_9_10, fs=sr)
    return blended_audio

class SpectralSynth:
    def __init__(self, file_path1, file_path2, file_path3, file_path4, file_path5, file_path6, file_path7, file_path8, file_path9, file_path10, audio, mixer, sample_rate=44100):
        # Initialize audio system

        self.audio = audio
        self.mixer = mixer
        
    
        self.audio1, self.sr1 = librosa.load(file_path1, sr=sample_rate)
        self.audio2, self.sr2 = librosa.load(file_path2, sr=sample_rate)
        self.audio3, self.sr3 = librosa.load(file_path3, sr=sample_rate)
        self.audio4, self.sr4 = librosa.load(file_path4, sr=sample_rate)
        self.audio5, self.sr5 = librosa.load(file_path5, sr=sample_rate)
        self.audio6, self.sr6 = librosa.load(file_path6, sr=sample_rate)
        self.audio7, self.sr7 = librosa.load(file_path7, sr=sample_rate)
        self.audio8, self.sr8 = librosa.load(file_path8, sr=sample_rate)
        self.audio9, self.sr9 = librosa.load(file_path9, sr=sample_rate)
        self.audio10, self.sr10 = librosa.load(file_path10, sr=sample_rate)

        
        # Ensure sample rates match
        assert self.sr1 == self.sr2 == self.sr3 == self.sr4 == self.sr5 == self.sr6 == self.sr7 == self.sr8 == self.sr9 == self.sr10, "Sample rates must match"
        self.sample_rate = self.sr1

        # Control track inclusion (needed for UI compatibility)
        self.is_audio1 = False
        self.is_audio2 = False
        self.is_audio3 = False
        self.is_audio4 = False
        self.is_audio5 = False
        self.is_audio6 = False
        self.is_audio7 = False
        self.is_audio8 = False
        self.is_audio9 = False
        self.is_audio10 = False
        
        # Pad audio2 and audio3 to match the length of audio1
        max_length = max(len(self.audio1), len(self.audio2), len(self.audio3), len(self.audio4), 
                        len(self.audio5), len(self.audio6), len(self.audio7), len(self.audio8), 
                        len(self.audio9), len(self.audio10))

        self.audio1 = np.pad(self.audio1, (0, max_length - len(self.audio1)))
        self.audio2 = np.pad(self.audio2, (0, max_length - len(self.audio2)))
        self.audio3 = np.pad(self.audio3, (0, max_length - len(self.audio3)))
        self.audio4 = np.pad(self.audio4, (0, max_length - len(self.audio4)))
        self.audio5 = np.pad(self.audio5, (0, max_length - len(self.audio5)))
        self.audio6 = np.pad(self.audio6, (0, max_length - len(self.audio6)))
        self.audio7 = np.pad(self.audio7, (0, max_length - len(self.audio7)))
        self.audio8 = np.pad(self.audio8, (0, max_length - len(self.audio8)))
        self.audio9 = np.pad(self.audio9, (0, max_length - len(self.audio9)))
        self.audio10 = np.pad(self.audio10, (0, max_length - len(self.audio10)))
        
        self.blended_audio = np.zeros_like(self.audio1)

        # Enhanced MIDI and polyphony support
        self.base_note = 60  # Middle C
        self.active_notes = {}  # Dictionary to store active notes and their streams
        self.midi_input = mido.open_input()
        print("MIDI input opened. Awaiting MIDI events...")
        Clock.schedule_interval(self.process_midi, 0.001)

        # Pre-calculate pitch shifts for common MIDI note range
        self.pitch_cache = {}
        self.initialize_pitch_cache()

        # Effect parameters
        self.delay_time = 0.1
        self.reverb_amount = 0
        self.high_shelf_gain = 0
        self.mid_freq_gain = 0
        self.low_shelf_gain = 0
        
        # Define EQ frequency ranges
        self.low_shelf_freq = 200
        self.mid_freq_center = 1000
        self.high_shelf_freq = 5000

    def change_audio(self, idx, wavefile):
        """Maintain compatibility with UI while updating internal audio"""
        try:
            new_audio, new_sr = load_audio(wavefile, self.sample_rate)
            if idx == 1:
                self.audio1 = new_audio
                self.sr1 = new_sr
            elif idx == 2:
                self.audio2 = new_audio
                self.sr2 = new_sr
            elif idx == 3:
                self.audio3 = new_audio
                self.sr3 = new_sr
            elif idx == 4:
                self.audio4 = new_audio
                self.sr4 = new_sr
            elif idx == 5:
                self.audio5 = new_audio
                self.sr5 = new_sr
            elif idx == 6:
                self.audio6 = new_audio
                self.sr6 = new_sr
            elif idx == 7:
                self.audio7 = new_audio
                self.sr7 = new_sr
            elif idx == 8:
                self.audio8 = new_audio
                self.sr8 = new_sr
            elif idx == 9:
                self.audio9 = new_audio
                self.sr9 = new_sr
            elif idx == 10:
                self.audio10 = new_audio
                self.sr10 = new_sr
            # Update the blend immediately
            self.update_blend()
            
            # Stop all currently playing notes
            self.stop_all_notes()
            
        except Exception as e:
            print(f"Error changing audio: {e}")

    def toggle_audio(self, audio_idx):
        """Maintain compatibility with UI while updating internal state"""
        if audio_idx == 1:
            self.is_audio1 = not self.is_audio1
        elif audio_idx == 2:
            self.is_audio2 = not self.is_audio2
        elif audio_idx == 3:
            self.is_audio3 = not self.is_audio3
        elif audio_idx == 4:
            self.is_audio4 = not self.is_audio4
        elif audio_idx == 5:
            self.is_audio5 = not self.is_audio5
        elif audio_idx == 6:
            self.is_audio6 = not self.is_audio6
        elif audio_idx == 7:
            self.is_audio7 = not self.is_audio7
        elif audio_idx == 8:
            self.is_audio8 = not self.is_audio8
        elif audio_idx == 9:
            self.is_audio9 = not self.is_audio9
        elif audio_idx == 10:
            self.is_audio10 = not self.is_audio10
        # Update the blend and stop all currently playing notes
        self.update_blend()
        self.stop_all_notes()

    def stop_all_notes(self):
        """Stop all currently playing notes."""
        for note in list(self.active_notes.keys()):
            if note in self.active_notes:
                note_data = self.active_notes[note]
                
                # Stop the generator playback
                if 'generator' in note_data:
                    generator = note_data['generator']
                    if generator in self.mixer.generators:
                        self.mixer.remove(generator)
                    else:
                        print(f"Warning: Generator {generator} not found in mixer.generators")
                
                # Remove the envelope from the mixer
                if 'envelope' in note_data:
                    envelope = note_data['envelope']
                    if envelope in self.mixer.generators:  # Assuming the envelope is part of the generators list
                        self.mixer.remove(envelope)
                    else:
                        print(f"Warning: Envelope {envelope} not found in mixer.generators")
                
                # Remove the note from the active notes dictionary
                del self.active_notes[note]


    def initialize_pitch_cache(self):
        """Pre-calculate pitch shift amounts for common MIDI notes"""
        midi_range = range(36, 97)  # From C2 to C7
        for note in midi_range:
            self.pitch_cache[note] = note - self.base_note

    def handle_midi_message(self, message):
        """Handle MIDI messages for polyphonic playback"""
        try:
            if message.type == 'note_on' and message.velocity > 0:
                print(f"Note ON: {message.note}")
                if not any([self.is_audio1, self.is_audio2, self.is_audio3]):
                    return
                    
                note = message.note
                velocity = message.velocity / 127.0
                
                # Save current blend to temp file (if not already done)
                if not hasattr(self, 'base_wave_buffer') or self.base_wave_buffer is None:
                    temp_path = 'current_blend.wav'
                    sf.write(temp_path, self.blended_audio, self.sample_rate)
                    self.base_wave_buffer = WaveBuffer(temp_path, 0, -1)
                
                # Create generators and envelope
                wave_gen = WaveGenerator(self.base_wave_buffer, loop=True)
                speed_mod = SpeedModulator(wave_gen, speed=2**((note - self.base_note)/12))
                envelope = Envelope2(
                    input_generator=speed_mod,
                    attack_time=0.05,
                    decay_time=self.reverb_amount + 0.5,
                    n2=0
                )
                
                # Add to mixer and start playback
                self.mixer.add(envelope)
                wave_gen.play()
                
                # Store generators
                self.active_notes[note] = {
                    'generator': wave_gen,
                    'envelope': envelope
                }
                    
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                if message.note in self.active_notes:
                    note_data = self.active_notes[message.note]
                    # Stop the generator and remove it from the mixer
                    if self.reverb_amount <= 0.2:
                        note_data['generator'].release()  # Stop playback
                        self.mixer.remove(note_data['envelope'])  # Remove envelope from the mixer
                        del self.active_notes[message.note]  # Remove the note from active_notes
        except Exception as e:
            print(f"Error handling MIDI message: {e}")

    def audio_callback(self, outdata, frames, time, status, note):
        """Callback for audio output streams"""
        try:
            if status:
                print(status)
            
            if note not in self.active_notes:
                outdata[:] = np.zeros((frames, 1))
                return
                
            note_data = self.active_notes[note]
            position = note_data['position']
            audio = note_data['audio']  # This is already pitched
            velocity = note_data['velocity']
            
            chunk = audio[position:position + frames]
            if len(chunk) < frames:
                # Loop the audio if we reach the end
                remaining = frames - len(chunk)
                chunk = np.concatenate([chunk, audio[:remaining]])
            
            self.active_notes[note]['position'] = (position + frames) % len(audio)  # Loop position
            outdata[:] = (chunk * velocity).reshape(-1, 1)
            
        except Exception as e:
            print(f"Error in audio callback: {e}")
            outdata[:] = np.zeros((frames, 1))

    """ def get_pitched_audio(self, note):
        try:
            pitch_shift = note - self.base_note
            pitched_audio = librosa.effects.pitch_shift(
                self.blended_audio,
                sr=self.sample_rate,
                n_steps=pitch_shift
            )
            return pitched_audio
        except Exception as e:
            print(f"Error in pitch shifting: {e}")
            return self.blended_audio """

    def update_blend(self, alpha1=0.5, alpha2=0.5):
        """Update the blend based on active tracks"""
        # Stop all currently playing notes before updating blend
        self.stop_all_notes()
        audios = [
                        self.audio1, self.audio2, self.audio3, self.audio4, self.audio5,
                        self.audio6, self.audio7, self.audio8, self.audio9, self.audio10
                    ]
        active = [
                        self.is_audio1, self.is_audio2, self.is_audio3, self.is_audio4, self.is_audio5,
                        self.is_audio6, self.is_audio7, self.is_audio8, self.is_audio9, self.is_audio10
                    ]
        audio_list = []
        true_set = set()
        for idx, audio in enumerate(audios):
            if active[idx]:
                audio_list.append(audio)
                true_set.add(idx)
            else:
                audio_list.append(np.zeros_like(self.audio1))
        if len(audio_list)==1:
            self.blended_audio = audios[true_set.pop()]
        elif len(audio_list) == 0:
            self.blended_audio = np.zeros_like(self.audio1)
        else:
            self.blended_audio = spectral_blend3(audio_list, self.sample_rate, alpha1, alpha2)
        # if self.is_audio1 and self.is_audio2 and self.is_audio3:
        #     self.blended_audio = spectral_blend3(self.audio1, self.audio2, self.audio3, self.sample_rate, alpha1, alpha2)
        # elif self.is_audio1 and self.is_audio2:
        #     self.blended_audio = spectral_blend3(self.audio1, self.audio2, np.zeros_like(self.audio3), self.sample_rate, alpha1, alpha2)
        # elif self.is_audio1 and self.is_audio3:
        #     self.blended_audio = spectral_blend3(self.audio1, np.zeros_like(self.audio2), self.audio3, self.sample_rate, alpha1, alpha2)
        # elif self.is_audio2 and self.is_audio3:
        #     self.blended_audio = spectral_blend3(np.zeros_like(self.audio1), self.audio2, self.audio3, self.sample_rate, alpha1, alpha2)
        # elif self.is_audio1:
        #     self.blended_audio = self.audio1
        # elif self.is_audio2:
        #     self.blended_audio = self.audio2
        # elif self.is_audio3:
        #     self.blended_audio = self.audio3
        # else:
        #     self.blended_audio = np.zeros_like(self.audio1)
            
        # Apply effects to the blended audio
        self.blended_audio = self.process_audio_effects()

        # Create new base WaveBuffer for the blend
        temp_path = 'current_blend.wav'
        sf.write(temp_path, self.blended_audio, self.sample_rate)
        self.base_wave_buffer = WaveBuffer(temp_path, 0, -1)
        
        # Stop all currently playing notes and update them with new blend
        #self.stop_all_notes()
    
    def apply_delay(self, audio, delay_time, sample_rate):
        """Apply a simple delay effect."""
        try:
            delay_samples = int(delay_time * sample_rate)
            if delay_samples == 0:
                return audio
                
            # Create delayed signal
            delayed_audio = np.zeros_like(audio)
            delayed_audio[delay_samples:] = audio[:-delay_samples]
            
            # Mix original and delayed signal
            result = audio + 0.5 * delayed_audio
            
            # Check for invalid values
            if not np.all(np.isfinite(result)):
                print("Warning: Invalid values in delay processing, returning original audio")
                return audio
                
            return result
        except Exception as e:
            print(f"Error in delay processing: {e}")
            return audio

    def apply_reverb(self, audio, reverb_amount):
        """Apply a simple reverb effect using a feedback loop."""
        try:
            if reverb_amount == 0:
                return audio
                
            # Ensure audio is valid
            if not np.all(np.isfinite(audio)):
                return audio

            feedback = np.clip(reverb_amount, 0, 0.9)  # Limit feedback to prevent instability
            b = [1]  # Feed-forward coefficients
            a = [1] + [-feedback] * 10  # Feedback coefficients
            
            result = lfilter(b, a, audio)
            
            # Check for invalid values
            if not np.all(np.isfinite(result)):
                print("Warning: Invalid values in reverb processing, returning original audio")
                return audio
                
            # Normalize to prevent clipping
            max_val = np.max(np.abs(result))
            if max_val > 1.0:
                result = result / max_val
                
            return result
        except Exception as e:
            print(f"Error in reverb processing: {e}")
            return audio

    def apply_eq(self, audio):
        """Apply 3-band EQ to the audio signal."""
        try:
            if all(gain == 0 for gain in [self.low_shelf_gain, self.mid_freq_gain, self.high_shelf_gain]):
                return audio

            # Convert gain from dB to linear scale
            low_gain = 10 ** (self.low_shelf_gain / 20)
            mid_gain = 10 ** (self.mid_freq_gain / 20)
            high_gain = 10 ** (self.high_shelf_gain / 20)

            # Calculate frequencies for filtering
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            fft_data = np.fft.rfft(audio)

            # Create frequency masks
            low_mask = 1 / (1 + (freqs / self.low_shelf_freq) ** 2)
            high_mask = 1 / (1 + (self.high_shelf_freq / freqs) ** 2)
            mid_mask = np.exp(-((freqs - self.mid_freq_center) ** 2) / (2 * (self.mid_freq_center/2) ** 2))

            # Apply EQ gains
            fft_data *= (1 + (low_gain - 1) * low_mask)
            fft_data *= (1 + (high_gain - 1) * high_mask)
            fft_data *= (1 + (mid_gain - 1) * mid_mask)

            # Convert back to time domain
            eq_audio = np.fft.irfft(fft_data)
            
            # Normalize to prevent clipping
            eq_audio = eq_audio / np.max(np.abs(eq_audio))
            
            return eq_audio
        except Exception as e:
            print(f"Error in EQ processing: {e}")
            return audio

    def process_audio_effects(self):
        """Apply all effects (EQ, delay, and reverb) to the blended audio."""
        try:
            processed_audio = np.copy(self.blended_audio)
            
            # Ensure audio is valid before processing
            if not np.all(np.isfinite(processed_audio)):
                print("Warning: Invalid values in input audio")
                return self.blended_audio
            
            # Apply EQ first
            processed_audio = self.apply_eq(processed_audio)
            
            # Apply delay and reverb as before
            if self.delay_time > 0:
                processed_audio = self.apply_delay(processed_audio, self.delay_time, self.sample_rate)
                
            if self.reverb_amount > 0:
                processed_audio = self.apply_reverb(processed_audio, self.reverb_amount)
            
            # Final check for valid output
            if not np.all(np.isfinite(processed_audio)):
                print("Warning: Invalid values in processed audio, returning original")
                return self.blended_audio
            
            return processed_audio
        except Exception as e:
            print(f"Error in audio processing: {e}")
            return self.blended_audio

    def process_midi(self, dt):
        """Process MIDI input messages."""
        for message in self.midi_input.iter_pending():
            self.handle_midi_message(message)

    def play_note_with_pitch(self, midi_note, velocity=1.0):
        """Plays the blended note transposed to match the MIDI note pitch."""
        try:
            # Calculate pitch shift in semitones
            semitone_shift = midi_note - self.base_note
            
            # Apply pitch shift using librosa
            transposed_audio = librosa.effects.pitch_shift(self.blended_audio, sr=self.sample_rate, n_steps=semitone_shift)
            
            # Check for valid audio before processing effects
            if not np.all(np.isfinite(transposed_audio)):
                print("Warning: Invalid values after pitch shift")
                return
            
            # Apply effects
            processed = self.apply_delay(self.blended_audio, self.delay_time, self.sample_rate)
            processed = self.apply_reverb(processed, self.reverb_amount)
            
            # Final validity check before playing
            if np.all(np.isfinite(processed)):
                sd.play(velocity * processed, samplerate=self.sample_rate)
            else:
                print("Warning: Invalid values in processed audio")
                sd.play(velocity * transposed_audio, samplerate=self.sample_rate)
                
        except Exception as e:
            print(f"Error in playback: {e}")

    def stop_audio(self):
        """Stops the playback."""
        sd.stop()

    def set_mix(self, is_audio1=True, is_audio2=True, is_audio3=True):
        """Set which audio tracks should be in the mix based on the small circles."""
        self.is_audio1 = is_audio1
        self.is_audio2 = is_audio2
        self.is_audio3 = is_audio3
        
        # Update the blend based on the active tracks
        if self.is_audio1 and self.is_audio2 and self.is_audio3:
            self.blended_audio = spectral_blend3(self.audio1, self.audio2, self.audio3, self.sample_rate)
        elif self.is_audio1 and self.is_audio2:
            print("3 not playing")
            self.blended_audio = spectral_blend3(self.audio1, self.audio2, np.zeros_like(self.audio3), self.sample_rate)
        elif self.is_audio1 and self.is_audio3:
            self.blended_audio = spectral_blend3(self.audio1, np.zeros_like(self.audio2), self.audio3, self.sample_rate)
        elif self.is_audio2 and self.is_audio3:
            self.blended_audio = spectral_blend3(np.zeros_like(self.audio1), self.audio2, self.audio3, self.sample_rate)
        elif self.is_audio1:
            self.blended_audio = self.audio1  # Only audio1
        elif self.is_audio2:
            self.blended_audio = self.audio2  # Only audio2
        elif self.is_audio3:
            self.blended_audio = self.audio3  # Only audio3
        else:
            self.blended_audio = np.zeros_like(self.audio1)  # No tracks in the mix


class AudioMixerWidget(Widget):
    def __init__(self, synth, sound_box, **kwargs):
        super().__init__(**kwargs)
        self.synth = synth
        self.sound_box = sound_box

        # Use Window size to center the circles
        main_circle_radius = 100
        small_circle_radius = 50
        
        # Calculate the positions
        self.main_circle_pos = (Window.width / 2 - main_circle_radius, Window.height / 2 - main_circle_radius)
        self.circle1_pos = (Window.width / 4 - small_circle_radius, Window.height * 3 / 4 - small_circle_radius)
        self.circle2_pos = (Window.width * 3 / 4 - small_circle_radius, Window.height * 3 / 4 - small_circle_radius)
        self.circle3_pos = (Window.width / 2 - small_circle_radius, Window.height / 4 - small_circle_radius)

        # Draw the circles
        with self.canvas:
            Color(1, 1, 1)
            self.main_circle = Ellipse(pos=self.main_circle_pos, size=(main_circle_radius * 2, main_circle_radius * 2))  # Main pot circle
            self.circle1 = Ellipse(pos=self.circle1_pos, size=(small_circle_radius * 2, small_circle_radius * 2))  # Audio 1
            self.circle2 = Ellipse(pos=self.circle2_pos, size=(small_circle_radius * 2, small_circle_radius * 2))  # Audio 2
            self.circle3 = Ellipse(pos=self.circle3_pos, size=(small_circle_radius * 2, small_circle_radius * 2))  # Audio 3

        # Labels for each circle
        self.label1 = Label(text="Audio 1", pos=(self.circle1_pos[0], self.circle1_pos[1] + 60), color=(1, 1, 1, 1))
        self.label2 = Label(text="Audio 2", pos=(self.circle2_pos[0], self.circle2_pos[1] + 60), color=(1, 1, 1, 1))
        self.label3 = Label(text="Audio 3", pos=(self.circle3_pos[0], self.circle3_pos[1] + 60), color=(1, 1, 1, 1))

        self.add_widget(self.label1)
        self.add_widget(self.label2)
        self.add_widget(self.label3)
        
    def on_touch_down(self, touch):
        """Handle touch/click events."""
        if self._is_touch_inside_circle(touch, self.circle1_pos, 50):
            self.synth.toggle_audio(1)
            print("1")
        elif self._is_touch_inside_circle(touch, self.circle2_pos, 50):
            self.synth.toggle_audio(2)
            print("2")
        elif self._is_touch_inside_circle(touch, self.circle3_pos, 50):
            self.synth.toggle_audio(3)
            print("3")
        self.update_sound_box()
        return super().on_touch_down(touch)

    def _is_touch_inside_circle(self, touch, circle_pos, radius):
        """Check if the touch is inside the circle's radius."""
        x, y = circle_pos
        distance = np.sqrt((touch.x - (x + radius)) ** 2 + (touch.y - (y + radius)) ** 2)
        return distance <= radius
    
    def update_sound_box(self):
        """Update the top box to show which sounds are in the mix."""
        sounds_in_pot = []
        if self.synth.is_audio1:
            sounds_in_pot.append("Audio 1")
        if self.synth.is_audio2:
            sounds_in_pot.append("Audio 2")
        if self.synth.is_audio3:
            sounds_in_pot.append("Audio 3")

        # self.sound_box.text = "Sounds in the Mix: " + ", ".join(sounds_in_pot)
        self.sound_box.text = ''


class AudioMixerWidget(Widget):
    def __init__(self, synth, sound_box, **kwargs):
        super().__init__(**kwargs)
        self.synth = synth
        self.sound_box = sound_box

        # Use Window size to center the circles
        main_circle_radius = 100
        small_circle_radius = 50
        
        # Calculate the positions
        self.main_circle_pos = (Window.width / 2 - main_circle_radius, Window.height / 2 - main_circle_radius)
        self.circle1_pos = (Window.width / 4 - small_circle_radius, Window.height * 3 / 4 - small_circle_radius)
        self.circle2_pos = (Window.width * 3 / 4 - small_circle_radius, Window.height * 3 / 4 - small_circle_radius)
        self.circle3_pos = (Window.width / 2 - small_circle_radius, Window.height / 4 - small_circle_radius)

        # Draw the circles
        with self.canvas:
            Color(1, 1, 1)
            self.main_circle = Ellipse(pos=self.main_circle_pos, size=(main_circle_radius * 2, main_circle_radius * 2))  # Main pot circle
            self.circle1 = Ellipse(pos=self.circle1_pos, size=(small_circle_radius * 2, small_circle_radius * 2))  # Audio 1
            self.circle2 = Ellipse(pos=self.circle2_pos, size=(small_circle_radius * 2, small_circle_radius * 2))  # Audio 2
            self.circle3 = Ellipse(pos=self.circle3_pos, size=(small_circle_radius * 2, small_circle_radius * 2))  # Audio 3

        # Labels for each circle
        self.label1 = Label(text="Audio 1", pos=(self.circle1_pos[0], self.circle1_pos[1] + 60), color=(1, 1, 1, 1))
        self.label2 = Label(text="Audio 2", pos=(self.circle2_pos[0], self.circle2_pos[1] + 60), color=(1, 1, 1, 1))
        self.label3 = Label(text="Audio 3", pos=(self.circle3_pos[0], self.circle3_pos[1] + 60), color=(1, 1, 1, 1))

        self.add_widget(self.label1)
        self.add_widget(self.label2)
        self.add_widget(self.label3)
        
    def on_touch_down(self, touch):
        """Handle touch/click events."""
        if self._is_touch_inside_circle(touch, self.circle1_pos, 50):
            self.synth.toggle_audio(1)
            print("1")
        elif self._is_touch_inside_circle(touch, self.circle2_pos, 50):
            self.synth.toggle_audio(2)
            print("2")
        elif self._is_touch_inside_circle(touch, self.circle3_pos, 50):
            self.synth.toggle_audio(3)
            print("3")
        self.update_sound_box()
        return super().on_touch_down(touch)

    def _is_touch_inside_circle(self, touch, circle_pos, radius):
        """Check if the touch is inside the circle's radius."""
        x, y = circle_pos
        distance = np.sqrt((touch.x - (x + radius)) ** 2 + (touch.y - (y + radius)) ** 2)
        return distance <= radius
    
    def update_sound_box(self):
        """Update the top box to show which sounds are in the mix."""
        sounds_in_pot = []
        if self.synth.is_audio1:
            sounds_in_pot.append("Audio 1")
        if self.synth.is_audio2:
            sounds_in_pot.append("Audio 2")
        if self.synth.is_audio3:
            sounds_in_pot.append("Audio 3")

        # self.sound_box.text = "Sounds in the Mix: " + ", ".join(sounds_in_pot)
        self.sound_box.text = ''

from kivy.graphics import Color, RoundedRectangle

class BubbleButton(Button):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.background_color = (0, 0, 0, 0)  # Fully transparent background
        self.background_normal = ""
        self.canvas.before.clear()
        with self.canvas.before:
            # Base bubble shape
            Color(0.8, 0.9, 1, 1)  # Light pastel blue (adjust for the style)
            self.bg_rect = RoundedRectangle(size=self.size, pos=self.pos, radius=[30])  # Rounded corners

            # Glossy highlight
            Color(1, 1, 1, 0.3)  # Semi-transparent white
            self.gloss = RoundedRectangle(size=(self.width * 0.9, self.height * 0.5),
                                          pos=(self.x + self.width * 0.05, self.y + self.height * 0.5),
                                          radius=[15])  # Slightly rounded gloss

        # Bind position and size updates
        self.bind(pos=self.update_graphics, size=self.update_graphics)

    def update_graphics(self, *args):
        # Adjust the rectangular shape and gloss when resized
        self.bg_rect.size = self.size
        self.bg_rect.pos = self.pos
        self.gloss.size = (self.width * 0.9, self.height * 0.5)
        self.gloss.pos = (self.x + self.width * 0.05, self.y + self.height * 0.5)

class SoundPopup(Popup):
    def __init__(self, canvas_widget, **kwargs):
        super().__init__(**kwargs)
        tan_color = (0.94, 0.87, 0.77, 1)  # Light tan
        green_color = (0.6, 0.8, 0.6, 1)   # Light green
        
        self.title = "Select Sounds"
        self.size_hint = (0.8, 0.8)
        self.background = ""  # Removes default texture
        self.background_color = tan_color
        self.canvas_widget = canvas_widget

        layout = BoxLayout(orientation="vertical", spacing=10)

        # Scrollable area
        scroll_view = ScrollView(size_hint=(1, 0.8))
        grid = GridLayout(cols=3, size_hint_y=None, row_default_height=100,spacing=(10, 10))
        grid.bind(minimum_height=grid.setter("height"))

        # Placeholder sounds
        sounds = [ # image, wave file, position
            {"image": "images/dreamy.png", "title": "Dreamy Synth", "file": "dreamy_synth_C.wav"},
            {"image": "images/cinematic.png", "title": "Cinematic Drone", "file": "cinematic_C.wav"},
            {"image": "images/seventh.png", "title": "7th Player", "file": "sounds/7th-player.wav"},
            {"image": "images/bass.png", "title": "Deep Bass Pattern", "file": "sounds/deep_bass_pattern.wav"},
            {"image": "images/drift.png", "title": "Distant Drift Synth", "file": "sounds/distant_drift_synth.wav"},
            {"image": "images/bells.webp", "title": "Dreamy Bells", "file": "sounds/dreamy_bells.wav"},
            {"image": "images/flute.png", "title": "Flute", "file": "sounds/flute.wav"},
            {"image": "images/wobbly.png", "title": "Heavy Wobble Bass", "file": "sounds/heavy_wobble_bass.wav"},
            {"image": "images/piano.png", "title": "Piano", "file": "sounds/piano.wav"},
            {"image": "images/string.png", "title": "Plucked String", "file": "sounds/plucked_string.wav"},
            {"image": "images/space.png", "title": "Edge of Space", "file": "sounds/edge_of_space.wav"},
            {"image": "images/game.png", "title": "Gamelan Bells", "file": "sounds/gamelan_bells.wav"},
            {"image": "images/wave.png", "title": "Bass Wave Cycles", "file": "sounds/bass_wave_cycles.wav"},
            {"image": "images/wood.png", "title": "Wooden Chimes", "file": "sounds/wooden_chimes.wav"},
            {"image": "images/phase.png", "title": "Phasing Bells", "file": "sounds/phasing_bells.wav"},
            {"image": "images/sheep.png", "title": "Electric Sheep Synth", "file": "sounds/electric_sheep_synth.wav"},
            {"image": "smiley.png", "title": "Hold r to record", "file": "recording.wav"}
        ]

        for sound in sounds:
            grid.add_widget(Image(source=sound["image"], size_hint=(0.2, 1)))
            grid.add_widget(Label(text=sound["title"], color=(0, 0, 0, 1), size_hint=(0.6, 1)))
            # "+" Button with BubbleButton style
            add_button = BubbleButton(
                text="+",
                size_hint=(0.2, 1),
                font_size=24,  # Slightly larger font
                color=(0, 0, 0, 1),  # Black text
            )
            add_button.bind(on_press=lambda _, s=sound: self.canvas_widget.add_sound_to_canvas(s))
            grid.add_widget(add_button)
            with grid.canvas:
                Color(0, 0, 0, 1)  # Black color
        
  

        scroll_view.add_widget(grid)
        layout.add_widget(scroll_view)

        # Close button
        close_button = BubbleButton(
            text="Close",
            size_hint=(1, 0.1),
            font_size=30,  # Adjust font size for proportion
            color=(0, 0, 0, 1),  # Black text
        )
        close_button.bind(on_press=self.dismiss)
        layout.add_widget(close_button)

        self.content = layout

class CanvasWidget(Widget):
    def __init__(self, mainwidget, **kwargs):
        super().__init__(**kwargs)
        self.main_widget = mainwidget
        self.active_sounds = []
        self.sound_limit = 10
        self.open_ids = [i+1 for i in range(self.sound_limit)]
        self.sound_x_position = 150  # Fixed x position for all sounds
        self.slot_height = Window.height / 12  # Divide the screen into 5 equal slots
        tan_color = (0.94, 0.87, 0.77, 1)  # Light tan
        green_color = (0.6, 0.8, 0.6, 1)   # Light green

        # Replace `Button` with `BubbleButton` for the "Sounds" button
        sounds_button = BubbleButton(
            text="Sounds",
            size_hint=(None, None),
            size=(170, 90),  # More rectangular shape
            pos=(Window.width * 0.85, Window.height * 0.88),  # Adjusted for new shape
            font_size=30,  # Larger font for better proportion
            color=(0, 0, 0, 1),  # Text color
        )
        sounds_button.bind(on_press=self.open_sound_popup)
        self.add_widget(sounds_button)

    def open_sound_popup(self, _):
        popup = SoundPopup(self)
        popup.open()

    def add_sound_to_canvas(self, sound):
        print(sound)
        # Check if we have reached the sound limit
        if len(self.main_widget.instrument_list) >= self.sound_limit:
            print("Cannot add more than 10 sounds.")
            self.show_alert_popup()  # Show the alert popup when the limit is reached
            return
        
        # Calculate the position for the new sound (stacked vertically in the 5 slots)
        idxslot = self.open_ids.pop(0)
        y_position = self.slot_height * (idxslot-1)+ (self.slot_height) / 2
        # The (self.slot_height - 50) / 2 ensures the circle is vertically centered in each slot

        # Create and draw the circle at the calculated position
        # with self.canvas:
        #     color = Color(np.random.random(), np.random.random(), np.random.random())
        #     circle = Ellipse(pos=(self.sound_x_position, y_position), size=(50, 50))
        inst = UI.InstrumentObject(sound['image'],(self.sound_x_position, y_position),sound['file'], idxslot, self.main_widget.pot,(Window.width - 100, 100), self.main_widget.add_audio, self.main_widget.rem_audio,self.remove_sound_from_canvas)
        self.main_widget.instrument_list[idxslot] = inst
        self.main_widget.anim_group.add(inst)
        # Create a red remove button positioned near the circle
        # remove_button = Button(
        #     text="-",
        #     size_hint=(None, None),
        #     size=(20, 20),
        #     pos=(self.sound_x_position + 40, y_position + 40),
        #     background_color=(1, 0, 0, 1),
        # )
        # remove_button.bind(on_press=lambda _: self.remove_sound_from_canvas(inst, remove_button))
        # self.add_widget(remove_button)
        # remove_button = Button(
        #     text="-",
        #     size_hint=(None, None),
        #     size=(20, 20),
        #     pos=(self.sound_x_position + 40, y_position + 40),
        #     background_color=(1, 0, 0, 1),
        # )
        # remove_button.bind(on_press=lambda _: self.remove_sound_from_canvas(inst, remove_button))
        # self.add_widget(remove_button)

        # Store references to the circle and button
        # self.active_sounds.append({"sound": sound, "circle": inst, "button": remove_button})
        self.active_sounds.append({"sound": sound, "circle": inst})

    def remove_sound_from_canvas(self, sound):
        self.main_widget.anim_group.remove(sound)
        del self.main_widget.instrument_list[sound.idx]
        self.open_ids.append(sound.idx)
    #     # Remove the circle from the canvas
    #     if circle in self.canvas.children:
    #         self.canvas.remove(circle)

    #     # Remove the button from the widget
    #     if button in self.children:
    #         self.remove_widget(button)

    #     # Update the active_sounds list to exclude the removed elements
    #     self.active_sounds = [
    #         s for s in self.active_sounds if s["circle"] != circle and s["button"] != button
    #     ]

    #     # Reposition all remaining sounds to fill the gap
    #     self.reposition_sounds()

    def reposition_sounds(self):
        # Recalculate the position of each sound to stack them up vertically
        for idx, sound in enumerate(self.active_sounds):
            y_position = self.slot_height * idx + (self.slot_height - 50) / 2  # Center within each slot
            sound["circle"].ellipse.cpos = (self.sound_x_position, y_position)
            sound["circle"].pos = (self.sound_x_position + 40, y_position + 40)

    def show_alert_popup(self):
        tan_color = (0.94, 0.87, 0.77, 1)  # Light tan
        green_color = (0.6, 0.8, 0.6, 1)   # Light green

        # Create a popup layout
        popup_layout = BoxLayout(orientation='vertical', padding=20, spacing=20)

        # Create a label with the alert message
        message = Label(
            text="Cannot add more than 10 sounds in the mix.",
            size_hint=(1, None),
            height=100,
            text_size=(800, None),  # Allow text to wrap within the width
            halign='center',  # Horizontally center the text
            valign='middle',  # Vertically center the text
            color=(0, 0, 0, 1)  # Black text for readability
        )
        message.bind(size=message.setter('text_size'))  # Bind size for dynamic text wrapping
        popup_layout.add_widget(message)

        # Create a BubbleButton for the Close button
        close_button = BubbleButton(
            text="Close",
            size_hint=(None, None),
            size=(200, 60),  # Larger and more prominent button
            font_size=20,
            color=(0, 0, 0, 1),  # Black text
            padding=(10, 10),  # Internal padding for a balanced look
            background_color=green_color,
            border=(20, 20, 20, 20),  # Rounded corners
        )
        close_button.bind(on_press=self.close_alert_popup)
        popup_layout.add_widget(close_button)

        # Create the popup and open it
        self.popup = Popup(
            title="Sound Limit Reached",
            title_align='center',
            title_color=(0, 0, 0, 1),  # Black title for readability
            background="",  # Removes the default background texture
            background_color=tan_color,  # Sets the custom tan color
            content=popup_layout,
            size_hint=(None, None),
            size=(800, 300),
            auto_dismiss=False  # Prevent automatic closing
        )
        self.popup.open()

    def close_alert_popup(self, _):
        self.popup.dismiss()  # Close the popup when the close button is pressed
        
import sys, os
sys.path.insert(0, os.path.abspath('..'))


from kivy.core.window import Window

import numpy as np
import sounddevice as sd
import wave

NUM_INPUT_CHANNELS = 1
NUM_OUTPUT_CHANNELS = 2

# Same WaveSource interface, but is given audio data explicitly.
class WaveArray(object):
    def __init__(self, np_array, num_channels):
        super(WaveArray, self).__init__()

        self.data = np_array
        self.num_channels = num_channels

    # start and end args are in units of frames,
    # so take into account num_channels when accessing sample data
    def get_frames(self, start_frame, num_frames):
        start_sample = start_frame * self.num_channels
        end_sample = (start_frame + num_frames) * self.num_channels
        return self.data[start_sample : end_sample]

    def get_length(self): 
        return len(self.data) / self.num_channels

    def get_num_channels(self):
        return self.num_channels

def convert_data_channels(data, in_channels, out_channels):
    if in_channels == out_channels:
        return data

    if in_channels == 1:
        out = np.empty(len(data) * out_channels)
        for c in range(out_channels):
            out[c::out_channels] = data
        return out

    if out_channels == 1:
        separated = [data[c::in_channels] for c in range(in_channels)]
        stack = np.array(separated)
        return np.mean(stack, axis=0)

    assert False, "don't know how to convert {} channels to {} channels".format(
        in_channels, out_channels
    )

class AudioMixerApp(App):
    def build(self):
        layout = FloatLayout()
        
        # Initialize synthesizer
        file_path1 = 'dreamy_synth_C.wav'
        file_path2 = 'cinematic_C.wav'
        file_path3 = 'synth_bass_C.wav'
        # synth = SpectralSynth(file_path1, file_path2, file_path3)
        # Create a label box for displaying active sounds
        self.main_widget = UI.MainWidget1()
        layout.add_widget(self.main_widget)
        sound_box = Label(text="", size_hint=(0.5, 0.1), pos_hint={'x': 0.25, 'y': 0.9}, color=(1, 1, 1, 1))
        layout.add_widget(sound_box)
        
        # Add audio mixer widget
        # mixer_widget = AudioMixerWidget(synth, sound_box)
        # layout.add_widget(mixer_widget)
        
        # Add CanvasWidget to the layout
        canvas_widget = CanvasWidget(self.main_widget)
        layout.add_widget(canvas_widget)

        # Set up recording attributes
        self.recording = []
        self.is_recording = False
        self.stream = None

        # Bind key events
        Window.bind(on_key_down=self.on_key_down)
        Window.bind(on_key_up=self.on_key_up)

        print("Layout children: ", layout.children)
        print("CanvasWidget size:", layout.children[0].size)
        print("CanvasWidget position:", layout.children[0].pos)

        
    

        return layout   
    def on_key_down(self, window, key, scancode, codepoint, modifier):
            if key == ord("r"):  # Start recording when 'r' is pressed
                if not self.is_recording:
                    self.start_recording()

    def on_key_up(self, window, key, scancode):
        if key == ord("r"):  # Stop recording when 'r' is released
            if self.is_recording:
                self.stop_recording()

    def start_recording(self):
        """Start recording audio."""
        self.is_recording = True
        self.recording = []

        def callback(indata, frames, time, status):
            if status:
                print(f"Stream status: {status}")
            if self.is_recording:
                # Append the recorded frames to a list
                self.recording.append(indata.copy())

        # Set a sample rate that matches your hardware
        self.samplerate = 44100  # Check and match your microphone's native rate
        self.stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=NUM_INPUT_CHANNELS,
            dtype='int16',  # Match with the format for the .wav file
            callback=callback
        )
        self.stream.start()
        print("Recording started...")

    def stop_recording(self):
        """Stop recording audio and save to a .wav file."""
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        print("Recording stopped.")

        # Combine recorded chunks into a single numpy array
        audio_data = np.concatenate(self.recording, axis=0)

        # Save the recording as a .wav file
        filename = "recording.wav"
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(NUM_INPUT_CHANNELS)  # Mono recording
            wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
            wf.setframerate(self.samplerate)  # Match the recording rate
            wf.writeframes(audio_data.tobytes())

        print(f"Recording saved as {filename}")

    def save_to_wav(self, file_name, audio_data):
        """Save numpy array as a .wav file."""
        with wave.open(file_name, "wb") as wf:
            wf.setnchannels(NUM_INPUT_CHANNELS)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(44100)
            wf.writeframes(audio_data.tobytes())
if __name__ == "__main__":
    AudioMixerApp().run()



# previous code

# import numpy as np
# import librosa
# import soundfile as sf
# from scipy.signal import stft, istft
# import sounddevice as sd
# from scipy.ndimage import gaussian_filter
# import mido

# # kivy for graphics for now
# from kivy.app import App
# from kivy.uix.widget import Widget
# from kivy.uix.button import Button
# from kivy.graphics import Ellipse, Color
# from kivy.uix.floatlayout import FloatLayout
# from kivy.uix.label import Label

# def load_audio(file_path, sample_rate=44100):
#     """Load audio file and return audio data and sample rate."""
#     audio, sr = librosa.load(file_path, sr=sample_rate)
#     return audio, sr

# def spectral_blend1(audio1, audio2, sr, alpha=0.5, harmonic_boost=1.0, sigma=1.5):
#     """
#     Blend spectra with harmonic matching, dynamic filtering, and weighted phase.
    
#     Parameters:
#     - audio1, audio2: audio signals to blend
#     - sr: sample rate of the audio
#     - alpha: blending factor (0.0 = all audio1, 1.0 = all audio2)
#     - harmonic_boost: factor to emphasize shared harmonics in the blend
#     - sigma: smoothing factor for dynamic filtering
    
#     Returns:
#     - blended_audio: spectrally blended audio signal
#     """
#     # Calculate STFT for both audio signals
#     _, _, Zxx1 = stft(audio1, fs=sr, nperseg=1024)
#     _, _, Zxx2 = stft(audio2, fs=sr, nperseg=1024)

#     min_time_bins = min(Zxx1.shape[1], Zxx2.shape[1])
#     Zxx1, Zxx2 = Zxx1[:, :min_time_bins], Zxx2[:, :min_time_bins]

#     # Magnitudes and phases of the original signals
#     mag1, mag2 = np.abs(Zxx1), np.abs(Zxx2)
#     phase1, phase2 = np.angle(Zxx1), np.angle(Zxx2)

#     # Harmonic Matching (adjust harmonic_boost for testing)
#     harmonic_mask = np.minimum(mag1, mag2) ** harmonic_boost
#     mag1 = mag1 * (1 - harmonic_mask) + harmonic_mask
#     mag2 = mag2 * (1 - harmonic_mask) + harmonic_mask

#     # Dynamic Filtering: Smooth magnitudes to reduce noise
#     smoothed_mag1 = gaussian_filter(mag1, sigma=sigma)
#     smoothed_mag2 = gaussian_filter(mag2, sigma=sigma)
#     blended_magnitude = (1 - alpha) * smoothed_mag1 + alpha * smoothed_mag2

#     # Weighted Phase Blending
#     phase_weight = mag1 / (mag1 + mag2 + 1e-10)
#     blended_phase = phase_weight * phase1 + (1 - phase_weight) * phase2

#     # Combine magnitude and phase to produce final blend
#     blended_Zxx = blended_magnitude * np.exp(1j * blended_phase)

#     # Inverse STFT to convert back to time domain
#     _, blended_audio = istft(blended_Zxx, fs=sr)
#     return blended_audio

# def spectral_blend2(audio1, audio2, sr, alpha=0.5):
#     """Blend spectra with weighted phase based on magnitude."""
#     _, _, Zxx1 = stft(audio1, fs=sr, nperseg=1024)
#     _, _, Zxx2 = stft(audio2, fs=sr, nperseg=1024)

#     min_time_bins = min(Zxx1.shape[1], Zxx2.shape[1])
#     Zxx1, Zxx2 = Zxx1[:, :min_time_bins], Zxx2[:, :min_time_bins]

#     mag1, mag2 = np.abs(Zxx1), np.abs(Zxx2)
#     phase_weight = mag1 / (mag1 + mag2 + 1e-10)  # Avoid division by zero

#     blended_magnitude = (1 - alpha) * mag1 + alpha * mag2
#     blended_phase = phase_weight * np.angle(Zxx1) + (1 - phase_weight) * np.angle(Zxx2)
#     blended_Zxx = blended_magnitude * np.exp(1j * blended_phase)

#     _, blended_audio = istft(blended_Zxx, fs=sr)
#     return blended_audio

# def spectral_blend3(audio1, audio2, audio3, sr, alpha1=0.5, alpha2=0.5):
#     """
#     Blend spectra of three audio signals with weighted phase.
    
#     Parameters:
#     - audio1, audio2, audio3: audio signals to blend
#     - sr: sample rate of the audio
#     - alpha1: blending factor between audio1 and audio2
#     - alpha2: blending factor for blending the result with audio3
    
#     Returns:
#     - blended_audio: spectrally blended audio signal of three sources
#     """
#     # Step 1: Perform STFT for each audio signal
#     _, _, Zxx1 = stft(audio1, fs=sr, nperseg=1024)
#     _, _, Zxx2 = stft(audio2, fs=sr, nperseg=1024)
#     _, _, Zxx3 = stft(audio3, fs=sr, nperseg=1024)

#     # Ensure all spectrograms have the same shape by matching time bins
#     min_time_bins = min(Zxx1.shape[1], Zxx2.shape[1], Zxx3.shape[1])
#     Zxx1, Zxx2, Zxx3 = Zxx1[:, :min_time_bins], Zxx2[:, :min_time_bins], Zxx3[:, :min_time_bins]

#     # Step 2: Blend audio1 and audio2 first
#     mag1, mag2 = np.abs(Zxx1), np.abs(Zxx2)
#     phase_weight1 = mag1 / (mag1 + mag2 + 1e-10)  # Avoid division by zero

#     blended_magnitude1 = (1 - alpha1) * mag1 + alpha1 * mag2
#     blended_phase1 = phase_weight1 * np.angle(Zxx1) + (1 - phase_weight1) * np.angle(Zxx2)
#     blended_Zxx1_2 = blended_magnitude1 * np.exp(1j * blended_phase1)

#     # Step 3: Blend the result of (audio1 + audio2) with audio3
#     mag_blended1_2, mag3 = np.abs(blended_Zxx1_2), np.abs(Zxx3)
#     phase_weight2 = mag_blended1_2 / (mag_blended1_2 + mag3 + 1e-10)  # Avoid division by zero

#     blended_magnitude_final = (1 - alpha2) * mag_blended1_2 + alpha2 * mag3
#     blended_phase_final = phase_weight2 * np.angle(blended_Zxx1_2) + (1 - phase_weight2) * np.angle(Zxx3)
#     blended_Zxx_final = blended_magnitude_final * np.exp(1j * blended_phase_final)

#     # Inverse STFT to convert back to time domain
#     _, blended_audio = istft(blended_Zxx_final, fs=sr)
#     return blended_audio

# class SpectralSynth:
#     def __init__(self, file_path1, file_path2, file_path3, sample_rate=44100, alpha1=0.5, alpha2=0.5, base_note=60):
#         # Load audio files and print their paths to confirm
#         self.audio1, self.sr1 = load_audio(file_path1, sample_rate)
#         self.audio2, self.sr2 = load_audio(file_path2, sample_rate)
#         self.audio3, self.sr3 = load_audio(file_path3, sample_rate)
#         print(f"Loaded files: {file_path1}, {file_path2}, {file_path3}")
        
#         # Ensure sample rates match
#         assert self.sr1 == self.sr2 == self.sr3, "Sample rates must match"
#         self.sample_rate = self.sr1
#         self.base_note = base_note  # MIDI note number for the original pitch
        
#         # Perform spectral blending with the given alpha values
#         self.blended_audio = spectral_blend3(self.audio1, self.audio2, self.audio3, self.sample_rate, alpha1, alpha2)
        
#         self.canvas.add("Hi")
        
    # def play_note_with_pitch(self, midi_note, velocity=1.0):
    #     """Plays the blended note transposed to match the MIDI note pitch."""
    #     # Calculate the pitch shift in semitones
    #     semitone_shift = midi_note - self.base_note
        
    #     # Apply pitch shift using librosa
    #     transposed_audio = librosa.effects.pitch_shift(self.blended_audio, sr=self.sample_rate, n_steps=semitone_shift)
        
    #     # Adjust volume by velocity and play
    #     sd.play(velocity * transposed_audio, samplerate=self.sample_rate)
    
#     def stop_note(self):
#         """Stops playback."""
#         sd.stop()
        
#     def handle_midi_message(self, message):
#         """Handle incoming MIDI messages for note on/off."""
#         if message.type == 'note_on' and message.velocity > 0:
#             self.play_note_with_pitch(midi_note=message.note, velocity=message.velocity / 127)
#         elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
#             self.stop_note()


# def main(file_path1, file_path2, file_path3):
#     # Initialize the synthesizer with the two audio files
#     synth = SpectralSynth(file_path1, file_path2, file_path3)

#     # Open default MIDI input
#     with mido.open_input() as midi_input:
#         print("MIDI Synth is ready. Play your MIDI keyboard to control the synth.")
        
#         # MIDI processing loop
#         try:
#             while True:
#                 for message in midi_input.iter_pending():
#                     synth.handle_midi_message(message)
#         except KeyboardInterrupt:
#             print("Exiting...")
            
    

# if __name__ == "__main__":
#     # Paths to audio files for blending
#     file_path1 = 'dreamy_synth_C.wav'
#     file_path2 = 'cinematic_C.wav'
#     file_path3 = 'synth_bass_C.wav'
    
#     main(file_path1, file_path2, file_path3)