from kivy.app import App
from kivy.uix.widget import Widget
from imslib.audio import Audio
from imslib.mixer import Mixer
from imslib.wavegen import WaveGenerator, SpeedModulator
from imslib.wavesrc import WaveBuffer
import mido
from kivy.clock import Clock

class SimplePlayer:
    def __init__(self, audio_file):
        # Initialize audio system
        self.audio = Audio(2)
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        
        # Create wave buffer from file
        self.wave_buffer = WaveBuffer(audio_file, 0, -1)
        
        # MIDI setup
        self.base_note = 60  # Middle C
        self.active_notes = {}  # Keep track of playing notes
        self.midi_input = mido.open_input()
        print("MIDI initialized")
        
        # Start MIDI processing
        Clock.schedule_interval(self.process_midi, 1/60.0)

    def process_midi(self, dt):
        for message in self.midi_input.iter_pending():
            print(f"MIDI message: {message}")  # Debug print
            if message.type == 'note_on' and message.velocity > 0:
                print(f"Playing note {message.note}")  # Debug print
                # Create new generator for this note
                wave_gen = WaveGenerator(self.wave_buffer, loop=True)
                speed_mod = SpeedModulator(wave_gen, speed=2**((message.note - self.base_note)/12))
                
                # Add to mixer and play
                self.mixer.add(speed_mod)
                wave_gen.play()
                
                # Store reference
                self.active_notes[message.note] = (wave_gen, speed_mod)
                
            elif message.type == 'note_off' or (message.type == 'note_on' and message.velocity == 0):
                if message.note in self.active_notes:
                    wave_gen, speed_mod = self.active_notes[message.note]
                    wave_gen.release()
                    self.mixer.remove(speed_mod)
                    del self.active_notes[message.note]

    def update(self, dt):
        self.audio.on_update()

class MainWidget(Widget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.player = SimplePlayer('new_sounds/piano.wav')  # Use any wav file
        Clock.schedule_interval(self.player.update, 1/60.0)

class SimpleApp(App):
    def build(self):
        return MainWidget()

if __name__ == '__main__':
    SimpleApp().run()