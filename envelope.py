import sys, os
sys.path.insert(0, os.path.abspath('..'))

from imslib.core import BaseWidget, run
from imslib.audio import Audio
from imslib.writer import AudioWriter
from imslib.gfxutil import topleft_label


#Used a different envelope that worked well for my purpose. Used both attack and decay, but in this part I kept the attack very low.
class Envelope2(object):
    def __init__(self, input_generator, attack_time=0.05, decay_time=2.0, n2=0.0):
        self.input_generator = input_generator
        self.attack_time = attack_time
        self.decay_time = decay_time
        self.n2 = n2
        self.sample_rate = Audio.sample_rate
        self.current_frame = 0
        self.attack_frames = int(self.attack_time * self.sample_rate)
        self.decay_frames = int(self.decay_time * self.sample_rate)

    def set_decay_time(self, decay_time):
        self.decay_time = decay_time
        self.decay_frames = int(self.decay_time * self.sample_rate)

    def set_attack_time(self, attack_time):
        self.attack_time = attack_time
        self.attack_frames = int(self.attack_time * self.sample_rate)

    def generate(self, num_frames, num_channels):
        input_signal, continue_flag = self.input_generator.generate(num_frames, num_channels)

        if self.current_frame < self.attack_frames:
            # In the attack phase, slowly increase volume
            envelope = (self.current_frame / self.attack_frames)
        elif self.current_frame < self.attack_frames + self.decay_frames:
            # After attack, we move to the decay phase
            decay_progress = (self.current_frame - self.attack_frames) / self.decay_frames
            envelope = 1.0 - (decay_progress * (1.0 - self.n2))
        else:
            # If beyond decay phase, just hold at n2 level (usually 0)
            envelope = self.n2

        # Apply envelope to the input signal
        output = input_signal * envelope

        self.current_frame += num_frames
        if self.current_frame >= self.attack_frames + self.decay_frames:
            continue_flag = False

        return (output, continue_flag)



