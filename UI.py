import sys, os
sys.path.insert(0, os.path.abspath('..'))
import main as main
import mido
from threading import Thread
from imslib.core import BaseWidget, run, lookup
from imslib.gfxutil import topleft_label, CEllipse, KFAnim, AnimGroup, CRectangle, CLabelRect

from imslib.audio import Audio
from imslib.mixer import Mixer
from imslib.note import NoteGenerator, Envelope
from imslib.wavegen import WaveGenerator, SpeedModulator
from imslib.wavesrc import WaveBuffer, WaveFile, make_wave_buffers

from kivy.core.window import Window
from kivy.clock import Clock as kivyClock
from kivy.uix.label import Label
from kivy.graphics.instructions import InstructionGroup
from kivy.graphics import Color, Ellipse, Rectangle, Line
from kivy.core.image import Image
from kivy.graphics import PushMatrix, PopMatrix, Translate, Scale, Rotate

from random import randint, random
import numpy as np

from kivy.uix.slider import Slider
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
import soundfile as sf


from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.slider import Slider
from kivy.graphics import Color, Rectangle, Line

from LoopManager import LoopManager


class AudioSlider(BoxLayout):
    def __init__(self, title, min_value=0, max_value=1, value=0, **kwargs):
        super(AudioSlider, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (None, None)
        self.size = (120, 250)  # Increased overall size
        self.padding = [10, 15]  # Added more padding
        self.spacing = 10  # Space between label and slider
        self.last_touch_time = 0  # Track the time of the last touch event
        self.double_click_interval = 0.3  # Max interval (seconds) for a double-click

        # Create a container for the label with background
        self.label_container = BoxLayout(
            orientation='vertical',
            size_hint_y=0.2,
            padding=[5, 5]
        )

        # Enhanced label with better styling
        self.label = Label(
            text=title,
            font_size='16sp',
            bold=True,
            color=(1, 1, 1, 1),
            size_hint_y=None,
            height=-400
        )

        # Value label to show current value
        self.value_label = Label(
            text=f"{value:0.1f}",
            font_size='14sp',
            color=(0.9, 0.9, 0.9, 1),
            size_hint_y=None,
            height=20
        )

        # Enhanced slider with custom graphics
        self.slider = Slider(
            min=min_value,
            max=max_value,
            value=value,
            orientation='vertical',
            size_hint_y=0.7,
            cursor_size=(30, 30),
            background_width='12dp',  # Make the slider bar thicker
            value_track=True,  # Show the value track
            value_track_width='3dp'
        )

        # Bind the slider to update the value label
        self.slider.bind(value=self.update_value_label)

        # Add widgets to the layout
        self.label_container.add_widget(self.label)
        self.label_container.add_widget(self.value_label)
        self.add_widget(self.label_container)
        self.add_widget(self.slider)

        # Add custom graphics
        with self.canvas.before:
            Color(0.2, 0.2, 0.2, 0.8)  # Dark semi-transparent background
            self.rect = Rectangle(size=self.size, pos=self.pos)

            # Add a border
            Color(0.3, 0.3, 0.3, 1)  # Border color
            self.border = Line(rectangle=(self.x, self.y, self.width, self.height), width=1.5)

        # Bind the layout size and position to update the graphics
        self.bind(size=self._update_rect, pos=self._update_rect)

    def update_value_label(self, instance, value):
        """Update the label displaying the current slider value."""
        self.value_label.text = f"{value:0.1f}"

    def _update_rect(self, instance, value):
        """Update the rectangle and border when resized."""
        self.rect.size = self.size
        self.rect.pos = self.pos
        self.border.rectangle = (self.x, self.y, self.width, self.height)

    def on_touch_down(self, touch):
        """Handle touch events for the slider."""
        if self.slider.collide_point(*touch.pos):
            current_time = Clock.get_time()
            # Detect double click
            if current_time - self.last_touch_time <= self.double_click_interval:
                self.reset_slider()  # Reset the slider value on double click
            self.last_touch_time = current_time
        return super(AudioSlider, self).on_touch_down(touch)

    def reset_slider(self):
        """Reset the slider to its default value (0) and update the label."""
        self.slider.value = 0
        self.update_value_label(self.slider, 0)



class MainWidget1(BaseWidget):
    def __init__(self):
        super(MainWidget1, self).__init__()

        # Initialize audio system first
        self.audio = Audio(2)
        self.mixer = Mixer()
        self.audio.set_generator(self.mixer)
        self.loop_manager = LoopManager(self.mixer)  # Initialize loop manager

        file_path1 = 'sounds/7th-player.wav'
        file_path2 = 'cinematic_C.wav'
        file_path3 = 'synth_bass_C.wav'
        file_path4 = 'sounds/piano.wav'
        file_path5 = 'sounds/flute.wav'
        file_path6 = 'sounds/7th-player.wav'
        file_path7 = 'cinematic_C.wav'
        file_path8 = 'synth_bass_C.wav'
        file_path9 = 'sounds/piano.wav'
        file_path10 = 'sounds/flute.wav'
        self.included_audio = []
        self.synth = main.SpectralSynth(file_path1, file_path2, file_path3, file_path4, file_path5,
                                file_path6, file_path7, file_path8, file_path9, file_path10, self.audio, self.mixer)
        
        # Store references to the rectangles for resizing later
        self.background_rect = CRectangle(
            cpos=(Window.width / 2, Window.height / 2 - 50),
            csize=(Window.width + 100, Window.height + 100),
            texture=Image('images/background.png').texture
        )
        self.recipe_rect = CRectangle(
            cpos=(Window.width / 11, 1 / 2 * Window.height),
            csize=(Window.width / 7, Window.height * 0.98),
            texture=Image('recipe.png').texture
        )
        self.logo_rect = CRectangle(
            cpos=(Window.width / 2, 57 * Window.height / 64),
            csize=(Window.width * 0.3, Window.height * 0.2),
            texture=Image('clefchef.png').texture
        )
        self.stove_rect = CRectangle(
            cpos=(Window.width / 2, 3 * Window.height / 12),
            csize=(Window.width * 0.6, Window.height * 0.4),
            texture=Image('stove.png').texture
        )
        # Add rectangles to the canvas
        self.canvas.before.add(self.background_rect)
        self.canvas.before.add(self.recipe_rect)
        self.canvas.before.add(self.logo_rect)
        self.canvas.before.add(self.stove_rect)
        
        self.pot = PotObject()
        self.canvas.before.add(self.pot)

        self.instrument_list = {}
        self.anim_group = AnimGroup()
        self.canvas.after.add(self.anim_group)
        self.canvas.add(Color(1,1,1))
        self.selected_inst = None

        # Create the slider layout with adjusted size
        self.slider_layout = BoxLayout(
            orientation='horizontal',
            size_hint=(None, None),
            size=(650, 280),  # Increased size to accommodate larger sliders
            pos=(Window.width/2 - 400, Window.height/32),  # Adjusted position to center
            spacing=50  # Reduced spacing between sliders
        )
        
        # Create sliders with custom ranges and initial values
        self.reverb_slider = AudioSlider(
            title='Reverb',
            min_value=0,
            max_value=5,
            value=0.0,
            pos = (500,100)
        )
        self.delay_slider = AudioSlider(
            title='Delay',
            min_value=0,
            max_value=1,
            value=0.1,
            pos = (700,100)
        )
        self.high_shelf_slider = AudioSlider(
            title='High Shelf',
            min_value=-24,
            max_value=24,
            value=0,
            pos = (900,100)
        )
        self.mid_freq_slider = AudioSlider(
            title='Mid Freq',
            min_value=-24,
            max_value=24,
            value=0,
            pos = (1100,100)
        )
        self.low_shelf_slider = AudioSlider(
            title='Low Shelf',
            min_value=-24,
            max_value=24,
            value=0,
            pos = (1300,100)
        )
        
        # # Add custom background for slider group
        # with self.slider_layout.canvas.before:
        #     Color(0.15, 0.15, 0.15, 0.9)  # Very dark semi-transparent background
        #     Rectangle(size=self.slider_layout.size, pos=self.slider_layout.pos)
        
        # Add all sliders to layout
        self.slider_layout.add_widget(self.reverb_slider)
        self.slider_layout.add_widget(self.delay_slider)
        self.slider_layout.add_widget(self.high_shelf_slider)
        self.slider_layout.add_widget(self.mid_freq_slider)
        self.slider_layout.add_widget(self.low_shelf_slider)
        
        # Bind slider events (same as before)
        self.reverb_slider.slider.bind(value=self.on_reverb_change)
        self.delay_slider.slider.bind(value=self.on_delay_change)
        self.high_shelf_slider.slider.bind(value=self.on_high_shelf_change)
        self.mid_freq_slider.slider.bind(value=self.on_mid_freq_change)
        self.low_shelf_slider.slider.bind(value=self.on_low_shelf_change)
        
        # Add slider layout to the widget
        self.add_widget(self.slider_layout)


        self.info = topleft_label(font_size='50sp')
        self.audio_list = CLabelRect(cpos=(Window.width*.5,Window.height*.88),font_size=21)
        self.canvas.add(Color(1,1,1))
        self.canvas.add(self.audio_list)
        self.add_widget(self.info)
        
        # Trash Area
        self.trash_area = CRectangle(
            cpos=(Window.width - 100, 100), 
            csize=(100, 100), 
            texture=Image("images/trash.png").texture
        )
        self.canvas.add(self.trash_area)
        
        # Window.bind(on_resize=self.on_resize())
        self.on_resize((Window.width, Window.height))
        
    def on_resize(self, win_size):
        """Adjust the positions and sizes of elements proportionally."""
        width, height =  win_size
        print(f"Window resized to width: {width}, height: {height}")
        
       
        # Update positions and sizes of the rectangles
        self.background_rect.cpos = (width / 2, height / 2 - 50)
        self.background_rect.csize = (width + 100, height + 100)
        
        self.recipe_rect.cpos = (width / 11, 1 / 2 * height)
        self.recipe_rect.csize = (width / 7, height * 0.98)
        
        self.logo_rect.cpos = (width / 2, 57 * height / 64)
        self.logo_rect.csize = (width * 0.3, height * 0.2)
        
        self.stove_rect.cpos = (width / 2, 3 * height / 12)
        self.stove_rect.csize = (width * 0.6, height * 0.4)

    def on_reverb_change(self, instance, value):
        if hasattr(self, 'synth'):
            self.synth.reverb_amount = value
            # Update decay time for any currently playing notes
            for note_data in self.synth.active_notes.values():
                note_data['envelope'].set_decay_time(value + 0.5)
            
    def on_delay_change(self, instance, value):
        if hasattr(self, 'synth'):
            self.synth.delay_time = value
    
    def on_high_shelf_change(self, instance, value):
        if hasattr(self, 'synth'):
            self.synth.high_shelf_gain = value
            self.synth.update_blend()

    def on_mid_freq_change(self, instance, value):
        if hasattr(self, 'synth'):
            self.synth.mid_freq_gain = value
            self.synth.update_blend()

    def on_low_shelf_change(self, instance, value):
        if hasattr(self, 'synth'):
            self.synth.low_shelf_gain = value
            self.synth.update_blend()

    def on_touch_down(self, touch):
        # Check if touch is in slider layout first
        if self.slider_layout.collide_point(*touch.pos):
            return super(MainWidget1, self).on_touch_down(touch)
        
        # Existing instrument handling
        for idx,inst in self.instrument_list.items():
            dist = ((touch.pos[0]-inst.ellipse.cpos[0])**2+(touch.pos[1]-inst.ellipse.cpos[1])**2)**(.5)
            if dist<inst.ellipse.csize[0]:
                inst.grabbed = True
                self.selected_inst = inst
                return True
        return super(MainWidget1, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        # Check if touch is in slider layout first
        if self.slider_layout.collide_point(*touch.pos):
            return super(MainWidget1, self).on_touch_move(touch)
            
        # Existing instrument movement
        if self.selected_inst:
            self.selected_inst.ellipse.cpos = touch.pos
            return True
        return super(MainWidget1, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        # Check if touch is in slider layout first
        if self.slider_layout.collide_point(*touch.pos):
            return super(MainWidget1, self).on_touch_up(touch)
            
        # Existing touch up code
        # if self.selected_inst:
        #     if (self.trash_area.cpos[0] - self.trash_area.csize[0]/2 <= touch.pos[0] <= self.trash_area.cpos[0] + self.trash_area.csize[0]/2 and
        #         self.trash_area.cpos[1] - self.trash_area.csize[1]/2 <= touch.pos[1] <= self.trash_area.cpos[1] + self.trash_area.csize[1]/2):
        #         if self.selected_inst in self.instrument_list:
        #             self.instrument_list.remove(self.selected_inst)
        if self.selected_inst:
            self.selected_inst.added = False
            self.selected_inst.stopped = False
            self.selected_inst.grav_const = -3600.
            self.selected_inst.gravity = np.array((0., self.selected_inst.grav_const))
            self.selected_inst.vel = np.array((0., 0.), dtype=float)
            self.selected_inst.pos = np.array(touch.pos, dtype=float)
            self.selected_inst.grabbed = False
            # if not self.selected_inst.added and self.selected_inst.ellipse.cpos[0] > self.pot.x1 and self.selected_inst.ellipse.cpos[0] < self.pot.x2 and self.selected_inst.ellipse.cpos[1]>(Window.height/2+self.pot.rad/8):
            #     self.synth.change_audio(self.selected_inst.idx, self.selected_inst.audiopath)
            #     self.synth.toggle_audio(self.selected_inst.idx)
            #     self.included_audio.append(self.selected_inst.audiopath)
            # elif self.selected_inst.added and not (self.selected_inst.ellipse.cpos[0] > self.pot.x1 and self.selected_inst.ellipse.cpos[0] < self.pot.x2 and self.selected_inst.ellipse.cpos[1]>(Window.height/2+self.pot.rad/8)):
            #     self.synth.toggle_audio(self.selected_inst.idx)
            #     self.included_audio.remove(self.selected_inst.audiopath)
        self.selected_inst = None
        return super(MainWidget1, self).on_touch_up(touch)

    def on_update(self, dt = 0):
        # Add audio frames to LoopManager during recording
        if self.loop_manager.recording:
            num_frames = 1024
            num_channels = 2
            audio_frames, _ = self.mixer.generate(num_frames, num_channels)
            self.loop_manager.add_frames(audio_frames)

        self.info.text = ''
        self.audio_list.set_text('')
        self.anim_group.on_update()
        # Add audio system update with debug count
        if hasattr(self, 'audio'):
            self.audio.on_update()
        
        # Process MIDI messages
        if hasattr(self, 'synth'):
            messages = list(self.synth.midi_input.iter_pending())
            if messages:  # Only print if there are messages
                print(f"Found {len(messages)} MIDI messages")
            for message in messages:
                self.synth.handle_midi_message(message)

    # def on_resize(self, win_size):
        # self.canvas.remove(self.pot)
        # self.pot.on_resize(win_size)
        # self.canvas.add(self.pot)
        # Update slider layout position
        # self.slider_layout.pos = (win_size[0]/2 - 150, win_size[1]/5)

    def on_key_down(self, keycode, modifiers):
        # Toggle recording with 'l'
        if keycode[1] == 'l':
            if self.loop_manager.recording:
                self.loop_manager.stop_recording()
            else:
                self.loop_manager.start_recording()

        # Clear loops with 'c'
        elif keycode[1] == 'c':
            self.loop_manager.clear_loops()

        elif keycode[1] == 'c':
            print("Clearing all loops...")
            self.loop_manager.clear_loops()

    def add_obj(self, obj):
        self.canvas.add(obj)

    def rem_obj(self, obj):
        self.canvas.remove(obj)
    def add_audio(self, instrument):
        self.synth.change_audio(instrument.idx, instrument.audiopath)
        self.synth.toggle_audio(instrument.idx)
        self.included_audio.append(instrument.audiopath)
    def rem_audio(self, instrument):
        self.synth.toggle_audio(instrument.idx)
        self.included_audio.remove(instrument.audiopath)


class PotObject(InstructionGroup): #Circle which houses instruments
    def __init__(self):
        super(PotObject, self).__init__()
        self.rad = .25*Window.width
        self.x1 = Window.width/2-self.rad*.6
        self.x2 = Window.width/2+self.rad*.6
        self.pot = CRectangle(cpos = (Window.width/2,Window.height*5/8-10), csize = (Window.width*.8, Window.height*.9), texture = Image('pot.png').texture)
        self.add(self.pot)
        self.ellipse = CEllipse(cpos=(Window.width/2, Window.height*19/32+5), csize = (self.rad*1.3, self.rad/4+11))
        self.color = Color(97./255,219./255,250./255)
        self.color.a = .75
        self.add(self.color)
        self.add(self.ellipse)
    def on_resize(self, win_size):
        self.rad = .25*win_size[0]
        self.line = Line(ellipse=(win_size[0]/2, win_size[1]/2, self.rad, self.rad/8), width=3)
damping = .6
class InstrumentObject(InstructionGroup): # Should be updated with images of instruments
    def __init__(self, imagepath, cpos, audiopath, idx, pot, trash, add, rem, remove_sound):
        super(InstrumentObject, self).__init__()
        self.idx = idx
        self.pot = pot
        self.trash = trash
        self.audiopath = audiopath
        self.pos = np.array(cpos, dtype=float)
        self.ogpos = cpos
        self.ellipse = CEllipse(cpos = cpos, csize = (100,100), texture=Image(imagepath).texture)
        self.radius = 100
        self.grav_const = 0
        self.grabbed = False
        self.vel = np.array((0, 0), dtype=float)
        self.gravity = np.array((0., 0.)) 
        self.added = False
        self.stopped = False
        self.inside = False
        self.done = False
        self.trashed = False
        self.add(self.ellipse)
        self.add_audio = add
        self.rem_audio = rem
        self.remove_sound = remove_sound
        
    def on_update(self, dt):
        if not self.grabbed:
            self.vel += self.gravity * dt
            self.pos += self.vel * dt
            self.ellipse.cpos = self.pos
            # print(0)
            # print((self.pos[1] <= Window.height/2+self.pot.rad/8))
            # print(self.pos[0] > self.pot.x1 and self.pos[0] < self.pot.x2)
            # print(not self.added)
            # print(not self.stopped)
            # print(0)
            if (self.pos[1] <= Window.height*19/32+15) and (self.pos[0] > self.pot.x1 and self.pos[0] < self.pot.x2) and not self.added and not self.stopped:
                self.vel*=damping
                self.grav_const = -self.grav_const
                self.gravity = np.array((0., self.grav_const))
                self.added = True
            if (self.pos[1] >= Window.height*19/32+15) and (self.pos[0] > self.pot.x1 and self.pos[0] < self.pot.x2) and self.added and not self.stopped:
                self.vel*=damping
                self.grav_const = -self.grav_const
                self.gravity = np.array((0., self.grav_const))
                self.added = False
            if abs(self.vel[1]) < 20 and abs(self.pos[1]-(Window.height*19/32+15))<10 and not self.stopped:
                self.pos = (self.pos[0], self.pos[1])
                self.ellipse.cpos = self.pos
                self.vel = np.array((0, 0), dtype=float)
                self.grav_const = 0.
                self.gravity = np.array((0., self.grav_const))
                self.added = True
                self.stopped = True
            # dist_from_trash = ((self.pos[0]-self.trash[0])**2+(self.pos[1]-self.trash[1])**2)**.5
            # if dist_from_trash<60:
            dist_from_trash = ((self.pos[0]-self.trash[0])**2+(self.pos[1]-self.trash[1])**2)**.5
            if dist_from_trash<120:
                self.trashed = True
            if self.pos[1] < 0 and not self.done:
                if not self.trashed:
                    self.grabbed = True
                    self.ellipse.cpos = self.ogpos
                else:
                    self.remove_sound(self)
                    self.done = True
            dist_from_pot = ((self.pos[0]-self.pot.ellipse.cpos[0])**2+(self.pos[1]-self.pot.ellipse.cpos[1])**2)**.5
            # print(dist_from_pot)
            # print(self.pot.ellipse.csize[0])
            if not self.inside and dist_from_pot<self.pot.ellipse.csize[0]:
                # print(1)
                self.add_audio(self)
                self.inside = True
            elif self.inside and dist_from_pot>self.pot.ellipse.csize[0]:
                # print(2)
                self.rem_audio(self)
                self.inside = False
        else:
            self.ellipse.cpos = self.ellipse.cpos
        return self.pos[0]+self.radius >= 0 and self.pos[0]-self.radius <= Window.width and self.pos[1]+self.radius >= 0 
""" class Slider(InstructionGroup):
    def __init__(self, points, control):
        super(Slider, self).__init__()
        self.add(Color(0,0,0))
        self.rail = Line(points = points, width = 4)
        self.add(self.rail)
        self.add(Color(1,1,1))
        self.handle = CRectangle(cpos = (points[0],(points[3]+points[1])/2), csize = (50,20))
        self.add(self.handle)
        self.control = control
        self.percent = .5
    def adjust(self):
        self.percent = (self.handle.cpos[1]-self.rail.points[1])/(self.rail.points[3]-self.rail.points[1])
 """

if __name__ == "__main__":
    run(MainWidget1())