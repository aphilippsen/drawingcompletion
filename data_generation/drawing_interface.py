import matplotlib.pyplot as plt
import datetime
import numpy as np

class DrawingGenerationInterface(object):
    """
    Display a drawing plane and collect the strokes that the human inputs.
    """

    def __init__(self, ax, stroke_fct, drawing_fct, x_minmax = [-1, 1], y_minmax = [-1, 1]):
        self.ax = ax
        self.process_stroke_fct = stroke_fct
        self.process_drawing_fct = drawing_fct
        self.x_minmax = x_minmax
        self.y_minmax = y_minmax

        self.drawing = 0          # currently drawing or not
        self.current_stroke = []  # list of x,y coordinates of current stroke
        self.current_drawing = [] # list of current strokes

        # link events to class functions
        plt.connect('motion_notify_event', self.mouse_move)
        plt.connect('button_press_event', self.mouse_click)
        plt.connect('button_release_event', self.mouse_release)

        # initialize the drawing plane
        self.plot_initial()

    def plot_initial(self):
        self.ax.set_xlim(self.x_minmax)
        self.ax.set_ylim(self.y_minmax)
        plt.draw()
        plt.plot(0, 0, 'o')
        # here, other things can be done, e.g. displaying some background image

    def mouse_move(self, event):
        if not event.inaxes:
            return

        if self.drawing:
            x, y = event.xdata, event.ydata
            self.current_stroke.append([x,y])
            self.human_line = self.ax.plot(x,y, 'o', color='black', label='human')
            plt.draw()

    def mouse_click(self, event):
        if event.button == 1: # left click
            # drawing starts
            print("Start drawing stroke")
            self.current_stroke = []
            self.drawing = True
        elif event.button == 3: # right click
            # drawing is finished
            self.process_drawing()
            print("Clear drawing")
            self.current_stroke = []
            self.current_drawing = []
            self.ax.clear()
            self.plot_initial()

    def mouse_release(self, event):
        self.drawing = False
        if event.button == 1: # left mouse button release
            print("Finish drawing stroke")
            self.current_drawing.append(self.current_stroke)
            self.process_stroke()

    def process_stroke(self):
        print("process_stroke called")
        self.process_stroke_fct(self.current_stroke)

    def process_drawing(self):
        print("process_drawing called")
        self.process_drawing_fct(self.current_drawing)

# create figure and axis for drawing
#fig, ax = plt.subplots(nrows=1, ncols=1)

# define function for processing the drawing result
def process_stroke(stroke):
    print("Process stroke of length " + str(len(stroke)))

def process_drawing(drawing):
    print("Process drawing of length " + str(len(drawing)) + ", with strokes " + str([len(x) for x in drawing]))
    now = datetime.datetime.now()
    np.save('drawing-' + str(now.year).zfill(4) + "-" + str(now.month).zfill(2) + "-" + str(now.day).zfill(2) + "_" + str(now.hour).zfill(2) + "-" + str(now.minute).zfill(2) + "_" + str(now.microsecond).zfill(7) + '.npy', np.asarray(drawing))

# create interface and connect mouse actions
#interface = DrawingGenerationInterface(ax, process_stroke, process_drawing)

# start
#plt.show(block=False)
