import os
import threading
from math import *
from tkinter import *

import cv2
import numpy as np
import win32api

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

running_thread = False


class ImageGenerator:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((self.height, self.width, 3), np.uint8)
        self.n = 3
        self.imThread = self.imageThread(self.canvas)
        self.details = {"Square": 0, "Triangle": 0, "Circle": 0, "Cross": 0, "Tear": 0, "Heart": 0, "Oval": 0,
                        "Star": 0, "Moon": 0, "Background": None}

    def generate_n(self):
        self.n = np.random.choice(range(3, 20), p=[0.03, 0.05, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                                                   0.05, 0.05, 0.05, 0.05, 0.15, 0.05, 0.04, 0.03])
        return self.n

    @staticmethod
    def get_rand_shape():
        shapes = ["Square", "Triangle", "Circle", "Cross", "Tear", "Heart", "Oval", "Star", "Moon"]
        return np.random.choice(shapes, p=[0.20, 0.20, 0.20, 0.15, 0.10, 0.05, 0.05, 0.04, 0.01])

    @staticmethod
    def get_rand_colour():
        colours = [(102, 51, 0), (204, 102, 51), (153, 0, 0), (102, 102, 0), (204, 153, 0), (204, 51, 0),
                   (255, 51, 51), (153, 153, 102), (204, 204, 51), (255, 153, 0), (204, 51, 51), (102, 204, 0),
                   (255, 153, 153), (255, 51, 153), (0, 102, 0), (153, 255, 102), (255, 204, 204), (255, 51, 204),
                   (204, 0, 153), (0, 153, 0), (102, 255, 255), (102, 0, 255), (0, 153, 204), (0, 51, 255),
                   (0, 51, 153), (0, 0, 0)]

        return colours[np.random.choice(len(colours))]

    @staticmethod
    def get_rand_scale():
        scale = [0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
        return np.random.choice(scale, p=[0.05, 0.20, 0.25, 0.25, 0.20, 0.05]) * 3

    @staticmethod
    def get_rand_skew():
        return np.random.choice(range(0, 360))

    def get_rand_thickness(self):
        medium_scale = self.width / 100
        thickness = [0.2, 0.5, 1, 1.2, 1.5, 2]
        return round(medium_scale * np.random.choice(thickness, p=[0.05, 0.20, 0.25, 0.25, 0.20, 0.05]))

    def get_rand_location(self):
        frag = self.width / 32
        x = (np.random.choice(range(0, 16)) * 2 + 1) * frag
        y_tiles = self.height / frag
        y = (np.random.choice(range(0, int(y_tiles / 2))) * 2 + 1) * frag
        return int(x), int(y)
        # return int(np.random.choice(range(0, self.width))), int(np.random.choice(range(0, self.height)))

    @staticmethod
    def get_rand_bg_colour():
        colours = ["Black", "Red", "Blue", "White", "Green", "Orange", "Random", "Gradient 1", "Gradient 2",
                   "Gradient 3", "Gradient 4", "Gradient 5", "Gradient 6"]
        return np.random.choice(colours,
                                p=[0.10, 0.10, 0.20, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])

    def set_bg_colour(self, colour):
        self.details["Background"] = colour
        if colour == "Black":
            self.canvas[:] = (0, 0, 0)
        elif colour == "Red":
            self.canvas[:] = (0, 0, 255)
        elif colour == "Blue":
            self.canvas[:] = (255, 0, 0)
        elif colour == "White":
            self.canvas[:] = (255, 255, 255)
        elif colour == "Green":
            self.canvas[:] = (0, 255, 0)
        elif colour == "Orange":
            self.canvas[:] = (0, 128, 255)
        elif colour == "Gradient 1":
            self.canvas[:] = self.gradient_bg((17, 192, 134), (196, 148, 103))
        elif colour == "Gradient 2":
            self.canvas[:] = self.gradient_bg((124, 18, 234), (65, 159, 33))
        elif colour == "Gradient 3":
            self.canvas[:] = self.gradient_bg((210, 73, 184), (182, 5, 13))
        elif colour == "Gradient 4":
            self.canvas[:] = self.gradient_bg((159, 118, 211), (79, 252, 216))
        elif colour == "Gradient 5":
            self.canvas[:] = self.gradient_bg((152, 58, 27), (172, 167, 222))
        elif colour == "Gradient 6":
            self.canvas[:] = self.gradient_bg((36, 195, 154), (202, 54, 194))
        elif colour == "Random":
            start = np.array([np.random.choice(range(0, 256)), np.random.choice(range(0, 256)),
                              np.random.choice(range(0, 256))])
            end = np.array([255, 255, 255]) - start
            # array = self.get_gradient_3d(self.width, self.height, (0, 0, 192), (255, 255, 64), (True, False, False))
            array = self.get_gradient_3d(self.width, self.height, start, end, (
                np.random.choice([True, False]), np.random.choice([True, False]), np.random.choice([True, False])))
            self.canvas[:] = array

    def get_gradient_3d(self, width, height, start_list, stop_list, is_horizontal_list):
        result = np.zeros((height, width, len(start_list)), dtype=np.float64)

        for i, (start, stop, is_horizontal) in enumerate(zip(start_list, stop_list, is_horizontal_list)):
            result[:, :, i] = self.get_gradient_2d(start, stop, width, height, is_horizontal)

        return result

    @staticmethod
    def get_gradient_2d(start, stop, width, height, is_horizontal):
        if is_horizontal:
            return np.tile(np.linspace(start, stop, width), (height, 1))
        else:
            return np.tile(np.linspace(start, stop, height), (width, 1)).T

    def gradient_bg(self, colour1, colour2):
        xyz = (True, True, True)
        start = (0, 0, 0)
        end = (1, 1, 1)

        a = np.random.choice(range(0, 4))
        bg1 = np.zeros((self.height, self.width, 3), np.uint8)
        bg1[:] = colour1
        bg2 = np.zeros((self.height, self.width, 3), np.uint8)
        bg2[:] = colour2
        if a == 1:
            xyz = (False, False, False)
            start = (0, 0, 0)
            end = (1, 1, 1)
        elif a == 2:
            xyz = (True, True, True)
            start = (1, 1, 1)
            end = (0, 0, 0)
        elif a == 3:
            xyz = (False, False, False)
            start = (1, 1, 1)
            end = (0, 0, 0)
        mask1 = self.get_gradient_3d(self.width, self.height, start, end, xyz)
        mask2 = self.get_gradient_3d(self.width, self.height, end, start, xyz)
        return mask1 * bg1 + mask2 * bg2

    def draw_shape(self, shape, colour, scale, location, skew, thickness):
        if shape == "Square":
            self.draw_square(colour, scale, location, skew, thickness)
        elif shape == "Triangle":
            self.draw_triangle(colour, scale, location, skew, thickness)
        elif shape == "Circle":
            self.draw_circle(colour, scale, location, thickness)
        elif shape == "Cross":
            self.draw_cross(colour, scale, location, skew, thickness)
        elif shape == "Tear":
            self.draw_tear(colour, scale, location, skew, thickness)
        elif shape == "Heart":
            self.draw_heart(colour, scale, location, skew, thickness)
        elif shape == "Oval":
            self.draw_oval(colour, scale, location, skew, thickness)
        elif shape == "Star":
            self.draw_star(colour, scale, location, skew, thickness)
        elif shape == "Moon":
            self.draw_moon(colour, scale, location, skew, thickness)

    def draw_square(self, colour, scale, location, skew, thickness):
        skew = skew % 90
        side = int(self.width * scale)
        points = np.array(cv2.boxPoints([location, (side, side), float(skew)])).astype(np.int32)
        cv2.drawContours(self.canvas, [points], 0, colour, thickness, lineType=cv2.LINE_AA)

    def draw_triangle(self, colour, scale, location, skew, thickness):
        a = skew % 60
        r = int(self.width * scale / 2)
        x, y = location
        points = np.array([(x + r * cos(radians(a)), y - r * sin(radians(a))),
                           (x + r * cos(radians(a + 120)), y - r * sin(radians(a + 120))),
                           (x + r * cos(radians(a + 240)), y - r * sin(radians(a + 240)))]).astype(np.int32)
        cv2.polylines(self.canvas, [points], TRUE, colour, thickness, lineType=cv2.LINE_AA)

    def draw_circle(self, colour, scale, location, thickness):
        radius = int(self.width * scale / 2)
        cv2.circle(self.canvas, location, radius, colour, thickness, lineType=cv2.LINE_AA)

    def draw_cross(self, colour, scale, location, skew, thickness):
        a = skew % 90
        d = int(self.width * scale / 2)
        x = location[0]
        y = location[1]
        points = [(int(x + d * cos(radians(a))), int(y - d * sin(radians(a)))),
                  (int(x - d * cos(radians(a))), int(y + d * sin(radians(a)))),
                  (int(x + d * cos(radians(a + 90))), int(y - d * sin(radians(a + 90)))),
                  (int(x - d * cos(radians(a + 90))), int(y + d * sin(radians(a + 90))))
                  ]
        cv2.line(self.canvas, points[0], points[1], colour, thickness, lineType=cv2.LINE_AA)
        cv2.line(self.canvas, points[2], points[3], colour, thickness, lineType=cv2.LINE_AA)

    def draw_tear(self, colour, scale, location, a, thickness):
        tear_constant = 2
        tilt = 26
        r = int(self.width * scale)
        rr = int(self.width * scale / tear_constant)
        d = int(self.width * scale / 3)
        x = location[0]
        y = location[1]
        cv2.ellipse(self.canvas, (int(x + d * cos(radians(a))), int(y - d * sin(radians(a)))), (r, rr), tilt - a - 90,
                    168.5,
                    283, colour, thickness,
                    lineType=cv2.LINE_AA)
        cv2.ellipse(self.canvas, (int(x - d * cos(radians(a))), int(y + d * sin(radians(a)))),
                    (r, rr), 270 - a - tilt, 77,
                    191.5, colour, thickness,
                    lineType=cv2.LINE_AA)

    def draw_heart(self, colour, scale, location, a, thickness):
        heart_constant = 4
        bounds = -19
        tilt = 45

        r = int(self.width * scale / 2)
        rr = int(self.width * scale / heart_constant)
        cv2.ellipse(self.canvas, location, (rr, r), a, tilt - 180 + bounds, tilt + bounds, colour, thickness,
                    lineType=cv2.LINE_AA)
        cv2.ellipse(self.canvas, location, (rr, r), a + 2 * tilt, 180 - tilt - bounds, -tilt - bounds, colour,
                    thickness,
                    lineType=cv2.LINE_AA)

    def draw_oval(self, colour, scale, location, a, thickness):
        oval_constant = 3
        r = int(self.width * scale / 2)
        rr = int(self.width * scale / oval_constant)
        cv2.ellipse(self.canvas, location, (r, rr), a, 0, 360, colour, thickness, lineType=cv2.LINE_AA)

    def draw_star(self, colour, scale, location, skew, thickness):
        a = skew % 72
        star_constant = 4
        r = int(self.width * scale / 2)
        rr = int(self.width * scale / star_constant)
        x = location[0]
        y = location[1]
        points = np.array([(x + r * cos(radians(a)), y - r * sin(radians(a))),
                           (x + rr * cos(radians(a + 36)), y - rr * sin(radians(a + 36))),
                           (x + r * cos(radians(a + 72)), y - r * sin(radians(a + 72))),
                           (x + rr * cos(radians(a + 108)), y - rr * sin(radians(a + 108))),
                           (x + r * cos(radians(a + 144)), y - r * sin(radians(a + 144))),
                           (x + rr * cos(radians(a + 180)), y - rr * sin(radians(a + 180))),
                           (x + r * cos(radians(a + 216)), y - r * sin(radians(a + 216))),
                           (x + rr * cos(radians(a + 252)), y - rr * sin(radians(a + 252))),
                           (x + r * cos(radians(a + 288)), y - r * sin(radians(a + 288))),
                           (x + rr * cos(radians(a + 324)), y - rr * sin(radians(a + 324))),
                           ]).astype(np.int32)
        cv2.polylines(self.canvas, [points], TRUE, colour, thickness, lineType=cv2.LINE_AA)

    def draw_moon(self, colour, scale, location, a, thickness):
        moon_constant = 3
        r = int(self.width * scale / 2)
        d = int(self.width * scale / moon_constant)
        aa = degrees(acos((d / 2) / r))
        x = location[0]
        y = location[1]
        cv2.ellipse(self.canvas, location, (r, r), 0, a - (180 - aa), a + (180 - aa), colour, thickness,
                    lineType=cv2.LINE_AA)
        cv2.ellipse(self.canvas, (int(x - d * cos(radians(a))), int(y - d * sin(radians(a)))), (r, r), 0, a - aa,
                    a + aa, colour,
                    thickness, lineType=cv2.LINE_AA)

    def generate_image(self):
        self.set_bg_colour(self.get_rand_bg_colour())
        for i in range(self.n):
            shape = self.get_rand_shape()
            self.details[shape] += 1
            self.draw_shape(shape, self.get_rand_colour(), self.get_rand_scale(),
                            self.get_rand_location(), self.get_rand_skew(), self.get_rand_thickness())

    def show_image(self):
        global running_thread
        running_thread = True
        self.imThread.start()

    class imageThread(threading.Thread):
        def __init__(self, data):
            threading.Thread.__init__(self)
            self.master = self
            self.data = data

        def run(self):
            global running_thread
            cv2.imshow("Image", np.array(self.data, dtype=np.uint8))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            running_thread = False

    def download_image(self, file_name):
        path = f"{os.getcwd()}/generatedImages"
        if not os.path.exists(path):
            os.mkdir(path)
        cv2.imwrite(f"{path}/{file_name}", self.canvas)

    def reset_details(self):
        self.details = {"Square": 0, "Triangle": 0, "Circle": 0, "Cross": 0, "Tear": 0, "Heart": 0, "Oval": 0,
                        "Star": 0, "Moon": 0, "Background": None}

    def format_details(self):
        format_text = f"Background: {self.details['Background']}"
        format_text += f"\nNumber of Shapes: {sum(list(self.details.values())[:-1])}"
        for shape in list(self.details.keys())[:-1]:
            if self.details[shape] != 0:
                format_text += f"\n{shape}: {self.details[shape]}"
        return format_text


class GUI(Tk):
    def __init__(self):
        super(GUI, self).__init__()
        self.image_generator = None
        self.columnconfigure(0, weight=1)
        self.title("Shapes")
        self.iconbitmap("icon.ico")
        self.resizable(False, False)

        font = ("Courier", 20)
        self.size_label = Label(self, text="Size:", font=font)
        self.size_label.grid(row=0, column=0, sticky="e")

        self.width_entry = Entry(self, font=font, width=4)
        self.width_entry.grid(row=0, column=1)
        self.width_entry.insert(END, 512)

        self.x = Label(self, text="×", font=font)
        self.x.grid(row=0, column=2)

        self.height_entry = Entry(self, font=font, width=4)
        self.height_entry.grid(row=0, column=3)
        self.height_entry.insert(END, 512)

        self.generate_button = Button(self, text="Generate Image", font=font, command=self.generate_image)
        self.generate_button.grid(row=1, column=0, columnspan=4, sticky="nesw")

        self.save_button = Button(self, text="Save Image", font=font, command=lambda: self.download_image("image.png"))
        self.save_button.grid(row=2, column=0, columnspan=4, sticky="nesw")

        self.download_button = Button(self, text="Download Many", font=font, command=self.mass_download)
        self.download_button.grid(row=3, column=0, columnspan=4, sticky="nesw")

        self.quantity_label = Label(self, text="Qty:", font=font)
        self.quantity_label.grid(row=3, column=4)

        self.quantity_entry = Entry(self, font=font, width=3)
        self.quantity_entry.grid(row=3, column=5)
        self.quantity_entry.insert(END, 20)

        self.text_box = Text(self, height=11, width=23, state=DISABLED)
        self.text_box.grid(row=0, column=4, rowspan=3, columnspan=2)

    def generate_image(self):
        global running_thread
        if running_thread:
            win32api.MessageBox(0, 'Close the image window', 'Alert', 0x00001000)
        else:
            self.image_generator = ImageGenerator(width=int(self.width_entry.get()),
                                                  height=int(self.height_entry.get()))
            self.image_generator.generate_n()
            self.image_generator.generate_image()
            self.image_generator.show_image()
            im_details = self.image_generator.format_details()
            self.text_box.config(state=NORMAL)
            self.text_box.delete(1.0, END)
            self.text_box.insert(END, im_details)
            self.text_box.config(state=DISABLED)

    def download_image(self, file_name):
        try:
            self.image_generator.download_image(file_name)
        except AttributeError:
            pass

    def mass_download(self):
        qty = int(self.quantity_entry.get())
        for i in range(0, qty):
            self.image_generator = ImageGenerator(width=int(self.width_entry.get()),
                                                  height=int(self.height_entry.get()))
            self.image_generator.generate_n()
            self.image_generator.generate_image()
            self.download_image(f"image{i}.png")


gui = GUI()
gui.mainloop()
