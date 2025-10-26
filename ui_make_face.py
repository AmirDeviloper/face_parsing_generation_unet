from tkinter import *
from tkinter import filedialog, messagebox

from PIL import ImageTk, Image
import cv2
import numpy as np



global face_image_array
global face_image

global face_part_image_array
global face_part_image



def remove_label(part_name):
    # face part stored in blue label
    # face skin stored in red label

    label_color = 'blue' if part_name == 'face' else 'red'

    for widget in root.winfo_children():
        if isinstance(widget, Label) and widget.cget("foreground") == label_color:
            widget.destroy()


def open_face_part():
    try:

        global face_part_image_array
        global face_part_image

        file_path = filedialog.askopenfilename()

        face_part_image_array = cv2.imread(file_path, 0)
        face_part_image = Image.fromarray(face_part_image_array)

        face_part_image = ImageTk.PhotoImage(face_part_image)
        
        remove_label('face parts')
        panel = Label(root, foreground='red', image=face_part_image)

        panel.image = face_part_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face part image", "invalid iamge.")


def open_face_skin():
    try:

        global face_image_array
        global face_image

        file_path = filedialog.askopenfilename()

        face_image_array = cv2.imread(file_path, 0)
        face_image = Image.fromarray(face_image_array)

        face_image = ImageTk.PhotoImage(face_image)
    
        remove_label('face')
        panel = Label(root, foreground='blue', image=face_image)
        panel.image = face_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")


def expand_face_5px():
    try:

        global face_image_array
        global face_image
        padding_value = 5
        face_image_array = cv2.copyMakeBorder(face_image_array,
                                             padding_value, padding_value, padding_value, padding_value,
                                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
        face_image = Image.fromarray(face_image_array)

        face_image = ImageTk.PhotoImage(face_image)
    
        remove_label('face')
        panel = Label(root, foreground='blue', image=face_image)
        panel.image = face_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")

def expand_part_5px():
    try:

        global face_part_image_array
        global face_part_image
        padding_value = 5
        face_part_image_array = cv2.copyMakeBorder(face_part_image_array,
                                                   padding_value, padding_value, padding_value, padding_value,
                                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
        face_part_image = Image.fromarray(face_part_image_array)

        face_part_image = ImageTk.PhotoImage(face_part_image)
    
        remove_label('face parts')
        panel = Label(root, foreground='red', image=face_part_image)
        panel.image = face_part_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")


def crop_part_2px_UD():
    try:

        global face_part_image_array
        global face_part_image
        crop_value = 2
        face_part_image_array = face_part_image_array[:, crop_value:-crop_value]

        face_part_image = Image.fromarray(face_part_image_array)

        face_part_image = ImageTk.PhotoImage(face_part_image)
    
        remove_label('face parts')
        panel = Label(root, foreground='red', image=face_part_image)
        panel.image = face_part_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")

def crop_part_2px_LR():
    try:

        global face_part_image_array
        global face_part_image
        crop_value = 2
        face_part_image_array = face_part_image_array[crop_value:-crop_value, :]

        face_part_image = Image.fromarray(face_part_image_array)

        face_part_image = ImageTk.PhotoImage(face_part_image)
    
        remove_label('face parts')
        panel = Label(root, foreground='red', image=face_part_image)
        panel.image = face_part_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")

def crop_face_2px():
    try:

        global face_image_array
        global face_image
        crop_value = 2

        face_image_array = face_image_array[crop_value:-crop_value, crop_value:-crop_value]
        face_image = Image.fromarray(face_image_array)

        face_image = ImageTk.PhotoImage(face_image)
    
        remove_label('face')
        panel = Label(root, foreground='blue', image=face_image)
        panel.image = face_image
        panel.pack()
        
    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")

def crop_part_from_middle_2px():
    try:

        global face_part_image_array
        global face_part_image
        crop_value = 2
        
        

        w, h = face_part_image_array.shape
        left_columns = face_part_image_array[: w, :(h // 2) - crop_value]
        right_columns = face_part_image_array[: w, (h // 2) + crop_value:]
        face_part_image_array = cv2.hconcat([left_columns, right_columns])

        face_part_image = Image.fromarray(face_part_image_array)

        face_part_image = ImageTk.PhotoImage(face_part_image)
    
        remove_label('face parts')
        panel = Label(root, foreground='red', image=face_part_image)
        panel.image = face_part_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")

def expand_part_from_middle_2px():
    try:

        global face_part_image_array
        global face_part_image
        expand_value = 2

        w, h = face_part_image_array.shape

        middle_column = np.tile(face_part_image_array[: w, h // 2], (expand_value, 1))
        middle_column = middle_column.transpose()
        left_columns = face_part_image_array[: w, :(h // 2)]
        left_columns = cv2.hconcat([left_columns, middle_column])

        right_columns = face_part_image_array[: w, (h // 2): ]
        right_columns = cv2.hconcat([middle_column, right_columns])
        face_part_image_array = cv2.hconcat([left_columns, right_columns])

        face_part_image = Image.fromarray(face_part_image_array)

        face_part_image = ImageTk.PhotoImage(face_part_image)
    
        remove_label('face parts')
        panel = Label(root, foreground='red', image=face_part_image)
        panel.image = face_part_image
        panel.pack()

    except Exception as e:
        print(e)
        messagebox.showinfo("please select valid face image", "invalid iamge.")


root = Tk()
root.title('face generation')
root.resizable(False, False)
root.geometry('700x850')
root.configure(background='darkgray')

N = 40
M = 600
arrow_dw = 320

button_face = Button(root, text="SELECT FACE SKIN", bg="lime", fg='black', command=open_face_skin)
button_face.place(x=10 + 12, y=10 + 750)

button_facepart = Button(root, text="SELECT FACE PART", bg="darkblue", fg='white', command=open_face_part)
button_facepart.place(x=10 + 10, y=40 + 750)

button_facepart = Button(root, text="SAVE CHANGES", bg="violet", fg='black', command=open_face_part)
button_facepart.place(x=10 + 10, y=70 + 750)

button_facepart = Button(root, text="🢁", bg="cyan", fg='black', command=open_face_part)
button_facepart.place(x=15 + arrow_dw, y=70 + 670)

button_facepart = Button(root, text="⭮", bg="darkcyan", fg='black', command=crop_part_from_middle_2px)
button_facepart.place(x=5 + 305, y=70 + 670)

button_facepart = Button(root, text="⭯", bg="darkcyan", fg='black', command=expand_part_from_middle_2px)
button_facepart.place(x=25 + 335, y=70 + 670)

button_facepart = Button(root, text="🢀", bg="cyan", fg='black', command=open_face_part)
button_facepart.place(x=5 + 305, y=100 + 670)

button_facepart = Button(root, text="🢂", bg="cyan", fg='black', command=open_face_part)
button_facepart.place(x=25 + 335, y=100 + 670)

button_facepart = Button(root, text="🢃", bg="cyan", fg='black', command=open_face_part)
button_facepart.place(x=15 + arrow_dw, y=100 + 670)

button_facepart = Button(root, text="↕", bg="darkred", fg='black', command=crop_part_2px_LR)
button_facepart.place(x=15 + 372, y=70 + 670)

button_facepart = Button(root, text="↔", bg="darkred", fg='black', command=crop_part_2px_UD)
button_facepart.place(x=15 + 370, y=100 + 670)

button_facepart = Button(root, text="🢅", bg="darkgreen", fg='white', command=expand_face_5px)
button_facepart.place(x=15 + 270, y=70 + 670)

button_facepart = Button(root, text="🢆", bg="darkRED", fg='white', command=crop_face_2px)
button_facepart.place(x=15 + 270, y=100 + 670)

#button_facepart = Button(root, text="PADDING PART", bg="darkgreen", fg='white', command=expand_face_5px)
#button_facepart.place(x=15 + 560, y=160 + M)



#button_facepart = Button(root, text="CROPPING PART", bg="darkRED", fg='white', command=crop_face_2px)
#button_facepart.place(x=15 + 550, y=190 + M)

button_facepart = Button(root, text="SAVE AS FINAL", bg="green", fg='lime', command=crop_face_2px)
button_facepart.place(x=15 + 550, y=220 + M)


#face_image_array = cv2.imread('model_Attention_UNet_Conv2D_10000imgs_3epochs_transfer_learning1.h5_results\\skin\\low\\3885_skin.jpg', 0)
#face_image = Image.fromarray(face_image_array)

#w, h = face_image.size
#x, y = w/2, h/2

#my_canvas = Canvas(root, width=w, height=h, bg='white')
#my_canvas.pack(pady=20)

#img = ImageTk.PhotoImage(face_image)
#my_image = my_canvas.create_image(1, 1, anchor=NW, image=img)

root.mainloop()


#import cv2
#import numpy as np
#from matplotlib import pyplot as plt

#img = cv2.imread('panda1.jpg')

#blur = cv2.blur(img,(5,5))

#plt.imshow(img),plt.title('Original')
#plt.show()
#cv2.imwrite('orig.png', img)
#plt.xticks([]), plt.yticks([])
#plt.imshow(blur),plt.title('Blurred')
#plt.xticks([]), plt.yticks([])
#plt.show()
#cv2.imwrite('boxfil.png', blur)

