import tkinter as tk
from tkinter import messagebox
import os


def ai():
    messagebox.showinfo("Info", "You may close the window after use.")
    r.destroy()
    os.system(
        "C:/Users/hp/AppData/Local/Programs/Python/Python38/python.exe ai_technique.py")


def aruco():
    messagebox.showinfo("Info", "You may close the window after use.")
    r.destroy()
    os.system(
        "C:/Users/hp/AppData/Local/Programs/Python/Python38/python.exe aruco_technique.py")


r = tk.Tk()
icon = tk.PhotoImage(file="assets/PoshTrack_Logo_Mini.png")
r.geometry("1000x500")
r.iconphoto(False, icon)
r.title('PoshTrack')
text = tk.Label(r, text="Welcome to PoshTrack Computer Vision",
                font=("Poppins Medium", 18))
content = tk.Label(r, text="Please select any one detection method",
                   font=("Poppins Medium", 12))
b1 = tk.Button(r, text="AI Detection Method",
               font=("Poppins Medium", 12), width=40, command=ai)
b2 = tk.Button(r, text="Marker Detection Method",
               font=("Poppins Medium", 12), width=40, command=aruco)
img = tk.PhotoImage(file='assets/logo.png')
tk.Label(r, text="  ").pack()
tk.Label(r, image=img).pack()
tk.Label(r, text="  ").pack()
text.pack()
tk.Label(r, text="  ").pack()
content.pack()
b1.pack()
tk.Label(r, text="  ").pack()
b2.pack()
tk.Label(r, text="  ").pack()
r.mainloop()
