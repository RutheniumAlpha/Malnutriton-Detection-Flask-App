import tkinter as tk
import os


def ai():
    r.destroy()
    print("------------------------------------------")
    print("------ AI Detection Method Running -------")
    print("--- Please close this window after use ---")
    print("------------------------------------------")
    os.system(
        "C:/Users/hp/AppData/Local/Programs/Python/Python38/python.exe ai_technique.py")


def aruco():
    r.destroy()
    print("------------------------------------------")
    print("---- Marker Detection Method Running -----")
    print("--- Please close this window after use ---")
    print("------------------------------------------")
    os.system(
        "C:/Users/hp/AppData/Local/Programs/Python/Python38/python.exe aruco_technique.py")


r = tk.Tk()
r.geometry("1000x500")
r.title('PoshTrack')
text = tk.Label(r, text="Welcome to PoshTrack Computer Vision",
                font=("Poppins Medium", 18))
content = tk.Label(r, text="Please select any one detection method",
                   font=("Poppins Medium", 12))
b1 = tk.Button(r, text="AI Detection Method",
               font=("Poppins Medium", 12), width=40, command=ai)
b2 = tk.Button(r, text="Marker Detection Method",
               font=("Poppins Medium", 12), width=40, command=aruco)
img = tk.PhotoImage(file='logo.png')
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
