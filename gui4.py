"""
gui3.py
AREIS — Industrial Operations Assistant
Pure Python / CustomTkinter GUI.

Backend: gui3.py -> chat_engine3.py -> stream_reply()
Adds: webcam capture with live thumbnail preview, and an optional
"Live Monitor" mode that auto-captures + analyzes on an interval.
"""

import threading
import time
import customtkinter as ctk
from PIL import Image

from chat_engine4 import stream_reply, clear_memory, capture_webcam_image


ctk.set_appearance_mode("dark")

BG = "#171714"
PANEL = "#22211D"
PANEL_LIGHT = "#2A2924"
BORDER = "#454239"
MUD = "#625B49"
MUD_DARK = "#514B3D"
ACCENT = "#A18B5B"
ACCENT_HOVER = "#B19B68"

TEXT = "#E3DFD2"
TEXT_SECONDARY = "#A9A394"
TEXT_DIM = "#777267"

SUCCESS = "#8F9B78"
ERROR = "#A87565"

FONT = "Segoe UI"

MONITOR_PROMPT = (
    "You are watching a live camera feed snapshot. Briefly report anything "
    "notable, unsafe, or changed in this frame. If nothing stands out, "
    "just say 'No change.' Keep it to one short sentence."
)


def launch_app():

    app = ctk.CTk()
    app.title("AREIS - Industrial Operations Assistant")
    app.geometry("1050x720")
    app.minsize(900, 620)
    app.configure(fg_color=BG)

    app.grid_columnconfigure(1, weight=1)
    app.grid_rowconfigure(1, weight=1)

    pending_image = {"path": None}
    monitor_state = {"running": False, "interval": 5}

    header = ctk.CTkFrame(app, height=68, corner_radius=0, fg_color=PANEL,
                           border_width=1, border_color=BORDER)
    header.grid(row=0, column=0, columnspan=2, sticky="nsew")
    header.grid_columnconfigure(1, weight=1)

    title = ctk.CTkLabel(header, text="AREIS", font=(FONT, 24, "bold"), text_color=TEXT)
    title.grid(row=0, column=0, padx=(24, 8), pady=(10, 0), sticky="sw")

    subtitle = ctk.CTkLabel(header, text="INDUSTRIAL OPERATIONS ASSISTANT",
                             font=(FONT, 11), text_color=TEXT_SECONDARY)
    subtitle.grid(row=1, column=0, padx=(25, 8), pady=(0, 10), sticky="nw")

    status_frame = ctk.CTkFrame(header, fg_color="transparent")
    status_frame.grid(row=0, column=1, rowspan=2, padx=25, sticky="e")

    status_indicator = ctk.CTkLabel(status_frame, text="DOT", font=(FONT, 14), text_color=SUCCESS)
    status_indicator.pack(side="left", padx=(0, 7))

    status_text = ctk.CTkLabel(status_frame, text="SYSTEM ONLINE",
                                font=(FONT, 11, "bold"), text_color=TEXT_SECONDARY)
    status_text.pack(side="left")

    sidebar = ctk.CTkFrame(app, width=235, corner_radius=0, fg_color=PANEL,
                            border_width=1, border_color=BORDER)
    sidebar.grid(row=1, column=0, sticky="nsew")
    sidebar.grid_propagate(False)

    operations_label = ctk.CTkLabel(sidebar, text="OPERATIONS", font=(FONT, 11, "bold"),
                                     text_color=TEXT_SECONDARY)
    operations_label.pack(anchor="w", padx=20, pady=(25, 15))

    active_context = ctk.StringVar(value="INDUSTRY")

    def select_context(context):
        active_context.set(context)
        if context == "INDUSTRY":
            industry_button.configure(fg_color=MUD, text_color=TEXT)
            employee_button.configure(fg_color=PANEL_LIGHT, text_color=TEXT_SECONDARY)
            context_value.configure(text="INDUSTRY DATA")
        else:
            employee_button.configure(fg_color=MUD, text_color=TEXT)
            industry_button.configure(fg_color=PANEL_LIGHT, text_color=TEXT_SECONDARY)
            context_value.configure(text="EMPLOYEE DATA")

    industry_button = ctk.CTkButton(sidebar, text="INDUSTRY DATA MANAGEMENT", height=44,
                                     corner_radius=4, fg_color=MUD, hover_color=MUD_DARK,
                                     text_color=TEXT, font=(FONT, 11, "bold"), anchor="w",
                                     command=lambda: select_context("INDUSTRY"))
    industry_button.pack(fill="x", padx=15, pady=4)

    employee_button = ctk.CTkButton(sidebar, text="EMPLOYEE DATA MANAGEMENT", height=44,
                                     corner_radius=4, fg_color=PANEL_LIGHT, hover_color=MUD_DARK,
                                     text_color=TEXT_SECONDARY, font=(FONT, 11, "bold"), anchor="w",
                                     command=lambda: select_context("EMPLOYEE"))
    employee_button.pack(fill="x", padx=15, pady=4)

    separator = ctk.CTkFrame(sidebar, height=1, fg_color=BORDER)
    separator.pack(fill="x", padx=15, pady=(25, 20))

    thinking_label = ctk.CTkLabel(sidebar, text="INFERENCE MODE", font=(FONT, 10, "bold"),
                                   text_color=TEXT_SECONDARY)
    thinking_label.pack(anchor="w", padx=20, pady=(0, 10))

    think_mode = ctk.StringVar(value="auto")

    def set_mode(value):
        think_mode.set(value)

    for label, val in [("AUTO", "auto"), ("ALWAYS ON", "on"), ("ALWAYS OFF", "off")]:
        ctk.CTkRadioButton(sidebar, text=label, variable=think_mode, value=val,
                            command=lambda v=val: set_mode(v), font=(FONT, 11),
                            text_color=TEXT_SECONDARY, fg_color=ACCENT,
                            hover_color=ACCENT_HOVER, border_color=TEXT_DIM
                            ).pack(anchor="w", padx=20, pady=5)

    separator2 = ctk.CTkFrame(sidebar, height=1, fg_color=BORDER)
    separator2.pack(fill="x", padx=15, pady=(25, 20))

    vision_label = ctk.CTkLabel(sidebar, text="VISION / CAMERA", font=(FONT, 10, "bold"),
                                 text_color=TEXT_SECONDARY)
    vision_label.pack(anchor="w", padx=20, pady=(0, 10))

    preview_frame = ctk.CTkFrame(sidebar, width=195, height=140, fg_color=PANEL_LIGHT,
                                  border_width=1, border_color=BORDER)
    preview_frame.pack(padx=20, pady=(0, 10))
    preview_frame.pack_propagate(False)

    preview_label = ctk.CTkLabel(preview_frame, text="NO IMAGE", text_color=TEXT_DIM,
                                  font=(FONT, 10))
    preview_label.pack(expand=True)

    def update_preview(image_path):
        try:
            pil_img = Image.open(image_path)
            pil_img.thumbnail((190, 135))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            preview_label.configure(image=ctk_img, text="")
            preview_label.image = ctk_img
        except Exception:
            preview_label.configure(text="[preview error]", image=None)

    def on_capture_click():
        try:
            path = capture_webcam_image()
            pending_image["path"] = path
            update_preview(path)
            capture_status.configure(text="Attached to next message", text_color=SUCCESS)
        except Exception as e:
            capture_status.configure(text=f"Error: {e}", text_color=ERROR)

    capture_btn = ctk.CTkButton(sidebar, text="CAPTURE SNAPSHOT", height=36, corner_radius=4,
                                 fg_color=MUD, hover_color=MUD_DARK, text_color=TEXT,
                                 font=(FONT, 11, "bold"), command=on_capture_click)
    capture_btn.pack(fill="x", padx=20, pady=(0, 6))

    capture_status = ctk.CTkLabel(sidebar, text="", font=(FONT, 9), text_color=TEXT_DIM)
    capture_status.pack(anchor="w", padx=20, pady=(0, 15))

    monitor_row = ctk.CTkFrame(sidebar, fg_color="transparent")
    monitor_row.pack(fill="x", padx=20, pady=(0, 8))

    interval_menu = ctk.CTkOptionMenu(monitor_row, values=["5 sec", "30 sec"], width=90,
                                       fg_color=PANEL_LIGHT, button_color=MUD,
                                       button_hover_color=MUD_DARK, text_color=TEXT,
                                       font=(FONT, 10))
    interval_menu.set("5 sec")
    interval_menu.pack(side="left")

    def toggle_monitor():
        if not monitor_state["running"]:
            interval_str = interval_menu.get()
            monitor_state["interval"] = 5 if "5" in interval_str else 30
            monitor_state["running"] = True
            monitor_btn.configure(text="STOP MONITOR", fg_color=ERROR, hover_color="#8A5F52")
            monitor_status.configure(
                text=f"Monitoring every {monitor_state['interval']}s", text_color=SUCCESS
            )
            threading.Thread(target=monitor_loop, daemon=True).start()
        else:
            monitor_state["running"] = False
            monitor_btn.configure(text="START MONITOR", fg_color=MUD, hover_color=MUD_DARK)
            monitor_status.configure(text="Monitor stopped", text_color=TEXT_DIM)

    monitor_btn = ctk.CTkButton(monitor_row, text="START MONITOR", height=32, width=95,
                                 corner_radius=4, fg_color=MUD, hover_color=MUD_DARK,
                                 text_color=TEXT, font=(FONT, 10, "bold"),
                                 command=toggle_monitor)
    monitor_btn.pack(side="left", padx=(8, 0))

    monitor_status = ctk.CTkLabel(sidebar, text="Monitor off", font=(FONT, 9), text_color=TEXT_DIM)
    monitor_status.pack(anchor="w", padx=20, pady=(2, 0))

    def monitor_loop():
        while monitor_state["running"]:
            try:
                path = capture_webcam_image()
                app.after(0, lambda p=path: update_preview(p))
                app.after(0, lambda: append_text("\nMONITOR\n[auto-capture]\n"))

                for kind, data in stream_reply(MONITOR_PROMPT, force_think=False, image_path=path):
                    if kind == "token":
                        app.after(0, lambda t=data: append_text(t))
                    elif kind == "error":
                        app.after(0, lambda e=data: append_text(f"\n[Monitor error: {e}]\n"))
                    elif kind == "done":
                        app.after(0, lambda: append_text("\n\n"))
            except Exception as e:
                app.after(0, lambda err=e: append_text(f"\n[Monitor error: {err}]\n"))

            slept = 0
            while slept < monitor_state["interval"] and monitor_state["running"]:
                time.sleep(0.5)
                slept += 0.5

    sidebar_bottom = ctk.CTkFrame(sidebar, fg_color="transparent")
    sidebar_bottom.pack(side="bottom", fill="x", padx=15, pady=18)

    db_status = ctk.CTkLabel(sidebar_bottom, text="DATABASE CONNECTED", font=(FONT, 10), text_color=SUCCESS)
    db_status.pack(anchor="w", pady=2)

    engine_status = ctk.CTkLabel(sidebar_bottom, text="INFERENCE ENGINE ONLINE", font=(FONT, 10), text_color=SUCCESS)
    engine_status.pack(anchor="w", pady=2)

    content = ctk.CTkFrame(app, corner_radius=0, fg_color=BG)
    content.grid(row=1, column=1, sticky="nsew")
    content.grid_columnconfigure(0, weight=1)
    content.grid_rowconfigure(1, weight=1)

    content_header = ctk.CTkFrame(content, height=55, fg_color="transparent")
    content_header.grid(row=0, column=0, sticky="ew", padx=22, pady=(15, 0))
    content_header.grid_columnconfigure(0, weight=1)

    conversation_label = ctk.CTkLabel(content_header, text="OPERATOR CONSOLE",
                                       font=(FONT, 13, "bold"), text_color=TEXT)
    conversation_label.grid(row=0, column=0, sticky="w")

    context_value = ctk.CTkLabel(content_header, text="INDUSTRY DATA",
                                  font=(FONT, 10, "bold"), text_color=ACCENT)
    context_value.grid(row=1, column=0, sticky="w", pady=(2, 0))

    chat_box = ctk.CTkTextbox(content, wrap="word", corner_radius=4, fg_color=PANEL,
                               border_width=1, border_color=BORDER, text_color=TEXT,
                               font=(FONT, 13))
    chat_box.grid(row=1, column=0, sticky="nsew", padx=22, pady=(5, 12))
    chat_box.configure(state="disabled")

    def append_text(text):
        chat_box.configure(state="normal")
        chat_box.insert("end", text)
        chat_box.configure(state="disabled")
        chat_box.see("end")

    def clear_chat():
        clear_memory()
        chat_box.configure(state="normal")
        chat_box.delete("1.0", "end")
        chat_box.configure(state="disabled")

    input_container = ctk.CTkFrame(content, height=62, fg_color=PANEL, corner_radius=4,
                                    border_width=1, border_color=BORDER)
    input_container.grid(row=2, column=0, sticky="ew", padx=22, pady=(0, 10))
    input_container.grid_columnconfigure(0, weight=1)

    entry = ctk.CTkEntry(input_container, height=40, corner_radius=3, border_width=0,
                          fg_color=PANEL_LIGHT, text_color=TEXT,
                          placeholder_text="ENTER OPERATOR QUERY...",
                          placeholder_text_color=TEXT_DIM, font=(FONT, 12))
    entry.grid(row=0, column=0, sticky="ew", padx=(10, 8), pady=10)

    def send_message(event=None):
        user_msg = entry.get().strip()
        image_path = pending_image["path"]

        if not user_msg and not image_path:
            return
        if not user_msg and image_path:
            user_msg = "Describe what you see in this image."

        entry.delete(0, "end")
        entry.configure(state="disabled")

        suffix = " [image attached]" if image_path else ""
        append_text(f"\nOPERATOR\n{user_msg}{suffix}\n\n")

        pending_image["path"] = None
        capture_status.configure(text="")

        mode = think_mode.get()
        force_think = {"auto": None, "on": True, "off": False}[mode]

        def worker():
            app.after(0, lambda: append_text("AREIS\n"))
            for kind, data in stream_reply(user_msg, force_think=force_think, image_path=image_path):
                if kind == "thinking_start":
                    app.after(0, lambda: append_text("[PROCESSING] "))
                elif kind == "token":
                    app.after(0, lambda t=data: append_text(t))
                elif kind == "error":
                    app.after(0, lambda e=data: append_text(f"\n\nSYSTEM ERROR: {e}\n"))
                elif kind == "done":
                    app.after(0, lambda: append_text("\n\n"))

            app.after(0, lambda: entry.configure(state="normal"))
            app.after(0, lambda: entry.focus())

        threading.Thread(target=worker, daemon=True).start()

    send_btn = ctk.CTkButton(input_container, text="TRANSMIT", width=105, height=40,
                              corner_radius=3, fg_color=MUD, hover_color=MUD_DARK,
                              text_color=TEXT, font=(FONT, 11, "bold"), command=send_message)
    send_btn.grid(row=0, column=1, padx=(0, 10), pady=10)

    footer = ctk.CTkFrame(content, height=38, fg_color="transparent")
    footer.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 12))

    clear_btn = ctk.CTkButton(footer, text="PURGE MEMORY", width=125, height=30, corner_radius=3,
                               fg_color="transparent", hover_color=MUD_DARK, border_width=1,
                               border_color=BORDER, text_color=TEXT_SECONDARY,
                               font=(FONT, 10, "bold"), command=clear_chat)
    clear_btn.pack(side="right")

    entry.bind("<Return>", send_message)
    entry.focus()

    app.mainloop()