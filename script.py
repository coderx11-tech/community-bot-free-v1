import os
from pathlib import Path
import zipfile
import json
import urllib.request

# -------------------------------
# Base directories
# -------------------------------
BASE_DIR = Path("robot_full_package")
TASKS_DIR = BASE_DIR / "tasks"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
MOTORS_DIR = BASE_DIR / "motors"
WHEELS_DIR = BASE_DIR / "wheels"
VISION_DIR = BASE_DIR / "vision"
SPEAKER_DIR = BASE_DIR / "speaker"
LLM_DIR = BASE_DIR / "llm"

for folder in [TASKS_DIR, KNOWLEDGE_DIR, MOTORS_DIR, WHEELS_DIR, VISION_DIR, SPEAKER_DIR, LLM_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# -------------------------------
# Download EfficientDet-Lite model
# -------------------------------
TFLITE_URL = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/model_maker/efficientdet_lite0.tflite"
TFLITE_PATH = VISION_DIR / "efficientdet_lite.tflite"
urllib.request.urlretrieve(TFLITE_URL, TFLITE_PATH)
print(f"Downloaded EfficientDet-Lite to {TFLITE_PATH}")

# -------------------------------
# Download GPT4All small LLM
# -------------------------------
LLM_URL = "https://gpt4all.io/models/ggml/gpt4all-lora-quantized.bin"
LLM_PATH = LLM_DIR / "gpt4all-lora-quantized.bin"
urllib.request.urlretrieve(LLM_URL, LLM_PATH)
print(f"Downloaded GPT4All model to {LLM_PATH}")

# -------------------------------
# Example JSON files
# -------------------------------
(OBJECTS_FILE := KNOWLEDGE_DIR / "objects.json").write_text(json.dumps({}))
(EXAMPLE_TASK := TASKS_DIR / "example_task.json").write_text(json.dumps(["Pick up cup", "Pour water", "Release"]))

# -------------------------------
# Motor control
# -------------------------------
(MOTORS_DIR / "motor_control.py").write_text("""\
import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
ARM_PIN, WRIST_PIN, GRIPPER_PIN = 17, 27, 22
for pin in [ARM_PIN, WRIST_PIN, GRIPPER_PIN]: GPIO.setup(pin, GPIO.OUT)
arm_pwm = GPIO.PWM(ARM_PIN,50)
wrist_pwm = GPIO.PWM(WRIST_PIN,50)
gripper_pwm = GPIO.PWM(GRIPPER_PIN,50)
arm_pwm.start(0); wrist_pwm.start(0); gripper_pwm.start(0)
def move_arm(pos): arm_pwm.ChangeDutyCycle(pos)
def move_wrist(pos): wrist_pwm.ChangeDutyCycle(pos)
def gripper_open(): gripper_pwm.ChangeDutyCycle(5)
def gripper_close(): gripper_pwm.ChangeDutyCycle(10)
""")

# -------------------------------
# Wheel control
# -------------------------------
(WHEELS_DIR / "wheel_control.py").write_text("""\
import RPi.GPIO as GPIO
LEFT_PIN1, LEFT_PIN2, RIGHT_PIN1, RIGHT_PIN2 = 5,6,13,19
for pin in [LEFT_PIN1,LEFT_PIN2,RIGHT_PIN1,RIGHT_PIN2]: GPIO.setup(pin, GPIO.OUT)
def move_forward(): GPIO.output([LEFT_PIN1,RIGHT_PIN1],[True,True]); GPIO.output([LEFT_PIN2,RIGHT_PIN2],[False,False])
def stop(): [GPIO.output(pin,False) for pin in [LEFT_PIN1,LEFT_PIN2,RIGHT_PIN1,RIGHT_PIN2]]
""")

# -------------------------------
# Speaker wrapper
# -------------------------------
(SPEAKER_DIR / "speak.py").write_text("""\
import os
def say(text): os.system(f'espeak-ng "{text}"')
""")

# -------------------------------
# Vision using EfficientDet-Lite
# -------------------------------
(VISION_DIR / "vision.py").write_text("""\
import cv2, numpy as np, tflite_runtime.interpreter as tflite
MODEL_PATH = "vision/efficientdet_lite.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
cap = cv2.VideoCapture(0)
def classify_frame(frame):
    img=cv2.resize(frame,(300,300))
    input_data=np.expand_dims(img.astype(np.float32)/255.0,axis=0)
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output
def look1(): ret, frame = cap.read(); return classify_frame(frame) if ret else None
def look2(): ret, frame = cap.read(); return classify_frame(frame) if ret else None
""")

# -------------------------------
# LLM wrapper
# -------------------------------
(LLM_DIR / "llm_model.py").write_text("""\
from gpt4all import GPT4All
MODEL_PATH = "llm/gpt4all-lora-quantized.bin"
class LanguageModel:
    def __init__(self): self.model = GPT4All(model=MODEL_PATH)
    def ask(self,prompt): return self.model.generate(prompt)
llm = LanguageModel()
""")

# -------------------------------
# Main robot script with microphone
# -------------------------------
(BASE_DIR / "robot.py").write_text("""\
import os, json, random, time, re
import speech_recognition as sr
from motors import motor_control as motors
from wheels import wheel_control as wheels
from speaker import speak
from vision import vision
from llm.llm_model import llm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TASKS_DIR = os.path.join(BASE_DIR,"tasks")
KNOWLEDGE_DIR = os.path.join(BASE_DIR,"knowledge")
OBJECTS_FILE = os.path.join(KNOWLEDGE_DIR,"objects.json")
if os.path.exists(OBJECTS_FILE):
    with open(OBJECTS_FILE,"r") as f: objects_knowledge=json.load(f)
else: objects_knowledge={}

def save_task(name,steps):
    path=os.path.join(TASKS_DIR,f"{name}.json")
    with open(path,"w") as f: json.dump(steps,f)
    speak.say(f"Task {name} saved")

def execute_task(name):
    path=os.path.join(TASKS_DIR,f"{name}.json")
    if os.path.exists(path):
        with open(path,"r") as f: steps=json.load(f)
        speak.say(f"Executing task {name}")
        for step in steps:
            print(f"[TASK] {step}")
            if "pick" in step.lower():
                motors.gripper_open(); motors.move_arm(random.randint(10,90)); motors.gripper_close()
            elif "release" in step.lower(): motors.gripper_open()
            elif "move" in step.lower(): wheels.move_forward()
            time.sleep(0.5)
    else: speak.say(f"Task {name} not found")

def listen_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        speak.say("I'm listening...")
        audio = r.listen(source)
        try: text = r.recognize_google(audio)
        except: text = ""
    return text

def chat_mode():
    speak.say("Buddy mode activated with microphone!")
    while True:
        user_input = listen_command()
        if not user_input:
            continue
        print(f"You said: {user_input}")
        cmd=user_input.lower()
        if cmd in ["quit","exit","bye"]:
            speak.say("Goodbye buddy!")
            break

        if cmd.startswith("this is"):
            obj=cmd.replace("this is","").strip()
            speak.say("Tell me more about it")
            desc = listen_command()
            objects_knowledge[obj]={"description":desc}
            with open(OBJECTS_FILE,"w") as f: json.dump(objects_knowledge,f)
            speak.say(f"Got it. I saved {obj}")
            continue

        if "learn a task" in cmd:
            speak.say("What is the task name?")
            name = listen_command()
            steps=[]
            speak.say("Tell me steps one by one, say done to finish")
            while True:
                s = listen_command()
                if s.lower()=="done": break
                steps.append(s)
            save_task(name,steps)
            continue

        m=re.match(r"(execute|do)\s+(.*)",cmd)
        if m: execute_task(m.group(2)); continue
        if "look1" in cmd: label=vision.look1(); speak.say(f"I see {label} in your hand"); continue
        if "look2" in cmd: label=vision.look2(); speak.say(f"I see {label} you are pointing at"); continue

        # Otherwise chat
        answer=llm.ask(cmd)
        print(f"Robot: {answer}")
        speak.say(answer)

if __name__=="__main__":
    chat_mode()
""")

# -------------------------------
# Shell script to install deps and run
# -------------------------------
(BASE_DIR / "run_robot.sh").write_text("""\
#!/bin/bash
echo "Installing dependencies..."
sudo apt update
sudo apt install -y espeak portaudio19-dev python3-pyaudio
pip3 install opencv-python tflite-runtime RPi.GPIO gpt4all SpeechRecognition
echo "Running robot with buddy mode..."
python3 robot.py
""")
(BASE_DIR / "run_robot.sh").chmod(0o755)

# -------------------------------
# Create ZIP
# -------------------------------
zip_path = "robot_full_package.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for file in BASE_DIR.rglob("*"):
        zf.write(file, arcname=file.relative_to(BASE_DIR))

print(f"Full robot package with microphone, LLM, EfficientDet-Lite zipped: {zip_path}")
