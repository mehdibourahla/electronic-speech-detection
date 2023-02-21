import os
import json
from glob import glob

with open("config.json") as f:
    config = json.load(f)


root = config["my_recording"]
dest = os.path.join(root, "wav")
ext = config["raw_extension"]


if not os.path.exists(dest):
    os.mkdir(dest)

for filepath in glob(os.path.join(root, "*" + ext)):
    filename = filepath.split("\\")[-1]
    output_file = os.path.join(dest, filename.split(".")[0] + ".wav")

    cmd = 'ffmpeg -i "' + filepath + '" "' + output_file + '"'
    os.system(cmd)
