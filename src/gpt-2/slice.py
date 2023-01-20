lines = []
with open("../../data/opensubtitles/dialogue_length3_6.response",'r') as f:
    for i in range(500000):
        lines.append(f.readline())
    f.close()

with open("../../data/opensubtitles/small_dialogue.response", 'w') as f:
    for i in range(500000):
        f.write(lines[i])