import json

path = "/home/kaia/recall_1108/datasets_llama/race/train.jsonl"
for i,line in enumerate(open(path)):
    try:
        json.loads(line)
    except Exception as e:
        print(f"❌ Error at line {i+1}: {e}")
        print("Line content:", line[:200])
        break
