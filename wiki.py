import wikipedia
import time

target_pages = ['World War II', 'Bretton Woods Conference', 'Potsdam Declaration', 'Operation Overlord', 'Battle of France', 
                'Tanks in World War II', 'Aviation in World War II','Battle of the Bulge', 'Dunkirk evacuation',
                'Battle of Leyte Gulf', 'Battle of Britain', 'Attack on Pearl Harbor', 'Air warfare of World War II',
                'Japanese invasion of Manchuria', 'Causes of World War II', 'Operation Barbarossa', 'North African campaign',
                'Afrika Korps', 'Winston Churchill', 'Adolf Hitler', 'Manhattan Project']

all_content = []

for i in target_pages:

    text = wikipedia.page(i, auto_suggest=False)
    header = f"\n\n{"="*20}\nDOCUMENT: {text.title}\n{'='*20}\n"
    all_content.append(header + text.content)
    time.sleep(1)

mega = "".join(all_content)

with open("data/mega.txt", "w", encoding="utf-8") as f:
    f.write(mega)