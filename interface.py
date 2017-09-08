from ch2ch import interface
import os

model_path = os.environ.get("CHITCHAT_MODEL")
interface.init(model_path)

while True:
    q = input("[Ваш вопрос]> ")
    r = interface.send(q)
    print("[Ответ]> %s" % r)
