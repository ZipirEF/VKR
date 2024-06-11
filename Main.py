import vk_api
from vk_api.longpoll import VkLongPoll, VkEventType
from config import TOKEN

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
import torch

import sqlite3

# Подключение к базе данных
conn = sqlite3.connect('user_status.db')
cursor = conn.cursor()

# Создание таблицы для отслеживания статуса пользователей
cursor.execute('''CREATE TABLE IF NOT EXISTS user_status
                (user_id INTEGER PRIMARY KEY, greeted INTEGER)''')

conn.commit()

# Функция для отправки приветственного сообщения
def hello(user_id):
    cursor.execute("INSERT into user_status VALUES (?,1)", (user_id,))
    sender(user_id, "Привет, пользователь! Добро пожаловать в чат с ботом. Я умею определять эмоции текстовых сообщений. Отправь мне свое сообщение.")

# Загрузка обученной модели
model_ = BertForSequenceClassification.from_pretrained("model")
tokenizer_ = AutoTokenizer.from_pretrained("model")
# Подключение к VK
vk_session = vk_api.VkApi(token=TOKEN)
vk = vk_session.get_api()
longpoll = VkLongPoll(vk_session)
print(vk)
print(longpoll)
# Функция отправки сообщения пользователю
def sender(id, text):
    vk.messages.send(user_id=id, message=text, random_id=0)
# Токенизация
def tokenize(sentence):
    encoded_sentence = tokenizer_.encode_plus(
        sentence,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        max_length=256,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        truncation=True,
    )
    return(encoded_sentence)
# Основной цикл
for event in longpoll.listen():
    if event.type == VkEventType.MESSAGE_NEW:
        if event.to_me:
            if event.text:
                user_id = event.user_id
                cursor.execute("SELECT greeted FROM user_status WHERE user_id=?", (user_id,))
                result = cursor.fetchone()
                if result is None:
                    hello(user_id)
                else:
                    greeted = result[0]
                    if greeted == 0:
                        hello(user_id)
                    else:
                        sentence = event.text
                        id = event.user_id
                        encoded_sentence = tokenize(sentence)
                        input_ids = []
                        attention_masks = []

                        input_ids.append(encoded_sentence['input_ids'])
                        attention_masks.append(encoded_sentence['attention_mask'])

                        input_ids = torch.cat(input_ids, dim=0)
                        attention_masks = torch.cat(attention_masks, dim=0)

                        outputs = model_(input_ids, token_type_ids=None,
                                         attention_mask=attention_masks)
                        logits = outputs[0]
                        softmax = torch.nn.Softmax(dim=1)
                        probabilities = softmax(logits)
                        predicted_classes = torch.argmax(probabilities, dim=1)
                        emotion_labels = ['нейтральность', 'радость', 'грусть', 'страх', 'злость', 'заинтересованность', 'отвращение']
                        for i, index in enumerate(predicted_classes):
                            sender(id,f"Данная эмоция - {emotion_labels[index]} (с вероятностью {probabilities[i][index] * 100:.0f}%)")
            else:
                user_id = event.user_id
                cursor.execute("SELECT greeted FROM user_status WHERE user_id=?", (user_id,))
                result = cursor.fetchone()
                if result is None:
                    hello(user_id)
                else:
                    greeted = result[0]
                    if greeted == 0:
                        hello(user_id)
                    else:
                        id = event.user_id
                        sender(id, 'Вы ввели не текст =(, пожалуйства отправляйте мне только текстовые сообщения')