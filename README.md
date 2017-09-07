# Char-rnn is based on lstm for chit-chat

#### Model

wget http://share.ipavlov.mipt.ru:8080/repository/models/chitchat/ch2ch-chit-chat-v0.2.tgz

#### Example
```
from ch2ch import interface

interface.init('{PATH}/model')

answer = interface.send('Здравствуйте, нет заисления зарплаты по реестру 9 от 31.03.17 года, реестр висит со вчерашнего дня?')


```
