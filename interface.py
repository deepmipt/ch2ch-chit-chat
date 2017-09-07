from ch2ch import interface

interface.init('[path]/model')

answer = interface.send('Здравствуйте, нет заисления зарплаты по реестру 9 от 31.03.17 года, реестр висит со вчерашнего дня?')
print(answer)
