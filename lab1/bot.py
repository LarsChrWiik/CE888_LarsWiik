
from sopel import module
from emo.wdemotions import EmotionDetector

emo = EmotionDetector()

@module.rule('a')
def hi(bot, trigger):
    print(trigger)
