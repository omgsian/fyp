import enchant
from nltk.metrics import edit_distance

class SpellingReplacer(object):

    def __init__(self, dict_name='en', max_dist=2):
        self.spell_dict = enchant.Dict(dict_name)
        self.max_dist = 2


    def replace(self, word):
        if self.spell_dict.check(word):
            return word

        suggestions = self.spell_dict.suggest(word)

        if suggestions:
            for suggestion in suggestions:
                if edit_distance(word, suggestion) <= self.max_dist:
                    return suggestions[0]