import random

from hydra import compose


class NaivePromptGenerator:
    def __init__(self, config_name):
        cfg = compose(config_name=config_name)

        if "ciagen" in cfg:
            cfg = cfg["ciagen"]
        if "conf" in cfg:
            cfg = cfg["conf"]

        self.vocabulary = cfg["vocabulary"]
        self.prompt_templates = cfg["prompt_templates"]

    def _change_token(self, token):  # : str) -> str:
        # Search in the vocabulary for words in the same family, pick one
        # if possible and return it
        for vocabulary_class in self.vocabulary:
            if token in self.vocabulary[vocabulary_class]:
                other_tokens = list(filter(lambda word: token != word, self.vocabulary[vocabulary_class]))
                if other_tokens:
                    return random.choice(other_tokens)
        return token

    def prompts(self, num_prompts, phrase):  # : int, phrase: str) -> List[str]:
        """
        Generates unique and different prompts in the format of the given phrase.
        It works by tokenizing the given phrase and changing each token whenever
        a match is found

        :param int num_prompts: maximum number of prompts created
        :param str phrase: the phrase to be changed

        :return: all generated new phrases
        :rtype: list
        """

        phrase_list = []
        phrase_tokens = phrase.lower().split()

        for _ in range(num_prompts):
            # For each word in the phrase try to change it with one in the same class.
            new_phrase = [self._change_token(token) for token in phrase_tokens]
            new_phrase = " ".join(new_phrase)
            if new_phrase not in phrase_list:
                phrase_list.append(new_phrase)

        return phrase_list

    def max_template_prompts(self):  # -> int:
        """
        Counts the maximum number of template prompts that can be generated
        with the exisitng prompt_templates

        :return: maximum number of prompts that can be generated from prompt_templates
        :rtype: int
        """

        counter = 0
        for phrase in self.prompt_templates:
            phrase_counter = 1
            phrase_list = phrase.split()
            for phrase_word in phrase_list:
                vocab_counter = 0
                if "color" in phrase_word:
                    vocab_counter = len(self.vocabulary["color"])
                elif "opt" in phrase_word:
                    vocab_counter = len(self.vocabulary[phrase_word[4:]])
                if vocab_counter > 0:
                    phrase_counter = phrase_counter * vocab_counter
                    continue
            counter = counter + phrase_counter

        return counter

    def template_prompts(self, num_prompts):  # : int) -> list:
        """
        # Generates unique prompts from the template prompts
        Args: num_prompts; int; number of prompts that are required
        Returns: phrases; list; list of prompts
        """

        phrases = []
        num_phrases = min(num_prompts, self.max_template_prompts())

        while len(phrases) < num_phrases:
            phrase = random.choice(self.prompt_templates)
            color_count = phrase.count("opt_color")
            new_phrase = [
                phrase.replace("opt_gender", random.choice(self.vocabulary["gender"]))
                .replace("opt_age", random.choice(self.vocabulary["age"]))
                .replace("opt_size", random.choice(self.vocabulary["size"]))
                .replace("opt_height", random.choice(self.vocabulary["height"]))
                .replace("opt_clothes_top", random.choice(self.vocabulary["clothes_top"]))
                .replace(
                    "opt_clothes_bottom",
                    random.choice(self.vocabulary["clothes_bottom"]),
                )
                .replace("opt_accessories", random.choice(self.vocabulary["accessories"]))
                .replace("opt_ground", random.choice(self.vocabulary["ground"]))
                .replace("opt_background", random.choice(self.vocabulary["background"]))
            ]

            for c in range(color_count + 1):
                new_phrase = [new_phrase[0].replace(f"opt_color{c}", random.choice(self.vocabulary["color"]))]

            if new_phrase[0] not in phrases:
                phrases.extend(new_phrase)

        return phrases
