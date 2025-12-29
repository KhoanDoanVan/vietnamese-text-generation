import re
import unicodedata
from abc import ABC, abstractmethod



class TextNormalizer(ABC):
    @abstractmethod
    def normalize(
        self,
        text: str
    ) -> str:
        pass



class VietnameseNormalizer(TextNormalizer):

    def __init__(
            self,
            lowercase: bool = True,
            remove_tone: bool = False
    ):
        self.lowercase = lowercase
        self.remove_tone = remove_tone


    def normalize(
            self, 
            text: str
    ) -> str:
        text = unicodedata.normalize("NFC", text)


        if self.lowercase:
            text = text.lower()

        if self.remove_tone:
            text = self._remove_tone(text)


        text = re.sub(r"\s+", " ", text).strip()

        return text
    


    def _remove_tone(
            self,
            text: str
    ) -> str:
        
        tone_map = {
            "à|á|ả|ã|ạ|ă|ằ|ắ|ẳ|ẵ|ặ|â|ầ|ấ|ẩ|ẫ|ậ": "a",
            "è|é|ẻ|ẽ|ẹ|ê|ề|ế|ể|ễ|ệ": "e",
            "ì|í|ỉ|ĩ|ị": "i",
            "ò|ó|ỏ|õ|ọ|ô|ồ|ố|ổ|ỗ|ộ|ơ|ờ|ớ|ở|ỡ|ợ": "o",
            "ù|ú|ủ|ũ|ụ|ư|ừ|ứ|ử|ữ|ự": "u",
            "ỳ|ý|ỷ|ỹ|ỵ": "y",
            "đ": "d",
        }

        for pattern, repl in tone_map.items():
            text = re.sub(pattern, repl, text)

        return text