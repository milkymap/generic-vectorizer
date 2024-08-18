from abc import ABC, abstractmethod 
from typing import Any 

from google.protobuf import message as _message

class ABCStrategy(ABC):
    def __int__(self):
        pass 
        
    @abstractmethod
    def process(self, task_type:bytes, encoded_message:bytes) -> _message.Message:
        pass 